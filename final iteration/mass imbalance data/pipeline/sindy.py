"""
sindy.py – Standard SINDy fitting with STLSQ / SR3 / SSR / FROLS
            plus sklearn-based LASSO / Ridge / ElasticNet.

Fits qddot_i = Theta(q, qdot, qddot_j, qddot_k) @ xi
for each DOF independently.
"""

import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import warnings

from .preprocessing import Scalers, fit_scalers, apply_scalers, inverse_scale_y


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SINDyResult:
    """Result from fitting a single acceleration DOF."""
    dof: int
    dof_name: str
    model: Any                   # ps.SINDy or sklearn estimator
    scalers: Scalers
    feature_names: List[str]
    equation: str
    coefs: np.ndarray
    n_active_terms: int
    mse_train: float
    mse_val: float
    mse_test: float
    y_true: np.ndarray          # (N,) full timeline
    y_pred: np.ndarray          # (N,) full timeline (NaN where no data)
    optimizer_name: str = ""
    threshold: float = 0.0
    unit_filtered: bool = False


@dataclass
class SINDyFit:
    """Collection of 3 DOF results from one SINDy experiment."""
    results: Dict[str, SINDyResult] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def summary_table(self) -> str:
        opt = self.config.get("optimizer", "?")
        th = self.config.get("threshold", "?")
        uf = self.config.get("unit_filtered", False)
        lib_tag = "Unit-Consistent" if uf else "Full"
        lines = [
            f"  Optimizer: {opt}  |  Threshold: {th}  |  Library: {lib_tag}",
            "",
            f"  {'DOF':<12} {'#terms':>6} {'MSE_train':>12} {'MSE_val':>12} {'MSE_test':>12} {'Unit%':>6}",
            "  " + "-" * 68,
        ]
        from .unit_filter import unit_consistency_score
        for name, r in self.results.items():
            uc = unit_consistency_score(r.feature_names, r.coefs, r.dof)
            lines.append(f"  {name:<12} {r.n_active_terms:>6} "
                         f"{r.mse_train:>12.4e} {r.mse_val:>12.4e} {r.mse_test:>12.4e} "
                         f"{uc:>5.0%}")
        lines.append("")
        lines.append("  Discovered Equations:")
        from . import sindy as _s
        for name, r in self.results.items():
            lines.append(f"    {_s.ACCEL_NAMES[r.dof]} = {r.equation}")
        return "\n".join(lines)


# ── Optimizer factory ─────────────────────────────────────────────────────────

PYSINDY_PRESETS = {
    "STLSQ": lambda th: ps.STLSQ(threshold=th),
    "SR3_L0": lambda th: ps.SR3(reg_weight_lam=th, regularizer="L0"),
    "SR3_L1": lambda th: ps.SR3(reg_weight_lam=th, regularizer="L1"),
    "SR3_L2": lambda th: ps.SR3(reg_weight_lam=th, regularizer="L2"),
    "SSR_coef": lambda _: ps.SSR(criteria="coefficient_value"),
    "SSR_resid": lambda _: ps.SSR(criteria="model_residual"),
    "FROLS": lambda th: ps.FROLS(max_iter=int(th) if th > 1 else 10),
}

SKLEARN_PRESETS = {
    "LASSO": lambda th: Lasso(alpha=th, max_iter=10000, fit_intercept=False),
    "Ridge": lambda th: Ridge(alpha=th, fit_intercept=False),
    "ElasticNet": lambda th: ElasticNet(alpha=th, l1_ratio=0.5,
                                         max_iter=10000, fit_intercept=False),
}

OPTIMIZER_PRESETS = {**PYSINDY_PRESETS, **SKLEARN_PRESETS}


def _is_sklearn_optimizer(name: str) -> bool:
    return name in SKLEARN_PRESETS


def make_optimizer(name: str, threshold: float):
    factory = OPTIMIZER_PRESETS.get(name)
    if factory is None:
        raise ValueError(f"Unknown optimizer: {name}. "
                         f"Choose from {list(OPTIMIZER_PRESETS)}")
    return factory(threshold)


def _sklearn_equation(coefs: np.ndarray, feature_names: List[str],
                      tol: float = 1e-10) -> str:
    """Build equation string from sklearn coefficient vector."""
    terms = []
    for c, n in zip(coefs, feature_names):
        if abs(c) > tol:
            terms.append(f"{c:+.6f} {n}")
    return " ".join(terms) if terms else "0"


# ── Core fitting ──────────────────────────────────────────────────────────────

ACCEL_NAMES = ["xddot", "yddot", "thetaddot"]
DOF_LABELS = ["a_x", "a_y", "a_theta"]


def fit_single_dof(
    Theta: np.ndarray,           # (N, n_features) raw feature matrix
    feature_names: List[str],
    y: np.ndarray,               # (N,) target acceleration
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    dof: int,
    dt: float,
    optimizer_name: str = "STLSQ",
    threshold: float = 0.18,
    unit_filtered: bool = False,
) -> SINDyResult:
    """Fit SINDy for one acceleration DOF."""

    X_tr, X_va, X_te = Theta[idx_train], Theta[idx_val], Theta[idx_test]
    y_tr, y_va, y_te = y[idx_train], y[idx_val], y[idx_test]

    scalers = fit_scalers(X_tr, y_tr)
    X_tr_n, y_tr_n = apply_scalers(scalers, X_tr, y_tr)
    X_va_n = apply_scalers(scalers, X_va)
    X_te_n = apply_scalers(scalers, X_te)

    if _is_sklearn_optimizer(optimizer_name):
        # ── sklearn path ──
        model = make_optimizer(optimizer_name, threshold)
        model.fit(X_tr_n, y_tr_n)
        coefs = model.coef_.flatten()

        p_tr = inverse_scale_y(scalers, model.predict(X_tr_n))
        p_va = inverse_scale_y(scalers, model.predict(X_va_n))
        p_te = inverse_scale_y(scalers, model.predict(X_te_n))

        equation = _sklearn_equation(coefs, feature_names)

    else:
        # ── PySINDy path ──
        lib = ps.IdentityLibrary()
        opt = make_optimizer(optimizer_name, threshold)
        model = ps.SINDy(feature_library=lib, optimizer=opt)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_n, t=dt,
                      x_dot=y_tr_n.reshape(-1, 1),
                      feature_names=feature_names)

        p_tr = inverse_scale_y(scalers, np.array(model.predict(X_tr_n)).flatten())
        p_va = inverse_scale_y(scalers, np.array(model.predict(X_va_n)).flatten())
        p_te = inverse_scale_y(scalers, np.array(model.predict(X_te_n)).flatten())

        coefs = model.optimizer.coef_.flatten()
        equation = model.equations()[0] if hasattr(model, "equations") else ""

    N = len(y)
    y_pred = np.full(N, np.nan)
    y_pred[idx_train] = p_tr
    y_pred[idx_val] = p_va
    y_pred[idx_test] = p_te

    return SINDyResult(
        dof=dof,
        dof_name=DOF_LABELS[dof],
        model=model,
        scalers=scalers,
        feature_names=feature_names,
        equation=equation,
        coefs=coefs,
        n_active_terms=int(np.count_nonzero(coefs)),
        mse_train=mean_squared_error(y_tr, p_tr),
        mse_val=mean_squared_error(y_va, p_va),
        mse_test=mean_squared_error(y_te, p_te),
        y_true=y,
        y_pred=y_pred,
        optimizer_name=optimizer_name,
        threshold=threshold,
        unit_filtered=unit_filtered,
    )


def fit_all_dofs(
    q: np.ndarray,
    u: np.ndarray,
    qddot: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    dt: float,
    build_library_fn,
    optimizer_name: str = "STLSQ",
    threshold: float = 0.18,
    unit_filtered: bool = False,
    include_sign_qdot: bool = False,
) -> SINDyFit:
    """Fit SINDy for all 3 acceleration DOFs.

    Parameters
    ----------
    build_library_fn : callable(q, u, qddot, target_dof, **kwargs) -> (Theta, names)
    """
    fit = SINDyFit(config={
        "optimizer": optimizer_name,
        "threshold": threshold,
        "unit_filtered": unit_filtered,
    })

    lib_tag = "Unit-Consistent" if unit_filtered else "Full"
    print(f"  [{optimizer_name} | th={threshold} | {lib_tag}]")

    for dof in range(3):
        Theta, names = build_library_fn(
            q, u, qddot, target_dof=dof,
            include_sign_qdot=include_sign_qdot,
        )
        y = qddot[:, dof]
        result = fit_single_dof(
            Theta, names, y,
            idx_train, idx_val, idx_test,
            dof=dof, dt=dt,
            optimizer_name=optimizer_name,
            threshold=threshold,
            unit_filtered=unit_filtered,
        )
        fit.results[DOF_LABELS[dof]] = result
        print(f"    {DOF_LABELS[dof]}: {result.n_active_terms} terms | "
              f"MSE train={result.mse_train:.4e} val={result.mse_val:.4e} "
              f"test={result.mse_test:.4e}")
        print(f"      EQN: {ACCEL_NAMES[dof]} = {result.equation}")

    return fit
