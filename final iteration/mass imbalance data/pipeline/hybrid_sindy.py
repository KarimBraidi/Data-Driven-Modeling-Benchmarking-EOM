"""
hybrid_sindy.py – Hybrid (regime-switching) SINDy.

Trains separate SINDy models per contact regime (flight, rolling, bouncing)
and uses a switching rule based on contact force lambda_N to select the
correct model at each timestep.

Each regime gets its own set of 3 equations (one per DOF), trained on
pooled data from all runs of that regime.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable

from .data_loader import RunData, load_run
from .preprocessing import regime_aware_split, temporal_split_frac
from .sindy import fit_all_dofs, SINDyFit, SINDyResult, DOF_LABELS, ACCEL_NAMES


# ── Regime classification ────────────────────────────────────────────────────

REGIME_LABELS = ["flight", "rolling", "bouncing"]


def classify_regime(lambda_N: np.ndarray,
                    lambda_F: np.ndarray,
                    slip_threshold: float = 0.01,
                    ) -> np.ndarray:
    """Classify each timestep into a regime based on contact forces.

    Returns
    -------
    labels : (N,) array of ints  — 0=flight, 1=rolling, 2=bouncing
    """
    N = len(lambda_N)
    labels = np.zeros(N, dtype=int)

    in_contact = np.abs(lambda_N) > 1e-8

    # Rolling: in contact and friction force is small (no slipping)
    rolling = in_contact & (np.abs(lambda_F) < slip_threshold)
    # Bouncing/sliding: in contact with significant friction
    bouncing = in_contact & ~rolling

    labels[rolling] = 1
    labels[bouncing] = 2
    # flight stays 0

    return labels


def classify_regime_simple(lambda_N: np.ndarray) -> np.ndarray:
    """Simple classification: flight (lambda_N=0) vs contact (lambda_N>0).

    For datasets that are purely one regime, this maps the whole run
    to a single label based on the dominant regime.
    """
    in_contact = np.abs(lambda_N) > 1e-8
    frac_contact = np.mean(in_contact)

    if frac_contact < 0.1:
        return np.zeros(len(lambda_N), dtype=int)   # flight
    elif frac_contact > 0.9:
        return np.ones(len(lambda_N), dtype=int)    # rolling/contact
    else:
        return 2 * np.ones(len(lambda_N), dtype=int)  # bouncing (intermittent)


# ── Data pooling ─────────────────────────────────────────────────────────────

@dataclass
class PooledRegimeData:
    """Pooled data from multiple runs of the same regime."""
    regime: str
    q: np.ndarray          # (N_total, 3)
    u: np.ndarray          # (N_total, 3)
    qddot: np.ndarray      # (N_total, 3)
    lambda_N: np.ndarray   # (N_total,)
    lambda_F: np.ndarray   # (N_total,)
    dt: float
    N: int
    run_names: List[str]
    run_boundaries: List[int]   # cumulative sample counts


def pool_runs(run_names: List[str]) -> PooledRegimeData:
    """Load and concatenate multiple runs into one dataset."""
    qs, us, qddots, lNs, lFs = [], [], [], [], []
    boundaries = [0]
    dt = None

    for rn in run_names:
        rd = load_run(rn)
        qs.append(rd.q)
        us.append(rd.u)
        qddots.append(rd.qddot)
        lNs.append(rd.lambda_N)
        lFs.append(rd.lambda_F)
        boundaries.append(boundaries[-1] + rd.N)
        if dt is None:
            dt = rd.dt

    q = np.vstack(qs)
    u = np.vstack(us)
    qddot = np.vstack(qddots)
    lN = np.concatenate(lNs)
    lF = np.concatenate(lFs)

    regime = "flight" if "flight" in run_names[0] else \
             "rolling" if "rolling" in run_names[0] else "bouncing"

    return PooledRegimeData(
        regime=regime, q=q, u=u, qddot=qddot,
        lambda_N=lN, lambda_F=lF,
        dt=dt, N=q.shape[0],
        run_names=run_names,
        run_boundaries=boundaries,
    )


# ── Hybrid SINDy result ─────────────────────────────────────────────────────

@dataclass
class HybridSINDyResult:
    """Container for a full hybrid SINDy model."""
    regime_models: Dict[str, SINDyFit]       # regime_name -> SINDyFit
    regime_configs: Dict[str, dict]           # regime_name -> {optimizer, threshold, ...}
    regime_run_names: Dict[str, List[str]]    # which runs trained each regime

    def summary(self) -> str:
        lines = ["=" * 70, "  HYBRID SINDy SUMMARY", "=" * 70]
        for regime in REGIME_LABELS:
            if regime not in self.regime_models:
                continue
            fit = self.regime_models[regime]
            cfg = self.regime_configs[regime]
            runs = self.regime_run_names[regime]
            lines.append(f"\n  Regime: {regime.upper()}")
            lines.append(f"  Trained on: {', '.join(runs)}")
            lines.append(f"  Optimizer: {cfg['optimizer']}  |  "
                         f"Threshold: {cfg['threshold']}")
            lines.append(fit.summary_table())
        return "\n".join(lines)


# ── Hybrid prediction on a new run ───────────────────────────────────────────

def hybrid_predict(
    hybrid_result: HybridSINDyResult,
    test_run: RunData,
    build_library_fn: Callable,
    include_sign_qdot: bool = False,
) -> Dict[str, np.ndarray]:
    """Predict accelerations on a test run using regime-switching.

    Returns dict with keys 'pred' (N,3), 'regime_labels' (N,), and
    per-DOF MSE.
    """
    labels = classify_regime_simple(test_run.lambda_N)
    pred = np.zeros_like(test_run.qddot)

    for regime_id, regime_name in enumerate(REGIME_LABELS):
        mask = labels == regime_id
        if not np.any(mask):
            continue
        if regime_name not in hybrid_result.regime_models:
            continue

        fit = hybrid_result.regime_models[regime_name]
        idx = np.where(mask)[0]

        for dof in range(3):
            dof_name = DOF_LABELS[dof]
            res = fit.results[dof_name]

            Theta, _ = build_library_fn(
                test_run.q, test_run.u, test_run.qddot,
                target_dof=dof,
                include_sign_qdot=include_sign_qdot,
            )

            from .preprocessing import apply_scalers
            X_n = apply_scalers(res.scalers, Theta[idx])
            from .preprocessing import inverse_scale_y

            if hasattr(res.model, 'predict'):
                p = res.model.predict(X_n)
                if hasattr(p, 'flatten'):
                    p = p.flatten()
            else:
                p = X_n @ res.coefs

            pred[idx, dof] = inverse_scale_y(res.scalers, p)

    # Per-DOF MSE
    mse = {}
    for dof in range(3):
        mse[DOF_LABELS[dof]] = float(
            mean_squared_error(test_run.qddot[:, dof], pred[:, dof]))

    return {
        'pred': pred,
        'regime_labels': labels,
        'mse': mse,
    }


# ── Main fitting function ───────────────────────────────────────────────────

# Default regime → run mapping
DEFAULT_REGIME_RUNS = {
    "flight":   ["flight_1", "flight_2"],
    "rolling":  ["rolling_1", "rolling_2"],
    "bouncing": ["bouncing_ball_1", "bouncing_ball_2", "bouncing_ball_3"],
}


def fit_hybrid_sindy(
    regime_runs: Dict[str, List[str]] = None,
    build_library_fn: Callable = None,
    optimizer_name: str = "STLSQ",
    threshold: float = 0.18,
    unit_filtered: bool = False,
    include_sign_qdot: bool = False,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    regime_configs: Dict[str, dict] = None,
) -> HybridSINDyResult:
    """Fit separate SINDy models per regime.

    Parameters
    ----------
    regime_runs : dict mapping regime name -> list of run names to pool
    build_library_fn : callable(q, u, qddot, target_dof, ...) -> (Theta, names)
    optimizer_name : default optimizer for all regimes
    threshold : default threshold for all regimes
    regime_configs : optional per-regime overrides, e.g.
        {"flight": {"optimizer": "STLSQ", "threshold": 0.1},
         "rolling": {"optimizer": "LASSO", "threshold": 0.01}}
    """
    if regime_runs is None:
        regime_runs = DEFAULT_REGIME_RUNS
    if regime_configs is None:
        regime_configs = {}

    result = HybridSINDyResult(
        regime_models={},
        regime_configs={},
        regime_run_names=regime_runs,
    )

    for regime, runs in regime_runs.items():
        print(f"\n{'='*60}")
        print(f"  REGIME: {regime.upper()}  |  Runs: {', '.join(runs)}")
        print(f"{'='*60}")

        # Get per-regime config (or use defaults)
        cfg = regime_configs.get(regime, {})
        opt = cfg.get("optimizer", optimizer_name)
        th = cfg.get("threshold", threshold)
        uf = cfg.get("unit_filtered", unit_filtered)
        sign_qdot = cfg.get("include_sign_qdot", include_sign_qdot)

        result.regime_configs[regime] = {
            "optimizer": opt, "threshold": th,
            "unit_filtered": uf, "include_sign_qdot": sign_qdot,
        }

        # Pool data
        pooled = pool_runs(runs)
        print(f"  Pooled {pooled.N} samples from {len(runs)} runs")

        # Split
        sp = temporal_split_frac(pooled.N, train_frac=train_frac,
                                 val_frac=val_frac)

        # Fit
        fit = fit_all_dofs(
            pooled.q, pooled.u, pooled.qddot,
            sp.idx_train, sp.idx_val, sp.idx_test,
            dt=pooled.dt,
            build_library_fn=build_library_fn,
            optimizer_name=opt,
            threshold=th,
            unit_filtered=uf,
            include_sign_qdot=sign_qdot,
        )

        result.regime_models[regime] = fit

    return result


def sweep_hybrid_sindy(
    optimizer_configs: List[dict],
    regime_runs: Dict[str, List[str]] = None,
    build_library_fn: Callable = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Dict[str, HybridSINDyResult]:
    """Sweep over multiple optimizer/threshold combos for hybrid SINDy.

    Parameters
    ----------
    optimizer_configs : list of dicts, each with keys:
        "name": str label
        "regime_configs": dict mapping regime -> {"optimizer": ..., "threshold": ...}
        OR "optimizer"/"threshold" for uniform config across regimes

    Returns
    -------
    results : dict mapping config name -> HybridSINDyResult
    """
    results = {}

    for cfg in optimizer_configs:
        name = cfg["name"]
        print(f"\n{'#'*70}")
        print(f"#  CONFIG: {name}")
        print(f"{'#'*70}")

        if "regime_configs" in cfg:
            rc = cfg["regime_configs"]
        else:
            opt = cfg.get("optimizer", "STLSQ")
            th = cfg.get("threshold", 0.18)
            rc = {r: {"optimizer": opt, "threshold": th}
                  for r in (regime_runs or DEFAULT_REGIME_RUNS)}

        res = fit_hybrid_sindy(
            regime_runs=regime_runs,
            build_library_fn=build_library_fn,
            regime_configs=rc,
            train_frac=train_frac,
            val_frac=val_frac,
        )
        results[name] = res

    return results
