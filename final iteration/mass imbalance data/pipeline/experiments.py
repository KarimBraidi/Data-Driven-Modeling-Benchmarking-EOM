"""
experiments.py – Hyperparameter sweeps and experiment orchestration.

Runs systematic experiments across:
  - regimes (bouncing, rolling, flight)
  - optimizers (STLSQ, SR3, SSR, FROLS)
  - thresholds
  - library types (full vs unit-consistent)
  - smoothing parameters
  - with/without unit filtering

Logs all results for comparison.
"""

import numpy as np
import itertools
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .data_loader import RunData, load_run, load_regime, REGIME_FOLDERS
from .preprocessing import (
    smooth_savgol, validate_derivatives, numerical_acceleration,
    temporal_split, temporal_split_frac, regime_aware_split, TemporalSplit,
)
from .library import (
    build_features, build_full_state_features,
    build_unit_consistent_features,
)
from .sindy import fit_all_dofs, SINDyFit, DOF_LABELS
from .lagrange_sindy import fit_lagrange_sindy
from .evaluation import (
    ModelComparison, unit_consistency, structural_accuracy, parsimony,
)
from .unit_filter import unit_consistency_score


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    run_name: str
    library_type: str              # "full" or "unit_consistent"
    optimizer: str                 # "STLSQ", "SR3_L0", "LASSO", etc.
    threshold: float
    degree: int = 2                # polynomial degree for library
    smoothing: str = "none"        # "none", "savgol"
    savgol_window: int = 11
    savgol_polyorder: int = 3
    use_simulator_accel: bool = True  # True = X_save[0:3], False = finite diff
    include_sign_qdot: bool = False  # for slipping regime
    split_mode: str = "regime"       # "time", "frac", or "regime"
    t_train_end: float = 3.25
    t_val_end: float = 3.75
    train_frac: float = 0.60
    val_frac: float = 0.20


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    sindy_fit: Optional[SINDyFit] = None
    lagrange_result: Optional[object] = None
    deriv_report: Optional[dict] = None


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Run a single SINDy experiment."""
    run = load_run(cfg.run_name)
    q, u, dt, N, t = run.q, run.u, run.dt, run.N, run.t

    # ── Smoothing ──
    if cfg.smoothing == "savgol":
        q = smooth_savgol(q, window=cfg.savgol_window, polyorder=cfg.savgol_polyorder)
        u = smooth_savgol(u, window=cfg.savgol_window, polyorder=cfg.savgol_polyorder)

    # ── Accelerations ──
    if cfg.use_simulator_accel:
        qddot = run.qddot
    else:
        qddot = numerical_acceleration(u, dt)

    # ── Derivative validation ──
    deriv_report, _ = validate_derivatives(run.qddot, u, dt)

    # ── Temporal split ──
    if cfg.split_mode == "time":
        split = temporal_split(N, t, cfg.t_train_end, cfg.t_val_end)
    elif cfg.split_mode == "regime":
        split = regime_aware_split(N, run.lambda_N, cfg.train_frac, cfg.val_frac)
    else:
        split = temporal_split_frac(N, cfg.train_frac, cfg.val_frac)

    # ── Library selection ──
    if cfg.library_type == "unit_consistent":
        lib_fn = lambda q, u, qddot, target_dof, **kw: build_unit_consistent_features(
            q, u, qddot, target_dof, degree=cfg.degree,
            include_sign_qdot=kw.get("include_sign_qdot", False))
        unit_filtered = True
    else:
        lib_fn = lambda q, u, qddot, target_dof, **kw: build_full_state_features(
            q, u, qddot, target_dof, degree=cfg.degree,
            include_sign_qdot=kw.get("include_sign_qdot", False))
        unit_filtered = False

    # ── Fit ──
    sindy_fit = fit_all_dofs(
        q, u, qddot,
        split.idx_train, split.idx_val, split.idx_test,
        dt=dt,
        build_library_fn=lib_fn,
        optimizer_name=cfg.optimizer,
        threshold=cfg.threshold,
        unit_filtered=unit_filtered,
        include_sign_qdot=cfg.include_sign_qdot,
    )

    return ExperimentResult(
        config=cfg,
        sindy_fit=sindy_fit,
        deriv_report=deriv_report,
    )


def threshold_sweep(run_name: str, library_type: str, optimizer: str,
                    thresholds: List[float],
                    **kwargs) -> List[ExperimentResult]:
    """Sweep over thresholds for a fixed configuration."""
    results = []
    for th in thresholds:
        cfg = ExperimentConfig(
            run_name=run_name,
            library_type=library_type,
            optimizer=optimizer,
            threshold=th,
            **kwargs,
        )
        print(f"\n{'='*60}  {optimizer} threshold={th} ({library_type})")
        res = run_experiment(cfg)
        results.append(res)
    return results


def optimizer_sweep(run_name: str, library_type: str,
                    configs: List[Dict],
                    **kwargs) -> List[ExperimentResult]:
    """Sweep over multiple optimizer configurations.

    configs : list of dicts with keys "optimizer" and "threshold"
    """
    results = []
    for c in configs:
        cfg = ExperimentConfig(
            run_name=run_name,
            library_type=library_type,
            optimizer=c["optimizer"],
            threshold=c["threshold"],
            **kwargs,
        )
        print(f"\n{'='*60}  {c['optimizer']} th={c['threshold']} ({library_type})")
        res = run_experiment(cfg)
        results.append(res)
    return results


def full_comparison(run_name: str, thresholds: List[float] = None,
                    **kwargs) -> ModelComparison:
    """Run full vs unit-consistent comparison and build comparison table."""
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.3]

    comp = ModelComparison()

    for lib_type in ["full", "unit_consistent"]:
        for th in thresholds:
            cfg = ExperimentConfig(
                run_name=run_name,
                library_type=lib_type,
                optimizer="STLSQ",
                threshold=th,
                **kwargs,
            )
            print(f"\n{'='*60}  STLSQ th={th} ({lib_type})")
            res = run_experiment(cfg)

            for dof_name, r in res.sindy_fit.results.items():
                uc = unit_consistency_score(r.feature_names, r.coefs, r.dof)
                comp.add(
                    label=f"{lib_type}/STLSQ/{th}",
                    dof=dof_name,
                    n_terms=r.n_active_terms,
                    mse_train=r.mse_train,
                    mse_val=r.mse_val,
                    mse_test=r.mse_test,
                    unit_score=uc,
                    equation=r.equation,
                )

    return comp


# ── Full experiment grid ──────────────────────────────────────────────────────

@dataclass
class GridResult:
    """Stores all results from a full grid sweep."""
    results: List[ExperimentResult] = field(default_factory=list)

    def summary_rows(self) -> List[Dict]:
        """Flatten results into one row per DOF for easy tabulation."""
        rows = []
        for er in self.results:
            if er.sindy_fit is None:
                continue
            cfg = er.config
            for dof_name, r in er.sindy_fit.results.items():
                uc = unit_consistency_score(r.feature_names, r.coefs, r.dof)
                rows.append({
                    "run": cfg.run_name,
                    "library": cfg.library_type,
                    "optimizer": cfg.optimizer,
                    "threshold": cfg.threshold,
                    "degree": cfg.degree,
                    "dof": dof_name,
                    "n_terms": r.n_active_terms,
                    "mse_train": r.mse_train,
                    "mse_val": r.mse_val,
                    "mse_test": r.mse_test,
                    "unit_score": uc,
                    "equation": r.equation,
                })
        return rows

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.summary_rows())


def run_full_grid(
    run_name: str,
    library_types: List[str] = None,
    optimizers: List[str] = None,
    thresholds: List[float] = None,
    degrees: List[int] = None,
    verbose: bool = True,
    **kwargs,
) -> GridResult:
    """Run the full experiment grid: library × optimizer × threshold × degree.

    Parameters
    ----------
    run_name : which simulation run to use
    library_types : ["full", "unit_consistent"]
    optimizers : ["STLSQ", "LASSO", "Ridge", "ElasticNet", ...]
    thresholds : list of threshold / alpha values
    degrees : polynomial degrees [2, 3]
    **kwargs : extra fields passed to ExperimentConfig
    """
    if library_types is None:
        library_types = ["full", "unit_consistent"]
    if optimizers is None:
        optimizers = ["STLSQ", "LASSO", "Ridge", "ElasticNet"]
    if thresholds is None:
        thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    if degrees is None:
        degrees = [2, 3]

    grid = list(itertools.product(library_types, optimizers, thresholds, degrees))
    gr = GridResult()

    for i, (lib, opt, th, deg) in enumerate(grid):
        if verbose:
            print(f"\n[{i+1}/{len(grid)}] {lib} | {opt} | th={th} | deg={deg}")
        cfg = ExperimentConfig(
            run_name=run_name,
            library_type=lib,
            optimizer=opt,
            threshold=th,
            degree=deg,
            **kwargs,
        )
        try:
            res = run_experiment(cfg)
            gr.results.append(res)
        except Exception as e:
            if verbose:
                print(f"  FAILED: {e}")

    if verbose:
        print(f"\nGrid complete: {len(gr.results)}/{len(grid)} succeeded")
    return gr


def run_lagrange_experiment(run_name: str, alpha: float = 0.01,
                            **kwargs) -> ExperimentResult:
    """Run Lagrange SINDy on a single run."""
    run = load_run(run_name)
    q, u, dt, N, t = run.q, run.u, run.dt, run.N, run.t
    qddot = run.qddot

    split_mode = kwargs.get("split_mode", "regime")
    if split_mode == "time":
        split = temporal_split(N, t,
                               kwargs.get("t_train_end", 3.25),
                               kwargs.get("t_val_end", 3.75))
    elif split_mode == "regime":
        split = regime_aware_split(N, run.lambda_N,
                                   kwargs.get("train_frac", 0.60),
                                   kwargs.get("val_frac", 0.20))
    else:
        split = temporal_split_frac(N, kwargs.get("train_frac", 0.60),
                                    kwargs.get("val_frac", 0.20))

    print(f"\n{'='*60}  Lagrange SINDy (alpha={alpha})")
    lag_result = fit_lagrange_sindy(
        q, u, qddot,
        split.idx_train, split.idx_val, split.idx_test,
        alpha=alpha,
        lambda_N=run.lambda_N,
        lambda_F=run.lambda_F,
    )

    cfg = ExperimentConfig(
        run_name=run_name, library_type="lagrangian",
        optimizer="Lasso", threshold=alpha,
    )
    return ExperimentResult(config=cfg, lagrange_result=lag_result)
