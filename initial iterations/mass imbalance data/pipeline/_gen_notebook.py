#!/usr/bin/env python3
"""Generate the reworked SINDy_Pipeline.ipynb notebook."""
import json, os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SINDy_Pipeline.ipynb')

cells = []

def md(text):
    lines = text.strip('\n').split('\n')
    fmt = [l + '\n' for l in lines[:-1]] + [lines[-1]] if lines else []
    cells.append({"cell_type": "markdown", "metadata": {}, "source": fmt})

def code(text):
    lines = text.strip('\n').split('\n')
    fmt = [l + '\n' for l in lines[:-1]] + [lines[-1]] if lines else []
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": fmt})

# ═══════════════════════════════════════════════════════════════════════════
# CELL 1: Title
# ═══════════════════════════════════════════════════════════════════════════
md("""\
# SINDy EOM Discovery Pipeline — Full Sweep & Comparison

**3 Algorithms**: Standard SINDy · Lagrange SINDy · Hybrid SINDy

**4 Libraries**: Full · Unit-Consistent (UC) · Full + λ_N/λ_F · UC + λ_N/λ_F

**4 Optimizers**: STLSQ · LASSO · Ridge · ElasticNet

**Thresholds**: 10⁻⁴ → 0.5

**3 Metrics**: MSE · ATE · R²""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 2: Imports
# ═══════════════════════════════════════════════════════════════════════════
code("""\
import sys, os, warnings, importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML
warnings.filterwarnings('ignore')

# Ensure pipeline is importable
_nb_dir = os.path.abspath('.')
if os.path.basename(_nb_dir) == 'pipeline':
    sys.path.insert(0, os.path.dirname(_nb_dir))
else:
    sys.path.insert(0, _nb_dir)

from pipeline.data_loader import load_run, load_all, REGIME_FOLDERS
from pipeline.preprocessing import (
    smooth_savgol, validate_derivatives, numerical_acceleration,
    temporal_split_frac, regime_aware_split,
)
from pipeline.library import build_unit_consistent_features, build_full_state_features
from pipeline.sindy import (
    fit_all_dofs, fit_single_dof, SINDyFit, SINDyResult,
    DOF_LABELS, ACCEL_NAMES, OPTIMIZER_PRESETS,
)
from pipeline.lagrange_sindy import fit_lagrange_sindy_shared
from pipeline.hybrid_sindy import (
    fit_hybrid_sindy, hybrid_predict, pool_runs,
    DEFAULT_REGIME_RUNS, REGIME_LABELS, HybridSINDyResult,
)
from pipeline.evaluation import (
    compute_open_loop_ate, compute_lagrange_ate, lagrange_predict_qddot,
    EXPECTED_TERMS, PHYSICAL_PARAMS,
)
from pipeline.unit_filter import unit_consistency_score, filter_library, BASE_DIMS

# Reload modules
for _m in ['pipeline.unit_filter', 'pipeline.library', 'pipeline.sindy',
           'pipeline.lagrange_sindy', 'pipeline.hybrid_sindy', 'pipeline.evaluation']:
    importlib.reload(sys.modules.get(_m, importlib.import_module(_m)))
from pipeline.library import build_unit_consistent_features, build_full_state_features
from pipeline.sindy import fit_all_dofs, fit_single_dof, DOF_LABELS, ACCEL_NAMES

# ── ATE grading ──
def ate_grade(ate):
    if ate < 0.01: return 'A'
    if ate < 0.05: return 'B'
    if ate < 0.15: return 'C'
    if ate < 0.50: return 'D'
    return 'F'

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0

print('Pipeline loaded.')
print('ATE grading: A<0.01  B<0.05  C<0.15  D<0.50  F>=0.50')""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 3: Data Loading
# ═══════════════════════════════════════════════════════════════════════════
md("## §1 Data Loading & Preprocessing")

code("""\
all_runs = load_all()
for regime, runs in all_runs.items():
    print(f'\\n{regime.upper()} ({len(runs)} runs):')
    for run in runs:
        print(f'  {run.name}: N={run.N}, dt={run.dt:.6f}, t_max={run.t[-1]:.3f}s')

# Derivative validation
print('\\n=== Derivative Validation ===')
for regime, runs in all_runs.items():
    for run in runs:
        report, _ = validate_derivatives(run.qddot, run.u, run.dt)
        worst_corr = min(v['correlation'] for v in report.values())
        print(f'  {run.name}: worst corr={worst_corr:.4f}')""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 4: Sweep Configuration
# ═══════════════════════════════════════════════════════════════════════════
md("""\
## §2 Sweep Configuration

All three algorithms undergo the **same** libraries, optimizers, and thresholds for fair comparison.

| Parameter | Values |
|-----------|--------|
| Thresholds | 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5 |
| Optimizers | STLSQ, LASSO, Ridge, ElasticNet |
| Libraries | Full, UC, Full+λ, UC+λ |""")

code("""\
# ═════════════════════════════════════════════════════════════════
# SHARED SWEEP CONFIGURATION
# ═════════════════════════════════════════════════════════════════
THRESHOLDS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.5]
OPTIMIZERS = ['STLSQ', 'LASSO', 'Ridge', 'ElasticNet']
REGIMES    = ['bouncing', 'rolling', 'flight']
DOF_MAP    = {'a_x': 0, 'a_y': 1, 'a_theta': 2}

# ── Lambda-augmented library builder ──
def build_lambda_augmented(base_fn, lambda_N, lambda_F):
    def _augmented(q, u, qddot, target_dof, **kw):
        Theta, names = base_fn(q, u, qddot, target_dof, **kw)
        Theta_aug = np.hstack([Theta,
                               lambda_N.reshape(-1, 1),
                               lambda_F.reshape(-1, 1)])
        return Theta_aug, names + ['lambda_N', 'lambda_F']
    return _augmented

# ── Library definitions for SINDy / Hybrid ──
#    Each entry: (label, base_function, needs_lambda)
LIBRARY_DEFS = [
    ('Full',   build_full_state_features,       False),
    ('UC',     build_unit_consistent_features,   False),
    ('Full+λ', build_full_state_features,       True),
    ('UC+λ',   build_unit_consistent_features,   True),
]

# ── Lagrange optimizer name mapping ──
LAG_OPT = {'STLSQ': 'STLSQ', 'LASSO': 'Lasso', 'Ridge': 'Ridge', 'ElasticNet': 'ElasticNet'}

# ── Pre-load first run of each regime + split ──
regime_data = {}
for regime in REGIMES:
    run = all_runs[regime][0]
    sp  = regime_aware_split(run.N, run.lambda_N)
    regime_data[regime] = {'run': run, 'split': sp}
    print(f'{regime}: {run.name}  train={sp.sizes[0]} val={sp.sizes[1]} test={sp.sizes[2]}')

print(f'\\nTotal SINDy configs per regime: '
      f'{len(LIBRARY_DEFS)} libs × {len(OPTIMIZERS)} opts × {len(THRESHOLDS)} thresholds '
      f'= {len(LIBRARY_DEFS)*len(OPTIMIZERS)*len(THRESHOLDS)}')""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 5: SINDy Sweep
# ═══════════════════════════════════════════════════════════════════════════
md("## §3 Standard SINDy — Full Sweep")

code("""\
# ═════════════════════════════════════════════════════════════════
# STANDARD SINDy — sweep all libs × opts × thresholds × regimes
# ═════════════════════════════════════════════════════════════════
sindy_rows = []
total = len(REGIMES) * len(LIBRARY_DEFS) * len(OPTIMIZERS) * len(THRESHOLDS)
count = 0

for regime in REGIMES:
    run   = regime_data[regime]['run']
    split = regime_data[regime]['split']

    for lib_label, base_fn, needs_lam in LIBRARY_DEFS:
        if needs_lam:
            lib_fn = build_lambda_augmented(base_fn, run.lambda_N, run.lambda_F)
        else:
            lib_fn = base_fn

        for opt in OPTIMIZERS:
            for th in THRESHOLDS:
                count += 1
                if count % 80 == 0:
                    print(f'  [{count}/{total}] {regime}/{lib_label}/{opt}/th={th}')
                try:
                    fit = fit_all_dofs(
                        run.q, run.u, run.qddot,
                        split.idx_train, split.idx_val, split.idx_test,
                        dt=run.dt, build_library_fn=lib_fn,
                        optimizer_name=opt, threshold=th,
                        unit_filtered=('UC' in lib_label),
                    )
                    for dof_name, res in fit.results.items():
                        d = DOF_MAP[dof_name]
                        y_val  = run.qddot[split.idx_val, d]
                        y_test = run.qddot[split.idx_test, d]
                        r2_val  = compute_r2(y_val,  res.y_pred[split.idx_val])
                        r2_test = compute_r2(y_test, res.y_pred[split.idx_test])
                        ate = compute_open_loop_ate(
                            res.y_pred[split.idx_val],
                            run.q[split.idx_val, d],
                            run.u[split.idx_val, d], run.dt)
                        sindy_rows.append({
                            'algorithm': 'SINDy', 'regime': regime,
                            'library': lib_label, 'optimizer': opt,
                            'threshold': th, 'dof': dof_name,
                            'n_terms': res.n_active_terms,
                            'MSE_train': res.mse_train,
                            'MSE_val': res.mse_val,
                            'MSE_test': res.mse_test,
                            'R2_val': r2_val, 'R2_test': r2_test,
                            'ATE': ate, 'ate_grade': ate_grade(ate),
                            'equation': res.equation,
                        })
                except Exception as e:
                    for dof_name in DOF_LABELS:
                        sindy_rows.append({
                            'algorithm': 'SINDy', 'regime': regime,
                            'library': lib_label, 'optimizer': opt,
                            'threshold': th, 'dof': dof_name,
                            'n_terms': 0,
                            'MSE_train': np.nan, 'MSE_val': np.nan,
                            'MSE_test': np.nan,
                            'R2_val': np.nan, 'R2_test': np.nan,
                            'ATE': np.nan, 'ate_grade': 'F',
                            'equation': f'ERROR: {e}',
                        })

df_sindy = pd.DataFrame(sindy_rows)
print(f'\\nSINDy sweep complete: {len(df_sindy)} rows')
print(f'Grade distribution:\\n{df_sindy["ate_grade"].value_counts().sort_index().to_string()}')""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 6: Lagrange Sweep
# ═══════════════════════════════════════════════════════════════════════════
md("""\
## §4 Lagrange SINDy — Full Sweep

Lagrange SINDy uses its own physics-based T/V library. The ±λ variation
controls whether contact forces are included as external forces.
For STLSQ the sweep value is the sparsity threshold; for others it is α.""")

code("""\
# ═════════════════════════════════════════════════════════════════
# LAGRANGE SINDy — sweep ±λ × opts × thresholds × regimes
# ═════════════════════════════════════════════════════════════════
lagrange_rows = []
lag_cache = {}  # (regime, lib, opt, th) → SharedLagrangeSINDyResult

LAG_LIBS = [
    ('Lagrange',    False),
    ('Lagrange+λ',  True),
]

total_lag = len(REGIMES) * len(LAG_LIBS) * len(OPTIMIZERS) * len(THRESHOLDS)
count = 0

for regime in REGIMES:
    run   = regime_data[regime]['run']
    split = regime_data[regime]['split']

    for lib_label, with_lam in LAG_LIBS:
        lN = run.lambda_N if with_lam else None
        lF = run.lambda_F if with_lam else None

        for opt in OPTIMIZERS:
            lag_opt = LAG_OPT[opt]

            for th in THRESHOLDS:
                count += 1
                if count % 40 == 0:
                    print(f'  [{count}/{total_lag}] {regime}/{lib_label}/{opt}/th={th}')
                try:
                    kw = dict(
                        lambda_N=lN, lambda_F=lF,
                        optimizer=lag_opt,
                    )
                    if lag_opt == 'STLSQ':
                        kw['stlsq_threshold'] = th
                        kw['alpha'] = 0.01
                    else:
                        kw['alpha'] = th

                    res = fit_lagrange_sindy_shared(
                        run.q, run.u, run.qddot,
                        split.idx_train, split.idx_val, split.idx_test,
                        **kw,
                    )
                    lag_cache[(regime, lib_label, opt, th)] = res

                    # Predict accelerations for fair R²/MSE comparison
                    try:
                        qddot_pred = lagrange_predict_qddot(
                            res, run.q, run.u, lambda_N=lN, lambda_F=lF)
                    except Exception:
                        qddot_pred = np.full_like(run.qddot, np.nan)

                    # Compute ATE per DOF
                    try:
                        ate_list = compute_lagrange_ate(
                            res, run.q, run.u, split.idx_val, run.dt,
                            lambda_N=lN, lambda_F=lF)
                    except Exception:
                        ate_list = [np.nan, np.nan, np.nan]

                    for d, dof_name in enumerate(DOF_LABELS):
                        y_val  = run.qddot[split.idx_val, d]
                        y_test = run.qddot[split.idx_test, d]
                        p_val  = qddot_pred[split.idx_val, d]
                        p_test = qddot_pred[split.idx_test, d]

                        mse_val  = float(np.nanmean((y_val - p_val)**2))
                        mse_test = float(np.nanmean((y_test - p_test)**2))
                        mse_train_d = res.mse_train_per_dof.get(dof_name, np.nan)
                        r2_val  = compute_r2(y_val, p_val)
                        r2_test = compute_r2(y_test, p_test)
                        ate_d   = ate_list[d] if d < len(ate_list) else np.nan

                        lagrange_rows.append({
                            'algorithm': 'Lagrange', 'regime': regime,
                            'library': lib_label, 'optimizer': opt,
                            'threshold': th, 'dof': dof_name,
                            'n_terms': res.n_active,
                            'MSE_train': mse_train_d,
                            'MSE_val': mse_val, 'MSE_test': mse_test,
                            'R2_val': r2_val, 'R2_test': r2_test,
                            'ATE': ate_d,
                            'ate_grade': ate_grade(ate_d) if not np.isnan(ate_d) else 'F',
                            'equation': res.lagrangian_equation(),
                        })
                except Exception as e:
                    for dof_name in DOF_LABELS:
                        lagrange_rows.append({
                            'algorithm': 'Lagrange', 'regime': regime,
                            'library': lib_label, 'optimizer': opt,
                            'threshold': th, 'dof': dof_name,
                            'n_terms': 0,
                            'MSE_train': np.nan, 'MSE_val': np.nan,
                            'MSE_test': np.nan,
                            'R2_val': np.nan, 'R2_test': np.nan,
                            'ATE': np.nan, 'ate_grade': 'F',
                            'equation': f'ERROR: {e}',
                        })

df_lagrange = pd.DataFrame(lagrange_rows)
print(f'\\nLagrange sweep complete: {len(df_lagrange)} rows')
print(f'Grade distribution:\\n{df_lagrange["ate_grade"].value_counts().sort_index().to_string()}')""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 7: Hybrid Sweep
# ═══════════════════════════════════════════════════════════════════════════
md("""\
## §5 Hybrid SINDy — Full Sweep

Trains **separate SINDy models per regime** (flight / rolling / bouncing),
each on pooled data from all runs of that regime. Evaluated on first run per regime.""")

code("""\
# ═════════════════════════════════════════════════════════════════
# HYBRID SINDy — sweep all libs × opts × thresholds
# ═════════════════════════════════════════════════════════════════

def fit_hybrid_with_lambda(regime_runs, base_fn, opt, th):
    \"\"\"Fit Hybrid SINDy with lambda-augmented library.\"\"\"
    result = HybridSINDyResult(
        regime_models={}, regime_configs={}, regime_run_names=regime_runs)
    for regime, runs in regime_runs.items():
        pooled = pool_runs(runs)
        sp = temporal_split_frac(pooled.N, train_frac=0.70, val_frac=0.15)
        aug_fn = build_lambda_augmented(base_fn, pooled.lambda_N, pooled.lambda_F)
        fit = fit_all_dofs(
            pooled.q, pooled.u, pooled.qddot,
            sp.idx_train, sp.idx_val, sp.idx_test,
            dt=pooled.dt, build_library_fn=aug_fn,
            optimizer_name=opt, threshold=th,
        )
        result.regime_models[regime] = fit
        result.regime_configs[regime] = {'optimizer': opt, 'threshold': th}
    return result

hybrid_rows = []
total_hyb = len(LIBRARY_DEFS) * len(OPTIMIZERS) * len(THRESHOLDS)
count = 0

for lib_label, base_fn, needs_lam in LIBRARY_DEFS:
    for opt in OPTIMIZERS:
        for th in THRESHOLDS:
            count += 1
            if count % 20 == 0:
                print(f'  [{count}/{total_hyb}] {lib_label}/{opt}/th={th}')
            try:
                if needs_lam:
                    hres = fit_hybrid_with_lambda(
                        DEFAULT_REGIME_RUNS, base_fn, opt, th)
                else:
                    hres = fit_hybrid_sindy(
                        regime_runs=DEFAULT_REGIME_RUNS,
                        build_library_fn=base_fn,
                        optimizer_name=opt, threshold=th,
                        unit_filtered=('UC' in lib_label),
                    )

                # Evaluate on first run of each regime
                for regime in REGIMES:
                    test_run = regime_data[regime]['run']
                    sp       = regime_data[regime]['split']

                    if needs_lam:
                        pred_fn = build_lambda_augmented(
                            base_fn, test_run.lambda_N, test_run.lambda_F)
                    else:
                        pred_fn = base_fn

                    out = hybrid_predict(hres, test_run, pred_fn)
                    pred = out['pred']

                    for d, dof_name in enumerate(DOF_LABELS):
                        y_val  = test_run.qddot[sp.idx_val, d]
                        y_test = test_run.qddot[sp.idx_test, d]
                        p_val  = pred[sp.idx_val, d]
                        p_test = pred[sp.idx_test, d]

                        mse_val  = float(np.mean((y_val - p_val)**2))
                        mse_test = float(np.mean((y_test - p_test)**2))
                        mse_train = float(np.mean(
                            (test_run.qddot[sp.idx_train, d] - pred[sp.idx_train, d])**2))
                        r2_val  = compute_r2(y_val, p_val)
                        r2_test = compute_r2(y_test, p_test)
                        ate = compute_open_loop_ate(
                            p_val, test_run.q[sp.idx_val, d],
                            test_run.u[sp.idx_val, d], test_run.dt)

                        hybrid_rows.append({
                            'algorithm': 'Hybrid', 'regime': regime,
                            'library': lib_label, 'optimizer': opt,
                            'threshold': th, 'dof': dof_name,
                            'n_terms': out['mse'].get(dof_name, 0),
                            'MSE_train': mse_train,
                            'MSE_val': mse_val, 'MSE_test': mse_test,
                            'R2_val': r2_val, 'R2_test': r2_test,
                            'ATE': ate, 'ate_grade': ate_grade(ate),
                            'equation': '',
                        })

            except Exception as e:
                for regime in REGIMES:
                    for dof_name in DOF_LABELS:
                        hybrid_rows.append({
                            'algorithm': 'Hybrid', 'regime': regime,
                            'library': lib_label, 'optimizer': opt,
                            'threshold': th, 'dof': dof_name,
                            'n_terms': 0,
                            'MSE_train': np.nan, 'MSE_val': np.nan,
                            'MSE_test': np.nan,
                            'R2_val': np.nan, 'R2_test': np.nan,
                            'ATE': np.nan, 'ate_grade': 'F',
                            'equation': f'ERROR: {e}',
                        })

df_hybrid = pd.DataFrame(hybrid_rows)
print(f'\\nHybrid sweep complete: {len(df_hybrid)} rows')
print(f'Grade distribution:\\n{df_hybrid["ate_grade"].value_counts().sort_index().to_string()}')""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 8: Combine all results
# ═══════════════════════════════════════════════════════════════════════════
md("## §6 Combined Results")

code("""\
# ═════════════════════════════════════════════════════════════════
# COMBINE ALL RESULTS INTO ONE DATAFRAME
# ═════════════════════════════════════════════════════════════════
df_all = pd.concat([df_sindy, df_lagrange, df_hybrid], ignore_index=True)
df_all['ate_grade'] = df_all['ATE'].apply(lambda x: ate_grade(x) if not np.isnan(x) else 'F')

print(f'Combined results: {len(df_all)} rows')
print(f'\\nRows per algorithm:')
print(df_all.groupby('algorithm').size().to_string())
print(f'\\nOverall grade distribution:')
print(df_all['ate_grade'].value_counts().sort_index().to_string())
print(f'\\nGrade distribution per algorithm:')
print(df_all.groupby('algorithm')['ate_grade'].value_counts().unstack(fill_value=0).to_string())

# Save to CSV
df_all.to_csv('pipeline/full_sweep_results.csv', index=False)
print('\\nSaved to pipeline/full_sweep_results.csv')""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 9: SINDy Per-Algorithm Analysis
# ═══════════════════════════════════════════════════════════════════════════
md("""\
## §7 Per-Algorithm Analysis — Standard SINDy

Best configurations, threshold sweep curves, and discovered equations.""")

code("""\
# ═════════════════════════════════════════════════════════════════
# STANDARD SINDy — PER-ALGORITHM ANALYSIS
# ═════════════════════════════════════════════════════════════════
ds = df_sindy.copy()

# ── Best config per DOF per regime (lowest ATE) ──
print('='*80)
print('  STANDARD SINDy — BEST CONFIG PER DOF (by ATE)')
print('='*80)
for regime in REGIMES:
    print(f'\\n  ── {regime.upper()} ──')
    for dof in DOF_LABELS:
        sub = ds[(ds['regime']==regime) & (ds['dof']==dof)]
        sub_valid = sub.dropna(subset=['ATE'])
        if sub_valid.empty:
            print(f'    {dof}: no valid results')
            continue
        best = sub_valid.loc[sub_valid['ATE'].idxmin()]
        print(f'    {dof}: {best["library"]}/{best["optimizer"]} th={best["threshold"]:.4f} '
              f'| ATE={best["ATE"]:.4f} [{best["ate_grade"]}] '
              f'| MSE_val={best["MSE_val"]:.4e} | R²={best["R2_val"]:.4f} '
              f'| {int(best["n_terms"])}t')
        print(f'           EQN: {best["equation"][:100]}')

# ── Threshold sweep plot: MSE vs threshold per library/optimizer ──
fig, axes = plt.subplots(len(REGIMES), 3, figsize=(18, 4*len(REGIMES)), sharex=True)
for i, regime in enumerate(REGIMES):
    for j, dof in enumerate(DOF_LABELS):
        ax = axes[i, j]
        sub = ds[(ds['regime']==regime) & (ds['dof']==dof)]
        for lib_label, _, _ in LIBRARY_DEFS:
            for opt in OPTIMIZERS:
                mask = (sub['library']==lib_label) & (sub['optimizer']==opt)
                s = sub[mask].sort_values('threshold')
                if s.empty: continue
                style = '-' if 'UC' not in lib_label else '--'
                ax.plot(s['threshold'], s['MSE_val'],
                        label=f'{lib_label}/{opt}', linestyle=style, alpha=0.7)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'{regime} — {dof}', fontsize=9)
        ax.set_xlabel('Threshold'); ax.set_ylabel('MSE (val)')
        ax.grid(True, alpha=0.3)
        if i==0 and j==2:
            ax.legend(fontsize=5, ncol=2, loc='upper left')
fig.suptitle('Standard SINDy — MSE vs Threshold', fontsize=13)
plt.tight_layout(); plt.show()

# ── Grade heatmap: library × optimizer ──
fig2, axes2 = plt.subplots(1, len(REGIMES), figsize=(6*len(REGIMES), 4))
grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
for i, regime in enumerate(REGIMES):
    ax = axes2[i]
    sub = ds[ds['regime']==regime].copy()
    sub['grade_num'] = sub['ate_grade'].map(grade_map)
    # Best grade per lib × opt (across thresholds and DOFs)
    pivot = sub.groupby(['library', 'optimizer'])['grade_num'].max().unstack(fill_value=0)
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=4, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            g = {v: k for k, v in grade_map.items()}.get(pivot.values[r, c], '?')
            ax.text(c, r, g, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_title(f'{regime}', fontsize=10)
fig2.suptitle('SINDy — Best ATE Grade (Library × Optimizer)', fontsize=13)
plt.tight_layout(); plt.show()""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 10: Lagrange Per-Algorithm Analysis
# ═══════════════════════════════════════════════════════════════════════════
md("""\
## §8 Per-Algorithm Analysis — Lagrange SINDy

Best configurations, coefficient comparison with ground truth, and ATE grading.""")

code("""\
# ═════════════════════════════════════════════════════════════════
# LAGRANGE SINDy — PER-ALGORITHM ANALYSIS
# ═════════════════════════════════════════════════════════════════
dl = df_lagrange.copy()

GROUND_TRUTH = {
    'T:xdot^2':                     0.75,
    'T:ydot^2':                     0.75,
    'T:thetadot^2':                 0.0243,
    'T:xdot thetadot sin(theta)':  -0.09,
    'T:ydot thetadot cos(theta)':   0.09,
    'V:y':                         -1.5,
    'V:sin(theta)':                -0.09,
}

# ── Best config per DOF per regime ──
print('='*80)
print('  LAGRANGE SINDy — BEST CONFIG PER DOF (by ATE)')
print('='*80)
for regime in REGIMES:
    print(f'\\n  ── {regime.upper()} ──')
    for dof in DOF_LABELS:
        sub = dl[(dl['regime']==regime) & (dl['dof']==dof)].dropna(subset=['ATE'])
        if sub.empty:
            print(f'    {dof}: no valid results')
            continue
        best = sub.loc[sub['ATE'].idxmin()]
        print(f'    {dof}: {best["library"]}/{best["optimizer"]} th={best["threshold"]:.4f} '
              f'| ATE={best["ATE"]:.4f} [{best["ate_grade"]}] '
              f'| MSE_val={best["MSE_val"]:.4e} | R²={best["R2_val"]:.4f} '
              f'| {int(best["n_terms"])}t')

# ── Best coefficient comparison with ground truth ──
print('\\n' + '='*80)
print('  LAGRANGE COEFFICIENT COMPARISON — BEST MODEL PER REGIME')
print('='*80)
for regime in REGIMES:
    sub = dl[dl['regime']==regime].dropna(subset=['ATE'])
    if sub.empty: continue
    # Best overall (mean ATE across DOFs)
    mean_ate = sub.groupby(['library','optimizer','threshold'])['ATE'].mean()
    best_key = mean_ate.idxmin()
    lib_b, opt_b, th_b = best_key
    key = (regime, lib_b, opt_b, th_b)
    if key in lag_cache:
        res = lag_cache[key]
        print(f'\\n  {regime.upper()} — {lib_b}/{opt_b} th={th_b}')
        print(f'  {"Term":42s} {"Discovered":>12s} {"Truth":>10s} {"Rel Err":>10s}')
        print(f'  {"-"*76}')
        for i, name in enumerate(res.all_names):
            c  = res.coefs[i]
            gt = GROUND_TRUTH.get(name, 0.0)
            if abs(c) < 1e-8 and gt == 0: continue
            if gt != 0:
                err = f'{abs(c-gt)/abs(gt):>9.1%}'
            elif abs(c) > 1e-8:
                err = ' SPURIOUS'
            else:
                err = '  MISSING'
            print(f'  {name:42s} {c:>+12.6f} {gt:>+10.6f} {err}')

# ── Threshold sweep plot ──
fig, axes = plt.subplots(len(REGIMES), 3, figsize=(18, 4*len(REGIMES)), sharex=True)
for i, regime in enumerate(REGIMES):
    for j, dof in enumerate(DOF_LABELS):
        ax = axes[i, j]
        sub = dl[(dl['regime']==regime) & (dl['dof']==dof)]
        for lib_label, _ in LAG_LIBS:
            for opt in OPTIMIZERS:
                s = sub[(sub['library']==lib_label) & (sub['optimizer']==opt)].sort_values('threshold')
                if s.empty: continue
                style = '-' if '+' not in lib_label else '--'
                ax.plot(s['threshold'], s['MSE_val'], label=f'{lib_label}/{opt}',
                        linestyle=style, alpha=0.7)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'{regime} — {dof}', fontsize=9)
        ax.set_xlabel('Threshold/α'); ax.set_ylabel('Accel MSE (val)')
        ax.grid(True, alpha=0.3)
        if i==0 and j==2:
            ax.legend(fontsize=5, ncol=2, loc='upper left')
fig.suptitle('Lagrange SINDy — Acceleration MSE vs Threshold/α', fontsize=13)
plt.tight_layout(); plt.show()""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 11: Hybrid Per-Algorithm Analysis
# ═══════════════════════════════════════════════════════════════════════════
md("""\
## §9 Per-Algorithm Analysis — Hybrid SINDy

Cross-regime evaluation and best configurations.""")

code("""\
# ═════════════════════════════════════════════════════════════════
# HYBRID SINDy — PER-ALGORITHM ANALYSIS
# ═════════════════════════════════════════════════════════════════
dh = df_hybrid.copy()

# ── Best config per DOF per regime ──
print('='*80)
print('  HYBRID SINDy — BEST CONFIG PER DOF (by ATE)')
print('='*80)
for regime in REGIMES:
    print(f'\\n  ── {regime.upper()} ──')
    for dof in DOF_LABELS:
        sub = dh[(dh['regime']==regime) & (dh['dof']==dof)].dropna(subset=['ATE'])
        if sub.empty:
            print(f'    {dof}: no valid results')
            continue
        best = sub.loc[sub['ATE'].idxmin()]
        print(f'    {dof}: {best["library"]}/{best["optimizer"]} th={best["threshold"]:.4f} '
              f'| ATE={best["ATE"]:.4f} [{best["ate_grade"]}] '
              f'| MSE_val={best["MSE_val"]:.4e} | R²={best["R2_val"]:.4f}')

# ── Threshold sweep plot ──
fig, axes = plt.subplots(len(REGIMES), 3, figsize=(18, 4*len(REGIMES)), sharex=True)
for i, regime in enumerate(REGIMES):
    for j, dof in enumerate(DOF_LABELS):
        ax = axes[i, j]
        sub = dh[(dh['regime']==regime) & (dh['dof']==dof)]
        for lib_label, _, _ in LIBRARY_DEFS:
            for opt in OPTIMIZERS:
                s = sub[(sub['library']==lib_label) & (sub['optimizer']==opt)].sort_values('threshold')
                if s.empty: continue
                style = '-' if 'UC' not in lib_label else '--'
                ax.plot(s['threshold'], s['MSE_val'], label=f'{lib_label}/{opt}',
                        linestyle=style, alpha=0.7)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'{regime} — {dof}', fontsize=9)
        ax.set_xlabel('Threshold'); ax.set_ylabel('MSE (val)')
        ax.grid(True, alpha=0.3)
        if i==0 and j==2:
            ax.legend(fontsize=5, ncol=2, loc='upper left')
fig.suptitle('Hybrid SINDy — MSE vs Threshold', fontsize=13)
plt.tight_layout(); plt.show()

# ── Grade heatmap: library × optimizer ──
fig2, axes2 = plt.subplots(1, len(REGIMES), figsize=(6*len(REGIMES), 4))
grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
for i, regime in enumerate(REGIMES):
    ax = axes2[i]
    sub = dh[dh['regime']==regime].copy()
    sub['grade_num'] = sub['ate_grade'].map(grade_map)
    pivot = sub.groupby(['library', 'optimizer'])['grade_num'].max().unstack(fill_value=0)
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=4, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            g = {v: k for k, v in grade_map.items()}.get(pivot.values[r, c], '?')
            ax.text(c, r, g, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_title(f'{regime}', fontsize=10)
fig2.suptitle('Hybrid SINDy — Best ATE Grade (Library × Optimizer)', fontsize=13)
plt.tight_layout(); plt.show()""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 12: Cross-Algorithm ATE Comparison
# ═══════════════════════════════════════════════════════════════════════════
md("""\
## §10 Cross-Algorithm Comparison — ATE Grading

For each **(Algorithm, Library, Optimizer)** tuple, find the **best ATE**
across all thresholds and regimes, then grade it.

Grading: **A** (<0.01) · **B** (<0.05) · **C** (<0.15) · **D** (<0.50) · **F** (≥0.50)""")

code("""\
# ═════════════════════════════════════════════════════════════════
# CROSS-ALGORITHM ATE COMPARISON
# ═════════════════════════════════════════════════════════════════

# ── Table 1: Best ATE per (Algorithm, Library, Optimizer) ──
# Across all thresholds and regimes, pick the threshold giving the best
# mean ATE (averaged over DOFs and regimes)
grouped = df_all.groupby(['algorithm', 'library', 'optimizer', 'threshold'])['ATE'].mean()
best_th = grouped.groupby(level=[0,1,2]).idxmin()

rows_t1 = []
for (algo, lib, opt), key in best_th.items():
    mean_ate = grouped[key]
    if np.isnan(mean_ate):
        continue
    grade = ate_grade(mean_ate)
    rows_t1.append({
        'Algorithm': algo, 'Library': lib, 'Optimizer': opt,
        'Best Threshold': key[3], 'Mean ATE': mean_ate, 'Grade': grade,
    })

df_t1 = pd.DataFrame(rows_t1).sort_values('Mean ATE')
df_t1['Rank'] = range(1, len(df_t1)+1)

print('='*90)
print('  TABLE 1: Algorithm × Library × Optimizer — Ranked by ATE Grade')
print('='*90)

def _grade_style(val):
    colors = {'A': 'background-color:#c6efce;color:#006100',
              'B': 'background-color:#dff0d8;color:#3c763d',
              'C': 'background-color:#fcf8e3;color:#8a6d3b',
              'D': 'background-color:#f2dede;color:#a94442',
              'F': 'background-color:#d9534f;color:white'}
    return colors.get(val, '')

styled_t1 = (df_t1[['Rank','Algorithm','Library','Optimizer','Best Threshold','Mean ATE','Grade']]
             .style
             .applymap(_grade_style, subset=['Grade'])
             .format({'Mean ATE': '{:.4f}', 'Best Threshold': '{:.4f}'})
             .set_caption('Ranked: Algorithm × Library × Optimizer (Best ATE)')
             .set_table_styles([
                 {'selector': 'caption', 'props': 'font-size:14px;font-weight:bold;margin-bottom:8px'},
                 {'selector': 'th', 'props': 'background-color:#2c3e50;color:white;padding:6px 10px'},
                 {'selector': 'td', 'props': 'padding:4px 10px;border-bottom:1px solid #ddd'},
             ])
             .hide(axis='index'))
display(styled_t1)

# ── Table 2: Grade distribution per Algorithm ──
print('\\n')
print('='*70)
print('  TABLE 2: Grade Distribution per Algorithm')
print('='*70)
grade_dist = df_all.groupby('algorithm')['ate_grade'].value_counts().unstack(fill_value=0)
for g in ['A','B','C','D','F']:
    if g not in grade_dist.columns:
        grade_dist[g] = 0
grade_dist = grade_dist[['A','B','C','D','F']]
grade_dist['Total'] = grade_dist.sum(axis=1)
grade_dist['%A'] = (grade_dist['A'] / grade_dist['Total'] * 100).round(1)
grade_dist['%A+B'] = ((grade_dist['A'] + grade_dist['B']) / grade_dist['Total'] * 100).round(1)
print(grade_dist.to_string())

# ── Table 3: Grade distribution per Algorithm × Library ──
print('\\n')
print('='*70)
print('  TABLE 3: Grade Distribution per Algorithm × Library')
print('='*70)
gd2 = df_all.groupby(['algorithm','library'])['ate_grade'].value_counts().unstack(fill_value=0)
for g in ['A','B','C','D','F']:
    if g not in gd2.columns:
        gd2[g] = 0
gd2 = gd2[['A','B','C','D','F']]
gd2['Total'] = gd2.sum(axis=1)
gd2['%A'] = (gd2['A'] / gd2['Total'] * 100).round(1)
gd2['%A+B'] = ((gd2['A'] + gd2['B']) / gd2['Total'] * 100).round(1)

def _col_grade_bg(col):
    cmap = {'A':'#c6efce','B':'#dff0d8','C':'#fcf8e3','D':'#f2dede','F':'#d9534f'}
    bg = cmap.get(col.name, 'white')
    return [f'background-color:{bg}' if v > 0 else '' for v in col]

styled_gd2 = (gd2.style
              .apply(_col_grade_bg, axis=0, subset=['A','B','C','D','F'])
              .set_caption('Grade Distribution: Algorithm × Library')
              .set_table_styles([
                  {'selector':'caption','props':'font-size:13px;font-weight:bold;margin-bottom:6px'},
                  {'selector':'th','props':'background-color:#34495e;color:white;padding:5px 12px;text-align:center'},
                  {'selector':'td','props':'text-align:center;padding:4px 12px'},
              ]))
display(styled_gd2)

# ── Bar chart: mean ATE per algorithm ──
fig, ax = plt.subplots(figsize=(10, 5))
algo_ate = df_all.groupby('algorithm')['ATE'].mean().sort_values()
colors = ['#2196F3', '#4CAF50', '#FF9800']
bars = ax.bar(range(len(algo_ate)), algo_ate.values, color=colors[:len(algo_ate)])
ax.set_xticks(range(len(algo_ate)))
ax.set_xticklabels(algo_ate.index, fontsize=11)
ax.set_ylabel('Mean ATE (lower is better)')
ax.set_title('Mean ATE per Algorithm (across all configs)')
for bar, v in zip(bars, algo_ate.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
            f'{v:.4f} [{ate_grade(v)}]', ha='center', va='bottom', fontsize=9)
ax.axhline(0.01, color='green', ls=':', alpha=0.5, label='A boundary')
ax.axhline(0.05, color='orange', ls=':', alpha=0.5, label='B boundary')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 13: Detailed ATE per regime
# ═══════════════════════════════════════════════════════════════════════════
code("""\
# ═════════════════════════════════════════════════════════════════
# DETAILED: Per-Regime × Per-DOF ATE Winners
# ═════════════════════════════════════════════════════════════════
print('='*100)
print('  BEST MODEL PER REGIME × DOF (lowest ATE)')
print('='*100)

winner_rows = []
for regime in REGIMES:
    for dof in DOF_LABELS:
        sub = df_all[(df_all['regime']==regime) & (df_all['dof']==dof)].dropna(subset=['ATE'])
        if sub.empty: continue
        best = sub.loc[sub['ATE'].idxmin()]
        winner_rows.append({
            'Regime': regime, 'DOF': dof,
            'Algorithm': best['algorithm'], 'Library': best['library'],
            'Optimizer': best['optimizer'], 'Threshold': best['threshold'],
            'ATE': best['ATE'], 'Grade': best['ate_grade'],
            'MSE_val': best['MSE_val'], 'R2_val': best['R2_val'],
            'n_terms': int(best['n_terms']),
        })
        print(f'  {regime:>10s} {dof:>8s}: {best["algorithm"]:>9s} '
              f'{best["library"]:>10s} {best["optimizer"]:>10s} '
              f'th={best["threshold"]:.4f} | ATE={best["ATE"]:.4f} '
              f'[{best["ate_grade"]}] | R²={best["R2_val"]:.4f}')

df_winners = pd.DataFrame(winner_rows)

styled_w = (df_winners.style
            .applymap(_grade_style, subset=['Grade'])
            .format({'ATE': '{:.4f}', 'Threshold': '{:.4f}',
                     'MSE_val': '{:.4e}', 'R2_val': '{:.4f}'})
            .set_caption('Best Model per Regime × DOF')
            .set_table_styles([
                {'selector':'caption','props':'font-size:14px;font-weight:bold;margin-bottom:8px'},
                {'selector':'th','props':'background-color:#2c3e50;color:white;padding:6px 10px'},
                {'selector':'td','props':'padding:4px 10px;border-bottom:1px solid #ddd'},
            ])
            .hide(axis='index'))
display(styled_w)""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 14: R² Analysis
# ═══════════════════════════════════════════════════════════════════════════
md("""\
## §11 R² Analysis — All Algorithms

$R^2 = 1 - \\frac{\\text{MSE}}{\\text{Var}(y_{\\text{true}})}$

Higher is better. R² = 1 means perfect prediction.""")

code("""\
# ═════════════════════════════════════════════════════════════════
# R² ANALYSIS — ALL ALGORITHMS
# ═════════════════════════════════════════════════════════════════

# ── Table: Best R² per Algorithm × Library × Optimizer ──
r2_best = (df_all.groupby(['algorithm','library','optimizer'])['R2_val']
           .max().reset_index()
           .sort_values('R2_val', ascending=False))
r2_best['Rank'] = range(1, len(r2_best)+1)

print('='*80)
print('  TABLE: Best R² per (Algorithm, Library, Optimizer)')
print('='*80)
print(r2_best.head(20).to_string(index=False))

# ── R² heatmap: Algorithm × Library ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, regime in enumerate(REGIMES):
    ax = axes[i]
    sub = df_all[df_all['regime']==regime]
    pivot = sub.groupby(['algorithm','library'])['R2_val'].max().unstack(fill_value=0)
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            ax.text(c, r, f'{pivot.values[r,c]:.3f}', ha='center', va='center',
                    fontsize=8, fontweight='bold')
    ax.set_title(f'{regime}', fontsize=10)
    fig.colorbar(im, ax=ax, label='R² (val)')
fig.suptitle('Best R² per Algorithm × Library (across opts & thresholds)', fontsize=13)
plt.tight_layout(); plt.show()

# ── Box plot: R² distribution per algorithm ──
fig2, ax2 = plt.subplots(figsize=(10, 5))
algos = df_all['algorithm'].unique()
box_data = [df_all[df_all['algorithm']==a]['R2_val'].dropna().values for a in algos]
bp = ax2.boxplot(box_data, labels=algos, patch_artist=True)
colors = ['#2196F3', '#4CAF50', '#FF9800']
for patch, color in zip(bp['boxes'], colors[:len(algos)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('R² (validation)')
ax2.set_title('R² Distribution per Algorithm (all configs)')
ax2.axhline(0.9, color='green', ls=':', alpha=0.5, label='R²=0.9')
ax2.axhline(0.0, color='red', ls=':', alpha=0.5, label='R²=0')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

# ── R² per DOF breakdown ──
print('\\n' + '='*80)
print('  MEAN R² PER ALGORITHM × DOF (best config per combination)')
print('='*80)
r2_dof = df_all.groupby(['algorithm','dof'])['R2_val'].max().unstack()
print(r2_dof.to_string(float_format='{:.4f}'.format))""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 15: R² Tabulations
# ═══════════════════════════════════════════════════════════════════════════
code("""\
# ═════════════════════════════════════════════════════════════════
# R² DETAILED TABULATIONS
# ═════════════════════════════════════════════════════════════════

# ── Table: R² per Algorithm × Library × Optimizer × Regime ──
print('='*100)
print('  R² SUMMARY: Algorithm × Library × Optimizer (best threshold, averaged over DOFs & regimes)')
print('='*100)

r2_summary = (df_all.groupby(['algorithm','library','optimizer','threshold'])['R2_val']
              .mean().reset_index())
# Best threshold per (algo, lib, opt)
idx_best = r2_summary.groupby(['algorithm','library','optimizer'])['R2_val'].idxmax()
r2_top = r2_summary.loc[idx_best].sort_values('R2_val', ascending=False)
r2_top['Rank'] = range(1, len(r2_top)+1)

styled_r2 = (r2_top[['Rank','algorithm','library','optimizer','threshold','R2_val']]
             .style
             .background_gradient(cmap='RdYlGn', subset=['R2_val'], vmin=0, vmax=1)
             .format({'R2_val': '{:.4f}', 'threshold': '{:.4f}'})
             .set_caption('R² Ranking: Algorithm × Library × Optimizer')
             .set_table_styles([
                 {'selector':'caption','props':'font-size:14px;font-weight:bold;margin-bottom:8px'},
                 {'selector':'th','props':'background-color:#2c3e50;color:white;padding:6px 10px'},
                 {'selector':'td','props':'padding:4px 10px;border-bottom:1px solid #ddd'},
             ])
             .hide(axis='index'))
display(styled_r2)

# ── R² vs threshold line plot (per algorithm) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, algo in enumerate(df_all['algorithm'].unique()):
    ax = axes[i]
    sub = df_all[df_all['algorithm']==algo]
    mean_r2 = sub.groupby(['library','optimizer','threshold'])['R2_val'].mean().reset_index()
    for lib in mean_r2['library'].unique():
        for opt in OPTIMIZERS:
            s = mean_r2[(mean_r2['library']==lib) & (mean_r2['optimizer']==opt)].sort_values('threshold')
            if s.empty: continue
            ax.plot(s['threshold'], s['R2_val'], label=f'{lib}/{opt}', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Threshold'); ax.set_ylabel('Mean R² (val)')
    ax.set_title(algo, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.05)
    if i == 2:
        ax.legend(fontsize=5, ncol=2, loc='lower left')
fig.suptitle('R² vs Threshold — All Algorithms', fontsize=13)
plt.tight_layout(); plt.show()""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 16: MSE Analysis
# ═══════════════════════════════════════════════════════════════════════════
md("""\
## §12 MSE Analysis — All Algorithms

Lower MSE is better. We compare validation MSE across all configurations.""")

code("""\
# ═════════════════════════════════════════════════════════════════
# MSE ANALYSIS — ALL ALGORITHMS
# ═════════════════════════════════════════════════════════════════

# ── Table: Best MSE per Algorithm × Library × Optimizer ──
mse_summary = (df_all.groupby(['algorithm','library','optimizer','threshold'])['MSE_val']
               .mean().reset_index())
idx_best = mse_summary.groupby(['algorithm','library','optimizer'])['MSE_val'].idxmin()
mse_top = mse_summary.loc[idx_best].sort_values('MSE_val')
mse_top['Rank'] = range(1, len(mse_top)+1)

print('='*80)
print('  TABLE: Best MSE per (Algorithm, Library, Optimizer)')
print('='*80)

styled_mse = (mse_top[['Rank','algorithm','library','optimizer','threshold','MSE_val']]
              .style
              .background_gradient(cmap='RdYlGn_r', subset=['MSE_val'])
              .format({'MSE_val': '{:.4e}', 'threshold': '{:.4f}'})
              .set_caption('MSE Ranking: Algorithm × Library × Optimizer')
              .set_table_styles([
                  {'selector':'caption','props':'font-size:14px;font-weight:bold;margin-bottom:8px'},
                  {'selector':'th','props':'background-color:#2c3e50;color:white;padding:6px 10px'},
                  {'selector':'td','props':'padding:4px 10px;border-bottom:1px solid #ddd'},
              ])
              .hide(axis='index'))
display(styled_mse)

# ── MSE heatmap: Algorithm × Library ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, regime in enumerate(REGIMES):
    ax = axes[i]
    sub = df_all[df_all['regime']==regime]
    pivot = sub.groupby(['algorithm','library'])['MSE_val'].min().unstack(fill_value=1)
    im = ax.imshow(np.log10(pivot.values + 1e-20), cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            ax.text(c, r, f'{pivot.values[r,c]:.1e}', ha='center', va='center',
                    fontsize=7, fontweight='bold')
    ax.set_title(f'{regime}', fontsize=10)
    fig.colorbar(im, ax=ax, label='log10(MSE)')
fig.suptitle('Best MSE per Algorithm × Library (across opts & thresholds)', fontsize=13)
plt.tight_layout(); plt.show()""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 17: MSE Detailed Tabulations
# ═══════════════════════════════════════════════════════════════════════════
code("""\
# ═════════════════════════════════════════════════════════════════
# MSE DETAILED TABULATIONS
# ═════════════════════════════════════════════════════════════════

# ── MSE per DOF breakdown ──
print('='*80)
print('  BEST MSE PER ALGORITHM × DOF')
print('='*80)
mse_dof = df_all.groupby(['algorithm','dof'])['MSE_val'].min().unstack()
print(mse_dof.to_string(float_format='{:.4e}'.format))

# ── Box plot: MSE distribution per algorithm ──
fig, ax = plt.subplots(figsize=(10, 5))
algos = df_all['algorithm'].unique()
box_data = [np.log10(df_all[df_all['algorithm']==a]['MSE_val'].dropna().values + 1e-20)
            for a in algos]
bp = ax.boxplot(box_data, labels=algos, patch_artist=True)
colors = ['#2196F3', '#4CAF50', '#FF9800']
for patch, color in zip(bp['boxes'], colors[:len(algos)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('log10(MSE_val)')
ax.set_title('MSE Distribution per Algorithm (all configs)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

# ── MSE vs threshold (per algorithm, averaged over regimes/DOFs) ──
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
for i, algo in enumerate(df_all['algorithm'].unique()):
    ax = axes2[i]
    sub = df_all[df_all['algorithm']==algo]
    mean_mse = sub.groupby(['library','optimizer','threshold'])['MSE_val'].mean().reset_index()
    for lib in mean_mse['library'].unique():
        for opt in OPTIMIZERS:
            s = mean_mse[(mean_mse['library']==lib) & (mean_mse['optimizer']==opt)].sort_values('threshold')
            if s.empty: continue
            ax.plot(s['threshold'], s['MSE_val'], label=f'{lib}/{opt}', alpha=0.7)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Threshold'); ax.set_ylabel('Mean MSE (val)')
    ax.set_title(algo, fontsize=11)
    ax.grid(True, alpha=0.3)
    if i == 2:
        ax.legend(fontsize=5, ncol=2, loc='upper left')
fig2.suptitle('MSE vs Threshold — All Algorithms', fontsize=13)
plt.tight_layout(); plt.show()

# ── Train vs Val vs Test MSE (best config per algorithm) ──
print('\\n' + '='*80)
print('  TRAIN / VAL / TEST MSE — BEST CONFIG PER ALGORITHM')
print('='*80)
for algo in df_all['algorithm'].unique():
    sub = df_all[df_all['algorithm']==algo].dropna(subset=['MSE_val'])
    if sub.empty: continue
    best_idx = sub.groupby('dof')['MSE_val'].idxmin()
    print(f'\\n  {algo}:')
    for dof in DOF_LABELS:
        if dof not in best_idx.index: continue
        b = sub.loc[best_idx[dof]]
        print(f'    {dof}: train={b["MSE_train"]:.4e}  val={b["MSE_val"]:.4e}  '
              f'test={b["MSE_test"]:.4e}  '
              f'({b["library"]}/{b["optimizer"]} th={b["threshold"]:.4f})')""")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 18: Final Summary
# ═══════════════════════════════════════════════════════════════════════════
md("""\
## §13 Final Summary

Overall ranking of algorithms with best configurations for each metric.""")

code("""\
# ═════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════

print('='*90)
print('  FINAL SUMMARY — ALL ALGORITHMS')
print('='*90)

# ── Overall winner per metric ──
for metric, ascending in [('ATE', True), ('MSE_val', True), ('R2_val', False)]:
    print(f'\\n  ── Best by {metric} ──')
    sub = df_all.dropna(subset=[metric])
    best_per_algo = sub.groupby('algorithm')[metric].agg('min' if ascending else 'max')
    winner = best_per_algo.idxmin() if ascending else best_per_algo.idxmax()
    print(f'    Winner: {winner} ({metric}={best_per_algo[winner]:.4f})')
    for algo in sub['algorithm'].unique():
        val = best_per_algo[algo]
        print(f'      {algo}: {val:.4f}')

# ── Comprehensive ranking table ──
summary_rows = []
for algo in df_all['algorithm'].unique():
    sub = df_all[df_all['algorithm']==algo].dropna(subset=['ATE','MSE_val','R2_val'])
    if sub.empty: continue
    summary_rows.append({
        'Algorithm': algo,
        'Best ATE': sub['ATE'].min(),
        'ATE Grade': ate_grade(sub['ATE'].min()),
        'Best MSE': sub['MSE_val'].min(),
        'Best R²': sub['R2_val'].max(),
        '%A grades': f'{(sub["ate_grade"]=="A").mean()*100:.1f}%',
        '%A+B grades': f'{(sub["ate_grade"].isin(["A","B"])).mean()*100:.1f}%',
        'Total configs': len(sub),
    })

df_summary = pd.DataFrame(summary_rows)
styled_summary = (df_summary.style
                  .applymap(_grade_style, subset=['ATE Grade'])
                  .format({'Best ATE': '{:.4f}', 'Best MSE': '{:.4e}', 'Best R²': '{:.4f}'})
                  .set_caption('Algorithm Summary — All Metrics')
                  .set_table_styles([
                      {'selector':'caption','props':'font-size:14px;font-weight:bold;margin-bottom:8px'},
                      {'selector':'th','props':'background-color:#2c3e50;color:white;padding:6px 10px'},
                      {'selector':'td','props':'padding:4px 10px;border-bottom:1px solid #ddd'},
                  ])
                  .hide(axis='index'))
display(styled_summary)

# ── Pareto front: ATE vs Sparsity ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
algo_colors = {'SINDy': '#2196F3', 'Lagrange': '#4CAF50', 'Hybrid': '#FF9800'}
algo_markers = {'SINDy': 'o', 'Lagrange': 's', 'Hybrid': '^'}
for i, dof in enumerate(DOF_LABELS):
    ax = axes[i]
    for algo in df_all['algorithm'].unique():
        sub = df_all[(df_all['algorithm']==algo) & (df_all['dof']==dof)].dropna(subset=['ATE'])
        ax.scatter(sub['n_terms'], sub['ATE'],
                   c=algo_colors.get(algo, 'gray'),
                   marker=algo_markers.get(algo, 'x'),
                   alpha=0.3, s=20, label=algo)
    ax.set_xlabel('# Active Terms'); ax.set_ylabel('ATE')
    ax.set_yscale('log'); ax.set_title(dof, fontsize=11)
    ax.axhline(0.01, color='green', ls=':', alpha=0.4)
    ax.axhline(0.05, color='orange', ls=':', alpha=0.4)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)
fig.suptitle('Pareto: ATE vs Sparsity — All Algorithms', fontsize=13)
plt.tight_layout(); plt.show()

print('\\n[DONE] Full sweep and analysis complete.')
print(f'Total configurations evaluated: {len(df_all)}')
print(f'Results saved to: pipeline/full_sweep_results.csv')""")

# ═══════════════════════════════════════════════════════════════════════════
# WRITE NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Notebook written to: {NB_PATH}')
print(f'Total cells: {len(cells)}')
print(f'  Markdown: {sum(1 for c in cells if c["cell_type"]=="markdown")}')
print(f'  Code:     {sum(1 for c in cells if c["cell_type"]=="code")}')
