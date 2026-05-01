#!/usr/bin/env python3
"""Generate HoopSINDy_Pipeline.ipynb — matches SINDy_Pipeline.ipynb structure."""
import json, os

OUTPATH = r'C:\Users\braid\OneDrive\Desktop\Data Driven Modeling Project\hoop data\HoopSINDy_Pipeline.ipynb'
cells = []
def md(s):  cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})

# ═══════════════════════════════════════════════════════════════
# CELL 1: Title
# ═══════════════════════════════════════════════════════════════
md("""# Hoop SINDy Pipeline — EOM Discovery from IMU Data

**3 Algorithms**: Standard SINDy · Lagrange SINDy · Hybrid SINDy
**2 Libraries**: Unit-Consistent (UC) · Full (degree 2)
**4 Optimizers**: STLSQ · LASSO · Ridge · ElasticNet
**Threshold sweep**: 10⁻⁴ → 0.5
**3 Metrics**: MSE (train/val/test) · R² · ATE
**5 Runs**: Hoop IMU data at 120 Hz
""")

# ═══════════════════════════════════════════════════════════════
# CELL 2: Imports + Helpers
# ═══════════════════════════════════════════════════════════════
code("""\
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from scipy.signal import butter, filtfilt, savgol_filter, medfilt
from scipy.integrate import cumulative_trapezoid
import sys, os

sys.path.insert(0, str(Path(r"C:\\Users\\braid\\OneDrive\\Desktop\\Data Driven Modeling Project\\hoop data")))
from run_sindy_analysis import preprocess_hoop_data, detrend_custom

try:
    import pysindy as ps
    PYSINDY_AVAILABLE = True
except ImportError:
    PYSINDY_AVAILABLE = False

matplotlib.rcParams.update({'figure.max_open_warning': 50})

# ── Helper functions ──────────────────────────────────────────
def ate_grade(ate):
    if ate < 0.01: return 'A'
    if ate < 0.05: return 'B'
    if ate < 0.15: return 'C'
    if ate < 0.50: return 'D'
    return 'F'

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-30 else 0.0

def compute_open_loop_ate(pred_accel, true_pos, true_vel, dt):
    v = cumulative_trapezoid(pred_accel, dx=dt, initial=0) + true_vel[0]
    q = cumulative_trapezoid(v, dx=dt, initial=0) + true_pos[0]
    return float(np.mean(np.abs(q - true_pos)))

print(f"Imports OK | PySINDy: {PYSINDY_AVAILABLE}")
""")

# ═══════════════════════════════════════════════════════════════
# CELL 3: §1 header
# ═══════════════════════════════════════════════════════════════
md("## §1 — Load & Preprocess Data")

# ═══════════════════════════════════════════════════════════════
# CELL 4: Load all CSVs
# ═══════════════════════════════════════════════════════════════
code("""\
FS = 120
CUTOFF_ACC, CUTOFF_GYR = 20, 15
SG_WIN, SG_ORD = 11, 3
DATA_DIR = Path(r"C:\\Users\\braid\\OneDrive\\Desktop\\Data Driven Modeling Project\\hoop data")
CSV_FILES = {
    'run_1': 'OR_20250903_203926.csv',
    'run_2': 'OR_20250903_204044.csv',
    'run_3': 'OR_20250903_204119.csv',
    'run_4': 'OR_20250903_204158.csv',
    'run_5': 'OR_D422CD005685_20250903_204229.csv',
}

def butter_lp(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    if data.ndim == 1: return filtfilt(b, a, data)
    return np.column_stack([filtfilt(b, a, data[:, i]) for i in range(data.shape[1])])

def butter_hp(data, cutoff, fs, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high')
    return filtfilt(b, a, data)

def clip_iqr(data, k=3.0):
    out = data.copy()
    for i in range(data.shape[1]):
        q1, q3 = np.percentile(data[:, i], [25, 75])
        iqr = q3 - q1
        out[:, i] = np.clip(data[:, i], q1 - k * iqr, q3 + k * iqr)
    return out

def preprocess_csv(path):
    df = pd.read_csv(path, skiprows=6)
    acc_cols = ['Acc_X', 'Acc_Y', 'Acc_Z']
    gyr_cols = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']
    for c in acc_cols: df[c] = butter_lp(df[c].values, CUTOFF_ACC, FS)
    for c in gyr_cols: df[c] = butter_lp(df[c].values, CUTOFF_GYR, FS)
    arr = df[acc_cols].to_numpy()
    arr = clip_iqr(arr, k=3.0)
    for i, c in enumerate(acc_cols): df[c] = arr[:, i]
    proc = preprocess_hoop_data(df, fs=FS)
    t = proc['t']; pos = proc['position']; vel = proc['velocity']
    acc = proc['acceleration']; angles = proc['euler_angles']
    n_st = max(1, int(FS))
    acc = acc - np.median(acc[:n_st], axis=0)
    acc[:, 2] = butter_hp(acc[:, 2], 0.3, FS)
    dt = 1.0 / FS
    acc = savgol_filter(acc, SG_WIN, SG_ORD, axis=0)
    vel = savgol_filter(vel, SG_WIN, SG_ORD, axis=0)
    pos = savgol_filter(pos, SG_WIN, SG_ORD, axis=0)
    vr = np.zeros_like(acc)
    for i in range(3):
        vr[:, i] = detrend_custom(t, cumulative_trapezoid(acc[:, i], t, initial=0), degree=6)
    pr = np.zeros_like(vr)
    for i in range(3):
        pr[:, i] = detrend_custom(t, cumulative_trapezoid(vr[:, i], t, initial=0), degree=6)
    vel = savgol_filter(vr, SG_WIN, SG_ORD, axis=0)
    pos = savgol_filter(pr, SG_WIN, SG_ORD, axis=0)
    ts, te = max(5.0, float(t[0])), min(18.0, float(t[-1]))
    m = (t >= ts) & (t <= te)
    t, pos, vel, acc = t[m], pos[m], vel[m], acc[m]
    angles = angles[m]
    df2 = df.iloc[np.where(m)[0]].reset_index(drop=True)
    omega = np.deg2rad(df2[gyr_cols].to_numpy(dtype=float))
    for i in range(3): omega[:, i] = medfilt(omega[:, i], kernel_size=5)
    omega = savgol_filter(omega, SG_WIN, SG_ORD, axis=0)
    angles = np.asarray(angles, dtype=float)
    if np.nanmax(np.abs(angles)) > 7: angles = np.deg2rad(angles)
    N = len(t)
    return {'pos': pos, 'vel': vel, 'acc': acc, 'omega': omega, 'angles': angles,
            't': t, 'dt': dt, 'N': N,
            'idx_train': np.arange(0, int(0.60 * N)),
            'idx_val':   np.arange(int(0.60 * N), int(0.80 * N)),
            'idx_test':  np.arange(int(0.80 * N), N)}

all_runs = {}
for name, fname in CSV_FILES.items():
    try:
        all_runs[name] = preprocess_csv(DATA_DIR / fname)
        rd = all_runs[name]
        print(f"  {name}: N={rd['N']}, t=[{rd['t'][0]:.1f},{rd['t'][-1]:.1f}]s, "
              f"split={len(rd['idx_train'])}/{len(rd['idx_val'])}/{len(rd['idx_test'])}")
    except Exception as e:
        print(f"  {name}: FAILED - {e}")

RUNS = list(all_runs.keys())
print(f"\\nLoaded {len(all_runs)} / {len(CSV_FILES)} runs")
""")

# ═══════════════════════════════════════════════════════════════
# CELL 5: §1b header
# ═══════════════════════════════════════════════════════════════
md("""## §1b — Data Visualization & Train/Val/Test Splits

Overview of all 5 runs: positions, velocities, accelerations, angular velocities.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 6: Data overview plots
# ═══════════════════════════════════════════════════════════════
code("""\
state_labels = ['x (m)', 'y (m)', 'z (m)']
vel_labels   = ['x\\u0307 (m/s)', 'y\\u0307 (m/s)', 'z\\u0307 (m/s)']
acc_labels   = ['x\\u0308 (m/s\\u00b2)', 'y\\u0308 (m/s\\u00b2)', 'z\\u0308 (m/s\\u00b2)']
run_colors   = {r: c for r, c in zip(RUNS, plt.cm.tab10.colors)}

for sig_name, sig_key, labels in [('Position', 'pos', state_labels),
                                    ('Velocity', 'vel', vel_labels),
                                    ('Acceleration', 'acc', acc_labels)]:
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'{sig_name} - All Runs', fontsize=13, fontweight='bold')
    for rn in RUNS:
        rd = all_runs[rn]
        for i in range(3):
            axes[i].plot(rd['t'], rd[sig_key][:, i], lw=0.8,
                         color=run_colors[rn], alpha=0.7, label=rn)
    for i in range(3):
        axes[i].set_ylabel(labels[i], fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=8, ncol=len(RUNS))
    axes[2].set_xlabel('Time (s)')
    plt.tight_layout(); plt.show()

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
fig.suptitle('Angular Velocity - All Runs', fontsize=13, fontweight='bold')
for rn in RUNS:
    rd = all_runs[rn]
    for i, lbl in enumerate(['\\u03c9x (rad/s)', '\\u03c9y (rad/s)', '\\u03c9z (rad/s)']):
        axes[i].plot(rd['t'], rd['omega'][:, i], lw=0.8, color=run_colors[rn], alpha=0.7, label=rn)
        axes[i].set_ylabel(lbl, fontweight='bold'); axes[i].grid(True, alpha=0.3)
axes[0].legend(loc='upper right', fontsize=8, ncol=len(RUNS))
axes[2].set_xlabel('Time (s)')
plt.tight_layout(); plt.show()
print('Data overview plots complete.')
""")

# ═══════════════════════════════════════════════════════════════
# CELL 7: Split visualization
# ═══════════════════════════════════════════════════════════════
code("""\
split_colors = {'Train': '#1f77b4', 'Val': '#ff7f0e', 'Test': '#2ca02c'}
nr = len(RUNS)
fig, axes = plt.subplots(nr, 1, figsize=(14, 3 * nr), sharex=False)
fig.suptitle('Train / Validation / Test Splits - Acceleration (x)', fontsize=13, fontweight='bold')
if nr == 1: axes = [axes]
for ax, rn in zip(axes, RUNS):
    rd = all_runs[rn]
    for lbl, idx, c in [('Train', rd['idx_train'], split_colors['Train']),
                          ('Val',   rd['idx_val'],   split_colors['Val']),
                          ('Test',  rd['idx_test'],  split_colors['Test'])]:
        ax.plot(rd['t'][idx], rd['acc'][idx, 0], color=c, lw=0.8, label=lbl)
    ax.set_ylabel(f'{rn}\\nx\\u0308', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if ax is axes[0]: ax.legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel('Time (s)')
plt.tight_layout(); plt.show()

print('=' * 65)
print(f'  {"Run":<10} {"Train":>8} {"Val":>8} {"Test":>8} {"Total":>8}')
print('=' * 65)
for rn in RUNS:
    rd = all_runs[rn]
    tr, va, te = len(rd['idx_train']), len(rd['idx_val']), len(rd['idx_test'])
    print(f'  {rn:<10} {tr:>8d} {va:>8d} {te:>8d} {rd["N"]:>8d}')
print('=' * 65)
""")

# ═══════════════════════════════════════════════════════════════
# CELL 8: §2 header
# ═══════════════════════════════════════════════════════════════
md("""## §2 — Sweep Configuration

**Thresholds**: [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
**Optimizers**: STLSQ, LASSO, Ridge, ElasticNet
**Libraries**: UC (~80 feat), Full deg-2 (~160)
**Lagrange**: T+V Lagrangian library (~130 feat)
""")

# ═══════════════════════════════════════════════════════════════
# CELL 9: Config + library builders + fit function
# ═══════════════════════════════════════════════════════════════
code("""\
DOF_LABELS = ['a_x', 'a_y', 'a_z']
DOF_MAP = {'a_x': 0, 'a_y': 1, 'a_z': 2}
THRESHOLDS = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
OPTIMIZERS = ['STLSQ', 'LASSO', 'Ridge', 'ElasticNet']

# ── Library builders ─────────────────────────────────────────
def build_uc_library(pos, vel, omega, angles, acc, target_dof, **kw):
    N = len(pos)
    phi, theta, psi = angles[:, 0], angles[:, 1], angles[:, 2]
    ox, oy, oz = omega[:, 0], omega[:, 1], omega[:, 2]
    others = [j for j in range(3) if j != target_dof]
    F, Fn = [np.ones(N)], ['1']
    for i, n in enumerate(['x','y','z']): F.append(pos[:, i]); Fn.append(n)
    for i, n in enumerate(['xdot','ydot','zdot']): F.append(vel[:, i]); Fn.append(n)
    for i, n in enumerate(['wx','wy','wz']): F.append(omega[:, i]); Fn.append(n)
    for ang, l in [(phi,'phi'),(theta,'theta'),(psi,'psi')]:
        F.append(np.sin(ang)); Fn.append(f'sin({l})')
        F.append(np.cos(ang)); Fn.append(f'cos({l})')
    for i, n in enumerate(['wx','wy','wz']): F.append(omega[:, i]**2); Fn.append(f'{n}^2')
    F.append(ox*oy); Fn.append('wx*wy')
    F.append(ox*oz); Fn.append('wx*wz')
    F.append(oy*oz); Fn.append('wy*wz')
    for j in others:
        dn = ['xddot','yddot','zddot'][j]; F.append(acc[:, j]); Fn.append(dn)
    for ang, al in [(phi,'phi'),(theta,'theta'),(psi,'psi')]:
        for i, wl in enumerate(['wx','wy','wz']):
            F.append(np.sin(ang)*omega[:, i]); Fn.append(f'sin({al})*{wl}')
            F.append(np.cos(ang)*omega[:, i]); Fn.append(f'cos({al})*{wl}')
    for i, vl in enumerate(['xdot','ydot','zdot']):
        for j, wl in enumerate(['wx','wy','wz']):
            F.append(vel[:, i]*omega[:, j]); Fn.append(f'{vl}*{wl}')
    for i, n in enumerate(['xdot','ydot','zdot']): F.append(vel[:, i]**2); Fn.append(f'{n}^2')
    for ang, al in [(phi,'phi'),(theta,'theta')]:
        for i, vl in enumerate(['xdot','ydot','zdot']):
            F.append(np.sin(ang)*vel[:, i]); Fn.append(f'sin({al})*{vl}')
            F.append(np.cos(ang)*vel[:, i]); Fn.append(f'cos({al})*{vl}')
    for ang, al in [(phi,'phi'),(theta,'theta')]:
        for i, pl in enumerate(['x','y','z']):
            F.append(np.sin(ang)*pos[:, i]); Fn.append(f'sin({al})*{pl}')
            F.append(np.cos(ang)*pos[:, i]); Fn.append(f'cos({al})*{pl}')
    for i, vl in enumerate(['xdot','ydot','zdot']): F.append(np.sign(vel[:, i])); Fn.append(f'sign({vl})')
    for i, wl in enumerate(['wx','wy','wz']): F.append(np.sign(omega[:, i])); Fn.append(f'sign({wl})')
    return np.column_stack(F), Fn

def build_full_library(pos, vel, omega, angles, acc, target_dof, degree=2, **kw):
    N = len(pos)
    phi, theta, psi = angles[:, 0], angles[:, 1], angles[:, 2]
    others = [j for j in range(3) if j != target_dof]
    base = np.column_stack([pos, vel, omega, acc[:, others[0]:others[0]+1], acc[:, others[1]:others[1]+1]])
    bn = ['x','y','z','xdot','ydot','zdot','wx','wy','wz'] + [['xddot','yddot','zddot'][j] for j in others]
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Tp = poly.fit_transform(base)
    pn = poly.get_feature_names_out(bn).tolist()
    F, Fn, seen = list(Tp.T), list(pn), set(pn)
    td = [(np.sin(phi),'sin(phi)'),(np.cos(phi),'cos(phi)'),
          (np.sin(theta),'sin(theta)'),(np.cos(theta),'cos(theta)'),
          (np.sin(psi),'sin(psi)'),(np.cos(psi),'cos(psi)')]
    for val, tn in td:
        if tn not in seen: F.append(val); Fn.append(tn); seen.add(tn)
    for val, tn in td:
        for bi in range(base.shape[1]):
            cn = f'{tn}*{bn[bi]}'
            if cn not in seen: F.append(val*base[:, bi]); Fn.append(cn); seen.add(cn)
    for i in range(len(td)):
        for j in range(i+1, len(td)):
            cn = f'{td[i][1]}*{td[j][1]}'
            if cn not in seen: F.append(td[i][0]*td[j][0]); Fn.append(cn); seen.add(cn)
    for i, vl in enumerate(['xdot','ydot','zdot']):
        sn = f'sign({vl})'
        if sn not in seen: F.append(np.sign(vel[:, i])); Fn.append(sn); seen.add(sn)
    for i, wl in enumerate(['wx','wy','wz']):
        sn = f'sign({wl})'
        if sn not in seen: F.append(np.sign(omega[:, i])); Fn.append(sn); seen.add(sn)
    return np.column_stack(F), Fn

def build_lagrange_library(pos, vel, omega, angles, acc=None, target_dof=None, **kw):
    N = len(pos)
    phi, theta, psi = angles[:, 0], angles[:, 1], angles[:, 2]
    F, Fn = [np.ones(N)], ['1']
    for i, n in enumerate(['x','y','z']): F.append(pos[:, i]); Fn.append(n)
    for ang, l in [(phi,'phi'),(theta,'theta'),(psi,'psi')]:
        F.append(np.sin(ang)); Fn.append(f'sin({l})')
        F.append(np.cos(ang)); Fn.append(f'cos({l})')
    for i, n in enumerate(['x','y','z']): F.append(pos[:, i]**2); Fn.append(f'{n}^2')
    for ang, al in [(phi,'phi'),(theta,'theta')]:
        for i, pn_ in enumerate(['x','y','z']):
            F.append(pos[:, i]*np.sin(ang)); Fn.append(f'{pn_}*sin({al})')
            F.append(pos[:, i]*np.cos(ang)); Fn.append(f'{pn_}*cos({al})')
    va = np.column_stack([vel, omega])
    vn = ['xdot','ydot','zdot','wx','wy','wz']
    for i in range(6):
        for j in range(i, 6):
            vv = va[:, i] * va[:, j]
            F.append(vv); Fn.append(f'{vn[i]}*{vn[j]}')
            for ang, al in [(phi,'phi'),(theta,'theta')]:
                F.append(vv*np.sin(ang)); Fn.append(f'{vn[i]}*{vn[j]}*sin({al})')
                F.append(vv*np.cos(ang)); Fn.append(f'{vn[i]}*{vn[j]}*cos({al})')
    for i, n in enumerate(['xdot','ydot','zdot']): F.append(np.sign(vel[:, i])); Fn.append(f'sign({n})')
    for i, n in enumerate(['wx','wy','wz']): F.append(np.sign(omega[:, i])); Fn.append(f'sign({n}_w)')
    return np.column_stack(F), Fn

LIBRARY_DEFS = [
    ('UC', build_uc_library, {}),
    ('Full', build_full_library, {'degree': 2}),
]

# ── SINDy fitting ─────────────────────────────────────────────
def fit_sindy_single(X_tr, y_tr, X_val, y_val, X_test, y_test, opt_name, threshold, feat_names):
    sX = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr.reshape(-1, 1))
    Xn = sX.transform(X_tr); Xv = sX.transform(X_val); Xt = sX.transform(X_test)
    yn = sy.transform(y_tr.reshape(-1, 1)).ravel()
    if opt_name == 'STLSQ' and PYSINDY_AVAILABLE:
        opt = ps.STLSQ(threshold=threshold, max_iter=100)
        opt.fit(Xn, yn.reshape(-1, 1)); coefs = opt.coef_.ravel()
    elif opt_name == 'LASSO':
        m = Lasso(alpha=threshold, max_iter=10000, tol=1e-4); m.fit(Xn, yn); coefs = m.coef_
    elif opt_name == 'Ridge':
        m = Ridge(alpha=threshold); m.fit(Xn, yn); coefs = m.coef_
    elif opt_name == 'ElasticNet':
        m = ElasticNet(alpha=threshold, l1_ratio=0.5, max_iter=10000); m.fit(Xn, yn); coefs = m.coef_
    else:
        m = Ridge(alpha=1e-3); m.fit(Xn, yn); coefs = m.coef_.copy()
        for _ in range(30):
            mask = np.abs(coefs) > threshold
            if not mask.any(): break
            m2 = Ridge(alpha=1e-3); m2.fit(Xn[:, mask], yn)
            coefs = np.zeros(Xn.shape[1]); coefs[mask] = m2.coef_
    p_tr = sy.inverse_transform((Xn @ coefs).reshape(-1,1)).ravel()
    p_va = sy.inverse_transform((Xv @ coefs).reshape(-1,1)).ravel()
    p_te = sy.inverse_transform((Xt @ coefs).reshape(-1,1)).ravel()
    co = coefs * sy.scale_[0] / sX.scale_
    active = np.abs(co) > 1e-10; n_act = int(active.sum())
    terms = [f'{co[i]:.4f} {feat_names[i]}' for i in np.where(active)[0]]
    return {'coefs': co, 'n_active': n_act,
            'p_train': p_tr, 'p_val': p_va, 'p_test': p_te,
            'mse_train': float(mean_squared_error(y_tr, p_tr)),
            'mse_val':   float(mean_squared_error(y_val, p_va)),
            'mse_test':  float(mean_squared_error(y_test, p_te)),
            'equation':  ' + '.join(terms[:20]) if terms else '0',
            'scaler_X': sX, 'scaler_y': sy, 'coefs_scaled': coefs}

# ── Regime detection for Hybrid ───────────────────────────────
def detect_regimes(acc, window=15, percentile=60):
    mag = np.linalg.norm(acc, axis=1)
    mag_s = medfilt(mag, kernel_size=window)
    return (mag_s >= np.percentile(mag_s, percentile)).astype(int)

# ── Pre-build libraries ──────────────────────────────────────
run_data = {}
for rn in RUNS:
    rd = all_runs[rn]
    run_data[rn] = {'rd': rd, 'libs': {}, 'regimes': detect_regimes(rd['acc'])}
    for di, dn in enumerate(DOF_LABELS):
        run_data[rn]['libs'][dn] = {}
        for ll, fn, kw in LIBRARY_DEFS:
            Theta, names = fn(rd['pos'], rd['vel'], rd['omega'], rd['angles'], rd['acc'],
                              target_dof=di, **kw)
            run_data[rn]['libs'][dn][ll] = {'Theta': Theta, 'names': names}
        Theta_lag, names_lag = build_lagrange_library(rd['pos'], rd['vel'], rd['omega'], rd['angles'])
        run_data[rn]['libs'][dn]['Lagrange'] = {'Theta': Theta_lag, 'names': names_lag}

for rn in RUNS[:1]:
    for dn in DOF_LABELS:
        sizes = {ll: run_data[rn]['libs'][dn][ll]['Theta'].shape[1]
                 for ll in run_data[rn]['libs'][dn]}
        print(f"  {rn}/{dn}: {sizes}")
print(f"\\nLibraries built for {len(RUNS)} runs x {len(DOF_LABELS)} DOFs")
""")

# ═══════════════════════════════════════════════════════════════
# CELL 10: §3 header
# ═══════════════════════════════════════════════════════════════
md("## §3 — Standard SINDy Sweep")

# ═══════════════════════════════════════════════════════════════
# CELL 11: SINDy sweep
# ═══════════════════════════════════════════════════════════════
code("""\
sindy_rows = []
total = len(RUNS) * len(LIBRARY_DEFS) * len(OPTIMIZERS) * len(THRESHOLDS)
count = 0
for rn in RUNS:
    rd = all_runs[rn]
    itr, iva, ite = rd['idx_train'], rd['idx_val'], rd['idx_test']
    for ll, _, _ in LIBRARY_DEFS:
        for opt in OPTIMIZERS:
            for th in THRESHOLDS:
                count += 1
                if count % 100 == 0: print(f'  [{count}/{total}] {rn}/{ll}/{opt}/th={th}')
                for di, dn in enumerate(DOF_LABELS):
                    d = DOF_MAP[dn]; lib = run_data[rn]['libs'][dn][ll]
                    try:
                        res = fit_sindy_single(
                            lib['Theta'][itr], rd['acc'][itr, d],
                            lib['Theta'][iva], rd['acc'][iva, d],
                            lib['Theta'][ite], rd['acc'][ite, d],
                            opt, th, lib['names'])
                        r2v = compute_r2(rd['acc'][iva, d], res['p_val'])
                        r2t = compute_r2(rd['acc'][ite, d], res['p_test'])
                        ate = compute_open_loop_ate(
                            res['p_val'], rd['pos'][iva, d], rd['vel'][iva, d], rd['dt'])
                        sindy_rows.append({
                            'algorithm': 'SINDy', 'run': rn, 'library': ll, 'optimizer': opt,
                            'threshold': th, 'dof': dn, 'n_terms': res['n_active'],
                            'MSE_train': res['mse_train'], 'MSE_val': res['mse_val'],
                            'MSE_test': res['mse_test'],
                            'R2_val': r2v, 'R2_test': r2t, 'ATE': ate,
                            'ate_grade': ate_grade(ate), 'equation': res['equation']})
                    except Exception as e:
                        sindy_rows.append({
                            'algorithm': 'SINDy', 'run': rn, 'library': ll, 'optimizer': opt,
                            'threshold': th, 'dof': dn, 'n_terms': 0,
                            'MSE_train': np.nan, 'MSE_val': np.nan, 'MSE_test': np.nan,
                            'R2_val': np.nan, 'R2_test': np.nan, 'ATE': np.nan,
                            'ate_grade': 'F', 'equation': f'ERROR: {e}'})
df_sindy = pd.DataFrame(sindy_rows)
print(f'\\nSINDy sweep: {len(df_sindy)} rows')
print(df_sindy['ate_grade'].value_counts().sort_index().to_string())
""")

# ═══════════════════════════════════════════════════════════════
# CELL 12: §4 header
# ═══════════════════════════════════════════════════════════════
md("""## §4 — Lagrange SINDy Sweep

Lagrangian-structured library: T (kinetic energy) + V (potential energy) terms.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 13: Lagrange sweep
# ═══════════════════════════════════════════════════════════════
code("""\
lagrange_rows = []
total_lag = len(RUNS) * len(OPTIMIZERS) * len(THRESHOLDS)
count = 0
for rn in RUNS:
    rd = all_runs[rn]
    itr, iva, ite = rd['idx_train'], rd['idx_val'], rd['idx_test']
    for opt in OPTIMIZERS:
        for th in THRESHOLDS:
            count += 1
            if count % 40 == 0: print(f'  [{count}/{total_lag}] {rn}/Lagrange/{opt}/th={th}')
            for di, dn in enumerate(DOF_LABELS):
                d = DOF_MAP[dn]; lib = run_data[rn]['libs'][dn]['Lagrange']
                try:
                    res = fit_sindy_single(
                        lib['Theta'][itr], rd['acc'][itr, d],
                        lib['Theta'][iva], rd['acc'][iva, d],
                        lib['Theta'][ite], rd['acc'][ite, d],
                        opt, th, lib['names'])
                    r2v = compute_r2(rd['acc'][iva, d], res['p_val'])
                    r2t = compute_r2(rd['acc'][ite, d], res['p_test'])
                    ate = compute_open_loop_ate(
                        res['p_val'], rd['pos'][iva, d], rd['vel'][iva, d], rd['dt'])
                    lagrange_rows.append({
                        'algorithm': 'Lagrange', 'run': rn, 'library': 'Lagrange',
                        'optimizer': opt, 'threshold': th, 'dof': dn,
                        'n_terms': res['n_active'],
                        'MSE_train': res['mse_train'], 'MSE_val': res['mse_val'],
                        'MSE_test': res['mse_test'],
                        'R2_val': r2v, 'R2_test': r2t, 'ATE': ate,
                        'ate_grade': ate_grade(ate), 'equation': res['equation']})
                except Exception as e:
                    lagrange_rows.append({
                        'algorithm': 'Lagrange', 'run': rn, 'library': 'Lagrange',
                        'optimizer': opt, 'threshold': th, 'dof': dn,
                        'n_terms': 0,
                        'MSE_train': np.nan, 'MSE_val': np.nan, 'MSE_test': np.nan,
                        'R2_val': np.nan, 'R2_test': np.nan, 'ATE': np.nan,
                        'ate_grade': 'F', 'equation': f'ERROR: {e}'})
df_lagrange = pd.DataFrame(lagrange_rows)
print(f'\\nLagrange sweep: {len(df_lagrange)} rows')
print(df_lagrange['ate_grade'].value_counts().sort_index().to_string())
""")

# ═══════════════════════════════════════════════════════════════
# CELL 14: §5 header
# ═══════════════════════════════════════════════════════════════
md("""## §5 — Hybrid SINDy Sweep

Acceleration-magnitude regime detection (low/high dynamics), separate models per regime.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 15: Hybrid sweep
# ═══════════════════════════════════════════════════════════════
code("""\
hybrid_rows = []
total_hyb = len(RUNS) * len(LIBRARY_DEFS) * len(OPTIMIZERS) * len(THRESHOLDS)
count = 0
for rn in RUNS:
    rd = all_runs[rn]
    itr, iva, ite = rd['idx_train'], rd['idx_val'], rd['idx_test']
    labels = run_data[rn]['regimes']
    for ll, _, _ in LIBRARY_DEFS:
        for opt in OPTIMIZERS:
            for th in THRESHOLDS:
                count += 1
                if count % 100 == 0: print(f'  [{count}/{total_hyb}] {rn}/{ll}/{opt}/th={th}')
                for di, dn in enumerate(DOF_LABELS):
                    d = DOF_MAP[dn]
                    lib = run_data[rn]['libs'][dn][ll]
                    Theta = lib['Theta']; y = rd['acc'][:, d]
                    try:
                        pred_tr = np.zeros(len(itr))
                        pred_va = np.zeros(len(iva))
                        pred_te = np.zeros(len(ite))
                        ntot = 0
                        for reg_id in [0, 1]:
                            rmask = labels == reg_id
                            tr_m = np.array([k for k, idx in enumerate(itr) if rmask[idx]])
                            va_m = np.array([k for k, idx in enumerate(iva) if rmask[idx]])
                            te_m = np.array([k for k, idx in enumerate(ite) if rmask[idx]])
                            if len(tr_m) < 5:
                                fb = np.mean(y[itr])
                                if len(tr_m): pred_tr[tr_m] = fb
                                if len(va_m): pred_va[va_m] = fb
                                if len(te_m): pred_te[te_m] = fb
                                continue
                            r_itr = itr[tr_m]; r_iva = iva[va_m]; r_ite = ite[te_m]
                            if len(r_iva) < 2: r_iva = r_itr[-2:]
                            if len(r_ite) < 2: r_ite = r_itr[-2:]
                            rr = fit_sindy_single(
                                Theta[r_itr], y[r_itr],
                                Theta[r_iva], y[r_iva],
                                Theta[r_ite], y[r_ite],
                                opt, th, lib['names'])
                            pred_tr[tr_m] = rr['p_train'][:len(tr_m)]
                            if len(va_m): pred_va[va_m] = rr['p_val'][:len(va_m)]
                            if len(te_m): pred_te[te_m] = rr['p_test'][:len(te_m)]
                            ntot += rr['n_active']
                        mse_tr = float(mean_squared_error(y[itr], pred_tr))
                        mse_va = float(mean_squared_error(y[iva], pred_va))
                        mse_te = float(mean_squared_error(y[ite], pred_te))
                        r2v = compute_r2(y[iva], pred_va)
                        r2t = compute_r2(y[ite], pred_te)
                        ate = compute_open_loop_ate(
                            pred_va, rd['pos'][iva, d], rd['vel'][iva, d], rd['dt'])
                        hybrid_rows.append({
                            'algorithm': 'Hybrid', 'run': rn, 'library': ll,
                            'optimizer': opt, 'threshold': th, 'dof': dn,
                            'n_terms': ntot,
                            'MSE_train': mse_tr, 'MSE_val': mse_va, 'MSE_test': mse_te,
                            'R2_val': r2v, 'R2_test': r2t, 'ATE': ate,
                            'ate_grade': ate_grade(ate), 'equation': ''})
                    except Exception as e:
                        hybrid_rows.append({
                            'algorithm': 'Hybrid', 'run': rn, 'library': ll,
                            'optimizer': opt, 'threshold': th, 'dof': dn,
                            'n_terms': 0,
                            'MSE_train': np.nan, 'MSE_val': np.nan, 'MSE_test': np.nan,
                            'R2_val': np.nan, 'R2_test': np.nan, 'ATE': np.nan,
                            'ate_grade': 'F', 'equation': f'ERROR: {e}'})
df_hybrid = pd.DataFrame(hybrid_rows)
print(f'\\nHybrid sweep: {len(df_hybrid)} rows')
print(df_hybrid['ate_grade'].value_counts().sort_index().to_string())
""")

# ═══════════════════════════════════════════════════════════════
# CELL 16: §6 header
# ═══════════════════════════════════════════════════════════════
md("## §6 — Combined Results")

# ═══════════════════════════════════════════════════════════════
# CELL 17: Combine + save CSV
# ═══════════════════════════════════════════════════════════════
code("""\
df_all = pd.concat([df_sindy, df_lagrange, df_hybrid], ignore_index=True)
csv_path = os.path.join(str(DATA_DIR), 'hoop_full_sweep_results.csv')
df_all.to_csv(csv_path, index=False)

print(f'Combined results: {len(df_all)} rows')
print(f'\\nRows per algorithm:\\n{df_all["algorithm"].value_counts().to_string()}')
print(f'\\nGrade distribution:\\n{df_all["ate_grade"].value_counts().sort_index().to_string()}')
print(f'\\nGrade per algorithm:')
print(df_all.groupby('algorithm')['ate_grade'].value_counts().unstack(fill_value=0).to_string())
print(f'\\nSaved to {csv_path}')
""")

# ═══════════════════════════════════════════════════════════════
# CELL 18: §6b header
# ═══════════════════════════════════════════════════════════════
md("""## §6b — MSE Comparison: Train / Validation / Test

Grouped bar charts comparing MSE across algorithms and runs.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 19: MSE train/val/test plots
# ═══════════════════════════════════════════════════════════════
code("""\
algos = df_all['algorithm'].unique()
algo_colors = {'SINDy': '#1f77b4', 'Lagrange': '#ff7f0e', 'Hybrid': '#2ca02c'}

# 1) Overall MSE per algorithm (median, clipped)
fig, ax = plt.subplots(figsize=(10, 5))
metrics_plot = ['MSE_train', 'MSE_val', 'MSE_test']
x = np.arange(len(algos)); w = 0.25
for i, met in enumerate(metrics_plot):
    vals = [df_all[df_all['algorithm']==a][met].median() for a in algos]
    ax.bar(x + i*w, vals, w, label=met.replace('MSE_', ''))
ax.set_xticks(x + w); ax.set_xticklabels(algos)
ax.set_ylabel('Median MSE'); ax.set_title('Median MSE per Algorithm (Train/Val/Test)')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

# 2) Per-run MSE
n_runs = len(RUNS)
fig, axes = plt.subplots(1, min(n_runs, 5), figsize=(4*min(n_runs,5), 5), sharey=True)
if n_runs == 1: axes = [axes]
for ax, rn in zip(axes, RUNS[:5]):
    sub = df_all[df_all['run'] == rn]
    x = np.arange(len(algos)); w = 0.25
    for i, met in enumerate(metrics_plot):
        vals = [sub[sub['algorithm']==a][met].median() for a in algos]
        ax.bar(x + i*w, vals, w, label=met.replace('MSE_', '') if ax is axes[0] else '')
    ax.set_xticks(x + w); ax.set_xticklabels(algos, fontsize=8)
    ax.set_title(rn, fontweight='bold'); ax.grid(True, alpha=0.3, axis='y')
axes[0].set_ylabel('Median MSE')
if axes[0].get_legend_handles_labels()[1]: axes[0].legend(fontsize=8)
fig.suptitle('Median MSE per Algorithm x Run', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# 3) Overfitting ratio scatter
fig, ax = plt.subplots(figsize=(8, 5))
for algo in algos:
    sub = df_all[df_all['algorithm']==algo].dropna(subset=['MSE_train','MSE_val'])
    sub = sub[sub['MSE_train'] > 1e-15]
    ratio = sub['MSE_val'] / sub['MSE_train']
    ratio = ratio.clip(upper=100)
    ax.scatter(sub['threshold'], ratio, alpha=0.3, s=10, label=algo,
               color=algo_colors.get(algo, '#999'))
ax.axhline(1, color='red', ls='--', lw=1, label='No overfit')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Threshold'); ax.set_ylabel('MSE_val / MSE_train')
ax.set_title('Overfitting Ratio vs Threshold'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
print('MSE train/val/test comparison complete.')
""")

# ═══════════════════════════════════════════════════════════════
# CELL 20: §6c header
# ═══════════════════════════════════════════════════════════════
md("""## §6c — R² Comparison: Validation vs Test

Scatter plots, box plots, and heatmaps of R² across algorithms.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 21: R² plots
# ═══════════════════════════════════════════════════════════════
code("""\
algo_markers = {'SINDy': 'o', 'Lagrange': 's', 'Hybrid': '^'}

# 1) R² val vs test scatter
fig, ax = plt.subplots(figsize=(7, 6))
for algo in algos:
    sub = df_all[df_all['algorithm']==algo].dropna(subset=['R2_val','R2_test'])
    sub = sub[(sub['R2_val'] > -10) & (sub['R2_test'] > -10)]
    ax.scatter(sub['R2_val'], sub['R2_test'], alpha=0.3, s=10,
               marker=algo_markers.get(algo, 'o'),
               color=algo_colors.get(algo, '#999'), label=algo)
ax.plot([-10, 1], [-10, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('R² (validation)'); ax.set_ylabel('R² (test)')
ax.set_title('R² Validation vs Test'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# 2) R² box plots per algorithm
fig, ax = plt.subplots(figsize=(8, 5))
box_data = []
box_labels = []
for algo in algos:
    vals = df_all[df_all['algorithm']==algo]['R2_val'].dropna()
    vals = vals[vals > -10]
    box_data.append(vals.values)
    box_labels.append(algo)
bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
for patch, algo in zip(bp['boxes'], algos):
    patch.set_facecolor(algo_colors.get(algo, '#999'))
    patch.set_alpha(0.5)
ax.set_ylabel('R² (validation)'); ax.set_title('R² Distribution per Algorithm')
ax.axhline(0, color='red', ls='--', lw=1, alpha=0.5)
ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# 3) R² heatmap: algorithm x DOF (best per combo, averaged over runs)
r2_pivot = df_all.groupby(['algorithm', 'dof'])['R2_val'].max().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(r2_pivot.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(r2_pivot.columns))); ax.set_xticklabels(r2_pivot.columns)
ax.set_yticks(range(len(r2_pivot.index))); ax.set_yticklabels(r2_pivot.index)
for i in range(r2_pivot.shape[0]):
    for j in range(r2_pivot.shape[1]):
        ax.text(j, i, f'{r2_pivot.values[i,j]:.3f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax, label='R²')
ax.set_title('Best R² per Algorithm x DOF'); plt.tight_layout(); plt.show()
print('R² comparison complete.')
""")

# ═══════════════════════════════════════════════════════════════
# CELL 22: §6d header
# ═══════════════════════════════════════════════════════════════
md("""## §6d — Comprehensive Comparison Plots

ATE landscape, sparsity-accuracy tradeoff, library effect, optimizer sensitivity, radar chart.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 23: Comprehensive comparison plots
# ═══════════════════════════════════════════════════════════════
code("""\
# 1) ATE landscape: median ATE vs threshold per algorithm
fig, ax = plt.subplots(figsize=(10, 5))
for algo in algos:
    sub = df_all[df_all['algorithm']==algo]
    grouped = sub.groupby('threshold')['ATE'].median()
    thvals = grouped.index.values; med = grouped.values
    q25 = sub.groupby('threshold')['ATE'].quantile(0.25).values
    q75 = sub.groupby('threshold')['ATE'].quantile(0.75).values
    ax.plot(thvals, med, 'o-', label=algo, color=algo_colors.get(algo, '#999'))
    ax.fill_between(thvals, q25, q75, alpha=0.15, color=algo_colors.get(algo, '#999'))
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Threshold'); ax.set_ylabel('Median ATE')
ax.set_title('ATE Landscape: Median ATE vs Threshold')
ax.legend(); ax.grid(True, alpha=0.3, which='both')
plt.tight_layout(); plt.show()

# 2) Sparsity-accuracy tradeoff
fig, ax = plt.subplots(figsize=(8, 5))
for algo in algos:
    sub = df_all[df_all['algorithm']==algo].dropna(subset=['n_terms','R2_val'])
    sub = sub[sub['R2_val'] > -10]
    ax.scatter(sub['n_terms'], sub['R2_val'], alpha=0.3, s=12,
               color=algo_colors.get(algo, '#999'), label=algo,
               marker=algo_markers.get(algo, 'o'))
ax.set_xlabel('Number of Active Terms'); ax.set_ylabel('R² (validation)')
ax.set_title('Sparsity vs Accuracy'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# 3) Library effect: median ATE per library x algorithm
fig, ax = plt.subplots(figsize=(10, 5))
libs = df_all['library'].unique()
x = np.arange(len(libs)); w = 0.8 / len(algos)
for i, algo in enumerate(algos):
    vals = [df_all[(df_all['algorithm']==algo) & (df_all['library']==lb)]['ATE'].median()
            for lb in libs]
    ax.bar(x + i*w, vals, w, label=algo, color=algo_colors.get(algo, '#999'))
ax.set_xticks(x + w*(len(algos)-1)/2); ax.set_xticklabels(libs, rotation=15)
ax.set_ylabel('Median ATE'); ax.set_title('Library Effect on ATE')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

# 4) Optimizer sensitivity
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(OPTIMIZERS)); w = 0.8 / len(algos)
for i, algo in enumerate(algos):
    vals = [df_all[(df_all['algorithm']==algo) & (df_all['optimizer']==op)]['ATE'].median()
            for op in OPTIMIZERS]
    ax.bar(x + i*w, vals, w, label=algo, color=algo_colors.get(algo, '#999'))
ax.set_xticks(x + w*(len(algos)-1)/2); ax.set_xticklabels(OPTIMIZERS)
ax.set_ylabel('Median ATE'); ax.set_title('Optimizer Sensitivity')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

# 5) Radar chart
categories = ['1-ATE', 'R2_val', '1-MSE_val', 'Sparsity', '%Grade_A']
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
for algo in algos:
    sub = df_all[df_all['algorithm']==algo]
    ate_med = sub['ATE'].median()
    r2_med = max(0, sub['R2_val'].median())
    mse_med = sub['MSE_val'].median()
    sparsity = 1 - sub['n_terms'].median() / max(sub['n_terms'].max(), 1)
    pct_a = (sub['ate_grade'] == 'A').mean()
    vals = [max(0, 1-ate_med), r2_med, max(0, 1-min(mse_med, 1)), sparsity, pct_a]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', label=algo, color=algo_colors.get(algo, '#999'))
    ax.fill(angles, vals, alpha=0.1, color=algo_colors.get(algo, '#999'))
ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=9)
ax.set_title('Algorithm Comparison Radar', fontsize=13, fontweight='bold', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1)); plt.tight_layout(); plt.show()
print('Comprehensive comparison plots complete.')
""")

# ═══════════════════════════════════════════════════════════════
# CELL 24: §7a header
# ═══════════════════════════════════════════════════════════════
md("""## §7a — Standard SINDy: Best Configurations

Per-run best model by ATE, coefficient heatmap.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 25: SINDy per-run analysis
# ═══════════════════════════════════════════════════════════════
code("""\
print('=' * 80)
print('  STANDARD SINDy - BEST CONFIG PER DOF (by ATE)')
print('=' * 80)
for rn in RUNS:
    print(f'\\n  -- {rn.upper()} --')
    for dn in DOF_LABELS:
        sub = df_sindy[(df_sindy['run']==rn) & (df_sindy['dof']==dn)].dropna(subset=['ATE'])
        if sub.empty: print(f'    {dn}: no valid results'); continue
        best = sub.loc[sub['ATE'].idxmin()]
        print(f'    {dn}: {best["library"]}/{best["optimizer"]} th={best["threshold"]:.4f} '
              f'| ATE={best["ATE"]:.4f} [{best["ate_grade"]}] '
              f'| MSE_val={best["MSE_val"]:.4e} | R2={best["R2_val"]:.4f} | {best["n_terms"]}t')
        eq = best["equation"]
        if len(eq) > 80: eq = eq[:80] + '...'
        print(f'           EQN: {eq}')

# Threshold sensitivity heatmap for SINDy
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('SINDy: Median ATE per Threshold x Optimizer', fontsize=13, fontweight='bold')
for ax, dn in zip(axes, DOF_LABELS):
    sub = df_sindy[df_sindy['dof']==dn]
    piv = sub.groupby(['optimizer','threshold'])['ATE'].median().unstack(fill_value=np.nan)
    mat = np.log10(piv.values.clip(min=1e-6))
    im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f'{v:.0e}' for v in piv.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title(dn, fontweight='bold')
    plt.colorbar(im, ax=ax, label='log10(ATE)')
plt.tight_layout(); plt.show()
""")

# ═══════════════════════════════════════════════════════════════
# CELL 26: §7b header
# ═══════════════════════════════════════════════════════════════
md("""## §7b — Lagrange SINDy: Best Configurations

Per-run best Lagrange model by ATE.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 27: Lagrange per-run analysis
# ═══════════════════════════════════════════════════════════════
code("""\
print('=' * 80)
print('  LAGRANGE SINDy - BEST CONFIG PER DOF (by ATE)')
print('=' * 80)
for rn in RUNS:
    print(f'\\n  -- {rn.upper()} --')
    for dn in DOF_LABELS:
        sub = df_lagrange[(df_lagrange['run']==rn) & (df_lagrange['dof']==dn)].dropna(subset=['ATE'])
        if sub.empty: print(f'    {dn}: no valid results'); continue
        best = sub.loc[sub['ATE'].idxmin()]
        print(f'    {dn}: Lagrange/{best["optimizer"]} th={best["threshold"]:.4f} '
              f'| ATE={best["ATE"]:.4f} [{best["ate_grade"]}] '
              f'| MSE_val={best["MSE_val"]:.4e} | R2={best["R2_val"]:.4f} | {best["n_terms"]}t')

# Lagrange heatmap
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Lagrange: Median ATE per Threshold x Optimizer', fontsize=13, fontweight='bold')
for ax, dn in zip(axes, DOF_LABELS):
    sub = df_lagrange[df_lagrange['dof']==dn]
    piv = sub.groupby(['optimizer','threshold'])['ATE'].median().unstack(fill_value=np.nan)
    if piv.empty: ax.set_title(f'{dn} (no data)'); continue
    mat = np.log10(piv.values.clip(min=1e-6))
    im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f'{v:.0e}' for v in piv.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title(dn, fontweight='bold')
    plt.colorbar(im, ax=ax, label='log10(ATE)')
plt.tight_layout(); plt.show()
""")

# ═══════════════════════════════════════════════════════════════
# CELL 28: §7c header
# ═══════════════════════════════════════════════════════════════
md("""## §7c — Hybrid SINDy: Best Configurations

Per-run best Hybrid model by ATE.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 29: Hybrid per-run analysis
# ═══════════════════════════════════════════════════════════════
code("""\
print('=' * 80)
print('  HYBRID SINDy - BEST CONFIG PER DOF (by ATE)')
print('=' * 80)
for rn in RUNS:
    print(f'\\n  -- {rn.upper()} --')
    for dn in DOF_LABELS:
        sub = df_hybrid[(df_hybrid['run']==rn) & (df_hybrid['dof']==dn)].dropna(subset=['ATE'])
        if sub.empty: print(f'    {dn}: no valid results'); continue
        best = sub.loc[sub['ATE'].idxmin()]
        print(f'    {dn}: {best["library"]}/{best["optimizer"]} th={best["threshold"]:.4f} '
              f'| ATE={best["ATE"]:.4f} [{best["ate_grade"]}] '
              f'| MSE_val={best["MSE_val"]:.4e} | R2={best["R2_val"]:.4f}')

# Hybrid heatmap
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Hybrid: Median ATE per Threshold x Optimizer', fontsize=13, fontweight='bold')
for ax, dn in zip(axes, DOF_LABELS):
    sub = df_hybrid[df_hybrid['dof']==dn]
    piv = sub.groupby(['optimizer','threshold'])['ATE'].median().unstack(fill_value=np.nan)
    if piv.empty: ax.set_title(f'{dn} (no data)'); continue
    mat = np.log10(piv.values.clip(min=1e-6))
    im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f'{v:.0e}' for v in piv.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title(dn, fontweight='bold')
    plt.colorbar(im, ax=ax, label='log10(ATE)')
plt.tight_layout(); plt.show()
""")

# ═══════════════════════════════════════════════════════════════
# CELL 30: §8 header
# ═══════════════════════════════════════════════════════════════
md("""## §8 — Cross-Algorithm ATE Comparison

Grade distribution tables and ranked best configurations.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 31: ATE tables + grade chart
# ═══════════════════════════════════════════════════════════════
code("""\
print('=' * 80)
print('  TABLE 1: Grade Distribution per Algorithm')
print('=' * 80)
gd = df_all.groupby('algorithm')['ate_grade'].value_counts().unstack(fill_value=0)
gd['Total'] = gd.sum(axis=1)
gd['%A'] = (gd.get('A', 0) / gd['Total'] * 100).round(1)
gd['%A+B'] = ((gd.get('A', 0) + gd.get('B', 0)) / gd['Total'] * 100).round(1)
print(gd.to_string())

print('\\n' + '=' * 80)
print('  TABLE 2: Grade Distribution per Algorithm x Library')
print('=' * 80)
gd2 = df_all.groupby(['algorithm','library'])['ate_grade'].value_counts().unstack(fill_value=0)
styled_gd2 = gd2.style.background_gradient(cmap='YlGn', axis=1)
display(styled_gd2)

# Grade distribution bar chart
fig, ax = plt.subplots(figsize=(10, 5))
grade_colors = {'A': '#2ca02c', 'B': '#98df8a', 'C': '#ffbb78', 'D': '#ff7f0e', 'F': '#d62728'}
grades = ['A', 'B', 'C', 'D', 'F']
x = np.arange(len(algos))
bottom = np.zeros(len(algos))
for g in grades:
    vals = [gd.loc[a, g] if g in gd.columns and a in gd.index else 0 for a in algos]
    ax.bar(x, vals, bottom=bottom, label=f'Grade {g}', color=grade_colors[g])
    bottom += vals
ax.set_xticks(x); ax.set_xticklabels(algos)
ax.set_ylabel('Count'); ax.set_title('ATE Grade Distribution per Algorithm')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()
""")

# ═══════════════════════════════════════════════════════════════
# CELL 32: Best model per run x DOF
# ═══════════════════════════════════════════════════════════════
code("""\
print('=' * 90)
print('  BEST MODEL PER RUN x DOF (lowest ATE)')
print('=' * 90)
for rn in RUNS:
    for dn in DOF_LABELS:
        sub = df_all[(df_all['run']==rn) & (df_all['dof']==dn)].dropna(subset=['ATE'])
        if sub.empty: continue
        best = sub.loc[sub['ATE'].idxmin()]
        print(f'  {rn:>8}  {dn:>6}: {best["algorithm"]:>10} {best["library"]:>12} '
              f'{best["optimizer"]:>12} th={best["threshold"]:.4f} '
              f'| ATE={best["ATE"]:.4f} [{best["ate_grade"]}] | R2={best["R2_val"]:.4f}')

# Best model summary table
rows_t1 = []
for rn in RUNS:
    for dn in DOF_LABELS:
        sub = df_all[(df_all['run']==rn) & (df_all['dof']==dn)].dropna(subset=['ATE'])
        if sub.empty: continue
        best = sub.loc[sub['ATE'].idxmin()]
        rows_t1.append({'run': rn, 'dof': dn, 'algorithm': best['algorithm'],
                        'library': best['library'], 'optimizer': best['optimizer'],
                        'threshold': best['threshold'], 'ATE': best['ATE'],
                        'grade': best['ate_grade'], 'R2_val': best['R2_val']})
df_t1 = pd.DataFrame(rows_t1)
styled_t1 = df_t1.style.map(lambda v: f'background-color: {grade_colors.get(v, "")}' if v in grade_colors else '',
                              subset=['grade']).format({'ATE': '{:.4f}', 'R2_val': '{:.4f}', 'threshold': '{:.4f}'})
display(styled_t1)
""")

# ═══════════════════════════════════════════════════════════════
# CELL 33: §9 header
# ═══════════════════════════════════════════════════════════════
md("""## §9 — R² Analysis

Ranked R² tables and per-algorithm comparison.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 34: R² analysis
# ═══════════════════════════════════════════════════════════════
code("""\
print('=' * 80)
print('  TABLE: Best R2 per (Algorithm, Library, Optimizer)')
print('=' * 80)
r2_best = df_all.groupby(['algorithm','library','optimizer'])['R2_val'].max().reset_index()
r2_best = r2_best.sort_values('R2_val', ascending=False).head(20)
r2_best['Rank'] = range(1, len(r2_best)+1)
print(r2_best.to_string(index=False, float_format='{:.6f}'.format))

# R² box per algorithm x DOF
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('R² Distribution per Algorithm x DOF', fontsize=13, fontweight='bold')
for ax, dn in zip(axes, DOF_LABELS):
    data = []; labels = []
    for algo in algos:
        vals = df_all[(df_all['algorithm']==algo) & (df_all['dof']==dn)]['R2_val'].dropna()
        vals = vals[vals > -10]
        data.append(vals.values); labels.append(algo)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, algo in zip(bp['boxes'], algos):
        patch.set_facecolor(algo_colors.get(algo, '#999')); patch.set_alpha(0.5)
    ax.set_title(dn, fontweight='bold'); ax.axhline(0, color='red', ls='--', lw=1)
    ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

print('\\n' + '=' * 80)
print('  MEAN R2 PER ALGORITHM x DOF (best config)')
print('=' * 80)
mean_r2 = df_all.groupby(['algorithm','dof'])['R2_val'].max().unstack()
print(mean_r2.to_string(float_format='{:.4f}'.format))
""")

# ═══════════════════════════════════════════════════════════════
# CELL 35: R² per-run
# ═══════════════════════════════════════════════════════════════
code("""\
# R² heatmap: best R² per run x DOF
fig, axes = plt.subplots(1, len(algos), figsize=(5*len(algos), 5))
if len(algos) == 1: axes = [axes]
for ax, algo in zip(axes, algos):
    sub = df_all[df_all['algorithm']==algo]
    piv = sub.groupby(['run','dof'])['R2_val'].max().unstack(fill_value=-1)
    im = ax.imshow(piv.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f'{piv.values[i,j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_title(algo, fontweight='bold')
plt.colorbar(im, ax=axes, label='R²', shrink=0.8)
fig.suptitle('Best R² per Run x DOF', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
""")

# ═══════════════════════════════════════════════════════════════
# CELL 36: §10 header
# ═══════════════════════════════════════════════════════════════
md("""## §10 — MSE Analysis

Detailed MSE tables and comparison plots.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 37: MSE analysis
# ═══════════════════════════════════════════════════════════════
code("""\
print('=' * 80)
print('  MSE SUMMARY: Median per Algorithm')
print('=' * 80)
mse_summary = df_all.groupby('algorithm')[['MSE_train','MSE_val','MSE_test']].median()
print(mse_summary.to_string(float_format='{:.6f}'.format))

# MSE best per algorithm x DOF
print('\\n' + '=' * 80)
print('  BEST MSE_val per Algorithm x DOF')
print('=' * 80)
mse_best = df_all.groupby(['algorithm','dof'])['MSE_val'].min().unstack()
print(mse_best.to_string(float_format='{:.6e}'.format))

# MSE per DOF box plot
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('MSE_val Distribution per Algorithm x DOF', fontsize=13, fontweight='bold')
for ax, dn in zip(axes, DOF_LABELS):
    data = []; labels = []
    for algo in algos:
        vals = df_all[(df_all['algorithm']==algo) & (df_all['dof']==dn)]['MSE_val'].dropna()
        vals = vals[vals < vals.quantile(0.95)] if len(vals) > 10 else vals
        data.append(vals.values); labels.append(algo)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, algo in zip(bp['boxes'], algos):
        patch.set_facecolor(algo_colors.get(algo, '#999')); patch.set_alpha(0.5)
    ax.set_title(dn, fontweight='bold')
    ax.set_ylabel('MSE_val' if ax is axes[0] else ''); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
""")

# ═══════════════════════════════════════════════════════════════
# CELL 38: MSE detailed per run
# ═══════════════════════════════════════════════════════════════
code("""\
# MSE heatmap: best MSE per run x DOF
fig, axes = plt.subplots(1, len(algos), figsize=(5*len(algos), 5))
if len(algos) == 1: axes = [axes]
for ax, algo in zip(axes, algos):
    sub = df_all[df_all['algorithm']==algo]
    piv = sub.groupby(['run','dof'])['MSE_val'].min().unstack(fill_value=np.nan)
    log_mat = np.log10(piv.values.clip(min=1e-10))
    im = ax.imshow(log_mat, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f'{piv.values[i,j]:.2e}', ha='center', va='center', fontsize=7)
    ax.set_title(algo, fontweight='bold')
plt.colorbar(im, ax=axes, label='log10(MSE)', shrink=0.8)
fig.suptitle('Best MSE_val per Run x DOF', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
""")

# ═══════════════════════════════════════════════════════════════
# CELL 39: §11 header
# ═══════════════════════════════════════════════════════════════
md("""## §11 — Final Summary

Overall comparison, best models, and grade report.
""")

# ═══════════════════════════════════════════════════════════════
# CELL 40: Final summary
# ═══════════════════════════════════════════════════════════════
code("""\
print('=' * 80)
print('  FINAL SUMMARY - ALL ALGORITHMS')
print('=' * 80)

for metric, ascending in [('ATE', True), ('MSE_val', True), ('R2_val', False)]:
    best_per_algo = df_all.groupby('algorithm')[metric].min() if ascending else df_all.groupby('algorithm')[metric].max()
    winner = best_per_algo.idxmin() if ascending else best_per_algo.idxmax()
    best_val = best_per_algo.min() if ascending else best_per_algo.max()
    print(f'\\n  -- Best by {metric} --')
    print(f'    Winner: {winner} ({metric}={best_val:.4f})')
    for algo in algos:
        print(f'      {algo}: {best_per_algo[algo]:.4f}')

# Winners table per run x DOF
winner_rows = []
for rn in RUNS:
    for dn in DOF_LABELS:
        sub = df_all[(df_all['run']==rn) & (df_all['dof']==dn)].dropna(subset=['ATE'])
        if sub.empty: continue
        best = sub.loc[sub['ATE'].idxmin()]
        winner_rows.append({'Run': rn, 'DOF': dn, 'Algorithm': best['algorithm'],
                           'Library': best['library'], 'Optimizer': best['optimizer'],
                           'Threshold': best['threshold'],
                           'ATE': best['ATE'], 'Grade': best['ate_grade'],
                           'R2': best['R2_val'], 'MSE_val': best['MSE_val']})
df_winners = pd.DataFrame(winner_rows)

styled_w = df_winners.style.map(
    lambda v: f'background-color: {grade_colors.get(v, "")}' if v in grade_colors else '',
    subset=['Grade']
).format({'ATE': '{:.4f}', 'R2': '{:.4f}', 'MSE_val': '{:.4e}', 'Threshold': '{:.4f}'})
display(styled_w)

# Final grade summary chart
fig, ax = plt.subplots(figsize=(10, 5))
grades = ['A', 'B', 'C', 'D', 'F']
x = np.arange(len(algos))
gd_final = df_all.groupby('algorithm')['ate_grade'].value_counts().unstack(fill_value=0)
bottom = np.zeros(len(algos))
for g in grades:
    vals = [gd_final.loc[a, g] if g in gd_final.columns and a in gd_final.index else 0 for a in algos]
    pcts = [v / gd_final.loc[a].sum() * 100 if a in gd_final.index else 0 for v, a in zip(vals, algos)]
    ax.bar(x, pcts, bottom=bottom, label=f'Grade {g}', color=grade_colors[g])
    bottom += pcts
ax.set_xticks(x); ax.set_xticklabels(algos)
ax.set_ylabel('Percentage'); ax.set_title('ATE Grade Distribution (%)')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

print(f'\\n[DONE] Hoop SINDy Pipeline complete.')
print(f'Total configurations evaluated: {len(df_all)}')
csv_path = os.path.join(str(DATA_DIR), 'hoop_full_sweep_results.csv')
print(f'Results saved to: {csv_path}')
""")

# ═══════════════════════════════════════════════════════════════
# WRITE NOTEBOOK
# ═══════════════════════════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.7"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(OUTPATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Created: {OUTPATH}")
print(f"Total cells: {len(cells)}")
print(f"Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
print(f"Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
