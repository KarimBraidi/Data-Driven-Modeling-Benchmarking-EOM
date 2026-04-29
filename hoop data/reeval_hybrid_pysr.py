"""
reeval_hybrid_pysr.py

Re-evaluates Hybrid PySR equations (discovered on pooled data) on each
individual run's test set, replacing the single 'pooled' row with 5 per-run
rows — making Hybrid PySR comparable to regular PySR (5 runs × 3 DOFs × 2 parsimonies = 30).
"""
import sys
import numpy as np
import pandas as pd
import sympy as sp
from pathlib import Path
from scipy.signal import butter, filtfilt, savgol_filter, medfilt
from scipy.integrate import cumulative_trapezoid
from sklearn.metrics import mean_squared_error

# ── paths ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
sys.path.insert(0, str(DATA_DIR))
from run_sindy_analysis import preprocess_hoop_data, detrend_custom

CSV_FILES = {
    'run_1': 'OR_20250903_203926.csv',
    'run_2': 'OR_20250903_204044.csv',
    'run_3': 'OR_20250903_204119.csv',
    'run_4': 'OR_20250903_204158.csv',
    'run_5': 'OR_D422CD005685_20250903_204229.csv',
}

FS = 120
CUTOFF_ACC, CUTOFF_GYR = 20, 15
SG_WIN, SG_ORD = 11, 3

FEAT_NAMES_BASE = ['px', 'py', 'pz', 'vx', 'vy', 'vz']
FEAT_NAMES_FULL = FEAT_NAMES_BASE + ['wx', 'wy', 'wz',
                                      'sin_ex', 'sin_ey', 'sin_ez',
                                      'cos_ex', 'cos_ey', 'cos_ez']
DOF_LABELS = ['a_x', 'a_y', 'a_z']


# ── preprocessing (same as notebook) ────────────────────────────────────────
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
    ts, te = max(5.0, float(t[0])), min(21.0, float(t[-1]))
    m = (t >= ts) & (t <= te)
    t, pos, vel, acc = t[m], pos[m], vel[m], acc[m]
    angles = angles[m]
    df2 = df.iloc[np.where(m)[0]].reset_index(drop=True)
    omega = np.deg2rad(df2[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy(dtype=float))
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

def build_features(rd, with_full=False):
    X = np.hstack([rd['pos'], rd['vel']])
    if with_full:
        X = np.hstack([X, rd['omega'],
                       np.sin(rd['angles']), np.cos(rd['angles'])])
    return X

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def compute_open_loop_ate(y_pred_val, pos_val, vel_val, dt):
    pos_int = np.cumsum(y_pred_val) * dt
    vel_int = np.cumsum(pos_int) * dt
    ate_pos = np.mean(np.abs(pos_int - pos_val))
    ate_vel = np.mean(np.abs(vel_int - vel_val))
    return float(ate_pos + ate_vel)

def ate_grade(v):
    if v < 0.01: return 'A'
    if v < 0.05: return 'B'
    if v < 0.15: return 'C'
    if v < 0.50: return 'D'
    return 'F'

def make_lambdified(eq_str, feat_names):
    """Parse sympy equation string and return a numpy-callable function."""
    syms = {n: sp.Symbol(n) for n in feat_names}
    try:
        expr = sp.sympify(eq_str, locals=syms)
        fn = sp.lambdify(list(syms.values()), expr, modules='numpy')
        return fn, list(syms.values())
    except Exception as e:
        return None, str(e)

def eval_equation(fn, sym_list, X):
    """Evaluate lambdified equation on feature matrix X (N × F)."""
    try:
        args = [X[:, i] for i in range(len(sym_list))]
        out = fn(*args)
        if np.isscalar(out):
            out = np.full(len(X), float(out))
        return np.asarray(out, dtype=float)
    except Exception:
        return np.full(len(X), np.nan)


# ── load data ───────────────────────────────────────────────────────────────
print("Loading run data...")
all_runs = {}
for name, fname in CSV_FILES.items():
    all_runs[name] = preprocess_csv(DATA_DIR / fname)
    print(f"  {name}: N={all_runs[name]['N']}")

RUNS = list(all_runs.keys())

# ── load CSV ─────────────────────────────────────────────────────────────────
csv_path = DATA_DIR / 'hoop_pysr_sweep_results.csv'
df = pd.read_csv(csv_path)
print(f"\nLoaded {len(df)} rows from CSV")
print(f"Hybrid PySR rows: {(df['algorithm']=='Hybrid PySR').sum()}")

# ── re-evaluate Hybrid PySR on each run ─────────────────────────────────────
hybrid_rows = df[df['algorithm'] == 'Hybrid PySR'].copy()
new_rows = []

for _, row in hybrid_rows.iterrows():
    eq_str   = str(row['equation'])
    lib      = str(row['library'])
    pars     = float(row['parsimony'])
    mxsz     = int(row['maxsize'])
    dof      = str(row['dof'])
    d        = DOF_LABELS.index(dof)
    n_terms  = row['n_terms']
    mse_tr   = row['MSE_train']
    mse_val  = row['MSE_val']
    r2_val   = row['R2_val']
    with_full = (lib == 'Full')
    feat_names = FEAT_NAMES_FULL if with_full else FEAT_NAMES_BASE

    fn, sym_list = make_lambdified(eq_str, feat_names)
    if fn is None:
        print(f"  SKIP (parse error): {eq_str[:60]} — {sym_list}")
        continue

    for rn in RUNS:
        rd = all_runs[rn]
        X  = build_features(rd, with_full)
        y  = rd['acc'][:, d]

        yp_val  = eval_equation(fn, sym_list, X[rd['idx_val']])
        yp_te   = eval_equation(fn, sym_list, X[rd['idx_test']])

        mse_te  = float(mean_squared_error(y[rd['idx_test']], yp_te))
        r2_te   = compute_r2(y[rd['idx_test']], yp_te)
        ate_v   = compute_open_loop_ate(
            yp_val,
            rd['pos'][rd['idx_val'], d],
            rd['vel'][rd['idx_val'], d],
            rd['dt'])

        new_rows.append({
            'algorithm': 'Hybrid PySR', 'run': rn,
            'library': lib, 'parsimony': pars,
            'maxsize': mxsz, 'dof': dof,
            'n_terms': n_terms,
            'MSE_train': mse_tr, 'MSE_val': mse_val,
            'MSE_test': mse_te,
            'R2_val': r2_val, 'R2_test': r2_te,
            'ATE': ate_v, 'ate_grade': ate_grade(ate_v),
            'equation': eq_str,
        })
        print(f"  {rn} | {lib:5s} | p={pars} ms={mxsz} | {dof}: ATE={ate_v:.4f} ({ate_grade(ate_v)})")

# ── rebuild CSV: replace pooled Hybrid PySR with per-run rows ───────────────
df_other = df[df['algorithm'] != 'Hybrid PySR']
df_new   = pd.DataFrame(new_rows)
df_out   = pd.concat([df_other, df_new], ignore_index=True)

df_out.to_csv(csv_path, index=False)
print(f"\nSaved {len(df_out)} rows to {csv_path}")
print(f"Hybrid PySR rows: {(df_out['algorithm']=='Hybrid PySR').sum()} (was {len(hybrid_rows)})")
print(f"\nGrade dist (Hybrid PySR):")
print(df_new['ate_grade'].value_counts().sort_index().to_string())
