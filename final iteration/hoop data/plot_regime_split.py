"""
plot_regime_split.py — visualize Hybrid SINDy regime detection for all 5 runs
"""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.signal import butter, filtfilt, savgol_filter, medfilt
from scipy.integrate import cumulative_trapezoid
import pandas as pd

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
FS = 120; CUTOFF_ACC, CUTOFF_GYR = 20, 15; SG_WIN, SG_ORD = 11, 3

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
    arr = clip_iqr(df[acc_cols].to_numpy(), k=3.0)
    for i, c in enumerate(acc_cols): df[c] = arr[:, i]
    proc = preprocess_hoop_data(df, fs=FS)
    t = proc['t']; acc = proc['acceleration']; angles = proc['euler_angles']
    vel = proc['velocity']; pos = proc['position']
    n_st = max(1, int(FS))
    acc = acc - np.median(acc[:n_st], axis=0)
    acc[:, 2] = butter_hp(acc[:, 2], 0.3, FS)
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
    t, acc, vel, pos = t[m], acc[m], vel[m], pos[m]
    N = len(t)
    return {'t': t, 'acc': acc, 'vel': vel, 'pos': pos, 'N': N,
            'idx_train': np.arange(0, int(0.60 * N)),
            'idx_val':   np.arange(int(0.60 * N), int(0.80 * N)),
            'idx_test':  np.arange(int(0.80 * N), N)}

def detect_regimes(acc, window=15, percentile=60):
    mag = np.linalg.norm(acc, axis=1)
    mag_s = medfilt(mag, kernel_size=window)
    return (mag_s >= np.percentile(mag_s, percentile)).astype(int), mag_s, np.percentile(mag_s, percentile)

# ── Load all runs ────────────────────────────────────────────
print("Loading data...")
all_runs = {}
for name, fname in CSV_FILES.items():
    all_runs[name] = preprocess_csv(DATA_DIR / fname)
    print(f"  {name}: N={all_runs[name]['N']}")

RUNS = list(all_runs.keys())
DOF_LABELS = ['a_x', 'a_y', 'a_z']
colors_regime = {0: '#2196F3', 1: '#E91E63'}  # blue=low, pink=high

# ── Plot: one row per run, showing acceleration magnitude + regime labels ──
fig, axes = plt.subplots(len(RUNS), 1, figsize=(16, 3.5 * len(RUNS)), sharex=False)
fig.suptitle('Hybrid SINDy — Regime Detection per Run\n'
             'Blue = Low Activity (bottom 60%), Pink = High Activity (top 40%)',
             fontsize=15, fontweight='bold', y=1.01)

for ri, rn in enumerate(RUNS):
    rd = all_runs[rn]
    t = rd['t']
    labels, mag_s, threshold = detect_regimes(rd['acc'])

    ax = axes[ri]

    # shade background by regime
    for j in range(len(labels)):
        color = '#fce4ec' if labels[j] == 1 else '#e3f2fd'
        ax.axvspan(t[j], t[min(j+1, len(t)-1)], color=color, alpha=0.4, linewidth=0)

    # plot smoothed magnitude
    ax.plot(t, mag_s, color='#333', lw=1.2, label='||acc|| (smoothed)', zorder=3)
    ax.axhline(threshold, color='orange', lw=1.5, ls='--', label=f'60th pct = {threshold:.2f}', zorder=4)

    # mark train/val/test splits with vertical lines
    ax.axvline(t[rd['idx_val'][0]],  color='green',  lw=1.5, ls=':', alpha=0.8, label='train|val')
    ax.axvline(t[rd['idx_test'][0]], color='purple', lw=1.5, ls=':', alpha=0.8, label='val|test')

    # regime counts
    n_low  = int((labels == 0).sum())
    n_high = int((labels == 1).sum())
    ax.set_title(f'{rn}  —  Low: {n_low} pts ({100*n_low/len(labels):.0f}%)   '
                 f'High: {n_high} pts ({100*n_high/len(labels):.0f}%)',
                 fontsize=11, fontweight='bold', loc='left')
    ax.set_ylabel('||acc|| (m/s²)', fontsize=9)
    if ri == len(RUNS) - 1:
        ax.set_xlabel('Time (s)', fontsize=10)
    ax.grid(alpha=0.2)
    if ri == 0:
        ax.legend(fontsize=9, loc='upper right', ncol=4)

plt.tight_layout()
out = DATA_DIR.parent / 'hoop_ppt_plots' / 'hoop_regime_split.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out}")
