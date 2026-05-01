"""
Injects a unified recall heatmap cell into SINDy_Pipeline.ipynb.
Covers: SINDy, Lagrangian SINDy, Hybrid SINDy, Symbolic Regression, Hybrid SR.
"""
import json, os, uuid, re

NB_PATH = r"c:\Users\braid\OneDrive\Desktop\Data Driven Modeling Project\pipeline\SINDy_Pipeline.ipynb"

CELL_CODE = r"""
# ═════════════════════════════════════════════════════════════════
# UNIFIED RECALL HEATMAP
# SINDy · Lagrangian SINDy · Hybrid SINDy · Symbolic Regression · Hybrid SR
# ═════════════════════════════════════════════════════════════════
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE = r"c:\Users\braid\OneDrive\Desktop\Data Driven Modeling Project"
PLOT_DIR = os.path.join(BASE, "ppt_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, "pipeline"))
from pipeline.eom_comparison import score_lagrange_result

DOF_LABELS_LOCAL = ['a_x', 'a_y', 'a_theta']
REGIMES_LOCAL = ['bouncing', 'rolling', 'flight']

# ── 1. SINDy and Hybrid SINDy recall from df_all ──────────────────────────
sindy_hybrid = (
    df_all[df_all['algorithm'].isin(['SINDy', 'Hybrid'])]
    .dropna(subset=['recall'])
    .groupby(['algorithm', 'regime', 'dof'])['recall']
    .max()
    .reset_index()
)
sindy_hybrid['algorithm'] = sindy_hybrid['algorithm'].map(
    {'SINDy': 'SINDy', 'Hybrid': 'Hybrid SINDy'})

# ── 2. Lagrangian SINDy recall via score_lagrange_result ──────────────────
# Lagrange fits one shared T/V model per regime — recall = fraction of true
# Lagrangian terms (T:xdot^2, T:ydot^2, …) recovered. Same value for all DOFs.
lag_rows = []
for regime in REGIMES_LOCAL:
    sub = df_lagrange[df_lagrange['regime'] == regime].dropna(subset=['ATE'])
    if sub.empty:
        continue
    mean_ate = sub.groupby(['library', 'optimizer', 'threshold'])['ATE'].mean()
    if mean_ate.empty:
        continue
    best_key = mean_ate.idxmin()
    lib_b, opt_b, th_b = best_key
    key = (regime, lib_b, opt_b, th_b)
    if key not in lag_cache:
        continue
    res = lag_cache[key]
    score = score_lagrange_result(res, label=f'{regime}/{lib_b}/{opt_b}')
    recall_val = score['recall']
    for dof in DOF_LABELS_LOCAL:
        lag_rows.append({
            'algorithm': 'Lagrangian SINDy',
            'regime': regime,
            'dof': dof,
            'recall': recall_val,
        })
df_lag_recall = pd.DataFrame(lag_rows)

# ── 3. PySR / Hybrid PySR recall from CSV ────────────────────────────────
pysr_csv = os.path.join(BASE, 'pysr_sweep_results.csv')
df_sr = pd.read_csv(pysr_csv)
sr_recall = (
    df_sr.groupby(['algorithm', 'regime', 'dof'])['recall']
    .max()
    .reset_index()
)
sr_recall['algorithm'] = sr_recall['algorithm'].map(
    {'PySR': 'Symbolic Regression', 'Hybrid PySR': 'Hybrid SR'})

# ── 4. Combine ─────────────────────────────────────────────────────────────
df_combined = pd.concat([sindy_hybrid, df_lag_recall, sr_recall], ignore_index=True)

ALGO_ORDER = [
    'SINDy',
    'Lagrangian SINDy',
    'Hybrid SINDy',
    'Symbolic Regression',
    'Hybrid SR',
]
present = [a for a in ALGO_ORDER if a in df_combined['algorithm'].unique()]

# ── 5. Print summary table ────────────────────────────────────────────────
print('UNIFIED RECALL SUMMARY (best per algorithm × regime × DOF)\n')
for algo in present:
    sub = df_combined[df_combined['algorithm'] == algo]
    pivot = sub.groupby(['regime', 'dof'])['recall'].max().unstack(fill_value=0)
    for d in DOF_LABELS_LOCAL:
        if d not in pivot.columns:
            pivot[d] = 0.0
    pivot = pivot[DOF_LABELS_LOCAL]
    print(f'  {algo}')
    print(pivot.applymap(lambda v: f'{v:.0%}').to_string())
    print()

# ── 6. Heatmap ────────────────────────────────────────────────────────────
n_algo = len(present)
fig, axes = plt.subplots(1, n_algo, figsize=(4.5 * n_algo, 4.5))
if n_algo == 1:
    axes = [axes]

cmap = plt.cm.RdYlGn

for ax, algo in zip(axes, present):
    sub = df_combined[df_combined['algorithm'] == algo]
    pivot = sub.groupby(['regime', 'dof'])['recall'].max().unstack(fill_value=0)
    for d in DOF_LABELS_LOCAL:
        if d not in pivot.columns:
            pivot[d] = 0.0
    # Ensure consistent row order
    row_order = [r for r in REGIMES_LOCAL if r in pivot.index]
    pivot = pivot.loc[row_order, DOF_LABELS_LOCAL]

    im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(DOF_LABELS_LOCAL)))
    ax.set_xticklabels(['ẍ  (a_x)', 'ÿ  (a_y)', 'θ̈  (a_θ)'], fontsize=9)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([r.capitalize() for r in row_order], fontsize=10)

    for r in range(len(row_order)):
        for c in range(len(DOF_LABELS_LOCAL)):
            v = pivot.values[r, c]
            txt_col = 'white' if v < 0.45 else 'black'
            ax.text(c, r, f'{v:.0%}',
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', color=txt_col)

    ax.set_title(algo, fontsize=11, fontweight='bold', pad=10)

    # Only put colorbar on the last panel
    if algo == present[-1]:
        cb = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.04)
        cb.set_label('Recall (Jaccard)', fontsize=9)

fig.suptitle(
    'Structural Term Recall — Best Config per Algorithm × Regime × DOF\n'
    '100% = discovered exactly the right terms (no extras, no missing)',
    fontsize=13, fontweight='bold', y=1.03,
)
plt.tight_layout()

# Save to ppt_plots for use in presentations
save_path = os.path.join(PLOT_DIR, 'unified_recall_heatmap.png')
fig.savefig(save_path, dpi=180, bbox_inches='tight')
print(f'Saved: {save_path}')
plt.show()

print()
print('Notes:')
print('  Lagrangian SINDy: recall = fraction of true T/V Lagrangian terms recovered')
print('  (same shared model covers all 3 DOFs — value replicated across DOF columns)')
print('  SINDy / Hybrid SINDy: Jaccard recall on acceleration-domain features')
print('  Symbolic Regression / Hybrid SR: Jaccard recall from pysr_sweep_results.csv')
""".strip()

# Split into source lines
source_lines = [line + "\n" for line in CELL_CODE.split("\n")]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": uuid.uuid4().hex[:8],
    "metadata": {},
    "outputs": [],
    "source": source_lines,
}

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"].append(new_cell)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Added cell (id={new_cell['id']}) as cell #{len(nb['cells'])} in notebook.")
print("Re-open or reload the notebook to see it.")
