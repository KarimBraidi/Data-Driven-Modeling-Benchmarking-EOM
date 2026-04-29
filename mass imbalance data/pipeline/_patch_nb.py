"""Patch the last code cell of SINDy_Pipeline.ipynb with updated content."""
import json

NB_PATH = r"c:\Users\braid\OneDrive\Desktop\Data Driven Modeling Project\pipeline\SINDy_Pipeline.ipynb"

NEW_SOURCE = r'''from pipeline.eom_comparison import (
    print_eom_report, print_lagrange_report, print_hybrid_report,
    score_all, score_experiments, score_lagrange, score_lagrange_sweep,
    score_hybrid,
)
from pipeline.experiments import run_experiment, ExperimentConfig
import pandas as pd

# Normalize library names to lowercase with underscores
_LIB_NORM = {'Full': 'full', 'Unit-Consistent': 'unit_consistent'}

# =====================================================================
# 0. Matched SINDy Sweep - same configs as Hybrid, both libraries
#    5 configs x 3 regimes x 2 libs x 3 DOFs = 90 eqs
#    (Hybrid: 5 x 3 x 1 lib x 3 DOFs = 45 eqs - exactly half)
# =====================================================================
_regime_first_run = {
    regime: all_runs[regime][0].name
    for regime in ['flight', 'rolling', 'bouncing']
}

sindy_sweep_matched = []

for hcfg in hybrid_configs:
    cfg_name = hcfg["name"]
    rc = hcfg.get("regime_configs", {})
    if not rc:
        opt = hcfg.get("optimizer", "STLSQ")
        th  = hcfg.get("threshold", 0.18)
        rc = {r: {"optimizer": opt, "threshold": th} for r in _regime_first_run}

    for regime, run_name in _regime_first_run.items():
        opt = rc[regime]["optimizer"]
        th  = rc[regime]["threshold"]
        for lib_type in ['full', 'unit_consistent']:
            print(f"  {cfg_name} | {regime} | {lib_type} | {opt} th={th}")
            try:
                res = run_experiment(ExperimentConfig(
                    run_name=run_name,
                    library_type=lib_type,
                    optimizer=opt,
                    threshold=th,
                ))
                res.config.run_name = f"{cfg_name}/{regime}/{run_name}"
                sindy_sweep_matched.append(res)
            except Exception as e:
                print(f"    SKIPPED: {e}")

print(f"\n{len(sindy_sweep_matched)} SINDy fits completed "
      f"(expected {len(hybrid_configs) * 3 * 2})")

# =====================================================================
# BUILD COMBINED COMPARISON TABLE
# =====================================================================
all_dfs = []

# 1. Standard SINDy  (3 regimes x 2 libs x 3 DOFs = 18 eqs)
print_eom_report(all_sindy_results)
df_sindy = score_all(all_sindy_results)
df_sindy['library'] = df_sindy['library'].replace(_LIB_NORM)
df_sindy.insert(0, 'method', 'SINDy')
all_dfs.append(df_sindy)

# 2. SINDy Sweep - matched configs (90 eqs)
print_eom_report(sindy_sweep_matched)
df_sweep = score_experiments(sindy_sweep_matched)
df_sweep.insert(0, 'method', 'SINDy Sweep')
all_dfs.append(df_sweep)

# 3. Hybrid SINDy (45 eqs)
print_hybrid_report(hybrid_sweep)
df_hyb = score_hybrid(hybrid_sweep)
df_hyb['library'] = 'unit_consistent'
all_dfs.append(df_hyb)

# 4. Lagrange SINDy
df_lag = score_lagrange_sweep(lag_opt_sweep)
df_lag['library'] = 'Lagrange T/V'
all_dfs.append(df_lag)
print_lagrange_report(lag_best)

# =====================================================================
# COMBINED SUMMARY TABLE
# =====================================================================
print(f"\n\n{'='*80}")
print("  COMBINED GRADE SUMMARY - ALL METHODS")
print(f"{'='*80}\n")

df_all = pd.concat(all_dfs, ignore_index=True)

print("Equations scored per method x library:")
print(df_all.groupby(['method', 'library']).size().to_string())
print()

show_cols = ['method', 'library', 'grade', 'recall', 'precision']
extra = ['dof', 'regime', 'config', 'run', 'optimizer', 'threshold', 'alpha',
         'found', 'expected', 'spurious']
show_cols += [c for c in extra if c in df_all.columns]
display(df_all[show_cols])

print("\nGrade distribution per method x library:")
grade_pivot = df_all.groupby(['method', 'library'])['grade'].value_counts().unstack(fill_value=0)
for g in ['A', 'B', 'C', 'D', 'F']:
    if g not in grade_pivot.columns:
        grade_pivot[g] = 0
display(grade_pivot[['A', 'B', 'C', 'D', 'F']])

print("\nMean recall and precision per method x library:")
mean_metrics = df_all.groupby(['method', 'library'])[['recall', 'precision']].mean()
display(mean_metrics.style.format('{:.1%}'))

print("\n\nBest equations (highest recall) per method x library:")
for (method, library), sub in df_all.groupby(['method', 'library']):
    best = sub.sort_values(['recall', 'precision'], ascending=False).head(3)
    print(f"\n  {method} [{library}]:")
    for _, row in best.iterrows():
        id_parts = [f"{k}={row[k]}" for k in ['regime','config','run','optimizer','threshold','alpha']
                    if k in row.index and pd.notna(row[k])]
        print(f"    Grade {row['grade']}  recall={row['recall']:.0%}  prec={row['precision']:.0%}"
              f"  ({', '.join(id_parts)})")
'''

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the last code cell (cell 62)
last_code_idx = None
for i in range(len(nb['cells']) - 1, -1, -1):
    if nb['cells'][i]['cell_type'] == 'code':
        last_code_idx = i
        break

if last_code_idx is None:
    print("ERROR: No code cell found")
else:
    # Convert source to list of lines with \n
    lines = NEW_SOURCE.split('\n')
    source_lines = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        source_lines.append(lines[-1])
    else:
        # Remove trailing empty line's \n
        pass

    nb['cells'][last_code_idx]['source'] = source_lines
    nb['cells'][last_code_idx]['outputs'] = []
    nb['cells'][last_code_idx]['execution_count'] = None

    with open(NB_PATH, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Patched cell {last_code_idx} ({len(source_lines)} lines)")
    # Verify
    print("First 3 lines:", source_lines[:3])
    print("Contains 'SINDy Sweep':", any('SINDy Sweep' in l for l in source_lines))
    print("Contains 'Threshold Sweep':", any('Threshold Sweep' in l for l in source_lines))
