"""
visualization.py – Plotting utilities for SINDy experiment analysis.

Provides:
  - Pareto front (MSE vs sparsity)
  - MSE vs threshold sweep
  - Optimizer comparison bar charts
  - Equation clustering / heatmap
  - Trajectory rollout comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Optional, Tuple
from itertools import cycle


# ── Colour / style defaults ──────────────────────────────────────────────────

_OPT_COLORS = {
    "STLSQ": "tab:blue",
    "LASSO": "tab:orange",
    "Ridge": "tab:green",
    "ElasticNet": "tab:red",
    "SR3_L0": "tab:purple",
    "SR3_L1": "tab:brown",
    "SR3_L2": "tab:pink",
    "SSR_coef": "tab:gray",
    "SSR_resid": "tab:olive",
    "FROLS": "tab:cyan",
}

_LIB_MARKERS = {"full": "o", "unit_consistent": "s"}


def _opt_color(name: str) -> str:
    return _OPT_COLORS.get(name, "black")


# ── 1. Pareto front: MSE vs number of active terms ──────────────────────────

def pareto_front(df, dof: str = "a_x",
                 mse_col: str = "mse_val",
                 ax=None, figsize=(8, 5)):
    """Scatter plot of MSE vs #terms coloured by optimizer, shaped by library.

    Parameters
    ----------
    df : pandas DataFrame from GridResult.to_dataframe()
         Must contain columns: optimizer, library, n_terms, <mse_col>, dof
    """
    sub = df[df["dof"] == dof]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for (opt, lib), grp in sub.groupby(["optimizer", "library"]):
        ax.scatter(grp["n_terms"], grp[mse_col],
                   c=_opt_color(opt),
                   marker=_LIB_MARKERS.get(lib, "^"),
                   label=f"{opt} / {lib}",
                   alpha=0.7, edgecolors="k", linewidths=0.3, s=50)

    ax.set_xlabel("Number of active terms")
    ax.set_ylabel(mse_col.replace("_", " ").upper())
    ax.set_yscale("log")
    ax.set_title(f"Pareto Front — {dof}")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    return ax


# ── 2. MSE vs threshold (sweep curves) ──────────────────────────────────────

def mse_vs_threshold(df, dof: str = "a_x",
                     mse_col: str = "mse_val",
                     ax=None, figsize=(8, 5)):
    """Line plot of MSE vs threshold for each optimizer×library combo."""
    sub = df[df["dof"] == dof].copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for (opt, lib), grp in sub.groupby(["optimizer", "library"]):
        grp_sorted = grp.sort_values("threshold")
        ls = "-" if lib == "unit_consistent" else "--"
        ax.plot(grp_sorted["threshold"], grp_sorted[mse_col],
                marker=_LIB_MARKERS.get(lib, "^"), color=_opt_color(opt),
                linestyle=ls, label=f"{opt}/{lib}", alpha=0.8, markersize=5)

    ax.set_xlabel("Threshold / α")
    ax.set_ylabel(mse_col.replace("_", " ").upper())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"MSE vs Threshold — {dof}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    return ax


# ── 3. Optimizer comparison bar chart ────────────────────────────────────────

def optimizer_comparison(df, dof: str = "a_x",
                         metric: str = "mse_val",
                         best_per: str = "optimizer",
                         ax=None, figsize=(10, 5)):
    """Bar chart comparing best metric value per optimizer (or per optimizer×library).

    Selects the configuration (threshold, degree) that minimises the metric
    for each group.
    """
    sub = df[df["dof"] == dof]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if best_per == "optimizer":
        group_cols = ["optimizer"]
    else:
        group_cols = ["optimizer", "library"]

    best = sub.loc[sub.groupby(group_cols)[metric].idxmin()]
    best = best.sort_values(metric)

    labels = best.apply(
        lambda r: f"{r['optimizer']}\n{r.get('library','')}\nth={r['threshold']}", axis=1)
    colors = [_opt_color(o) for o in best["optimizer"]]

    bars = ax.bar(range(len(best)), best[metric], color=colors, edgecolor="k",
                  linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"Best {metric} per {best_per} — {dof}")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    # Annotate with #terms
    for bar, n in zip(bars, best["n_terms"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{n} terms", ha="center", va="bottom", fontsize=7)
    return ax


# ── 4. Equation coefficient heatmap ─────────────────────────────────────────

def equation_heatmap(results_list, labels: List[str],
                     dof: str = "a_x",
                     top_n: int = 20,
                     ax=None, figsize=(12, 6)):
    """Heatmap of coefficient values across experiments for a single DOF.

    Parameters
    ----------
    results_list : list of ExperimentResult objects
    labels : display label for each result
    dof : which DOF to plot
    top_n : show only the top_n most frequently active features
    """
    # Collect coefficient vectors and feature names
    all_names = set()
    coef_dicts = []
    for res in results_list:
        if res.sindy_fit is None or dof not in res.sindy_fit.results:
            coef_dicts.append({})
            continue
        r = res.sindy_fit.results[dof]
        d = {n: c for n, c in zip(r.feature_names, r.coefs)}
        coef_dicts.append(d)
        all_names.update(n for n, c in d.items() if abs(c) > 1e-12)

    if not all_names:
        print("No active features found.")
        return None

    # Rank features by frequency of appearance
    freq = {}
    for d in coef_dicts:
        for n in d:
            if abs(d[n]) > 1e-12:
                freq[n] = freq.get(n, 0) + 1
    sorted_names = sorted(freq, key=lambda n: freq[n], reverse=True)[:top_n]

    # Build matrix
    mat = np.zeros((len(results_list), len(sorted_names)))
    for i, d in enumerate(coef_dicts):
        for j, n in enumerate(sorted_names):
            mat[i, j] = d.get(n, 0.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    vmax = np.max(np.abs(mat)) or 1
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Coefficient Heatmap — {dof}")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Coefficient value")
    return ax


# ── 5. Trajectory rollout comparison ────────────────────────────────────────

def trajectory_rollout(t: np.ndarray,
                       y_true: np.ndarray,
                       predictions: Dict[str, np.ndarray],
                       dof_name: str = "x",
                       split_times: Tuple[float, float] = (3.25, 3.75),
                       ax=None, figsize=(12, 4)):
    """Plot true acceleration vs multiple model predictions.

    Parameters
    ----------
    t : (N,) time array
    y_true : (N,) true acceleration
    predictions : {label: y_pred (N,)} dict of model predictions
    dof_name : label for y-axis
    split_times : (t_train_end, t_val_end) for shading
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(t, y_true, "k-", lw=0.8, alpha=0.6, label="True")

    colors = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red",
                     "tab:purple", "tab:brown"])
    for label, y_p in predictions.items():
        c = next(colors)
        valid = ~np.isnan(y_p)
        ax.plot(t[valid], y_p[valid], color=c, lw=0.7, alpha=0.8, label=label)

    # Shade train / val / test
    t0, t1 = split_times
    ax.axvspan(t[0], t0, color="green", alpha=0.06, label="Train")
    ax.axvspan(t0, t1, color="orange", alpha=0.06, label="Val")
    ax.axvspan(t1, t[-1], color="red", alpha=0.06, label="Test")
    ax.axvline(t0, color="gray", ls="--", lw=0.5)
    ax.axvline(t1, color="gray", ls="--", lw=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Acceleration — {dof_name}")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    return ax


# ── 6. Multi-DOF summary dashboard ──────────────────────────────────────────

def summary_dashboard(df, mse_col: str = "mse_val", figsize=(16, 12)):
    """3×2 dashboard: Pareto front + MSE vs threshold for each DOF."""
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    dofs = ["a_x", "a_y", "a_theta"]

    for i, dof in enumerate(dofs):
        pareto_front(df, dof=dof, mse_col=mse_col, ax=axes[i, 0])
        mse_vs_threshold(df, dof=dof, mse_col=mse_col, ax=axes[i, 1])

    fig.tight_layout()
    return fig


# ── 7. Sparsity pattern comparison ──────────────────────────────────────────

def sparsity_pattern(results_list, labels: List[str],
                     dof: str = "a_x", ax=None, figsize=(12, 4)):
    """Binary heatmap showing which features are active (nonzero) per experiment."""
    all_names = set()
    coef_dicts = []
    for res in results_list:
        if res.sindy_fit is None or dof not in res.sindy_fit.results:
            coef_dicts.append({})
            continue
        r = res.sindy_fit.results[dof]
        d = {n: c for n, c in zip(r.feature_names, r.coefs)}
        coef_dicts.append(d)
        all_names.update(n for n, c in d.items() if abs(c) > 1e-12)

    if not all_names:
        return None

    sorted_names = sorted(all_names)
    mat = np.zeros((len(results_list), len(sorted_names)))
    for i, d in enumerate(coef_dicts):
        for j, n in enumerate(sorted_names):
            mat[i, j] = 1.0 if abs(d.get(n, 0.0)) > 1e-12 else 0.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(mat, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Sparsity Pattern — {dof}")
    return ax
