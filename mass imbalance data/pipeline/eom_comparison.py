"""
EOM Comparison: Score SINDy Equations Against Analytical EOM
=============================================================
Analytical EOM for an eccentric rolling disk (resolved per DOF):

  M(θ) · a = F_grav + F_vel + W_N·λ_N + W_F·λ_F

  Resolved (what SINDy is fitting):
    ẍ  = ε·R·sin(θ)·θ̈  + ε·R·cos(θ)·θ̇²  + λ_F / m
    ÿ  = −ε·R·cos(θ)·θ̈  + ε·R·sin(θ)·θ̇²  − g  + λ_N / m
    θ̈  = (ε/R)·sin(θ)·ẍ − (ε/R)·cos(θ)·ÿ  − (g·ε/R)·cos(θ)  + λ_F/(m·R)

  Contact forces (λ_N, λ_F) are NOT in the SINDy feature library,
  so those terms are inherently uncapturable.

Usage (in notebook):
    from pipeline.eom_comparison import (
        score_fit, score_all, score_experiments, score_hybrid,
        score_lagrange, print_eom_report, print_lagrange_report,
    )

    # Standard SINDy
    print_eom_report(fit)                    # one SINDyFit
    print_eom_report(all_sindy_results)      # {regime: {lib: SINDyFit}}
    print_eom_report(opt_sweep_uc)           # [ExperimentResult]

    # Hybrid SINDy
    df = score_hybrid(hybrid_sweep)          # {name: HybridSINDyResult}

    # Lagrange SINDy
    print_lagrange_report(lag_best)          # {run: (alpha, SharedResult)}
    df = score_lagrange(lag_best)            # DataFrame
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from .sindy import SINDyResult, SINDyFit, ACCEL_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# 1. EXPECTED ANALYTICAL TERMS
# ─────────────────────────────────────────────────────────────────────────────
# Each DOF maps to a list of (human_label, regex_pattern).
# Patterns match the feature names produced by library.py:
#   Unit-consistent: "sin(theta) thetaddot", "cos(theta) thetadot^2", ...
#   Full library:    "sin(theta) thetaddot", "cos(theta) thetadot", ...

EXPECTED_TERMS = {
    0: [  # xddot = ε·R·sin(θ)·θ̈ + ε·R·cos(θ)·θ̇² + λ_F/m
        ("ε·R·sin(θ)·θ̈  [Coriolis]",
         r"^sin\(theta\) thetaddot$"),
        ("ε·R·cos(θ)·θ̇² [centripetal]",
         r"^cos\(theta\) thetadot\^2$"),
    ],
    1: [  # yddot = −ε·R·cos(θ)·θ̈ + ε·R·sin(θ)·θ̇² − g + λ_N/m
        ("-ε·R·cos(θ)·θ̈  [Coriolis]",
         r"^cos\(theta\) thetaddot$"),
        ("ε·R·sin(θ)·θ̇² [centripetal]",
         r"^sin\(theta\) thetadot\^2$"),
        ("-g [gravity]",
         r"^1$"),
    ],
    2: [  # θ̈ = (ε/R)·sin(θ)·ẍ − (ε/R)·cos(θ)·ÿ − (g·ε/R)·cos(θ) + λ_F/(m·R)
        ("(ε/R)·sin(θ)·ẍ  [x-coupling]",
         r"^sin\(theta\) xddot$"),
        ("-(ε/R)·cos(θ)·ÿ [y-coupling]",
         r"^cos\(theta\) yddot$"),
        ("-(g·ε/R)·cos(θ) [grav torque]",
         r"^cos\(theta\)$"),
    ],
}

UNCAPTURABLE_TERMS = {
    0: ["λ_F/m (friction force — not in library)"],
    1: ["λ_N/m (normal force — not in library)"],
    2: ["λ_F/(m·R) (friction torque — not in library)"],
}

_COEF_TOL = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_equation(result: SINDyResult) -> dict:
    """
    Score a single SINDy equation against expected analytical terms.

    Returns dict with: dof, dof_name, grade, recall, precision,
    n_expected, n_found, n_active, n_spurious, n_zeroed, n_not_in_lib,
    found, missing, spurious, uncapturable, equation
    """
    dof = result.dof
    expected = EXPECTED_TERMS.get(dof, [])
    uncapturable = UNCAPTURABLE_TERMS.get(dof, [])

    feature_names = list(result.feature_names)
    coefs = np.asarray(result.coefs).ravel()

    # --- Match expected terms against actual features ---
    found = []
    missing = []
    matched_features = set()

    for label, pattern in expected:
        matches = [(i, fn) for i, fn in enumerate(feature_names)
                   if re.search(pattern, fn)]
        if not matches:
            missing.append({"term": label, "status": "not_in_library",
                            "feature": None})
        else:
            active = [(i, fn) for i, fn in matches
                      if abs(coefs[i]) > _COEF_TOL]
            if active:
                idx, fn = active[0]
                found.append({"term": label, "feature": fn,
                              "coef": float(coefs[idx])})
                matched_features.add(fn)
            else:
                missing.append({"term": label, "status": "zeroed_out",
                                "feature": matches[0][1]})
                matched_features.add(matches[0][1])

    # --- Spurious: active terms not matching any expected pattern ---
    spurious = []
    for fn, c in zip(feature_names, coefs):
        if abs(c) > _COEF_TOL and fn not in matched_features:
            spurious.append({"feature": fn, "coef": float(c)})

    # --- Metrics ---
    n_expected = len(expected)
    n_found = len(found)
    n_active = n_found + len(spurious)
    n_zeroed = sum(1 for m in missing if m["status"] == "zeroed_out")
    n_not_in_lib = sum(1 for m in missing if m["status"] == "not_in_library")

    recall = n_found / n_expected if n_expected > 0 else 1.0
    precision = n_found / n_active if n_active > 0 else 0.0

    # --- Grade ---
    if n_expected == 0:
        grade = "-"
    elif recall >= 1.0:
        grade = "A" if precision >= 0.5 else "B"
    elif recall >= 0.67:
        grade = "B" if precision >= 0.4 else "C"
    elif recall >= 0.33:
        grade = "C" if precision >= 0.3 else "D"
    else:
        grade = "F"

    return {
        "dof": dof,
        "dof_name": result.dof_name,
        "grade": grade,
        "recall": recall,
        "precision": precision,
        "n_expected": n_expected,
        "n_found": n_found,
        "n_active": n_active,
        "n_spurious": len(spurious),
        "n_zeroed": n_zeroed,
        "n_not_in_lib": n_not_in_lib,
        "found": found,
        "missing": missing,
        "spurious": spurious,
        "uncapturable": uncapturable,
        "equation": result.equation,
    }


def score_fit(fit: SINDyFit) -> pd.DataFrame:
    """Score all 3 DOFs in a SINDyFit. Returns a compact DataFrame."""
    rows = []
    for dof_name, result in fit.results.items():
        s = score_equation(result)
        rows.append({
            "dof": s["dof_name"],
            "grade": s["grade"],
            "recall": f"{s['recall']:.0%}",
            "precision": f"{s['precision']:.0%}",
            "found": s["n_found"],
            "expected": s["n_expected"],
            "active": s["n_active"],
            "spurious": s["n_spurious"],
            "missing_terms": ", ".join(
                m["term"].split("[")[0].strip()
                + (" [not in lib]" if m["status"] == "not_in_library"
                   else " [zeroed]")
                for m in s["missing"]
            ) or "-",
            "spurious_terms": ", ".join(
                sp["feature"] for sp in s["spurious"]
            ) or "-",
        })
    return pd.DataFrame(rows)


def score_experiments(results: list) -> pd.DataFrame:
    """Score a list of ExperimentResult objects. Returns one row per (config, DOF)."""
    rows = []
    for er in results:
        if er.sindy_fit is None:
            continue
        cfg = er.config
        for dof_name, result in er.sindy_fit.results.items():
            s = score_equation(result)
            rows.append({
                "run": cfg.run_name,
                "optimizer": cfg.optimizer,
                "threshold": cfg.threshold,
                "library": cfg.library_type,
                "dof": s["dof_name"],
                "grade": s["grade"],
                "recall": s["recall"],
                "precision": s["precision"],
                "found": s["n_found"],
                "expected": s["n_expected"],
                "active": s["n_active"],
                "spurious": s["n_spurious"],
                "mse_val": result.mse_val,
                "equation": s["equation"],
            })
    return pd.DataFrame(rows)


def score_grid(grid) -> pd.DataFrame:
    """Score a GridResult. Returns one row per (config, DOF)."""
    return score_experiments(grid.results)


def score_all(all_results: dict) -> pd.DataFrame:
    """Score a nested {regime: {lib_type: SINDyFit}} dict.

    Matches the all_sindy_results format from the notebook.
    """
    rows = []
    for regime, lib_dict in all_results.items():
        for lib_type, fit in lib_dict.items():
            for dof_name, result in fit.results.items():
                s = score_equation(result)
                rows.append({
                    "regime": regime,
                    "library": lib_type,
                    "dof": s["dof_name"],
                    "grade": s["grade"],
                    "recall": s["recall"],
                    "precision": s["precision"],
                    "found": s["n_found"],
                    "expected": s["n_expected"],
                    "active": s["n_active"],
                    "spurious": s["n_spurious"],
                    "mse_val": result.mse_val,
                    "missing_terms": ", ".join(
                        m["term"].split("[")[0].strip() for m in s["missing"]
                    ) or "-",
                    "equation": s["equation"],
                })
    return pd.DataFrame(rows)


def best_equations(results: list, metric: str = "recall") -> pd.DataFrame:
    """Find the best experiment config per DOF from a list of ExperimentResults."""
    df = score_experiments(results)
    if df.empty:
        return df
    return df.loc[df.groupby("dof")[metric].idxmax()]


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRETTY PRINTING
# ─────────────────────────────────────────────────────────────────────────────

def print_eom_report(fit_or_results, title: str = ""):
    """Pretty-print an EOM comparison report.

    Accepts:
      - SINDyFit
      - dict {regime: {lib_type: SINDyFit}}
      - list of ExperimentResult
    """
    if isinstance(fit_or_results, SINDyFit):
        _print_fit_report(fit_or_results, title)
    elif isinstance(fit_or_results, dict):
        for regime, lib_dict in fit_or_results.items():
            for lib_type, fit in lib_dict.items():
                _print_fit_report(fit, f"{regime} / {lib_type}")
    elif isinstance(fit_or_results, list):
        for er in fit_or_results:
            if er.sindy_fit is None:
                continue
            cfg = er.config
            _print_fit_report(
                er.sindy_fit,
                f"{cfg.optimizer} th={cfg.threshold} ({cfg.library_type})",
            )


def _print_fit_report(fit: SINDyFit, title: str = ""):
    """Print detailed report for one SINDyFit."""
    header = title or "EOM Scorecard"
    print(f"\n{'=' * 70}")
    print(f"  {header}")
    print(f"{'=' * 70}")

    for dof_name, result in fit.results.items():
        s = score_equation(result)
        accel = ACCEL_NAMES[s["dof"]]

        print(f"\n  {accel} ({s['dof_name']})"
              f"{'':>{44 - len(accel) - len(s['dof_name'])}}Grade: {s['grade']}")
        print(f"  {'-' * 66}")

        # Found terms
        for f in s["found"]:
            print(f"    + {f['term']:<38} -> {f['feature']}"
                  f"  (c={f['coef']:+.6f})")

        # Missing terms
        for m in s["missing"]:
            if m["status"] == "not_in_library":
                print(f"    X {m['term']:<38}   [not in library]")
            else:
                print(f"    X {m['term']:<38}   [zeroed out: {m['feature']}]")

        # Uncapturable
        for u in s["uncapturable"]:
            print(f"    ! {u}")

        # Spurious
        for sp in s["spurious"]:
            print(f"    ? SPURIOUS: {sp['feature']:<26}"
                  f"  (c={sp['coef']:+.6f})")

        print(f"\n    Recall: {s['recall']:.0%}  |  "
              f"Precision: {s['precision']:.0%}  |  "
              f"Active: {s['n_active']}  |  "
              f"Spurious: {s['n_spurious']}")
        print(f"    {result.equation}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. LAGRANGE SINDy SCORING
# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth Lagrangian terms (from the simulation model).

LAGRANGE_GROUND_TRUTH = {
    'T:xdot^2':                      0.75,
    'T:ydot^2':                      0.75,
    'T:thetadot^2':                  0.0243,
    'T:xdot thetadot sin(theta)':   -0.09,
    'T:ydot thetadot cos(theta)':    0.09,
    'V:y':                          -1.5,
    'V:sin(theta)':                 -0.09,
}


def score_lagrange_result(res, label: str = "",
                          is_flight: bool = False) -> dict:
    """Score a single SharedLagrangeSINDyResult against ground truth.

    Returns dict with found/missing/spurious T and V terms, recall, precision,
    grade, and coefficient comparison.
    """
    coefs = res.coefs.copy()
    rescaled = False
    if is_flight and abs(coefs[0]) > 1e-10:
        coefs = coefs * (0.75 / coefs[0])
        rescaled = True

    coef_map = {name: coefs[i] for i, name in enumerate(res.all_names)}
    gt = LAGRANGE_GROUND_TRUTH

    found = []
    missing = []
    for term, gt_val in gt.items():
        c = coef_map.get(term, 0.0)
        if abs(c) > 1e-8:
            rel_err = abs(c - gt_val) / abs(gt_val) if gt_val != 0 else float('inf')
            found.append({"term": term, "coef": float(c),
                          "truth": gt_val, "rel_err": rel_err})
        else:
            missing.append({"term": term, "truth": gt_val})

    spurious = []
    gt_names = set(gt.keys())
    for name, c in coef_map.items():
        if abs(c) > 1e-8 and name not in gt_names:
            spurious.append({"term": name, "coef": float(c)})

    n_expected = len(gt)
    n_found = len(found)
    n_active = n_found + len(spurious)
    recall = n_found / n_expected if n_expected > 0 else 1.0
    precision = n_found / n_active if n_active > 0 else 0.0

    mean_rel_err = (np.mean([f["rel_err"] for f in found])
                    if found else float('nan'))

    if recall >= 1.0:
        grade = "A" if precision >= 0.5 and mean_rel_err < 0.5 else "B"
    elif recall >= 0.71:
        grade = "B" if precision >= 0.4 else "C"
    elif recall >= 0.43:
        grade = "C" if precision >= 0.3 else "D"
    else:
        grade = "F"

    return {
        "label": label,
        "grade": grade,
        "recall": recall,
        "precision": precision,
        "n_expected": n_expected,
        "n_found": n_found,
        "n_active": n_active,
        "n_spurious": len(spurious),
        "mean_rel_err": mean_rel_err,
        "found": found,
        "missing": missing,
        "spurious": spurious,
        "rescaled": rescaled,
        "equation": res.lagrangian_equation(),
    }


def score_lagrange(lag_best: dict) -> pd.DataFrame:
    """Score {run_name: (best_alpha, SharedLagrangeSINDyResult)} dict."""
    rows = []
    for rn, (alpha, res) in lag_best.items():
        is_flight = 'flight' in rn
        s = score_lagrange_result(res, label=rn, is_flight=is_flight)
        rows.append({
            "method": "Lagrange",
            "run": rn,
            "optimizer": "best",
            "alpha": alpha,
            "grade": s["grade"],
            "recall": s["recall"],
            "precision": s["precision"],
            "found": s["n_found"],
            "expected": s["n_expected"],
            "spurious": s["n_spurious"],
            "mean_rel_err": s["mean_rel_err"],
            "missing_terms": ", ".join(m["term"] for m in s["missing"]) or "-",
        })
    return pd.DataFrame(rows)


def score_lagrange_sweep(lag_opt_sweep: dict) -> pd.DataFrame:
    """Score full Lagrange sweep: {run: {(opt, alpha): SharedResult}}.

    Returns one row per (run, optimizer, alpha).
    """
    rows = []
    for rn, sweep_dict in lag_opt_sweep.items():
        is_flight = 'flight' in rn
        for (opt, alpha), res in sweep_dict.items():
            s = score_lagrange_result(res, label=f"{rn}/{opt}/{alpha}",
                                      is_flight=is_flight)
            rows.append({
                "method": "Lagrange",
                "run": rn,
                "optimizer": opt,
                "alpha": alpha,
                "grade": s["grade"],
                "recall": s["recall"],
                "precision": s["precision"],
                "found": s["n_found"],
                "expected": s["n_expected"],
                "spurious": s["n_spurious"],
                "mean_rel_err": s["mean_rel_err"],
                "missing_terms": ", ".join(m["term"] for m in s["missing"]) or "-",
            })
    return pd.DataFrame(rows)


def print_lagrange_report(lag_best: dict):
    """Pretty-print Lagrange SINDy comparison report."""
    print(f"\n{'=' * 70}")
    print(f"  LAGRANGE SINDy — EOM SCORECARD")
    print(f"{'=' * 70}")

    for rn, (alpha, res) in lag_best.items():
        is_flight = 'flight' in rn
        s = score_lagrange_result(res, label=rn, is_flight=is_flight)

        print(f"\n  {rn}  (alpha={alpha})"
              f"{'':>{42 - len(rn)}}Grade: {s['grade']}")
        if s["rescaled"]:
            print(f"  (flight regime: coefficients rescaled to anchor xdot² = 0.75)")
        print(f"  {'-' * 66}")

        for f in s["found"]:
            err_str = f"{f['rel_err']:.1%}" if f['rel_err'] < 100 else ">100%"
            print(f"    + {f['term']:<38}"
                  f"  c={f['coef']:+.6f}  truth={f['truth']:+.6f}"
                  f"  err={err_str}")

        for m in s["missing"]:
            print(f"    X {m['term']:<38}  MISSING  (truth={m['truth']:+.6f})")

        for sp in s["spurious"]:
            print(f"    ? SPURIOUS: {sp['term']:<26}"
                  f"  (c={sp['coef']:+.6f})")

        print(f"\n    Recall: {s['recall']:.0%}  |  "
              f"Precision: {s['precision']:.0%}  |  "
              f"Mean Rel Err: {s['mean_rel_err']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. HYBRID SINDy SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_hybrid(hybrid_sweep: dict) -> pd.DataFrame:
    """Score {config_name: HybridSINDyResult} dict.

    Each HybridSINDyResult has regime_models: {regime: SINDyFit}.
    Returns one row per (config, regime, DOF).
    """
    rows = []
    for cfg_name, hres in hybrid_sweep.items():
        for regime, fit in hres.regime_models.items():
            for dof_name, result in fit.results.items():
                s = score_equation(result)
                cfg = hres.regime_configs.get(regime, {})
                rows.append({
                    "method": "Hybrid",
                    "config": cfg_name,
                    "regime": regime,
                    "optimizer": cfg.get("optimizer", "?"),
                    "threshold": cfg.get("threshold", "?"),
                    "dof": s["dof_name"],
                    "grade": s["grade"],
                    "recall": s["recall"],
                    "precision": s["precision"],
                    "found": s["n_found"],
                    "expected": s["n_expected"],
                    "active": s["n_active"],
                    "spurious": s["n_spurious"],
                    "mse_val": result.mse_val,
                    "missing_terms": ", ".join(
                        m["term"].split("[")[0].strip() for m in s["missing"]
                    ) or "-",
                    "equation": s["equation"],
                })
    return pd.DataFrame(rows)


def print_hybrid_report(hybrid_sweep: dict):
    """Pretty-print Hybrid SINDy EOM scorecard."""
    for cfg_name, hres in hybrid_sweep.items():
        print(f"\n{'=' * 70}")
        print(f"  HYBRID: {cfg_name}")
        print(f"{'=' * 70}")
        for regime, fit in hres.regime_models.items():
            cfg = hres.regime_configs.get(regime, {})
            _print_fit_report(
                fit,
                f"{regime} — {cfg.get('optimizer', '?')} "
                f"th={cfg.get('threshold', '?')}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. LAGRANGE SINDy — PER-DOF EOM RECALL
# ─────────────────────────────────────────────────────────────────────────────
# Maps each T/V Lagrangian term name to the DOF indices (0=x, 1=y, 2=theta)
# whose Euler-Lagrange equation it contributes to.
#
# A T:term contributes to DOF i if  d/dt(∂T/∂qdot_i) - ∂T/∂q_i ≠ 0.
# A V:term contributes to DOF i if  ∂V/∂q_i ≠ 0.

LAGRANGE_DOF_MAP: dict = {
    # Kinetic — pure quadratic: only the matching DOF
    "T:xdot^2":                      [0],
    "T:ydot^2":                      [1],
    "T:thetadot^2":                  [2],
    # Kinetic — cross-velocity (no trig): two DOFs
    "T:xdot ydot":                   [0, 1],
    "T:xdot thetadot":               [0, 2],
    "T:ydot thetadot":               [1, 2],
    # Kinetic — cross-velocity × trig: two DOFs
    "T:xdot thetadot sin(theta)":    [0, 2],
    "T:xdot thetadot cos(theta)":    [0, 2],
    "T:ydot thetadot sin(theta)":    [1, 2],
    "T:ydot thetadot cos(theta)":    [1, 2],
    # Kinetic — thetadot² × trig: theta EOM only
    "T:thetadot^2 sin(theta)":       [2],
    "T:thetadot^2 cos(theta)":       [2],
    # Potential — positional
    "V:y":                           [1],
    "V:sin(theta)":                  [2],
    "V:cos(theta)":                  [2],
    "V:y sin(theta)":                [1, 2],
    "V:y cos(theta)":                [1, 2],
    "V:sin(theta)^2":                [2],
    "V:cos(theta)^2":                [2],
    "V:y^2":                         [1],
    "V:x":                           [0],
}

# Ground-truth per-DOF: which LAGRANGE_GROUND_TRUTH terms belong to each DOF.
_LAGRANGE_GT_PER_DOF: dict = {
    dof: [t for t, _ in LAGRANGE_GROUND_TRUTH.items()
          if dof in LAGRANGE_DOF_MAP.get(t, [])]
    for dof in range(3)
}


def score_lagrange_per_dof(res, is_flight: bool = False) -> dict:
    """Compute Jaccard recall per DOF for a SharedLagrangeSINDyResult.

    Returns {dof_idx: recall} where dof_idx in {0, 1, 2}.

    Jaccard recall = |found ∩ expected| / |found ∪ expected|
                   = n_found / (n_expected + n_spurious_for_dof)

    'Spurious for DOF d' = active T/V terms (non-zero coef) that
    (a) are NOT in LAGRANGE_GROUND_TRUTH AND
    (b) contribute to DOF d's EOM (per LAGRANGE_DOF_MAP).
    """
    coefs = res.coefs.copy()
    if is_flight and abs(coefs[0]) > 1e-10:
        coefs = coefs * (0.75 / coefs[0])

    coef_map = {name: coefs[i] for i, name in enumerate(res.all_names)}
    gt_names = set(LAGRANGE_GROUND_TRUTH.keys())

    result = {}
    for dof in range(3):
        expected = _LAGRANGE_GT_PER_DOF[dof]  # GT terms for this DOF

        # How many expected terms were recovered?
        n_found = sum(1 for t in expected if abs(coef_map.get(t, 0.0)) > 1e-8)
        n_expected = len(expected)

        # How many spurious terms affect this DOF?
        n_spurious = sum(
            1 for name, c in coef_map.items()
            if abs(c) > 1e-8
            and name not in gt_names
            and dof in LAGRANGE_DOF_MAP.get(name, [])
        )

        union = n_expected + n_spurious   # = found + missed + spurious
        recall = n_found / union if union > 0 else 1.0
        result[dof] = recall

    return result
