"""
SINDy (Sparse Identification of Nonlinear Dynamics) Analysis
Recovers equations of motion for a rolling hula hoop with mass imbalance.
Fits each acceleration (x, y, theta) independently to guarantee 3 equations.
Saves all results to 'sindy_results/' for visualization in a notebook.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = 'sindy_results'
BASE_FEATURE_NAMES = ['q1', 'q2', 'q3', 'q1_dot', 'q2_dot', 'q3_dot']
ACCEL_NAMES = ['a1_x', 'a2_y', 'a3_theta']
DEGREE = 2
THRESHOLD = 0.1         # initial threshold (will be adapted per equation)
R2_TARGET = 0.90          # minimum acceptable R2 during threshold sweep
ALPHA = 0.05              # ridge regularization for STLSQ (tames multicollinearity)


# ============================================================================
# HELPERS
# ============================================================================
def load_data(data_path):
    X_save = np.load(data_path + 'X_save.npy')
    Q_save = np.load(data_path + 'q_save.npy')
    U_save = np.load(data_path + 'u_save.npy')
    return X_save, Q_save, U_save


def build_state_and_accels(X_save, Q_save, U_save):
    """Return state (N,6) and accelerations (N,3)."""
    q1, q2, q3 = Q_save[0], Q_save[1], Q_save[2]
    qd1, qd2, qd3 = U_save[0], U_save[1], U_save[2]
    a1, a2, a3 = X_save[0], X_save[1], X_save[2]
    state = np.column_stack((q1, q2, q3, qd1, qd2, qd3))
    accels = np.column_stack((a1, a2, a3))
    return state, accels


def get_feature_names_for(col_idx):
    """Return raw variable names for fitting equation col_idx.
    Includes base state features + the other two accelerations."""
    other_accel_names = [ACCEL_NAMES[j] for j in range(3) if j != col_idx]
    return BASE_FEATURE_NAMES + other_accel_names


def build_augmented_state(state, accels, col_idx):
    """Build augmented state: [q1,q2,q3,q1_dot,q2_dot,q3_dot, other_accel_1, other_accel_2]."""
    other_cols = [accels[:, j] for j in range(3) if j != col_idx]
    return np.column_stack([state] + other_cols)


def build_trig_feature_library(state_aug, var_names, degree):
    """Build a feature library that includes:
    - Polynomial features up to `degree`
    - sin(q3) and cos(q3)
    - Each polynomial feature multiplied by sin(q3) and cos(q3)
    Returns (feature_matrix, feature_name_list)."""

    # q3 is always column index 2 in state_aug
    q3_col = state_aug[:, 2]
    sin_q3 = np.sin(q3_col)
    cos_q3 = np.cos(q3_col)

    # Step 1: polynomial features
    poly_lib = PolynomialLibrary(degree=degree, include_bias=False)
    poly_features = poly_lib.fit_transform(state_aug)
    poly_names = poly_lib.get_feature_names(var_names)

    # Step 2: sin(q3) * each poly feature, cos(q3) * each poly feature
    sin_features = poly_features * sin_q3[:, np.newaxis]
    cos_features = poly_features * cos_q3[:, np.newaxis]
    sin_names = [f"sin(q3)*{n}" for n in poly_names]
    cos_names = [f"cos(q3)*{n}" for n in poly_names]

    # Step 3: standalone sin(q3) and cos(q3)
    sin_standalone = sin_q3.reshape(-1, 1)
    cos_standalone = cos_q3.reshape(-1, 1)

    # Combine all
    all_features = np.hstack([
        poly_features,
        sin_standalone, cos_standalone,
        sin_features, cos_features,
    ])
    all_names = poly_names + ['sin(q3)', 'cos(q3)'] + sin_names + cos_names

    return all_features, all_names


def fit_single_equation(feature_matrix, accel_col, threshold, alpha=ALPHA):
    """Fit STLSQ on a pre-built feature matrix for one acceleration.
    Normalizes features and target so the threshold is scale-independent.
    Uses ridge regularization (alpha) to handle multicollinearity.
    Sweeps thresholds upward to find the sparsest model with R2 >= R2_TARGET.
    """
    from sklearn.metrics import r2_score as _r2

    # Normalize target
    scale = np.std(accel_col)
    if scale < 1e-12:
        return np.zeros(feature_matrix.shape[1])
    accel_normed = accel_col / scale

    # Normalize features
    feat_scales = np.std(feature_matrix, axis=0)
    feat_scales[feat_scales < 1e-12] = 1.0
    feat_normed = feature_matrix / feat_scales

    # Sweep thresholds: start at given threshold and increase
    thresholds = np.arange(threshold, 0.51, 0.05)
    best_coefs = None
    best_nterms = feature_matrix.shape[1] + 1

    for th in thresholds:
        optimizer = STLSQ(threshold=th, alpha=alpha)
        optimizer.fit(feat_normed, accel_normed.reshape(-1, 1))
        coefs_normed = optimizer.coef_.flatten()
        coefs = (coefs_normed / feat_scales) * scale
        nterms = np.count_nonzero(np.abs(coefs) > 1e-8)
        if nterms == 0:
            continue
        pred = feature_matrix @ coefs
        r2 = _r2(accel_col, pred)
        if r2 >= R2_TARGET and nterms < best_nterms:
            best_coefs = coefs
            best_nterms = nterms

    # If no model met R2_TARGET, fall back to lowest threshold
    if best_coefs is None:
        optimizer = STLSQ(threshold=threshold, alpha=alpha)
        optimizer.fit(feat_normed, accel_normed.reshape(-1, 1))
        coefs_normed = optimizer.coef_.flatten()
        best_coefs = (coefs_normed / feat_scales) * scale

    return best_coefs


def equation_string_from_coefs(coefs, feat_names):
    """Build a human-readable equation string."""
    nonzero = np.where(np.abs(coefs) > 1e-8)[0]
    if len(nonzero) == 0:
        return "0"
    nonzero = nonzero[np.argsort(-np.abs(coefs[nonzero]))]
    parts = []
    for idx in nonzero:
        c = coefs[idx]
        sign = "+ " if c > 0 else "- "
        parts.append(f"{sign}{abs(c):.6f}*{feat_names[idx]}")
    return " ".join(parts)


def metrics_for_col(actual, predicted):
    return {
        'R2': float(r2_score(actual, predicted)),
        'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
        'MAE': float(mean_absolute_error(actual, predicted)),
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def process_dataset(data_path, dataset_label, out_dir):
    """Process one dataset: fit 3 separate SINDy models, save everything."""

    print(f"\n{'='*80}")
    print(f"  {dataset_label}")
    print(f"{'='*80}")

    X_save, Q_save, U_save = load_data(data_path)
    state, accels = build_state_and_accels(X_save, Q_save, U_save)
    print(f"  State shape : {state.shape}")
    print(f"  Accels shape: {accels.shape}")

    dataset_dir = os.path.join(out_dir, dataset_label)
    os.makedirs(dataset_dir, exist_ok=True)

    # Save raw data for the notebook
    np.save(os.path.join(dataset_dir, 'state.npy'), state)
    np.save(os.path.join(dataset_dir, 'accels.npy'), accels)

    summary = {'dataset': dataset_label, 'path': data_path, 'equations': {}, 'metrics': {}}
    all_predictions = np.zeros_like(accels)

    for col_idx, accel_name in enumerate(ACCEL_NAMES):
        print(f"\n  Fitting {accel_name} ...")
        accel_col = accels[:, col_idx]

        # Build augmented state: base state + the OTHER two accelerations
        state_aug = build_augmented_state(state, accels, col_idx)
        var_names = get_feature_names_for(col_idx)

        # Build feature library: polynomials + sin(q3)/cos(q3) * polynomials
        feat_matrix, feat_names = build_trig_feature_library(state_aug, var_names, DEGREE)
        print(f"    Variables: {var_names}")
        print(f"    Total library features: {len(feat_names)}")

        # Fit with STLSQ
        coefs = fit_single_equation(feat_matrix, accel_col, THRESHOLD)
        pred = feat_matrix @ coefs
        all_predictions[:, col_idx] = pred

        eq_str = equation_string_from_coefs(coefs, feat_names)
        met = metrics_for_col(accel_col, pred)

        summary['equations'][accel_name] = eq_str
        summary['metrics'][accel_name] = met

        # Save coefficients
        coef_df = pd.DataFrame({
            'feature': feat_names,
            'coefficient': coefs,
        })
        coef_df = coef_df[coef_df['coefficient'].abs() > 1e-8]
        coef_df.to_csv(os.path.join(dataset_dir, f'coefficients_{accel_name}.csv'), index=False)

        print(f"    {accel_name} = {eq_str}")
        print(f"    R2={met['R2']:.6f}  RMSE={met['RMSE']:.6f}  MAE={met['MAE']:.6f}")

    # Save predictions
    np.save(os.path.join(dataset_dir, 'predictions.npy'), all_predictions)

    # Save summary JSON
    with open(os.path.join(dataset_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  [OK] Results saved to {dataset_dir}/")
    return summary


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    datasets = {
        'bouncing_ball_1': 'mass imbalance data/mass imbalance data/Bouncing ball data/hula_hoop_2026-03-14_13-07-44/',
        'bouncing_ball_2': 'mass imbalance data/mass imbalance data/Bouncing ball data/hula_hoop_2026-03-14_13-31-19/',
        'bouncing_ball_3': 'mass imbalance data/mass imbalance data/Bouncing ball data/hula_hoop_2026-03-14_13-33-04/',
    }

    all_summaries = {}
    for label, path in datasets.items():
        all_summaries[label] = process_dataset(path, label, RESULTS_DIR)

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Dataset':<25} {'a1_x R2':<14} {'a2_y R2':<14} {'a3_theta R2':<14} {'Mean R2':<14}")
    print("-" * 81)
    for label, s in all_summaries.items():
        r2s = [s['metrics'][a]['R2'] for a in ACCEL_NAMES]
        print(f"{label:<25} {r2s[0]:<14.6f} {r2s[1]:<14.6f} {r2s[2]:<14.6f} {np.mean(r2s):<14.6f}")
    print("=" * 81)

    # Save master summary
    with open(os.path.join(RESULTS_DIR, 'all_summaries.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll results saved to '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()
