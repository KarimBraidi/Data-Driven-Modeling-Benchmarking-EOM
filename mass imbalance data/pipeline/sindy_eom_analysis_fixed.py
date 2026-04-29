"""
SINDy (Sparse Identification of Nonlinear Dynamics) Analysis
Recovers equations of motion for a rolling hula hoop with mass imbalance
from simulation data (positions, velocities, and accelerations)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Try to import pysindy, install if needed
try:
    from pysindy import SINDy
    from pysindy.feature_library import PolynomialLibrary
    from pysindy.optimizers import STLSQ
except ImportError:
    print("Installing pysindy...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pysindy'])
    from pysindy import SINDy
    from pysindy.feature_library import PolynomialLibrary
    from pysindy.optimizers import STLSQ


# ============================================================================
# 1. DATA LOADING
# ============================================================================
def load_data(data_path=None):
    """Load simulation data from .npy files"""
    if data_path is None:
        data_path = 'mass imbalance data/Rolling without slipping/hula_hoop_2026-03-14_13-37-06/'
    
    X_save = np.load(data_path + 'X_save.npy')  # Accelerations
    Q_save = np.load(data_path + 'q_save.npy')  # Positions
    U_save = np.load(data_path + 'u_save.npy')  # Velocities
    
    print("=" * 80)
    print("DATA LOADING")
    print("=" * 80)
    print(f"Path: {data_path}")
    print(f"X_save (accelerations) shape: {X_save.shape}")
    print(f"Q_save (positions) shape: {Q_save.shape}")
    print(f"U_save (velocities) shape: {U_save.shape}")
    print()
    
    return X_save, Q_save, U_save


# ============================================================================
# 2. DATA EXTRACTION
# ============================================================================
def extract_components(X_save, Q_save, U_save):
    """Extract individual components from the data arrays"""
    
    # Extract accelerations (first 3 rows of X)
    a1 = X_save[0]  # x acceleration
    a2 = X_save[1]  # y acceleration
    a3 = X_save[2]  # theta acceleration
    
    # Extract positions
    q1 = Q_save[0]  # x position
    q2 = Q_save[1]  # y position
    q3 = Q_save[2]  # theta (angle)
    
    # Extract velocities
    x_dot_1 = U_save[0]  # x velocity
    x_dot_2 = U_save[1]  # y velocity
    x_dot_3 = U_save[2]  # theta velocity
    
    print("=" * 80)
    print("DATA EXTRACTION")
    print("=" * 80)
    print(f"Positions (q): q1={q1.shape}, q2={q2.shape}, q3={q3.shape}")
    print(f"Velocities (q_dot): x_dot_1={x_dot_1.shape}, x_dot_2={x_dot_2.shape}, x_dot_3={x_dot_3.shape}")
    print(f"Accelerations (q_ddot): a1={a1.shape}, a2={a2.shape}, a3={a3.shape}")
    print()
    
    return a1, a2, a3, q1, q2, q3, x_dot_1, x_dot_2, x_dot_3


# ============================================================================
# 3. BUILD STATE AND DERIVATIVE MATRICES
# ============================================================================
def build_matrices(q1, q2, q3, x_dot_1, x_dot_2, x_dot_3, a1, a2, a3):
    """Build state and derivative matrices for SINDy"""
    
    # State matrix: [q1, q2, q3, x_dot_1, x_dot_2, x_dot_3]
    state = np.column_stack((q1, q2, q3, x_dot_1, x_dot_2, x_dot_3))
    
    # Derivative matrix (accelerations): [a1, a2, a3]
    derivatives = np.column_stack((a1, a2, a3))
    
    print("=" * 80)
    print("MATRICES BUILT")
    print("=" * 80)
    print(f"State matrix shape: {state.shape}")
    print(f"  (coordinates: q1, q2, q3, x_dot_1, x_dot_2, x_dot_3)")
    print(f"Derivatives matrix shape: {derivatives.shape}")
    print(f"  (accelerations: a1, a2, a3)")
    print()
    
    return state, derivatives


# ============================================================================
# 4. FIT SINDY MODEL
# ============================================================================
def fit_sindy_model(state, derivatives, threshold=0.1, degree=2):
    """Fit SINDy model to discover equations of motion"""
    
    print("=" * 80)
    print("FITTING SINDY MODEL")
    print("=" * 80)
    print(f"Feature library degree: {degree}")
    print(f"STLSQ threshold: {threshold} (lower = sparser, more aggressive)")
    print()
    
    # Create polynomial feature library
    poly_lib = PolynomialLibrary(degree=degree, include_bias=False)
    
    # Create SINDy model with STLSQ optimizer
    model = SINDy(
        optimizer=STLSQ(threshold=threshold),
        feature_library=poly_lib,
        differentiation_method=None  # We're providing derivatives directly
    )
    
    # Create time array for the data
    t = np.linspace(0, 1, state.shape[0])
    
    # Fit the model
    model.fit(state, t=t, x_dot=derivatives)
    
    print("[OK] Model fitted successfully!")
    print()
    
    return model, poly_lib


# ============================================================================
# 5. PRINT DISCOVERED EQUATIONS
# ============================================================================
def print_equations(model, poly_lib):
    """Print the discovered equations of motion"""
    
    print("=" * 80)
    print("DISCOVERED EQUATIONS OF MOTION")
    print("=" * 80)
    
    feature_names = ['q1', 'q2', 'q3', 'x_dot_1', 'x_dot_2', 'x_dot_3']
    
    # Get all feature names
    feature_lib_names = poly_lib.get_feature_names(feature_names)
    
    # Get coefficients
    coefficients = model.coefficients()
    
    # Print equations for each coordinate
    eq_names = ['a1 (x acceleration)', 'a2 (y acceleration)', 'a3 (theta acceleration)']
    
    for i, eq_name in enumerate(eq_names):
        print(f"\n{eq_name}:")
        print("-" * 80)
        
        coef_row = coefficients[i]
        nonzero_idx = np.where(np.abs(coef_row) > 1e-6)[0]
        
        if len(nonzero_idx) == 0:
            print("  (No terms found)")
            continue
        
        # Sort by magnitude
        sorted_idx = nonzero_idx[np.argsort(-np.abs(coef_row[nonzero_idx]))]
        
        equation_parts = []
        for feat_idx in sorted_idx:
            coef_val = coef_row[feat_idx]
            feature_name = feature_lib_names[feat_idx]
            
            sign = "+" if coef_val > 0 else "-"
            magnitude = abs(coef_val)
            
            equation_parts.append(f"{sign} {magnitude:.6f}*{feature_name}")
        
        equation_str = "\n  ".join(equation_parts)
        print(f"  {equation_str}")
    
    print("\n" + "=" * 80)
    print()


# ============================================================================
# 6. MODEL VALIDATION
# ============================================================================
def validate_model(model, state, derivatives):
    """Validate model performance"""
    
    print("=" * 80)
    print("MODEL VALIDATION")
    print("=" * 80)
    
    # Make predictions
    predictions = model.predict(state)
    residuals = derivatives - predictions
    
    # Calculate metrics
    r2_scores = [r2_score(derivatives[:, i], predictions[:, i]) for i in range(3)]
    rmse_scores = [np.sqrt(mean_squared_error(derivatives[:, i], predictions[:, i])) for i in range(3)]
    mae_scores = [mean_absolute_error(derivatives[:, i], predictions[:, i]) for i in range(3)]
    
    coord_names = ['a1 (x)', 'a2 (y)', 'a3 (theta)']
    
    print(f"\n{'Coordinate':<20} {'R2 Score':<15} {'RMSE':<15} {'MAE':<15}")
    print("-" * 65)
    for i, name in enumerate(coord_names):
        print(f"{name:<20} {r2_scores[i]:<15.6f} {rmse_scores[i]:<15.6f} {mae_scores[i]:<15.6f}")
    
    mean_r2 = np.mean(r2_scores)
    print("-" * 65)
    print(f"{'Mean':<20} {mean_r2:<15.6f}")
    print("=" * 80)
    print()
    
    return predictions, residuals, r2_scores


# ============================================================================
# 7. VISUALIZATION
# ============================================================================
def plot_predictions(derivatives, predictions, residuals, r2_scores):
    """Plot actual vs predicted accelerations and residuals"""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('SINDy Model: Predictions vs Actual', fontsize=14, fontweight='bold')
    
    coord_names = ['a1 (x)', 'a2 (y)', 'a3 (theta)']
    
    for i in range(3):
        # Actual vs Predicted
        axes[i, 0].plot(derivatives[:, i], label='Actual', alpha=0.7, linewidth=2)
        axes[i, 0].plot(predictions[:, i], label='Predicted', alpha=0.7, linewidth=1.5, linestyle='--')
        axes[i, 0].set_ylabel(f'{coord_names[i]} acceleration')
        axes[i, 0].set_title(f'Actual vs Predicted - {coord_names[i]}')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Residuals
        axes[i, 1].plot(residuals[:, i], label='Residuals', alpha=0.7, color='red')
        axes[i, 1].set_ylabel(f'Residual')
        axes[i, 1].set_title(f'Residuals - {coord_names[i]} (R2={r2_scores[i]:.4f})')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_individual_coords(derivatives, predictions):
    """Plot each coordinate separately"""
    
    coord_names = ['a1 (x acceleration)', 'a2 (y acceleration)', 'a3 (theta acceleration)']
    
    for i in range(3):
        plt.figure(figsize=(12, 5))
        plt.plot(derivatives[:, i], label='Actual', linewidth=2, alpha=0.8)
        plt.plot(predictions[:, i], label='SINDy Prediction', linewidth=2, alpha=0.8, linestyle='--')
        plt.xlabel('Time Step')
        plt.ylabel('Acceleration')
        plt.title(f'SINDy Model: {coord_names[i]}')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_coefficient_sparsity(model, poly_lib):
    """Visualize the sparsity of the discovered equations"""
    
    coefficients = model.coefficients()
    feature_names = poly_lib.get_feature_names(['q1', 'q2', 'q3', 'x_dot_1', 'x_dot_2', 'x_dot_3'])
    
    # Create heatmap of coefficients
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize for visualization
    coef_abs = np.abs(coefficients.T)
    im = ax.imshow(coef_abs, cmap='hot', aspect='auto')
    
    ax.set_xlabel('Acceleration Component')
    ax.set_ylabel('Feature')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['a1', 'a2', 'a3'])
    
    # Show only important features
    important_features = np.any(coef_abs > 1e-4, axis=1)
    selected_idx = np.where(important_features)[0]
    
    if len(selected_idx) > 0:
        ax.set_yticks(selected_idx)
        ax.set_yticklabels([feature_names[i] for i in selected_idx])
    
    plt.colorbar(im, ax=ax, label='|Coefficient|')
    plt.title('SINDy Model: Coefficient Sparsity Pattern')
    plt.tight_layout()
    plt.show()


# ============================================================================
# 8. PARAMETER SWEEP - TEST DIFFERENT THRESHOLDS
# ============================================================================
def test_thresholds(state, derivatives, thresholds=None, degree=3):
    """Test different sparsity thresholds to find optimal balance"""
    
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    print("=" * 80)
    print("THRESHOLD PARAMETER SWEEP")
    print("=" * 80)
    print(f"\n{'Threshold':<12} {'Mean R2':<12} {'Sparsity':<12} {'Active Terms':<15}")
    print("-" * 80)
    
    results = {}
    t = np.linspace(0, 1, state.shape[0])
    
    for threshold in thresholds:
        poly_lib = PolynomialLibrary(degree=degree, include_bias=False)
        model_temp = SINDy(
            optimizer=STLSQ(threshold=threshold),
            feature_library=poly_lib,
            differentiation_method=None
        )
        model_temp.fit(state, t=t, x_dot=derivatives)
        
        predictions_temp = model_temp.predict(state)
        r2_temp = np.mean([r2_score(derivatives[:, i], predictions_temp[:, i]) for i in range(3)])
        sparsity = np.sum(np.abs(model_temp.coefficients()) < 1e-5) / model_temp.coefficients().size
        n_terms = np.sum(np.abs(model_temp.coefficients()) > 1e-5)
        
        results[threshold] = {
            'model': model_temp,
            'r2': r2_temp,
            'sparsity': sparsity,
            'n_terms': n_terms
        }
        
        print(f"{threshold:<12.2f} {r2_temp:<12.6f} {sparsity:<12.1%} {n_terms:<15}")
    
    print("=" * 80)
    print("\nTip: Lower threshold = more terms, better accuracy")
    print("     Higher threshold = fewer terms, sparser model")
    print()
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main analysis pipeline"""
    
    print("\n")
    print("=" * 80)
    print("SINDy Analysis: Recovering Equations of Motion".center(80))
    print("Rolling Hula Hoop with Mass Imbalance".center(80))
    print("=" * 80)
    print()
    
    # Step 1: Load data
    X_save, Q_save, U_save = load_data()
    
    # Step 2: Extract components
    a1, a2, a3, q1, q2, q3, x_dot_1, x_dot_2, x_dot_3 = extract_components(X_save, Q_save, U_save)
    
    # Step 3: Build matrices
    state, derivatives = build_matrices(q1, q2, q3, x_dot_1, x_dot_2, x_dot_3, a1, a2, a3)
    
    # Step 4: Test different thresholds (optional but recommended)
    print("\nStep 4: Testing different sparsity thresholds...")
    threshold_results = test_thresholds(state, derivatives)
    
    # Step 5: Fit main model with chosen threshold (change this as needed)
    chosen_threshold = 0.1  # Adjust based on threshold sweep results
    print(f"\nStep 5: Fitting SINDy model with threshold={chosen_threshold}...")
    model, poly_lib = fit_sindy_model(state, derivatives, threshold=chosen_threshold, degree=3)
    
    # Step 6: Print discovered equations
    print("Step 6: Discovered Equations...")
    print_equations(model, poly_lib)
    
    # Step 7: Validate model
    print("Step 7: Validating model performance...")
    predictions, residuals, r2_scores = validate_model(model, state, derivatives)
    
    # Step 8: Visualize results
    print("Step 8: Generating plots...")
    plot_predictions(derivatives, predictions, residuals, r2_scores)
    plot_individual_coords(derivatives, predictions)
    plot_coefficient_sparsity(model, poly_lib)
    
    print("\n[OK] Analysis complete!")
    print("\nTo adjust the analysis:")
    print("  - Change 'chosen_threshold' to use different sparsity levels")
    print("  - Change 'degree' in fit_sindy_model() for higher polynomial orders")
    print("  - Modify feature library in fit_sindy_model() to include other features (e.g., Fourier)")


def analyze_multiple_datasets(dataset_paths):
    """Analyze multiple datasets and compare results"""
    
    print("\n")
    print("=" * 80)
    print("SINDy Multi-Dataset Analysis & Comparison".center(80))
    print("Rolling Hula Hoop with Mass Imbalance".center(80))
    print("=" * 80)
    print()
    
    results_dict = {}
    chosen_threshold = 0.1
    
    for idx, path in enumerate(dataset_paths):
        print(f"\n{'='*80}")
        print(f"ANALYZING DATASET {idx + 1}/{len(dataset_paths)}")
        print(f"{'='*80}\n")
        
        try:
            # Load data
            X_save, Q_save, U_save = load_data(data_path=path)
            
            # Extract components
            a1, a2, a3, q1, q2, q3, x_dot_1, x_dot_2, x_dot_3 = extract_components(X_save, Q_save, U_save)
            
            # Build matrices
            state, derivatives = build_matrices(q1, q2, q3, x_dot_1, x_dot_2, x_dot_3, a1, a2, a3)
            
            # Fit model
            model, poly_lib = fit_sindy_model(state, derivatives, threshold=chosen_threshold, degree=3)
            
            # Validate
            predictions, residuals, r2_scores = validate_model(model, state, derivatives)
            
            # Store results
            dataset_name = path.split('/')[-2]  # Get folder name before trailing /
            results_dict[dataset_name] = {
                'path': path,
                'model': model,
                'poly_lib': poly_lib,
                'r2_scores': r2_scores,
                'predictions': predictions,
                'derivatives': derivatives,
                'state': state
            }
            
            print(f"[OK] Dataset {idx + 1} processed successfully\n")
            
        except Exception as e:
            print(f"[ERROR] Error processing dataset {idx + 1}: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison
    print(f"\n{'='*80}")
    print("COMPARISON OF ALL DATASETS")
    print(f"{'='*80}\n")
    
    print(f"{'Dataset':<45} {'a1 R2':<12} {'a2 R2':<12} {'a3 R2':<12} {'Mean R2':<12}")
    print("-" * 93)
    
    for dataset_name, results in results_dict.items():
        r2 = results['r2_scores']
        mean_r2 = np.mean(r2)
        print(f"{dataset_name:<45} {r2[0]:<12.6f} {r2[1]:<12.6f} {r2[2]:<12.6f} {mean_r2:<12.6f}")
    
    print(f"{'='*80}\n")
    
    # Print equations for each dataset
    print(f"{'='*80}")
    print("DISCOVERED EQUATIONS BY DATASET")
    print(f"{'='*80}\n")
    
    for dataset_name, results in results_dict.items():
        print(f"\n{dataset_name}:")
        print("-" * 80)
        print_equations(results['model'], results['poly_lib'])
    
    return results_dict


if __name__ == "__main__":
    # Option 1: Analyze a single dataset
    # main()
    
    # Option 2: Analyze multiple datasets and compare
    bouncing_ball_datasets = [
        'mass imbalance data/mass imbalance data/Bouncing ball data/hula_hoop_2026-03-14_13-07-44/',
        'mass imbalance data/mass imbalance data/Bouncing ball data/hula_hoop_2026-03-14_13-31-19/',
        'mass imbalance data/mass imbalance data/Bouncing ball data/hula_hoop_2026-03-14_13-33-04/',
    ]
    
    results = analyze_multiple_datasets(bouncing_ball_datasets)
