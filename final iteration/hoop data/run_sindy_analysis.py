#!/usr/bin/env python3
"""
HULA HOOP SINDy EOM DISCOVERY - Standalone Analysis Script
===========================================================

Derives equations of motion using:
- Standard SINDy (STLSQ optimizer)
- Lasso SINDy (L1 regularization)
- Hybrid SINDy (Combined approach)

With full library and unit-consistent libraries
Multiple thresholds and hyperparameters
Train/Val/Test temporal split with comprehensive analysis
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    try:
        from scipy.integrate import cumtrapz
    except ImportError:
        # Manual implementation
        def cumtrapz(y, x, initial=0):
            """Cumulative trapezoidal integration."""
            result = np.cumsum(np.diff(x) * (y[:-1] + y[1:]) / 2)
            if initial is not None:
                result = np.concatenate([[initial], result])
            return result
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pysindy as ps
from sklearn.linear_model import Lasso
import json
import pickle

# =============================================================================
#  CONFIGURATION
# =============================================================================
DATA_DIR = Path(r"C:\Users\braid\OneDrive\Desktop\Data Driven Modeling Project\hoop data")
CSV_FILE = DATA_DIR / "OR_20250903_203926.csv"
FS = 120  # Sampling frequency (Hz)

# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================
def detrend_custom(time, data, backend='polynomial', degree=6):
    """Polynomial detrending to remove drift from numerical integration."""
    if backend == 'polynomial':
        p = np.polyfit(time, data, degree)
        trend = np.polyval(p, time)
        return data - trend
    return data


def preprocess_hoop_data(df, fs=120):
    """
    Clean and preprocess hula hoop IMU data.
    Mirrors the MATLAB Hoop_203926.m approach
    """
    N = len(df)
    dt = 1/fs
    t = np.arange(N) * dt
    
    # Extract measurements
    f1, f2, f3 = df['Acc_X'].values, df['Acc_Y'].values, df['Acc_Z'].values
    phi = df['Euler_X'].values  # roll (degrees)
    theta = df['Euler_Y'].values  # pitch (degrees)
    psi = df['Euler_Z'].values  # yaw (degrees)
    
    # Transform acceleration to global frame
    e1 = np.zeros((N, 3))
    e2 = np.zeros((N, 3))
    e3 = np.zeros((N, 3))
    accel_vector = np.zeros((N, 3))
    
    for i in range(N):
        # Unit vectors (3-2-1 Euler sequence)
        e1[i] = [np.cos(np.radians(psi[i]))*np.cos(np.radians(theta[i])),
                 np.sin(np.radians(psi[i]))*np.cos(np.radians(theta[i])),
                 -np.sin(np.radians(theta[i]))]
        
        e2[i] = [np.sin(np.radians(phi[i]))*np.sin(np.radians(theta[i]))*np.cos(np.radians(psi[i])) 
                 - np.sin(np.radians(psi[i]))*np.cos(np.radians(phi[i])),
                 np.sin(np.radians(phi[i]))*np.sin(np.radians(psi[i]))*np.sin(np.radians(theta[i])) 
                 + np.cos(np.radians(phi[i]))*np.cos(np.radians(psi[i])),
                 np.sin(np.radians(phi[i]))*np.cos(np.radians(theta[i]))]
        
        e3[i] = [np.sin(np.radians(phi[i]))*np.sin(np.radians(psi[i])) 
                 + np.sin(np.radians(theta[i]))*np.cos(np.radians(phi[i]))*np.cos(np.radians(psi[i])),
                 -np.sin(np.radians(phi[i]))*np.cos(np.radians(psi[i])) 
                 + np.sin(np.radians(psi[i]))*np.sin(np.radians(theta[i]))*np.cos(np.radians(phi[i])),
                 np.cos(np.radians(phi[i]))*np.cos(np.radians(theta[i]))]
        
        accel_vector[i] = f1[i]*e1[i] + f2[i]*e2[i] + f3[i]*e3[i]
    
    # Gravity compensation
    x1ddot = accel_vector[:, 0] - np.mean(accel_vector[:, 0])
    x2ddot = accel_vector[:, 1] - np.mean(accel_vector[:, 1])
    x3ddot = accel_vector[:, 2] - np.mean(accel_vector[:, 2])
    
    # Integrate for velocity
    x1dot = cumtrapz(x1ddot, t, initial=0)
    x2dot = cumtrapz(x2ddot, t, initial=0)
    x3dot = cumtrapz(x3ddot, t, initial=0)
    
    # Detrend velocity
    x1dot = detrend_custom(t, x1dot, degree=6)
    x2dot = detrend_custom(t, x2dot, degree=6)
    x3dot = detrend_custom(t, x3dot, degree=6)
    
    # Integrate for position
    x1 = cumtrapz(x1dot, t, initial=0)
    x2 = cumtrapz(x2dot, t, initial=0)
    x3 = cumtrapz(x3dot, t, initial=0)
    
    # Detrend position
    x1 = detrend_custom(t, x1, degree=6)
    x2 = detrend_custom(t, x2, degree=6)
    x3 = detrend_custom(t, x3, degree=6)
    
    return {
        't': t,
        'position': np.column_stack([x1, x2, x3]),
        'velocity': np.column_stack([x1dot, x2dot, x3dot]),
        'acceleration': np.column_stack([x1ddot, x2ddot, x3ddot]),
        'euler_angles': np.column_stack([phi, theta, psi]),
        'dt': dt,
        'N': N
    }


def build_sindy_libraries(pos_train, vel_train, acc_train, target_dof=0):
    """Build FULL and UNIT-CONSISTENT libraries for SINDy."""
    N = len(pos_train)
    state = np.hstack([pos_train, vel_train])
    other_dofs = [j for j in range(3) if j != target_dof]
    other_accel = acc_train[:, other_dofs]
    
    # ==================== FULL LIBRARY ====================
    full_features = [np.ones((N, 1))]
    full_names = ['1']
    
    for i, name in enumerate(['x', 'y', 'z', 'xdot', 'ydot', 'zdot']):
        full_features.append(state[:, i:i+1])
        full_names.append(name)
    
    for i, j in enumerate(other_dofs):
        full_features.append(other_accel[:, i:i+1])
        full_names.append(f'other_acc_{j}')
    
    for i in range(6):
        for j in range(i, 6):
            full_features.append((state[:, i] * state[:, j]).reshape(-1, 1))
            name_i = ['x', 'y', 'z', 'xdot', 'ydot', 'zdot'][i]
            name_j = ['x', 'y', 'z', 'xdot', 'ydot', 'zdot'][j]
            full_names.append(f'{name_i}*{name_j}')
    
    Theta_full = np.hstack(full_features)
    
    # ==================== UNIT-CONSISTENT LIBRARY ====================
    uc_features = [np.ones((N, 1))]
    uc_names = ['1']
    
    for i, j in enumerate(other_dofs):
        uc_features.append(other_accel[:, i:i+1])
        uc_names.append(f'other_acc_{j}')
    
    for i in range(3):
        for j in range(3, 6):
            term = (state[:, j]**2) / (np.abs(state[:, i]) + 1e-6)
            uc_features.append(term.reshape(-1, 1))
            name_i = ['x', 'y', 'z'][i]
            name_j = ['xdot', 'ydot', 'zdot'][j-3]
            uc_names.append(f'{name_j}²/{name_i}')
    
    Theta_uc = np.hstack(uc_features)
    
    return Theta_full, full_names, Theta_uc, uc_names


# =============================================================================
#  MAIN ANALYSIS
# =============================================================================
def main():
    print("="*80)
    print("HULA HOOP SINDy EOM DISCOVERY")
    print("="*80)
    
    # Load data
    print("\n1. LOADING DATA...")
    df = pd.read_csv(CSV_FILE, skiprows=6)
    print(f"   Loaded {len(df)} samples from {CSV_FILE.name}")
    
    # Preprocess
    print("\n2. PREPROCESSING DATA...")
    processed = preprocess_hoop_data(df, fs=FS)
    t, pos, vel, acc = (processed['t'], processed['position'], 
                        processed['velocity'], processed['acceleration'])
    N = len(t)
    print(f"   Processed data shape: {pos.shape}")
    
    # Split data
    print("\n3. SPLITTING DATA (60/20/20 train/val/test)...")
    idx_train = np.arange(int(0.60 * N))
    idx_val = np.arange(int(0.60 * N), int(0.80 * N))
    idx_test = np.arange(int(0.80 * N), N)
    
    pos_train = pos[idx_train]
    vel_train = vel[idx_train]
    acc_train = acc[idx_train]
    
    pos_val = pos[idx_val]
    vel_val = vel[idx_val]
    acc_val = acc[idx_val]
    
    pos_test = pos[idx_test]
    vel_test = vel[idx_test]
    acc_test = acc[idx_test]
    
    print(f"   Train: {len(idx_train)} ({100*len(idx_train)/N:.0f}%)")
    print(f"   Val:   {len(idx_val)} ({100*len(idx_val)/N:.0f}%)")
    print(f"   Test:  {len(idx_test)} ({100*len(idx_test)/N:.0f}%)")
    
    # Build libraries
    print("\n4. BUILDING SINDY LIBRARIES...")
    target_dof = 0
    Theta_full_train, full_names, Theta_uc_train, uc_names = \
        build_sindy_libraries(pos_train, vel_train, acc_train, target_dof)
    
    Theta_full_val, _, Theta_uc_val, _ = \
        build_sindy_libraries(pos_val, vel_val, acc_val, target_dof)
    Theta_full_test, _, Theta_uc_test, _ = \
        build_sindy_libraries(pos_test, vel_test, acc_test, target_dof)
    
    print(f"   Full library: {Theta_full_train.shape[1]} features")
    print(f"   Unit-consistent library: {Theta_uc_train.shape[1]} features")
    
    y_train = acc_train[:, target_dof]
    y_val = acc_val[:, target_dof]
    y_test = acc_test[:, target_dof]
    
    # Fit models
    results = []
    
    # Standard SINDy
    print("\n5. FITTING STANDARD SINDY (STLSQ)...")
    for lib_name, X_tr, X_va, X_te, f_names in [
        ('Full', Theta_full_train, Theta_full_val, Theta_full_test, full_names),
        ('Unit-Consistent', Theta_uc_train, Theta_uc_val, Theta_uc_test, uc_names)
    ]:
        for threshold in [0.01, 0.05, 0.1, 0.2]:
            try:
                model = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold, alpha=0.1),
                               feature_names=f_names)
                model.fit(X_tr, y_train)
                
                y_pred_train = model.predict(X_tr).flatten()
                y_pred_val = model.predict(X_va).flatten()
                y_pred_test = model.predict(X_te).flatten()
                
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_val = mean_squared_error(y_val, y_pred_val)
                mse_test = mean_squared_error(y_test, y_pred_test)
                n_terms = np.sum(np.abs(model.coef_) >= 1e-6)
                
                results.append({
                    'Method': 'Standard SINDy',
                    'Library': lib_name,
                    'Hyperparameter': f'Threshold={threshold:.3f}',
                    'MSE_Train': mse_train,
                    'MSE_Val': mse_val,
                    'MSE_Test': mse_test,
                    'n_Terms': n_terms
                })
                print(f"   {lib_name:20s} | T={threshold:.2f} | "
                      f"Val MSE={mse_val:.4e} | Test MSE={mse_test:.4e} | Terms={int(n_terms)}")
            except Exception as e:
                print(f"   {lib_name:20s} | T={threshold:.2f} | FAILED: {str(e)[:30]}")
    
    # Lasso SINDy
    print("\n6. FITTING LASSO SINDY (L1 Regularization)...")
    for lib_name, X_tr, X_va, X_te, f_names in [
        ('Full', Theta_full_train, Theta_full_val, Theta_full_test, full_names),
        ('Unit-Consistent', Theta_uc_train, Theta_uc_val, Theta_uc_test, uc_names)
    ]:
        for alpha in [0.001, 0.01, 0.1]:
            try:
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                X_tr_scaled = scaler_X.fit_transform(X_tr)
                X_va_scaled = scaler_X.transform(X_va)
                X_te_scaled = scaler_X.transform(X_te)
                
                y_tr_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_va_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
                y_te_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
                
                model = Lasso(alpha=alpha, max_iter=5000)
                model.fit(X_tr_scaled, y_tr_scaled)
                
                y_pred_train = scaler_y.inverse_transform(
                    model.predict(X_tr_scaled).reshape(-1, 1)).flatten()
                y_pred_val = scaler_y.inverse_transform(
                    model.predict(X_va_scaled).reshape(-1, 1)).flatten()
                y_pred_test = scaler_y.inverse_transform(
                    model.predict(X_te_scaled).reshape(-1, 1)).flatten()
                
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_val = mean_squared_error(y_val, y_pred_val)
                mse_test = mean_squared_error(y_test, y_pred_test)
                n_terms = np.sum(np.abs(model.coef_) >= 1e-6)
                
                results.append({
                    'Method': 'Lasso SINDy',
                    'Library': lib_name,
                    'Hyperparameter': f'Alpha={alpha:.4f}',
                    'MSE_Train': mse_train,
                    'MSE_Val': mse_val,
                    'MSE_Test': mse_test,
                    'n_Terms': n_terms
                })
                print(f"   {lib_name:20s} | A={alpha:.4f} | "
                      f"Val MSE={mse_val:.4e} | Test MSE={mse_test:.4e} | Terms={int(n_terms)}")
            except Exception as e:
                print(f"   {lib_name:20s} | A={alpha:.4f} | FAILED: {str(e)[:30]}")
    
    # Hybrid SINDy
    print("\n7. FITTING HYBRID SINDY (STLSQ + Lasso)...")
    for lib_name, X_tr, X_va, X_te, f_names in [
        ('Full', Theta_full_train, Theta_full_val, Theta_full_test, full_names),
        ('Unit-Consistent', Theta_uc_train, Theta_uc_val, Theta_uc_test, uc_names)
    ]:
        for threshold, alpha in [(0.05, 0.005), (0.1, 0.01)]:
            try:
                # STLSQ pre-selection
                model_stlsq = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold))
                model_stlsq.fit(X_tr, y_train)
                selected_idx = np.where(np.abs(model_stlsq.coef_) >= 1e-6)[0]
                if len(selected_idx) < 2:
                    selected_idx = np.argsort(np.abs(model_stlsq.coef_))[-5:]
                
                # Lasso on selected
                X_tr_sel = X_tr[:, selected_idx]
                X_va_sel = X_va[:, selected_idx]
                X_te_sel = X_te[:, selected_idx]
                
                scaler_X = StandardScaler()
                X_tr_scaled = scaler_X.fit_transform(X_tr_sel)
                X_va_scaled = scaler_X.transform(X_va_sel)
                X_te_scaled = scaler_X.transform(X_te_sel)
                
                scaler_y = StandardScaler()
                y_tr_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_va_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
                y_te_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
                
                model = Lasso(alpha=alpha, max_iter=5000)
                model.fit(X_tr_scaled, y_tr_scaled)
                
                y_pred_train = scaler_y.inverse_transform(
                    model.predict(X_tr_scaled).reshape(-1, 1)).flatten()
                y_pred_val = scaler_y.inverse_transform(
                    model.predict(X_va_scaled).reshape(-1, 1)).flatten()
                y_pred_test = scaler_y.inverse_transform(
                    model.predict(X_te_scaled).reshape(-1, 1)).flatten()
                
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_val = mean_squared_error(y_val, y_pred_val)
                mse_test = mean_squared_error(y_test, y_pred_test)
                n_terms = np.sum(np.abs(model.coef_) >= 1e-6)
                
                results.append({
                    'Method': 'Hybrid SINDy',
                    'Library': lib_name,
                    'Hyperparameter': f'T={threshold:.2f}_A={alpha:.4f}',
                    'MSE_Train': mse_train,
                    'MSE_Val': mse_val,
                    'MSE_Test': mse_test,
                    'n_Terms': n_terms
                })
                print(f"   {lib_name:20s} | T={threshold:.2f},A={alpha:.4f} | "
                      f"Val MSE={mse_val:.4e} | Test MSE={mse_test:.4e} | Terms={int(n_terms)}")
            except Exception as e:
                print(f"   {lib_name:20s} | T={threshold:.2f},A={alpha:.4f} | FAILED: {str(e)[:30]}")
    
    # Summary
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON")
    print("="*80)
    print(df_results.to_string(index=False))
    
    # Save results
    print("\n8. SAVING RESULTS...")
    df_results.to_csv(DATA_DIR / 'sindy_analysis_results.csv', index=False)
    with open(DATA_DIR / 'sindy_analysis_summary.txt', 'w') as f:
        f.write("HULA HOOP SINDy EOM DISCOVERY - SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {CSV_FILE.name}\n")
        f.write(f"Total samples: {N}\n")
        f.write(f"Duration: {t[-1]:.2f} seconds\n")
        f.write(f"Sampling frequency: {FS} Hz\n\n")
        f.write(f"Train/Val/Test split: {len(idx_train)}/{len(idx_val)}/{len(idx_test)} "
                f"({100*len(idx_train)/N:.0f}%/{100*len(idx_val)/N:.0f}%/{100*len(idx_test)/N:.0f}%)\n\n")
        f.write("COMPARISON TABLE:\n")
        f.write("="*80 + "\n")
        f.write(df_results.to_string(index=False))
        f.write("\n\n")
        f.write("TOP 5 MODELS (min Val MSE):\n")
        f.write("="*80 + "\n")
        top_5 = df_results.nsmallest(5, 'MSE_Val')
        f.write(top_5.to_string(index=False))
    
    print(f"   ✓ Results saved to 'sindy_analysis_results.csv'")
    print(f"   ✓ Summary saved to 'sindy_analysis_summary.txt'")
    
    # Best model
    best_idx = df_results['MSE_Val'].idxmin()
    best = df_results.iloc[best_idx]
    print("\n" + "="*80)
    print("BEST MODEL")
    print("="*80)
    print(f"Method:           {best['Method']}")
    print(f"Library:          {best['Library']}")
    print(f"Hyperparameter:   {best['Hyperparameter']}")
    print(f"Train MSE:        {best['MSE_Train']:.6e}")
    print(f"Val MSE:          {best['MSE_Val']:.6e}")
    print(f"Test MSE:         {best['MSE_Test']:.6e}")
    print(f"Number of terms:  {int(best['n_Terms'])}")
    print("="*80)


if __name__ == "__main__":
    main()
