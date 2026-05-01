"""
preprocessing.py – Smoothing, derivative validation, normalization, temporal splitting.
"""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Tuple


# ── Smoothing ────────────────────────────────────────────────────────────────

def smooth_savgol(x: np.ndarray, window: int = 11, polyorder: int = 3,
                  axis: int = 0) -> np.ndarray:
    """Savitzky-Golay filter along *axis*."""
    return savgol_filter(x, window_length=window, polyorder=polyorder, axis=axis)


def smooth_tv(x: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """1-D total variation denoising (proximal gradient, per-column)."""
    from scipy.optimize import minimize
    out = np.empty_like(x)
    for col in range(x.shape[1]):
        y = x[:, col]
        n = len(y)

        def objective(z):
            fid = 0.5 * np.sum((z - y) ** 2)
            tv = weight * np.sum(np.abs(np.diff(z)))
            return fid + tv

        res = minimize(objective, y.copy(), method="L-BFGS-B",
                       options={"maxiter": 500})
        out[:, col] = res.x
    return out


# ── Derivative validation ────────────────────────────────────────────────────

def numerical_acceleration(u: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference acceleration from velocity array (N, 3)."""
    return np.gradient(u, dt, axis=0)


def validate_derivatives(qddot_given: np.ndarray, u: np.ndarray, dt: float):
    """Compare simulator accelerations with numerical ones.
    Returns dict with per-DOF max error and correlation."""
    qddot_num = numerical_acceleration(u, dt)
    labels = ["xddot", "yddot", "thetaddot"]
    report = {}
    for i, lab in enumerate(labels):
        err = qddot_given[:, i] - qddot_num[:, i]
        report[lab] = {
            "max_abs_error": float(np.max(np.abs(err))),
            "rms_error": float(np.sqrt(np.mean(err ** 2))),
            "correlation": float(np.corrcoef(qddot_given[:, i],
                                              qddot_num[:, i])[0, 1]),
        }
    return report, qddot_num


# ── Temporal splitting ───────────────────────────────────────────────────────

@dataclass
class TemporalSplit:
    """Index arrays for a forward-in-time train/val/test split."""
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray

    @property
    def sizes(self):
        return len(self.idx_train), len(self.idx_val), len(self.idx_test)


def temporal_split(N: int, t: np.ndarray,
                   t_train_end: float, t_val_end: float) -> TemporalSplit:
    """Forward-in-time split at given time boundaries (seconds).

    Parameters
    ----------
    N : total timesteps
    t : (N,) time vector
    t_train_end : end of training window
    t_val_end   : end of validation window (rest is test)
    """
    idx_train = np.where(t < t_train_end)[0]
    idx_val = np.where((t >= t_train_end) & (t < t_val_end))[0]
    idx_test = np.where(t >= t_val_end)[0]
    return TemporalSplit(idx_train, idx_val, idx_test)


def temporal_split_frac(N: int, train_frac: float = 0.70,
                        val_frac: float = 0.15) -> TemporalSplit:
    """Forward-in-time split by fraction (no shuffling)."""
    n_train = int(train_frac * N)
    n_val = int(val_frac * N)
    return TemporalSplit(
        np.arange(0, n_train),
        np.arange(n_train, n_train + n_val),
        np.arange(n_train + n_val, N),
    )


def regime_aware_split(
    N: int,
    lambda_N: np.ndarray,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    contact_threshold: float = 1e-6,
    min_group_size: int = 50,
) -> TemporalSplit:
    """Split preserving proportional contact/flight representation.

    1. Label each timestep as *contact* (|λ_N| > threshold) or *flight*.
    2. Collect the ordered indices for each label.
    3. Within each label's index array, take the first ``train_frac``
       for train, next ``val_frac`` for val, remainder for test.
       If a group has fewer than ``min_group_size`` points, assign
       all of them to the training set to avoid wasting scarce data.
    4. Merge and sort.

    This ensures every split sees both contact and flight dynamics (when
    both exist), and temporal ordering is preserved within each regime.

    Parameters
    ----------
    N : total timesteps
    lambda_N : (N,) normal contact force
    train_frac, val_frac : fractions (must sum < 1)
    contact_threshold : magnitude below which λ_N counts as flight
    min_group_size : groups smaller than this go entirely to training
    """
    is_contact = np.abs(lambda_N) > contact_threshold
    contact_idx = np.where(is_contact)[0]
    flight_idx = np.where(~is_contact)[0]

    idx_train, idx_val, idx_test = [], [], []

    for group_idx in [contact_idx, flight_idx]:
        n = len(group_idx)
        if n == 0:
            continue
        # Small minority group: put all into training
        if n < min_group_size:
            idx_train.append(group_idx)
            continue
        n_tr = int(train_frac * n)
        n_va = int(val_frac * n)
        # Ensure at least 1 in each non-empty set
        n_tr = max(1, n_tr)
        n_va = max(1, min(n_va, n - n_tr - 1))
        n_te = n - n_tr - n_va

        idx_train.append(group_idx[:n_tr])
        idx_val.append(group_idx[n_tr:n_tr + n_va])
        if n_te > 0:
            idx_test.append(group_idx[n_tr + n_va:])

    idx_train = np.sort(np.concatenate(idx_train)) if idx_train else np.arange(0)
    idx_val = np.sort(np.concatenate(idx_val)) if idx_val else np.arange(0)
    idx_test = np.sort(np.concatenate(idx_test)) if idx_test else np.arange(0)

    return TemporalSplit(idx_train, idx_val, idx_test)


# ── Normalization ────────────────────────────────────────────────────────────

@dataclass
class Scalers:
    scaler_X: StandardScaler
    scaler_y: StandardScaler


def fit_scalers(X_train: np.ndarray, y_train: np.ndarray) -> Scalers:
    sX = StandardScaler().fit(X_train)
    sy = StandardScaler().fit(y_train.reshape(-1, 1))
    return Scalers(sX, sy)


def apply_scalers(scalers: Scalers, X: np.ndarray,
                  y: np.ndarray = None) -> Tuple:
    Xn = scalers.scaler_X.transform(X)
    if y is not None:
        yn = scalers.scaler_y.transform(y.reshape(-1, 1)).flatten()
        return Xn, yn
    return Xn


def inverse_scale_y(scalers: Scalers, y_scaled: np.ndarray) -> np.ndarray:
    return scalers.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
