"""
evaluation.py – Evaluation metrics and comparison utilities.

Metrics:
  A. Structural accuracy (correct physical terms present)
  B. Unit consistency score
  C. Parameter accuracy (vs analytical ground truth)
  D. Forward simulation error
  E. Model parsimony
"""

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ── Ground truth coefficients ────────────────────────────────────────────────
# From the simulation physics:
#   M(θ) q̈ = f_applied + f_velocity + W_N λ_N + W_F λ_F
#
# In flight (no contact):
#   m ẍ - m ε R sin(θ) θ̈ = m ε R θ̇² cos(θ)
#   m ÿ + m ε R cos(θ) θ̈ = -mg + m ε R θ̇² sin(θ)
#   -m ε R sin(θ) ẍ + m ε R cos(θ) ÿ + m R² θ̈ = -mgd cos(θ)
#
# Parameters: m=1.5, R=0.18, ε=1/3, d=0.06, g=1 (non-dimensionalized)

PHYSICAL_PARAMS = {
    "m": 1.5, "R": 0.18, "epsilon": 1/3, "d": 0.06,
    "I": 0.0432, "g": 1.0,
    "mR2": 1.5 * 0.18**2,                    # = 0.0486
    "m_eps_R": 1.5 * (1/3) * 0.18,           # = 0.09
    "mg": 1.5 * 1.0,                          # = 1.5
    "mgd": 1.5 * 1.0 * 0.06,                 # = 0.09
}

# Expected active terms for flight regime (no contact forces):
#  xddot ~ thetaddot*sin(θ) + θ̇²*cos(θ)
#  yddot ~ -g + thetaddot*cos(θ) + θ̇²*sin(θ)   (after solving M⁻¹)
#  thetaddot ~ sin(θ)*xddot - cos(θ)*yddot - (mgd/mR²)*cos(θ)  (from 3rd row)
EXPECTED_TERMS = {
    "flight": {
        "a_x": ["sin(theta)", "cos(theta)", "thetadot^2", "thetaddot"],
        "a_y": ["1", "sin(theta)", "cos(theta)", "thetadot^2", "thetaddot"],
        "a_theta": ["cos(theta)", "xddot", "yddot"],
    },
}


# ── Metric A: Structural accuracy ───────────────────────────────────────────

def structural_accuracy(discovered_names: List[str], coefs: np.ndarray,
                        expected_terms: List[str]) -> Dict:
    """Check which expected physical terms appear in the discovered model."""
    active_idx = np.nonzero(coefs)[0]
    active_names = [discovered_names[i] for i in active_idx]

    found = [t for t in expected_terms if any(t in a for a in active_names)]
    missing = [t for t in expected_terms if t not in found]
    spurious = [a for a in active_names
                if not any(t in a for t in expected_terms)]

    return {
        "expected": expected_terms,
        "found": found,
        "missing": missing,
        "spurious": spurious,
        "recall": len(found) / max(len(expected_terms), 1),
        "precision": len(found) / max(len(active_names), 1),
    }


# ── Metric B: Unit consistency (delegated to unit_filter.py) ─────────────────

def unit_consistency(feature_names, coefs, dof):
    from .unit_filter import unit_consistency_score
    return unit_consistency_score(feature_names, coefs, dof)


# ── Metric C: Parameter accuracy ────────────────────────────────────────────

def parameter_accuracy(discovered_coefs: Dict[str, float],
                       expected_coefs: Dict[str, float]) -> Dict:
    """Compare discovered coefficients against ground truth."""
    report = {}
    for term, expected in expected_coefs.items():
        found = discovered_coefs.get(term, 0.0)
        abs_err = abs(found - expected)
        rel_err = abs_err / max(abs(expected), 1e-12)
        report[term] = {
            "expected": expected, "found": found,
            "abs_error": abs_err, "rel_error": rel_err,
        }
    return report


# ── Metric D: Forward simulation error ──────────────────────────────────────

def simulate_forward(sindy_results: Dict, q0: np.ndarray, u0: np.ndarray,
                     t_span: Tuple[float, float], dt: float,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the discovered system forward in time.

    This requires solving the coupled system:
      qdot = u
      M(q) qddot = f_discovered(q, u)

    For now we use a simplified approach: since the SINDy models predict
    qddot directly (including cross-coupling via other accelerations),
    we iterate: predict qddot → integrate → step.

    Returns
    -------
    t_sim : (n_steps,) time points
    q_sim : (n_steps, 3) simulated positions
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t_eval)
    q_sim = np.zeros((n_steps, 3))
    u_sim = np.zeros((n_steps, 3))
    q_sim[0] = q0
    u_sim[0] = u0

    for k in range(n_steps - 1):
        # Predict accelerations at current state
        qddot_k = np.zeros(3)
        for dof in range(3):
            res = sindy_results.get(["a_x", "a_y", "a_theta"][dof])
            if res is None:
                continue
            # Build feature vector at current state
            # This is simplified — in general we need the same library transform
            qddot_k[dof] = 0  # placeholder

        # Semi-implicit Euler
        u_sim[k + 1] = u_sim[k] + dt * qddot_k
        q_sim[k + 1] = q_sim[k] + dt * u_sim[k + 1]

    return t_eval, q_sim


def simulation_error(q_true: np.ndarray, q_sim: np.ndarray) -> Dict:
    """Compute trajectory error metrics."""
    n = min(len(q_true), len(q_sim))
    labels = ["x", "y", "theta"]
    report = {}
    for i, lab in enumerate(labels):
        err = q_true[:n, i] - q_sim[:n, i]
        report[lab] = {
            "max_abs": float(np.max(np.abs(err))),
            "rms": float(np.sqrt(np.mean(err ** 2))),
            "mse": float(mean_squared_error(q_true[:n, i], q_sim[:n, i])),
        }
    return report


# ── Metric D2: Open-loop ATE (Absolute Trajectory Error) ────────────────────

def compute_open_loop_ate(y_pred_segment: np.ndarray,
                          q_true_segment: np.ndarray,
                          u_true_segment: np.ndarray,
                          dt: float) -> float:
    """
    Open-loop Absolute Trajectory Error for one DOF.

    Integrates predicted accelerations forward from true initial conditions
    using semi-implicit Euler and compares the resulting trajectory to
    ground truth positions.

    Parameters
    ----------
    y_pred_segment : shape (n,)
        Predicted accelerations for one DOF over a contiguous time segment.
    q_true_segment : shape (n,)
        True positions for the same DOF over the same segment.
    u_true_segment : shape (n,)
        True velocities for the same DOF over the same segment.
    dt : float
        Time step size.

    Returns
    -------
    ate : float
        RMS position error between integrated and true trajectory.
    """
    n = len(y_pred_segment)
    if n < 2:
        return 0.0

    q_sim = np.zeros(n)
    u_sim = np.zeros(n)
    q_sim[0] = q_true_segment[0]
    u_sim[0] = u_true_segment[0]

    for i in range(n - 1):
        u_sim[i + 1] = u_sim[i] + y_pred_segment[i] * dt
        q_sim[i + 1] = q_sim[i] + u_sim[i + 1] * dt

    return float(np.sqrt(np.mean((q_true_segment - q_sim) ** 2)))


# ── Lagrange → predicted accelerations (for ATE) ────────────────────────────

def lagrange_predict_qddot(res, q, u, lambda_N=None, lambda_F=None,
                           R_hoop=0.18):
    """Predict accelerations from a SharedLagrangeSINDyResult.

    Solves  M(q) @ qddot = Q_ext - h(q, qdot)  at each timestep,
    where M is the mass matrix and h is the non-acceleration forcing,
    both derived from the discovered Lagrangian.

    Parameters
    ----------
    res : SharedLagrangeSINDyResult
    q, u : (N, 3) state arrays
    lambda_N, lambda_F : (N,) contact forces or None
    R_hoop : hoop radius

    Returns
    -------
    qddot_pred : (N, 3) predicted accelerations
    """
    from pipeline.lagrange_sindy import euler_lagrange_rhs_analytical

    N = q.shape[0]
    theta = q[:, 2]
    s = np.sin(theta)
    c_th = np.cos(theta)

    # Evaluate EL columns with qddot=0 to isolate non-acceleration terms
    qddot_zero = np.zeros_like(q)
    EL_noaccel, full_names, T_names, V_names = euler_lagrange_rhs_analytical(
        q, u, qddot_zero)

    # Map result coefs onto the full (possibly un-pruned) feature set
    full_coefs = np.zeros(len(full_names))
    res_map = {name: res.coefs[i] for i, name in enumerate(res.all_names)}
    for j, fname in enumerate(full_names):
        if fname in res_map:
            full_coefs[j] = res_map[fname]

    # External forces
    Q_ext = np.zeros((N, 3))
    if lambda_N is not None:
        Q_ext[:, 1] += lambda_N
    if lambda_F is not None:
        Q_ext[:, 0] += lambda_F
        Q_ext[:, 2] += R_hoop * lambda_F

    # RHS = Q_ext - (non-accel EL terms) @ coefs
    rhs = np.zeros((N, 3))
    for d in range(3):
        rhs[:, d] = Q_ext[:, d] - EL_noaccel[d] @ full_coefs

    # Build mass matrix M(θ) from kinetic coefs
    # T order: 0:xd², 1:yd², 2:thd², 3:xd*yd, 4:xd*thd, 5:yd*thd,
    #   6:xd*thd*sin, 7:xd*thd*cos, 8:yd*thd*sin, 9:yd*thd*cos,
    #   10:thd²*sin, 11:thd²*cos
    n_T = len(T_names)
    c_T = full_coefs[:n_T]

    M = np.zeros((N, 3, 3))
    M[:, 0, 0] = 2 * c_T[0]
    M[:, 1, 1] = 2 * c_T[1]
    M[:, 2, 2] = 2 * c_T[2] + 2 * c_T[10] * s + 2 * c_T[11] * c_th
    M[:, 0, 1] = c_T[3];          M[:, 1, 0] = c_T[3]
    M[:, 0, 2] = c_T[4] + c_T[6] * s + c_T[7] * c_th
    M[:, 2, 0] = c_T[4] + c_T[6] * s + c_T[7] * c_th
    M[:, 1, 2] = c_T[5] + c_T[8] * s + c_T[9] * c_th
    M[:, 2, 1] = c_T[5] + c_T[8] * s + c_T[9] * c_th

    # Solve M @ qddot = rhs at each timestep
    qddot_pred = np.zeros((N, 3))
    for t in range(N):
        try:
            qddot_pred[t] = np.linalg.solve(M[t], rhs[t])
        except np.linalg.LinAlgError:
            qddot_pred[t] = np.linalg.lstsq(M[t], rhs[t], rcond=None)[0]

    return qddot_pred


def compute_lagrange_ate(res, q, u, idx, dt,
                         lambda_N=None, lambda_F=None):
    """Compute per-DOF ATE for a SharedLagrangeSINDyResult.

    Returns list of 3 floats (one ATE per DOF).
    """
    qddot_pred = lagrange_predict_qddot(res, q, u, lambda_N, lambda_F)
    ates = []
    for dof in range(3):
        ate = compute_open_loop_ate(
            qddot_pred[idx, dof], q[idx, dof], u[idx, dof], dt)
        ates.append(ate)
    return ates


# ── Metric E: Parsimony ─────────────────────────────────────────────────────

def parsimony(coefs: np.ndarray) -> int:
    return int(np.count_nonzero(coefs))


# ── Comparison table ─────────────────────────────────────────────────────────

@dataclass
class ModelComparison:
    """Side-by-side comparison of multiple SINDy experiments."""
    rows: List[Dict] = field(default_factory=list)

    def add(self, label: str, dof: str, n_terms: int,
            mse_train: float, mse_val: float, mse_test: float,
            unit_score: float, equation: str = ""):
        self.rows.append({
            "label": label, "dof": dof, "n_terms": n_terms,
            "mse_train": mse_train, "mse_val": mse_val, "mse_test": mse_test,
            "unit_score": unit_score, "equation": equation,
        })

    def table(self) -> str:
        header = (f"{'Label':<30} {'DOF':<8} {'#':>4} "
                  f"{'MSE_train':>12} {'MSE_val':>12} {'MSE_test':>12} {'UC%':>5}")
        lines = [header, "-" * len(header)]
        for r in self.rows:
            lines.append(
                f"{r['label']:<30} {r['dof']:<8} {r['n_terms']:>4} "
                f"{r['mse_train']:>12.4e} {r['mse_val']:>12.4e} "
                f"{r['mse_test']:>12.4e} {r['unit_score']:>4.0%}"
            )
        return "\n".join(lines)
