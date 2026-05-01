"""
lagrange_sindy.py – Lagrangian-constrained SINDy.

Instead of directly regressing qddot = Θ·ξ, we:
1. Build a library of candidate Lagrangian terms  L(q, qdot)
2. Symbolically compute the Euler-Lagrange equations  d/dt(∂L/∂qdot) - ∂L/∂q = Q_ext
3. Fit the combined EOM to observed accelerations

For the hula-hoop with mass imbalance:
  Kinetic energy:  T = ½ m (ẋ² + ẏ²) + ½ I θ̇² + m ε R θ̇ (−ẋ sinθ + ẏ cosθ)
  Potential energy: V = m g y + m g d cosθ
  → we learn candidates for T and V coefficients with sparse regression.

The key structural constraint: T must be quadratic in velocities, V depends only on q.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class LagrangeSINDyResult:
    """Result from Lagrangian SINDy fit."""
    kinetic_coefs: np.ndarray      # coefficients for kinetic energy candidates
    potential_coefs: np.ndarray    # coefficients for potential energy candidates
    kinetic_names: List[str]
    potential_names: List[str]
    mse_train: Dict[str, float] = field(default_factory=dict)
    mse_val: Dict[str, float] = field(default_factory=dict)
    mse_test: Dict[str, float] = field(default_factory=dict)
    equations: Dict[str, str] = field(default_factory=dict)
    y_preds: Dict[str, np.ndarray] = field(default_factory=dict)


def build_lagrangian_library(q: np.ndarray, u: np.ndarray
                             ) -> Tuple[np.ndarray, List[str],
                                        np.ndarray, List[str]]:
    """Build candidate terms for T(q, qdot) and V(q).

    Kinetic energy candidates (quadratic in velocities):
      ẋ², ẏ², θ̇², ẋẏ, ẋθ̇, ẏθ̇
      and velocity² × trig: ẋ²sin(θ), ẏ²cos(θ), θ̇²sin(θ), θ̇²cos(θ), ...
      and cross terms × trig: ẋθ̇sin(θ), ẏθ̇cos(θ), ẋθ̇cos(θ), ẏθ̇sin(θ)

    Potential energy candidates (depend on q only):
      y, sin(θ), cos(θ), y·sin(θ), y·cos(θ), sin²(θ), cos²(θ), y²

    Returns
    -------
    T_lib   : (N, n_T) kinetic energy candidate values
    T_names : names
    V_lib   : (N, n_V) potential energy candidate values
    V_names : names
    """
    N = q.shape[0]
    x, y_pos, theta = q[:, 0], q[:, 1], q[:, 2]
    xd, yd, thd = u[:, 0], u[:, 1], u[:, 2]
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    # ── Kinetic energy candidates (quadratic in velocities) ──
    T_cols = []
    T_names = []

    # Pure quadratic
    for vi, vn in [(xd, "xdot"), (yd, "ydot"), (thd, "thetadot")]:
        T_cols.append(vi ** 2)
        T_names.append(f"{vn}^2")

    # Cross terms
    T_cols.append(xd * yd);       T_names.append("xdot ydot")
    T_cols.append(xd * thd);      T_names.append("xdot thetadot")
    T_cols.append(yd * thd);      T_names.append("ydot thetadot")

    # Velocity cross × trig (captures the eccentricity coupling)
    T_cols.append(xd * thd * sin_th);  T_names.append("xdot thetadot sin(theta)")
    T_cols.append(xd * thd * cos_th);  T_names.append("xdot thetadot cos(theta)")
    T_cols.append(yd * thd * sin_th);  T_names.append("ydot thetadot sin(theta)")
    T_cols.append(yd * thd * cos_th);  T_names.append("ydot thetadot cos(theta)")

    # θ̇² × trig
    T_cols.append(thd ** 2 * sin_th);  T_names.append("thetadot^2 sin(theta)")
    T_cols.append(thd ** 2 * cos_th);  T_names.append("thetadot^2 cos(theta)")

    T_lib = np.column_stack(T_cols)

    # ── Potential energy candidates ──
    V_cols = []
    V_names = []

    V_cols.append(y_pos);              V_names.append("y")
    V_cols.append(sin_th);             V_names.append("sin(theta)")
    V_cols.append(cos_th);             V_names.append("cos(theta)")
    V_cols.append(y_pos * sin_th);     V_names.append("y sin(theta)")
    V_cols.append(y_pos * cos_th);     V_names.append("y cos(theta)")
    V_cols.append(sin_th ** 2);        V_names.append("sin(theta)^2")
    V_cols.append(cos_th ** 2);        V_names.append("cos(theta)^2")
    V_cols.append(y_pos ** 2);         V_names.append("y^2")
    V_cols.append(x);                  V_names.append("x")

    V_lib = np.column_stack(V_cols)

    return T_lib, T_names, V_lib, V_names


def euler_lagrange_rhs(q: np.ndarray, u: np.ndarray, qddot: np.ndarray,
                       dt: float
                       ) -> Tuple[np.ndarray, List[str],
                                  np.ndarray, List[str]]:
    r"""Compute Euler-Lagrange equation columns for regression.

    For each kinetic candidate T_k(q, qdot) and potential candidate V_k(q):

      EOM_i :  Σ_k  c_k^T  [d/dt(∂T_k/∂qdot_i) - ∂T_k/∂q_i]
             - Σ_j  c_j^V  ∂V_j/∂q_i  =  0

    We compute each EL column numerically:
      For T_k:  col_i = d/dt(∂T_k/∂qdot_i) - ∂T_k/∂q_i
      For V_j:  col_i = -∂V_j/∂q_i

    Then we stack them and regress against the external forces (here, zero
    for conservative system, or the contact forces if available).

    Returns EL_matrix (N, n_T + n_V) for each DOF i stacked, plus names.
    """
    N = q.shape[0]
    eps = 1e-6

    T_lib, T_names, V_lib, V_names = build_lagrangian_library(q, u)
    n_T = T_lib.shape[1]
    n_V = V_lib.shape[1]

    # We'll build the EL matrix for all 3 DOFs simultaneously: (3*N, n_T + n_V)
    # But it's more useful to return per-DOF: EL_cols[dof] = (N, n_T + n_V)
    EL_per_dof = {}

    for dof in range(3):
        cols = []

        # ── Kinetic energy terms ──
        for k in range(n_T):
            # ∂T_k/∂qdot_i via finite difference in velocity space
            u_plus = u.copy();  u_plus[:, dof] += eps
            u_minus = u.copy(); u_minus[:, dof] -= eps

            T_lib_plus, _, _, _ = build_lagrangian_library(q, u_plus)
            T_lib_minus, _, _, _ = build_lagrangian_library(q, u_minus)
            dT_dqdot = (T_lib_plus[:, k] - T_lib_minus[:, k]) / (2 * eps)

            # d/dt(∂T_k/∂qdot_i) via central difference in time
            d_dt_dT_dqdot = np.gradient(dT_dqdot, dt, axis=0)

            # ∂T_k/∂q_i via finite difference in position space
            q_plus = q.copy();  q_plus[:, dof] += eps
            q_minus = q.copy(); q_minus[:, dof] -= eps

            T_lib_qp, _, _, _ = build_lagrangian_library(q_plus, u)
            T_lib_qm, _, _, _ = build_lagrangian_library(q_minus, u)
            dT_dq = (T_lib_qp[:, k] - T_lib_qm[:, k]) / (2 * eps)

            cols.append(d_dt_dT_dqdot - dT_dq)

        # ── Potential energy terms ──
        for j in range(n_V):
            q_plus = q.copy();  q_plus[:, dof] += eps
            q_minus = q.copy(); q_minus[:, dof] -= eps

            _, _, V_lib_plus, _ = build_lagrangian_library(q_plus, u)
            _, _, V_lib_minus, _ = build_lagrangian_library(q_minus, u)
            dV_dq = (V_lib_plus[:, j] - V_lib_minus[:, j]) / (2 * eps)

            cols.append(-dV_dq)

        EL_per_dof[dof] = np.column_stack(cols)

    all_names = [f"T:{n}" for n in T_names] + [f"V:{n}" for n in V_names]
    return EL_per_dof, all_names, T_names, V_names


def fit_lagrange_sindy(
    q: np.ndarray, u: np.ndarray, qddot: np.ndarray,
    idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray,
    dt: float = None,
    alpha: float = 0.01,
    external_forces: np.ndarray = None,
    lambda_N: np.ndarray = None,
    lambda_F: np.ndarray = None,
    remove_cos2: bool = True,
) -> LagrangeSINDyResult:
    """Fit Lagrangian SINDy using analytical EL and shared coefficients.

    Uses closed-form Euler-Lagrange derivatives (no finite differences).
    Fits a single shared coefficient vector across all 3 DOFs, since one
    Lagrangian governs the entire system.

    For contact data (lambda_N/lambda_F provided):
      EL @ c = Q_ext   (direct Lasso regression)
    For flight data (no contact forces):
      Anchor approach — fix xdot^2 coeff = 1 to avoid trivial c=0.
    """
    N = q.shape[0]
    DOF_NAMES = ["a_x", "a_y", "a_theta"]
    R_hoop = 0.18

    print("  Computing Euler-Lagrange columns (analytical)...")
    EL_per_dof, all_names, T_names, V_names = euler_lagrange_rhs_analytical(
        q, u, qddot
    )
    n_T = len(T_names)
    n_V = len(V_names)

    # Optionally remove cos²θ (degenerate with sin²θ via trig identity)
    if remove_cos2 and "cos(theta)^2" in V_names:
        cos2_idx = n_T + V_names.index("cos(theta)^2")
        keep = [i for i in range(n_T + n_V) if i != cos2_idx]
        for dof in range(3):
            EL_per_dof[dof] = EL_per_dof[dof][:, keep]
        all_names = [all_names[i] for i in keep]
        V_names = [v for v in V_names if v != "cos(theta)^2"]
        n_V -= 1

    n_total = n_T + n_V

    # ── Build RHS: external forces ──
    Q_ext = np.zeros((N, 3))
    if lambda_N is not None:
        Q_ext[:, 1] += lambda_N          # W_N = [0, 1, 0]
    if lambda_F is not None:
        Q_ext[:, 0] += lambda_F          # W_F = [1, 0, R]
        Q_ext[:, 2] += R_hoop * lambda_F

    has_contact = np.any(np.abs(Q_ext) > 1e-8)

    # ── Stack all 3 DOFs for shared-coefficient fitting ──
    EL_stacked_tr = np.vstack([EL_per_dof[d][idx_train] for d in range(3)])
    y_stacked_tr = np.concatenate([Q_ext[idx_train, d] for d in range(3)])

    if has_contact:
        # Direct Lasso: EL @ c = Q_ext
        sc_EL = StandardScaler().fit(EL_stacked_tr)
        sc_y = StandardScaler().fit(y_stacked_tr.reshape(-1, 1))

        EL_n = sc_EL.transform(EL_stacked_tr)
        y_n = sc_y.transform(y_stacked_tr.reshape(-1, 1)).flatten()

        model = Lasso(alpha=alpha, max_iter=10000, fit_intercept=True)
        model.fit(EL_n, y_n)

        coefs = model.coef_ * sc_y.scale_[0] / sc_EL.scale_
    else:
        # Anchor approach: fix xdot^2 coeff = 1, move to RHS
        anchor_idx = 0  # T:xdot^2
        anchor_col = EL_stacked_tr[:, anchor_idx]
        remain_idx = [i for i in range(n_total) if i != anchor_idx]
        EL_remain = EL_stacked_tr[:, remain_idx]

        y_anchor = -anchor_col

        sc_EL = StandardScaler().fit(EL_remain)
        sc_y = StandardScaler().fit(y_anchor.reshape(-1, 1))

        EL_n = sc_EL.transform(EL_remain)
        y_n = sc_y.transform(y_anchor.reshape(-1, 1)).flatten()

        model = Lasso(alpha=alpha, max_iter=10000, fit_intercept=False)
        model.fit(EL_n, y_n)

        coefs_remain = model.coef_ * sc_y.scale_[0] / sc_EL.scale_
        coefs = np.zeros(n_total)
        coefs[anchor_idx] = 1.0
        for i, ri in enumerate(remain_idx):
            coefs[ri] = coefs_remain[i]

    # ── Compute per-DOF MSE (train / val / test) ──
    result = LagrangeSINDyResult(
        kinetic_coefs=coefs[:n_T],
        potential_coefs=coefs[n_T:],
        kinetic_names=T_names,
        potential_names=V_names,
    )

    for dof in range(3):
        EL = EL_per_dof[dof]
        actual_tr = Q_ext[idx_train, dof]
        actual_va = Q_ext[idx_val, dof]
        actual_te = Q_ext[idx_test, dof]

        pred_tr = EL[idx_train] @ coefs
        pred_va = EL[idx_val] @ coefs
        pred_te = EL[idx_test] @ coefs

        y_pred = np.full(N, np.nan)
        y_pred[idx_train] = pred_tr
        y_pred[idx_val] = pred_va
        y_pred[idx_test] = pred_te

        result.mse_train[DOF_NAMES[dof]] = mean_squared_error(actual_tr, pred_tr)
        result.mse_val[DOF_NAMES[dof]] = mean_squared_error(actual_va, pred_va)
        result.mse_test[DOF_NAMES[dof]] = mean_squared_error(actual_te, pred_te)
        result.y_preds[DOF_NAMES[dof]] = y_pred

        # Build equation string from shared coefficients
        eqn_parts = []
        for j in np.nonzero(coefs)[0]:
            eqn_parts.append(f"{coefs[j]:+.4f}*{all_names[j]}")
        result.equations[DOF_NAMES[dof]] = " ".join(eqn_parts) if eqn_parts else "0"

        n_active = int(np.count_nonzero(coefs))
        print(f"  {DOF_NAMES[dof]}: {n_active} active terms | "
              f"MSE train={result.mse_train[DOF_NAMES[dof]]:.4e} "
              f"val={result.mse_val[DOF_NAMES[dof]]:.4e} "
              f"test={result.mse_test[DOF_NAMES[dof]]:.4e}")

    return result


# ── Analytical Euler-Lagrange computation ────────────────────────────────────

@dataclass
class SharedLagrangeSINDyResult:
    """Result from shared-coefficient Lagrangian SINDy."""
    coefs: np.ndarray              # (n_T + n_V,) shared coefficient vector
    all_names: List[str]           # T:name and V:name labels
    kinetic_names: List[str]
    potential_names: List[str]
    n_T: int
    n_V: int
    mse_per_dof: Dict[str, float] = field(default_factory=dict)
    mse_train_per_dof: Dict[str, float] = field(default_factory=dict)
    mse_val_per_dof: Dict[str, float] = field(default_factory=dict)
    residual_norm: float = 0.0

    @property
    def kinetic_coefs(self):
        return self.coefs[:self.n_T]

    @property
    def potential_coefs(self):
        return self.coefs[self.n_T:]

    @property
    def n_active(self):
        return int(np.count_nonzero(self.coefs))

    def lagrangian_equation(self, threshold: float = 1e-8) -> str:
        parts_T, parts_V = [], []
        for i, name in enumerate(self.kinetic_names):
            c = self.coefs[i]
            if abs(c) > threshold:
                parts_T.append(f"{c:+.6f}*{name}")
        for j, name in enumerate(self.potential_names):
            c = self.coefs[self.n_T + j]
            if abs(c) > threshold:
                parts_V.append(f"{c:+.6f}*{name}")
        T_str = " ".join(parts_T) if parts_T else "0"
        V_str = " ".join(parts_V) if parts_V else "0"
        return f"T = {T_str}\nV = {V_str}"


def euler_lagrange_rhs_analytical(
    q: np.ndarray, u: np.ndarray, qddot: np.ndarray,
) -> Tuple[Dict[int, np.ndarray], List[str], List[str], List[str]]:
    r"""Compute Euler-Lagrange columns analytically (closed-form).

    For each library term, compute the EL contribution to each DOF:
      T_k:  d/dt(∂T_k/∂qdot_i) - ∂T_k/∂q_i
      V_j:  -∂V_j/∂q_i

    This avoids numerical derivatives entirely.

    Returns
    -------
    EL_per_dof : {0: (N, n_T+n_V), 1: ..., 2: ...}
    all_names  : combined name list
    T_names    : kinetic names
    V_names    : potential names
    """
    N = q.shape[0]
    x, y_pos, theta = q[:, 0], q[:, 1], q[:, 2]
    xd, yd, thd = u[:, 0], u[:, 1], u[:, 2]
    xdd, ydd, thdd = qddot[:, 0], qddot[:, 1], qddot[:, 2]
    s = np.sin(theta)
    c = np.cos(theta)
    z = np.zeros(N)

    T_names = [
        "xdot^2", "ydot^2", "thetadot^2",
        "xdot ydot", "xdot thetadot", "ydot thetadot",
        "xdot thetadot sin(theta)", "xdot thetadot cos(theta)",
        "ydot thetadot sin(theta)", "ydot thetadot cos(theta)",
        "thetadot^2 sin(theta)", "thetadot^2 cos(theta)",
    ]
    V_names = [
        "y", "sin(theta)", "cos(theta)",
        "y sin(theta)", "y cos(theta)",
        "sin(theta)^2", "cos(theta)^2",
        "y^2", "x",
    ]

    # ── Kinetic EL columns: d/dt(∂T/∂qdot_i) - ∂T/∂q_i ──

    # T0: xdot^2
    T0 = {0: 2*xdd, 1: z.copy(), 2: z.copy()}

    # T1: ydot^2
    T1 = {0: z.copy(), 1: 2*ydd, 2: z.copy()}

    # T2: thetadot^2
    T2 = {0: z.copy(), 1: z.copy(), 2: 2*thdd}

    # T3: xdot*ydot
    T3 = {0: ydd, 1: xdd, 2: z.copy()}

    # T4: xdot*thetadot
    T4 = {0: thdd, 1: z.copy(), 2: xdd}

    # T5: ydot*thetadot
    T5 = {0: z.copy(), 1: thdd, 2: ydd}

    # T6: xdot*thetadot*sin(theta)
    T6 = {0: thdd*s + thd**2*c, 1: z.copy(), 2: xdd*s}

    # T7: xdot*thetadot*cos(theta)
    T7 = {0: thdd*c - thd**2*s, 1: z.copy(), 2: xdd*c}

    # T8: ydot*thetadot*sin(theta)
    T8 = {0: z.copy(), 1: thdd*s + thd**2*c, 2: ydd*s}

    # T9: ydot*thetadot*cos(theta)
    T9 = {0: z.copy(), 1: thdd*c - thd**2*s, 2: ydd*c}

    # T10: thetadot^2*sin(theta)
    T10 = {0: z.copy(), 1: z.copy(), 2: 2*thdd*s + thd**2*c}

    # T11: thetadot^2*cos(theta)
    T11 = {0: z.copy(), 1: z.copy(), 2: 2*thdd*c - thd**2*s}

    T_EL = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11]

    # ── Potential EL columns: -∂V/∂q_i ──

    # V0: y
    V0 = {0: z.copy(), 1: -np.ones(N), 2: z.copy()}

    # V1: sin(theta)
    V1 = {0: z.copy(), 1: z.copy(), 2: -c}

    # V2: cos(theta)
    V2 = {0: z.copy(), 1: z.copy(), 2: s}

    # V3: y*sin(theta)
    V3 = {0: z.copy(), 1: -s, 2: -y_pos*c}

    # V4: y*cos(theta)
    V4 = {0: z.copy(), 1: -c, 2: y_pos*s}

    # V5: sin(theta)^2
    V5 = {0: z.copy(), 1: z.copy(), 2: -2*s*c}

    # V6: cos(theta)^2
    V6 = {0: z.copy(), 1: z.copy(), 2: 2*s*c}

    # V7: y^2
    V7 = {0: z.copy(), 1: -2*y_pos, 2: z.copy()}

    # V8: x
    V8 = {0: -np.ones(N), 1: z.copy(), 2: z.copy()}

    V_EL = [V0, V1, V2, V3, V4, V5, V6, V7, V8]

    # ── Assemble per-DOF matrices ──
    n_T = len(T_EL)
    n_V = len(V_EL)
    EL_per_dof = {}
    for dof in range(3):
        cols = []
        for t_el in T_EL:
            cols.append(t_el[dof])
        for v_el in V_EL:
            cols.append(v_el[dof])
        EL_per_dof[dof] = np.column_stack(cols)

    all_names = [f"T:{n}" for n in T_names] + [f"V:{n}" for n in V_names]
    return EL_per_dof, all_names, T_names, V_names


def fit_lagrange_sindy_shared(
    q: np.ndarray, u: np.ndarray, qddot: np.ndarray,
    idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray,
    lambda_N: Optional[np.ndarray] = None,
    lambda_F: Optional[np.ndarray] = None,
    alpha: float = 0.01,
    remove_cos2: bool = True,
    optimizer: str = 'Lasso',
    stlsq_threshold: float = 0.01,
    stlsq_max_iter: int = 30,
) -> SharedLagrangeSINDyResult:
    r"""Shared-coefficient Lagrangian SINDy with analytical EL.

    Stack EL equations from all 3 DOFs into one tall system:
      [EL_0]         [Q_ext_0]
      [EL_1] @ c  =  [Q_ext_1]
      [EL_2]         [Q_ext_2]

    where Q_ext includes contact forces via W_N and W_F Jacobians.

    For flight data: Q_ext = 0, so we use the anchor approach (move
    one known column to RHS) to avoid the trivial c=0 solution.
    For contact data: Q_ext = λ_N·W_N + λ_F·W_F ≠ 0, so Lasso works.

    Parameters
    ----------
    remove_cos2 : if True, drop cos²θ from V library (degenerate with sin²θ)
    optimizer : 'Lasso', 'Ridge', 'ElasticNet', or 'STLSQ'
    stlsq_threshold : hard-threshold for STLSQ (terms below this are zeroed)
    stlsq_max_iter : max iterations for STLSQ sequential thresholding
    """
    N = q.shape[0]
    DOF_NAMES = ["a_x", "a_y", "a_theta"]

    print("  Computing Euler-Lagrange columns (analytical)...")
    EL_per_dof, all_names, T_names, V_names = euler_lagrange_rhs_analytical(q, u, qddot)
    n_T = len(T_names)
    n_V = len(V_names)

    # Optionally remove cos²θ (index 6 in V, overall index n_T + 6)
    if remove_cos2 and "cos(theta)^2" in V_names:
        cos2_idx = n_T + V_names.index("cos(theta)^2")
        keep = [i for i in range(n_T + n_V) if i != cos2_idx]
        for dof in range(3):
            EL_per_dof[dof] = EL_per_dof[dof][:, keep]
        all_names = [all_names[i] for i in keep]
        V_names = [v for v in V_names if v != "cos(theta)^2"]
        n_V -= 1

    n_total = n_T + n_V

    # ── Build RHS: external forces ──
    # Contact Jacobians for hula-hoop:
    #   W_N = [0, 1, 0]^T  (normal force along y)
    #   W_F = [1, 0, R]^T  (friction along x, with moment arm R on θ)
    R_hoop = 0.18

    Q_ext = np.zeros((N, 3))
    if lambda_N is not None:
        Q_ext[:, 1] += lambda_N  # W_N = [0,1,0]
    if lambda_F is not None:
        Q_ext[:, 0] += lambda_F  # W_F[0] = 1
        Q_ext[:, 2] += R_hoop * lambda_F  # W_F[2] = R

    # Check if we have non-trivial RHS
    has_contact = np.any(np.abs(Q_ext) > 1e-8)

    # ── Stack all 3 DOFs ──
    EL_stacked = np.vstack([EL_per_dof[d][idx_train] for d in range(3)])
    y_stacked = np.concatenate([Q_ext[idx_train, d] for d in range(3)])

    def _make_model(fit_intercept):
        if optimizer == 'Lasso':
            return Lasso(alpha=alpha, max_iter=10000, fit_intercept=fit_intercept)
        elif optimizer == 'Ridge':
            return Ridge(alpha=alpha, max_iter=10000, fit_intercept=fit_intercept)
        elif optimizer == 'ElasticNet':
            return ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000,
                              fit_intercept=fit_intercept)
        elif optimizer == 'STLSQ':
            return Ridge(alpha=alpha, max_iter=10000, fit_intercept=fit_intercept)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def _stlsq_fit(X, y, fit_intercept):
        """Sequential thresholded least squares."""
        active = np.ones(X.shape[1], dtype=bool)
        for _ in range(stlsq_max_iter):
            model = Ridge(alpha=alpha, max_iter=10000, fit_intercept=fit_intercept)
            model.fit(X[:, active], y)
            full_coef = np.zeros(X.shape[1])
            full_coef[active] = model.coef_
            small = np.abs(full_coef) < stlsq_threshold
            if not np.any(small & active):
                break
            active &= ~small
            if not np.any(active):
                break
        return full_coef

    if has_contact:
        # Direct regression: EL @ c = Q_ext
        sc_EL = StandardScaler().fit(EL_stacked)
        sc_y = StandardScaler().fit(y_stacked.reshape(-1, 1))

        EL_n = sc_EL.transform(EL_stacked)
        y_n = sc_y.transform(y_stacked.reshape(-1, 1)).flatten()

        if optimizer == 'STLSQ':
            coefs_scaled = _stlsq_fit(EL_n, y_n, fit_intercept=True)
        else:
            model = _make_model(fit_intercept=True)
            model.fit(EL_n, y_n)
            coefs_scaled = model.coef_

        coefs = coefs_scaled * sc_y.scale_[0] / sc_EL.scale_
    else:
        # Anchor approach for flight: move xdot^2 column (idx 0) to RHS
        anchor_idx = 0  # T:xdot^2
        anchor_col = EL_stacked[:, anchor_idx]
        remain_idx = [i for i in range(n_total) if i != anchor_idx]
        EL_remain = EL_stacked[:, remain_idx]

        y_anchor = -anchor_col

        sc_EL = StandardScaler().fit(EL_remain)
        sc_y = StandardScaler().fit(y_anchor.reshape(-1, 1))

        EL_n = sc_EL.transform(EL_remain)
        y_n = sc_y.transform(y_anchor.reshape(-1, 1)).flatten()

        if optimizer == 'STLSQ':
            coefs_remain_scaled = _stlsq_fit(EL_n, y_n, fit_intercept=False)
        else:
            model = _make_model(fit_intercept=False)
            model.fit(EL_n, y_n)
            coefs_remain_scaled = model.coef_

        coefs_remain = coefs_remain_scaled * sc_y.scale_[0] / sc_EL.scale_
        coefs = np.zeros(n_total)
        coefs[anchor_idx] = 1.0
        for i, ri in enumerate(remain_idx):
            coefs[ri] = coefs_remain[i]

    # ── Compute MSE per DOF (train / val / test) ──
    mse_per_dof = {}
    mse_train_per_dof = {}
    mse_val_per_dof = {}
    for dof in range(3):
        name = DOF_NAMES[dof]
        EL_d = EL_per_dof[dof]
        mse_train_per_dof[name] = float(mean_squared_error(
            Q_ext[idx_train, dof], EL_d[idx_train] @ coefs))
        mse_val_per_dof[name] = float(mean_squared_error(
            Q_ext[idx_val, dof], EL_d[idx_val] @ coefs))
        mse_per_dof[name] = float(mean_squared_error(
            Q_ext[idx_test, dof], EL_d[idx_test] @ coefs))

    residual_all = np.vstack([EL_per_dof[d] for d in range(3)]) @ coefs - \
                   np.concatenate([Q_ext[:, d] for d in range(3)])
    residual_norm = float(np.linalg.norm(residual_all))

    result = SharedLagrangeSINDyResult(
        coefs=coefs,
        all_names=all_names,
        kinetic_names=T_names,
        potential_names=V_names,
        n_T=n_T,
        n_V=n_V,
        mse_per_dof=mse_per_dof,
        mse_train_per_dof=mse_train_per_dof,
        mse_val_per_dof=mse_val_per_dof,
        residual_norm=residual_norm,
    )

    print(f"  Active terms: {result.n_active}")
    print(f"  Residual norm: {residual_norm:.4e}")
    for dof_name in DOF_NAMES:
        print(f"  {dof_name}  train MSE: {mse_train_per_dof[dof_name]:.4e}  "
              f"val MSE: {mse_val_per_dof[dof_name]:.4e}  "
              f"test MSE: {mse_per_dof[dof_name]:.4e}")
    print(f"\n{result.lagrangian_equation()}")

    return result
