"""
library.py – Build candidate feature libraries for SINDy regression.

Provides both a full (unfiltered) and unit-consistent (filtered) library.
Supports regime-specific terms: sign(qdot) for slipping, etc.
"""

import numpy as np
import re
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import PolynomialFeatures


def _poly_feature_names(names: List[str], degree: int,
                        include_bias: bool) -> List[str]:
    """Generate polynomial feature names matching sklearn output order."""
    pf = PolynomialFeatures(degree=degree, include_bias=include_bias,
                            interaction_only=False)
    pf.fit(np.zeros((1, len(names))))
    return pf.get_feature_names_out(names).tolist()


def build_features(q: np.ndarray, u: np.ndarray, qddot: np.ndarray,
                   target_dof: int,
                   degree: int = 2,
                   include_trig: bool = True,
                   include_sign_qdot: bool = False,
                   include_bias: bool = True,
                   ) -> Tuple[np.ndarray, List[str]]:
    """Build the full candidate feature matrix Θ for predicting qddot[target_dof].

    Parameters
    ----------
    q      : (N, 3)  positions [x, y, theta]
    u      : (N, 3)  velocities [xdot, ydot, thetadot]
    qddot  : (N, 3)  accelerations
    target_dof : which DOF we are predicting (0, 1, or 2)
    degree : polynomial degree
    include_trig : add sin(theta), cos(theta) and their products
    include_sign_qdot : add sign(xdot), sign(thetadot) (for slipping regime)
    include_bias : include constant "1" column

    Returns
    -------
    Theta : (N, n_features) feature matrix
    names : list of feature name strings
    """
    N = q.shape[0]
    theta = q[:, 2]
    sin_th = np.sin(theta).reshape(-1, 1)
    cos_th = np.cos(theta).reshape(-1, 1)
    thdot2 = (u[:, 2] ** 2).reshape(-1, 1)

    # Other two accelerations (cross-coupling features)
    other_idx = [j for j in range(3) if j != target_dof]
    other_accel = qddot[:, other_idx]
    other_names = [["xddot", "yddot", "thetaddot"][j] for j in other_idx]

    # Base features for polynomial expansion
    base_cols = [sin_th, cos_th, thdot2, other_accel]
    base_names = ["sin(theta)", "cos(theta)", "thetadot^2"] + other_names
    base = np.hstack(base_cols)

    # Polynomial features on base
    pf = PolynomialFeatures(degree=degree, include_bias=include_bias,
                            interaction_only=False)
    Theta = pf.fit_transform(base)
    names = pf.get_feature_names_out(base_names).tolist()

    # Clean up names: "1" for bias, remove spaces around ^
    names = [re.sub(r"\s*\^\s*", "^", n) for n in names]
    # Fix sklearn naming: "sin(theta) cos(theta)" etc
    # sklearn uses space-separated products: keep as-is since unit_filter tokenizes on spaces

    if include_sign_qdot:
        sign_xdot = np.sign(u[:, 0]).reshape(-1, 1)
        sign_thdot = np.sign(u[:, 2]).reshape(-1, 1)
        Theta = np.hstack([Theta, sign_xdot, sign_thdot])
        names += ["sign(xdot)", "sign(thetadot)"]

    return Theta, names


def build_full_state_features(q: np.ndarray, u: np.ndarray,
                              qddot: np.ndarray, target_dof: int,
                              degree: int = 2,
                              include_trig: bool = True,
                              include_sign_qdot: bool = False,
                              ) -> Tuple[np.ndarray, List[str]]:
    """Build feature matrix using ALL state variables (x, y, theta, xdot, ydot, thetadot)
    plus other accelerations — the 'full' unfiltered library.

    This is the baseline library that does NOT enforce unit consistency.
    """
    N = q.shape[0]
    other_idx = [j for j in range(3) if j != target_dof]
    other_accel = qddot[:, other_idx]
    other_names = [["xddot", "yddot", "thetaddot"][j] for j in other_idx]

    # All 8 inputs: q(3) + u(3) + other_accel(2)
    X_all = np.hstack([q, u, other_accel])
    feat_names = ["x", "y", "theta", "xdot", "ydot", "thetadot"] + other_names

    # Polynomial features
    pf = PolynomialFeatures(degree=degree, include_bias=True,
                            interaction_only=False)
    Theta = pf.fit_transform(X_all)
    names = pf.get_feature_names_out(feat_names).tolist()

    # Add trig terms and their products with polynomials
    if include_trig:
        theta_col = q[:, 2]
        sin_th = np.sin(theta_col)
        cos_th = np.cos(theta_col)
        # Add sin(theta), cos(theta) standalone
        Theta = np.hstack([Theta, sin_th.reshape(-1, 1), cos_th.reshape(-1, 1)])
        names += ["sin(theta)", "cos(theta)"]
        # Add sin * poly and cos * poly (degree-1 products)
        pf1 = PolynomialFeatures(degree=max(1, degree - 1), include_bias=False)
        poly1 = pf1.fit_transform(X_all)
        names1 = pf1.get_feature_names_out(feat_names).tolist()
        sin_prods = poly1 * sin_th.reshape(-1, 1)
        cos_prods = poly1 * cos_th.reshape(-1, 1)
        Theta = np.hstack([Theta, sin_prods, cos_prods])
        names += [f"sin(theta) {n}" for n in names1]
        names += [f"cos(theta) {n}" for n in names1]

    if include_sign_qdot:
        sign_xdot = np.sign(u[:, 0]).reshape(-1, 1)
        sign_thdot = np.sign(u[:, 2]).reshape(-1, 1)
        Theta = np.hstack([Theta, sign_xdot, sign_thdot])
        names += ["sign(xdot)", "sign(thetadot)"]

    return Theta, names


def build_unit_consistent_features(q: np.ndarray, u: np.ndarray,
                                   qddot: np.ndarray, target_dof: int,
                                   degree: int = 2,
                                   include_sign_qdot: bool = False,
                                   ) -> Tuple[np.ndarray, List[str]]:
    """Build the pre-filtered unit-consistent library.

    Generates polynomial features from {sin(θ), cos(θ), θ̇², accel_j, accel_k},
    then applies the dimensional filter to remove any product terms whose
    combined units don't match the target acceleration DOF.
    """
    from pipeline.unit_filter import filter_library

    Theta, names = build_features(
        q, u, qddot, target_dof,
        degree=degree,
        include_trig=True,
        include_sign_qdot=include_sign_qdot,
        include_bias=True,
    )

    # Filter out polynomial products with inconsistent units
    keep_idx, keep_names, reject_names = filter_library(names, target_dof)
    Theta = Theta[:, keep_idx]

    return Theta, keep_names
