"""
unit_filter.py – Dimensional-consistency filter for candidate library terms.

Physical units for the hula-hoop system
---------------------------------------
  q = [x, y, theta]          →  [L, L, 1]            (theta is dimensionless angle)
  qdot = [xdot, ydot, thdot] →  [L/T, L/T, 1/T]
  qddot                       →  [L/T², L/T², 1/T²]

We express each term's dimension as (L_power, T_power).
A term is compatible with acceleration DOF i if its dimension equals:
  xddot  → (1, -2)
  yddot  → (1, -2)
  thddot → (0, -2)

Trig functions sin(θ), cos(θ) are dimensionless → (0, 0).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import re


@dataclass(frozen=True)
class Dim:
    """Dimensional exponents (L, T)."""
    L: float = 0.0
    T: float = 0.0

    def __add__(self, other):
        return Dim(self.L + other.L, self.T + other.T)

    def __mul__(self, other):
        """Multiply dimensions (for products of terms)."""
        return Dim(self.L + other.L, self.T + other.T)

    def __eq__(self, other):
        if other is None or not isinstance(other, Dim):
            return NotImplemented
        return abs(self.L - other.L) < 1e-12 and abs(self.T - other.T) < 1e-12

    def __hash__(self):
        return hash((round(self.L, 10), round(self.T, 10)))

    def __repr__(self):
        return f"L^{self.L} T^{self.T}"


# Target dimensions for each acceleration DOF
ACCEL_DIM = {
    0: Dim(1, -2),   # xddot  [L/T²]
    1: Dim(1, -2),   # yddot  [L/T²]
    2: Dim(0, -2),   # thddot [1/T²]
}

# Allowed feature dimensions per DOF.
# Coefficients can carry dimensions from physical constants:
#   εR → [L],  ε/R or 1/R → [1/L],  g → [L/T²]
# So a feature is admissible if its dimension, combined with one of these
# coefficient dimensions, yields the target acceleration dimension.
#
# For a_x, a_y [L/T²]:
#   (1,-2) with coeff [1]     → L/T²  ✓  (e.g. yddot)
#   (0,-2) with coeff [L]=εR  → L/T²  ✓  (e.g. sin(θ)·θ̈, cos(θ)·θ̇²)
#   (0, 0) with coeff [L/T²]=g → L/T²  ✓  (e.g. constant "1" for gravity)
#
# For a_θ [1/T²]:
#   (0,-2) with coeff [1]     → 1/T²  ✓  (e.g. θ̇², thetaddot)
#   (1,-2) with coeff [1/L]   → 1/T²  ✓  (e.g. sin(θ)·ẍ, cos(θ)·ÿ)
#   (0, 0) with coeff [1/T²]=gε/R → 1/T²  ✓  (e.g. cos(θ) for gravity coupling)
ALLOWED_FEATURE_DIMS = {
    0: [Dim(1, -2), Dim(0, -2), Dim(0, 0)],
    1: [Dim(1, -2), Dim(0, -2), Dim(0, 0)],
    2: [Dim(0, -2), Dim(1, -2), Dim(0, 0)],
}

# Base variable dimensions
BASE_DIMS: Dict[str, Dim] = {
    "x":        Dim(1, 0),
    "y":        Dim(1, 0),
    "theta":    Dim(0, 0),    # angle is dimensionless
    "xdot":     Dim(1, -1),
    "ydot":     Dim(1, -1),
    "thetadot": Dim(0, -1),
    # Trig terms
    "sin(theta)": Dim(0, 0),
    "cos(theta)": Dim(0, 0),
    # Accelerations (when used as features for cross-coupling)
    "xddot":      Dim(1, -2),
    "yddot":       Dim(1, -2),
    "thetaddot":  Dim(0, -2),
    # Velocity squared terms
    "thetadot^2": Dim(0, -2),
    # Constant
    "1":          Dim(0, 0),
    # Contact forces (force/mass = acceleration, [L/T²])
    "lambda_N":   Dim(1, -2),
    "lambda_F":   Dim(1, -2),
}


def get_dim(var_name: str) -> Optional[Dim]:
    """Return dimension for a variable name, or None if unknown."""
    if var_name in BASE_DIMS:
        return BASE_DIMS[var_name]
    return None


def compute_term_dim(term_name: str, var_dims: Dict[str, Dim]) -> Optional[Dim]:
    """Parse a PySINDy-style term name and compute its dimension.

    Handles patterns like:
      "sin(theta) cos(theta)", "thetadot^2 yddot", "1", "sin(theta)"
      "x y", "xdot^2", etc.

    Products are space-separated. Powers are "var^n".
    """
    term_name = term_name.strip()
    if term_name == "1":
        return Dim(0, 0)

    dim = Dim(0, 0)

    # Tokenize: split by spaces, but keep "sin(theta)" and "cos(theta)" as single tokens
    tokens = _tokenize(term_name)

    for tok in tokens:
        # Handle power: "var^n"
        if "^" in tok:
            base, exp_str = tok.rsplit("^", 1)
            try:
                exp = int(exp_str)
            except ValueError:
                # Not a valid power; treat whole token as variable name
                d = var_dims.get(tok)
                if d is None:
                    return None
                dim = dim + d
                continue
            base_dim = var_dims.get(base)
            if base_dim is None:
                return None
            dim = Dim(dim.L + base_dim.L * exp, dim.T + base_dim.T * exp)
        else:
            d = var_dims.get(tok)
            if d is None:
                return None
            dim = dim + d

    return dim


def _tokenize(term: str) -> List[str]:
    """Split a term string into factor tokens. Keeps sin(...) and cos(...) intact."""
    tokens = []
    i = 0
    while i < len(term):
        if term[i] == " ":
            i += 1
            continue
        # Check for sin( or cos(
        for func in ("sin(", "cos("):
            if term[i:].startswith(func):
                j = term.index(")", i) + 1
                tokens.append(term[i:j])
                i = j
                break
        else:
            # Regular token: read until space
            j = i
            while j < len(term) and term[j] != " ":
                j += 1
            tokens.append(term[i:j])
            i = j
    return tokens


def filter_library(feature_names: List[str], target_dof: int,
                   var_dims: Optional[Dict[str, Dim]] = None,
                   verbose: bool = False) -> Tuple[List[int], List[str], List[str]]:
    """Filter a feature library for unit consistency with a target acceleration DOF.

    Parameters
    ----------
    feature_names : list of PySINDy-generated feature names
    target_dof    : 0=xddot, 1=yddot, 2=thetaddot
    var_dims      : custom dimension map (default: BASE_DIMS)

    Returns
    -------
    keep_idx      : column indices that pass the filter
    keep_names    : names of kept features
    reject_names  : names of rejected features
    """
    dims = var_dims or BASE_DIMS
    allowed = ALLOWED_FEATURE_DIMS[target_dof]
    keep_idx, keep_names, reject_names = [], [], []

    for i, name in enumerate(feature_names):
        d = compute_term_dim(name, dims)
        if d is None:
            reject_names.append(name)
            if verbose:
                print(f"  REJECT (unknown dim): {name}")
            continue

        # Allow if dimension matches any physically plausible target,
        # or if it's the constant "1" (coefficient absorbs all dimensions).
        if d in allowed or name == "1":
            keep_idx.append(i)
            keep_names.append(name)
        else:
            reject_names.append(name)
            if verbose:
                print(f"  REJECT: {name} has dim {d}, allowed {allowed}")

    if verbose:
        print(f"  Unit filter: {len(keep_idx)}/{len(feature_names)} terms kept "
              f"for DOF {target_dof}")

    return keep_idx, keep_names, reject_names


def unit_consistency_score(feature_names: List[str], coefs: np.ndarray,
                           target_dof: int,
                           var_dims: Optional[Dict[str, Dim]] = None) -> float:
    """Fraction of active (nonzero-coefficient) terms that are unit-consistent."""
    dims = var_dims or BASE_DIMS
    allowed = ALLOWED_FEATURE_DIMS[target_dof]
    active = np.nonzero(coefs)[0]
    if len(active) == 0:
        return 1.0
    consistent = 0
    for idx in active:
        name = feature_names[idx]
        d = compute_term_dim(name, dims)
        if (d is not None and d in allowed) or name == "1":
            consistent += 1
    return consistent / len(active)
