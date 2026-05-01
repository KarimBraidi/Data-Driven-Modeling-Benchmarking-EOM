"""
data_loader.py – Load simulation data for each regime.

Each run folder contains:
  q_save.npy   (3, N)  generalized coordinates  [x, y, theta]
  u_save.npy   (3, N)  generalized velocities   [xdot, ydot, thetadot]
  X_save.npy  (14, N)  full solution; rows 0-2 are accelerations [xddot, yddot, thetaddot]
  contacts.mat (5, N)  contact indicator rows A-E
  lambdaN_save.npy (1, N)  normal contact force
  lambdaF_save.npy (1, N)  friction force
  params.json          simulation parameters

Regime catalogue
----------------
  bouncing_ball_1  Bouncing ball data / hula_hoop_2026-03-14_13-07-44
  bouncing_ball_2  Bouncing ball data / hula_hoop_2026-03-14_13-31-19
  bouncing_ball_3  Bouncing ball data / hula_hoop_2026-03-14_13-33-04
  rolling_1        Rolling without slipping / hula_hoop_2026-03-14_13-37-06
  rolling_2        Rolling without slipping / hula_hoop_2026-03-14_13-43-00
  flight_1         Flight / hula_hoop_2026-03-14_13-51-04
  flight_2         Flight / hula_hoop_2026-03-14_13-53-02
"""

import json
import pathlib
import numpy as np
import scipy.io as sio
from dataclasses import dataclass, field
from typing import Dict, Optional

# ── catalogue ────────────────────────────────────────────────────────────────
DATA_ROOT = pathlib.Path(
    r"C:\Users\braid\OneDrive\Desktop\Data Driven Modeling Project"
    r"\mass imbalance data\mass imbalance data"
)

REGIME_FOLDERS: Dict[str, str] = {
    "bouncing_ball_1": "Bouncing ball data/hula_hoop_2026-03-14_13-07-44",
    "bouncing_ball_2": "Bouncing ball data/hula_hoop_2026-03-14_13-31-19",
    "bouncing_ball_3": "Bouncing ball data/hula_hoop_2026-03-14_13-33-04",
    "rolling_1":       "Rolling without slipping/hula_hoop_2026-03-14_13-37-06",
    "rolling_2":       "Rolling without slipping/hula_hoop_2026-03-14_13-43-00",
    "flight_1":        "Flight/hula_hoop_2026-03-14_13-51-04",
    "flight_2":        "Flight/hula_hoop_2026-03-14_13-53-02",
}


@dataclass
class RunData:
    """Container for one simulation run."""
    name: str
    regime: str                       # "bouncing", "rolling", "flight"
    q: np.ndarray                     # (N, 3)  positions
    u: np.ndarray                     # (N, 3)  velocities
    qddot: np.ndarray                # (N, 3)  accelerations (from simulator)
    contacts: np.ndarray              # (5, N)  contact flags A-E
    lambda_N: np.ndarray              # (N,)    normal contact force
    lambda_F: np.ndarray              # (N,)    friction force
    dt: float
    N: int
    t: np.ndarray                     # (N,) time vector
    params: dict = field(default_factory=dict)


def _regime_label(name: str) -> str:
    if name.startswith("bouncing"):
        return "bouncing"
    if name.startswith("rolling"):
        return "rolling"
    if name.startswith("flight"):
        return "flight"
    return "unknown"


def load_run(name: str, data_root: Optional[pathlib.Path] = None) -> RunData:
    """Load a single run by catalogue name."""
    root = data_root or DATA_ROOT
    folder = root / REGIME_FOLDERS[name]

    q = np.load(folder / "q_save.npy").T        # (N, 3)
    u = np.load(folder / "u_save.npy").T        # (N, 3)
    X = np.load(folder / "X_save.npy")          # (14, N)
    qddot = X[:3].T                             # (N, 3)

    cmat = sio.loadmat(str(folder / "contacts.mat"))
    ckey = [k for k in cmat if not k.startswith("_")][0]
    contacts = cmat[ckey]                        # (5, N)

    lN = np.load(folder / "lambdaN_save.npy").flatten()
    lF = np.load(folder / "lambdaF_save.npy").flatten()

    with open(folder / "params.json") as f:
        params = json.load(f)

    dt = params["dtime"]
    N = q.shape[0]
    t = np.arange(N) * dt

    return RunData(
        name=name,
        regime=_regime_label(name),
        q=q, u=u, qddot=qddot,
        contacts=contacts,
        lambda_N=lN, lambda_F=lF,
        dt=dt, N=N, t=t,
        params=params,
    )


def load_regime(regime: str, data_root: Optional[pathlib.Path] = None):
    """Load all runs for a given regime ('bouncing', 'rolling', 'flight')."""
    return [
        load_run(name, data_root)
        for name in REGIME_FOLDERS
        if _regime_label(name) == regime
    ]


def load_all(data_root: Optional[pathlib.Path] = None):
    """Load every run, grouped by regime."""
    all_runs = {}
    for name in REGIME_FOLDERS:
        regime = _regime_label(name)
        all_runs.setdefault(regime, []).append(load_run(name, data_root))
    return all_runs
