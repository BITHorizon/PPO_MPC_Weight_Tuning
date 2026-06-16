from dataclasses import dataclass
import numpy as np

@dataclass
class UAVParams:
    dt: float = 0.08
    m: float = 0.8
    g: float = 9.81
    Ixx: float = 0.021
    Iyy: float = 0.022
    Izz: float = 0.040
    radius: float = 0.12
    thrust_min: float = 0.25 * 0.8 * 9.81
    thrust_max: float = 1.85 * 0.8 * 9.81
    tau_phi_max: float = 0.18
    tau_theta_max: float = 0.18
    tau_psi_max: float = 0.10
    phi_max: float = np.deg2rad(35)
    theta_max: float = np.deg2rad(35)
    psi_max: float = np.pi
    pqr_max: float = 4.0

@dataclass
class MPCConfig:
    horizon: int = 8
    maxiter: int = 80
    # Clearance is computed as distance - (obstacle radius + UAV radius + safe_margin).
    # Keep this fixed for fair comparison; do not increase it just to enlarge metrics.
    safe_margin: float = 0.18
    terminal_scale: float = 4.0
    # default fixed weights used before RL mapping or for baseline
    q_pos: float = 8.0
    q_vel: float = 1.5
    q_att: float = 0.2
    r_thrust: float = 0.08
    r_torque: float = 0.08
    rd_u: float = 0.03
    beta_track: float = 1.0
    beta_static: float = 4.0
    beta_dynamic: float = 4.0

@dataclass
class EnvConfig:
    max_steps: int = 180
    target: tuple = (5.0, 5.0, 3.0)
    reference_kind: str = 'smooth_s'
    dynamic_obstacles: bool = False
    model_mismatch: bool = True
    disturbance: bool = True
    random_init: bool = True
    success_dist: float = 0.45
