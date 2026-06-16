import numpy as np
from .config import UAVParams

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rot_matrix(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [cps*cth, cps*sphi*sth-cphi*sps, sphi*sps+cphi*cps*sth],
        [cth*sps, cphi*cps+sphi*sps*sth, cphi*sps*sth-cps*sphi],
        [-sth,    cth*sphi,                 cphi*cth]
    ], dtype=float)

class QuadFullModel:
   
    def __init__(self, params: UAVParams, model_mismatch=False, disturbance=False, seed=0):
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.model_mismatch = model_mismatch
        self.disturbance = disturbance
        self.m_real = params.m * (1.10 if model_mismatch else 1.0)
        self.I_real = np.array([params.Ixx, params.Iyy, params.Izz]) * (np.array([1.12, 0.90, 1.08]) if model_mismatch else 1.0)
        self.drag = 0.10 if model_mismatch else 0.02

    def _clip_state(self, x):
        x = np.asarray(x, dtype=float).copy()
        x[6] = np.clip(wrap_angle(x[6]), -self.p.phi_max, self.p.phi_max)
        x[7] = np.clip(wrap_angle(x[7]), -self.p.theta_max, self.p.theta_max)
        x[8] = wrap_angle(x[8])
        x[9:12] = np.clip(x[9:12], -self.p.pqr_max, self.p.pqr_max)
        return x

    def step(self, state, action):
        return self._step_with_params(state, action, self.m_real, self.I_real, real=True)

    def nominal_step(self, state, action):
        return self._step_with_params(state, action, self.p.m, np.array([self.p.Ixx, self.p.Iyy, self.p.Izz]), real=False)

    def _step_with_params(self, state, action, mass, inertia, real=False):
        p = self.p; dt = p.dt
        x = np.asarray(state, dtype=float)
        u = np.asarray(action, dtype=float)
        T = np.clip(u[0], p.thrust_min, p.thrust_max)
        tau = np.array([
            np.clip(u[1], -p.tau_phi_max, p.tau_phi_max),
            np.clip(u[2], -p.tau_theta_max, p.tau_theta_max),
            np.clip(u[3], -p.tau_psi_max, p.tau_psi_max),
        ])
        pos = x[0:3]; vel = x[3:6]
        phi, theta, psi = x[6:9]
        omega = x[9:12]
        R = rot_matrix(phi, theta, psi)
        acc = (R @ np.array([0.0, 0.0, T])) / mass - np.array([0.0, 0.0, p.g])
        acc -= self.drag * vel
        if real and self.disturbance:
            wind = np.array([0.04*np.sin(0.7*pos[1]), 0.05*np.sin(0.5*pos[0]), 0.015*np.sin(0.3*(pos[0]+pos[1]))])
            acc += wind + self.rng.normal(0.0, 0.01, size=3)
        # simplified Euler angle kinematics, acceptable for small/moderate angles
        euler_dot = omega
        omega_dot = tau / inertia
        pos_next = pos + vel * dt
        vel_next = vel + acc * dt
        euler_next = np.array([phi, theta, psi]) + euler_dot * dt
        omega_next = omega + omega_dot * dt
        xn = np.concatenate([pos_next, vel_next, euler_next, omega_next])
        return self._clip_state(xn)
