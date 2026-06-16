import numpy as np
from scipy.optimize import minimize
from .config import UAVParams, MPCConfig
from .uav import QuadFullModel, wrap_angle

class UAVMPC:
    def __init__(self, params: UAVParams, cfg: MPCConfig):
        self.p = params
        self.cfg = cfg
        # MPC prediction uses the nominal model.
        # The real plant can still contain model mismatch and disturbance in envs.py.
        self.nominal = QuadFullModel(params, model_mismatch=False, disturbance=False)
        self.last_u = None

    def set_weights_from_action(self, action):
        
        a = np.asarray(action, dtype=float).reshape(-1)
        a = np.clip(a, -1.0, 1.0)

        # Tracking weights.
        # Compared with the previous narrow mapping, this range gives PPO enough
        # freedom to trade tracking accuracy and obstacle avoidance in challenging scenes.
        self.cfg.q_pos = 7.0 + 10.0 * (a[0] + 1.0) / 2.0       # 7 ~ 17
        self.cfg.q_vel = 0.8 + 2.4 * (a[1] + 1.0) / 2.0        # 0.8 ~ 3.2
        self.cfg.q_att = 0.18 + 0.52 * (a[2] + 1.0) / 2.0      # 0.18 ~ 0.70

        # Control penalty.
        # Keep a stable lower bound, but allow PPO to reduce/increase effort
        # penalties when a stronger avoidance maneuver is needed.
        self.cfg.r_thrust = 0.12 + 0.28 * (a[3] + 1.0) / 2.0   # 0.12 ~ 0.40
        self.cfg.r_torque = 0.12 + 0.28 * (a[4] + 1.0) / 2.0   # 0.12 ~ 0.40

        # Track/obstacle balance.
        # The key change is to enlarge the obstacle-weight range while keeping
        # the lower bound reasonable. This lets PPO become more conservative
        # near obstacles without forcing every trajectory to be over-conservative.
        self.cfg.beta_track = 0.8 + 1.8 * (a[5] + 1.0) / 2.0     # 0.8 ~ 2.6
        self.cfg.beta_static = 6.0 + 26.0 * (a[6] + 1.0) / 2.0   # 6 ~ 32
        self.cfg.beta_dynamic = 6.0 + 30.0 * (a[7] + 1.0) / 2.0  # 6 ~ 36


    def predict_next(self, x, u):
        
        xn = self.nominal.nominal_step(x, u)
        xn[6:9] = [wrap_angle(v) for v in xn[6:9]]
        return xn

    def rollout(self, x0, z):
        N = self.cfg.horizon
        X = [np.asarray(x0, dtype=float)]
        x = X[0].copy()
        for k in range(N):
            u = z[4*k:4*k+4]
            x = self.predict_next(x, u)
            X.append(x)
        return np.asarray(X)

    def _static_penalty(self, pos, obstacles):
        penalty = 0.0
        for ob in obstacles:
            horiz_clear = np.hypot(pos[0]-ob.x, pos[1]-ob.y) - (ob.radius + self.p.radius + self.cfg.safe_margin)
            # Height gate: only penalize horizontal distance when UAV is below/near the cylinder top.
            height_gate = 1.0 / (1.0 + np.exp(5.0 * (pos[2] - (ob.height + 0.20))))
            penalty += height_gate / (horiz_clear + 0.35)**2 if horiz_clear > -0.25 else 80.0 + 50.0*(-horiz_clear)
        return penalty

    def _dynamic_penalty(self, pos, obstacles, t, k):
        penalty = 0.0
        for ob in obstacles:
            clear = np.linalg.norm(pos - ob.position(t+k, self.p.dt)) - (ob.radius + self.p.radius + self.cfg.safe_margin)
            penalty += 1.0 / (clear + 0.35)**2 if clear > -0.25 else 80.0 + 50.0*(-clear)
        return penalty

    def solve(self, x0, ref, static_obs=None, dynamic_obs=None, t=0):
        
        static_obs = static_obs or []
        dynamic_obs = dynamic_obs or []
        N = self.cfg.horizon
        p = self.p
        hover = p.m * p.g

        fallback_u = np.array([hover, 0.0, 0.0, 0.0], dtype=float)

        if self.last_u is None or not np.all(np.isfinite(self.last_u)):
            z0 = np.tile(fallback_u, N)
        else:
            z0 = np.tile(self.last_u, N)

        bounds = []
        for _ in range(N):
            bounds.extend([
                (p.thrust_min, p.thrust_max),
                (-p.tau_phi_max, p.tau_phi_max),
                (-p.tau_theta_max, p.tau_theta_max),
                (-p.tau_psi_max, p.tau_psi_max),
            ])

        def cost(z):
            z = np.asarray(z, dtype=float)
            if not np.all(np.isfinite(z)):
                return 1e12

            X = self.rollout(x0, z)
            if not np.all(np.isfinite(X)):
                return 1e12

            J = 0.0
            prev_u = self.last_u if self.last_u is not None else fallback_u

            for k in range(N):
                e = X[k] - ref[k]
                e[6:9] = [wrap_angle(v) for v in e[6:9]]

                terminal = self.cfg.terminal_scale if k == N - 1 else 1.0
                pos_cost = np.sum(e[0:3] ** 2)
                vel_cost = np.sum(e[3:6] ** 2)
                att_cost = np.sum(e[6:12] ** 2)

                u = z[4 * k:4 * k + 4]
                du = u - prev_u

                # Normalize torque effort so r_torque has a meaningful scale.
                torque_norm = np.array([
                    p.tau_phi_max,
                    p.tau_theta_max,
                    p.tau_psi_max,
                ], dtype=float)
                u_cost = (
                    self.cfg.r_thrust * ((u[0] - hover) / hover) ** 2
                    + self.cfg.r_torque * np.sum((u[1:] / torque_norm) ** 2)
                )

                smooth = self.cfg.rd_u * np.sum(
                    (du / np.array([hover, p.tau_phi_max, p.tau_theta_max, p.tau_psi_max])) ** 2
                )

                obs_cost = self.cfg.beta_static * self._static_penalty(X[k, 0:3], static_obs)
                obs_cost += self.cfg.beta_dynamic * self._dynamic_penalty(X[k, 0:3], dynamic_obs, t, k)

                J += terminal * self.cfg.beta_track * (
                    self.cfg.q_pos * pos_cost
                    + self.cfg.q_vel * vel_cost
                    + self.cfg.q_att * att_cost
                )
                J += u_cost + smooth + obs_cost
                prev_u = u

            if not np.isfinite(J):
                return 1e12
            return float(J)

        init_cost = cost(z0)

        try:
            res = minimize(
                cost,
                z0,
                method='SLSQP',
                bounds=bounds,
                options={
                    'maxiter': self.cfg.maxiter,
                    'ftol': 1e-3,
                    'disp': False,
                },
            )
        except Exception as exc:
            self.last_status = {
                'success': False,
                'acceptable': False,
                'message': f'exception: {exc}',
                'nit': 0,
                'init_cost': float(init_cost) if np.isfinite(init_cost) else 1e12,
                'final_cost': 1e12,
            }
            pred = self.rollout(x0, np.tile(fallback_u, N))
            return fallback_u.copy(), pred, 1e12, False

        final_cost = float(res.fun) if np.isfinite(res.fun) else 1e12
        has_finite_solution = hasattr(res, 'x') and np.all(np.isfinite(res.x))

       
        cost_improved = (
            np.isfinite(init_cost)
            and np.isfinite(final_cost)
            and final_cost < 0.95 * init_cost
        )

        hit_maxiter_but_improved = (
            has_finite_solution
            and np.isfinite(final_cost)
            and final_cost < 1e11
            and int(getattr(res, 'nit', 0)) >= max(1, self.cfg.maxiter - 1)
            and cost_improved
        )

        ok = bool(
            has_finite_solution
            and np.isfinite(final_cost)
            and final_cost < 1e11
            and (res.success or cost_improved or hit_maxiter_but_improved)
        )

        if ok:
            z = np.asarray(res.x, dtype=float)
            u0 = z[:4].copy()
            self.last_u = u0.copy()
            pred = self.rollout(x0, z)
        else:
            # Do not update warm-start with a failed solution.
            u0 = fallback_u.copy() if self.last_u is None else np.asarray(self.last_u, dtype=float).copy()
            z = np.tile(u0, N)
            pred = self.rollout(x0, z)

        self.last_status = {
            'success': bool(res.success),
            'acceptable': bool(ok),
            'relaxed_accept': bool(ok and not res.success),
            'message': str(getattr(res, 'message', '')),
            'nit': int(getattr(res, 'nit', 0)),
            'init_cost': float(init_cost) if np.isfinite(init_cost) else 1e12,
            'final_cost': float(final_cost),
        }

        return u0, pred, final_cost, ok
