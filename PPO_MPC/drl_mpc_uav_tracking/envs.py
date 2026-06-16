import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import UAVParams, MPCConfig, EnvConfig
from .uav import QuadFullModel
from .mpc import UAVMPC
from .reference import UAVReference
from .obstacles import static_obstacles_default, dynamic_obstacles_default, _valid_scene


class UAVMPCTuningEnv(gym.Env):

    metadata = {'render_modes': []}

    def __init__(self, max_steps=220, reference_kind='curve3d',
                 dynamic_obstacles=True, model_mismatch=True, disturbance=True,
                 eval_mode=False, seed=0, action_dim=8,
                 randomize_obstacles=True):
        super().__init__()
        self.p = UAVParams()
        self.mpc_cfg = MPCConfig()
        self.env_cfg = EnvConfig(max_steps=max_steps, reference_kind=reference_kind,
                                 dynamic_obstacles=dynamic_obstacles,
                                 model_mismatch=model_mismatch, disturbance=disturbance)

        self.eval_mode = eval_mode
        self.randomize_obstacles = bool(randomize_obstacles)
        self.rng = np.random.default_rng(seed)

        self.reference = UAVReference(
            total_steps=max_steps + 1,
            target=self.env_cfg.target,
            kind=reference_kind
        )

        self.static_obstacles = static_obstacles_default(
            rng=self.rng,
            randomize=(self.randomize_obstacles and not self.eval_mode)
        )
        self.dynamic_obstacles = (
            dynamic_obstacles_default(
                rng=self.rng,
                randomize=(self.randomize_obstacles and not self.eval_mode)
            ) if dynamic_obstacles else []
        )

        self.plant = QuadFullModel(
            self.p,
            model_mismatch=model_mismatch,
            disturbance=disturbance,
            seed=seed
        )

        self.mpc = UAVMPC(self.p, self.mpc_cfg)

        self.action_dim = action_dim
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(32,),
            dtype=np.float32
        )

        self.reset(seed=seed)

    # ============================================================
    # Scenario randomization
    # ============================================================
    def _reset_obstacles(self):
        
        do_rand = self.randomize_obstacles and (not self.eval_mode)

        for _ in range(200):
            static_obs = static_obstacles_default(
                rng=self.rng,
                randomize=do_rand
            )

            dynamic_obs = (
                dynamic_obstacles_default(
                    rng=self.rng,
                    randomize=do_rand
                )
                if self.env_cfg.dynamic_obstacles else []
            )

            if _valid_scene(
                static_obs,
                dynamic_obs,
                total_steps=self.env_cfg.max_steps,
                dt=self.p.dt,
                min_gap=0.05,
            ):
                self.static_obstacles = static_obs
                self.dynamic_obstacles = dynamic_obs
                return

        raise RuntimeError(
            "Failed to sample a collision-free obstacle scene after 200 attempts. "
            "Please reduce obstacle radius/randomization range or increase the gap."
        )

    # ============================================================
    # Basic metrics
    # ============================================================
    def _pos(self):
        return self.state[0:3]

    def _goal_dist(self):
        return float(np.linalg.norm(self._pos() - self.reference.target))

    def _min_clearance(self):
        """Minimum clearance to static/dynamic obstacles.

        Static obstacles are cylinders with height. If UAV is above the obstacle
        top with a small safety margin, this obstacle is treated as fly-over safe.
        """
        pos = self._pos()
        vals = []

        for ob in self.static_obstacles:
            horiz = np.hypot(pos[0] - ob.x, pos[1] - ob.y) - (
                ob.radius + self.p.radius + self.mpc_cfg.safe_margin
            )

            # Allow flying over low obstacles.
            # 0.20 is retained from the original file as top-height tolerance.
            if pos[2] <= ob.height + 0.20:
                vals.append(horiz)

        for ob in self.dynamic_obstacles:
            vals.append(
                np.linalg.norm(pos - ob.position(self.t, self.p.dt))
                - (ob.radius + self.p.radius + self.mpc_cfg.safe_margin)
            )

        return float(min(vals)) if vals else 9.99

    def _collision(self):
        return self._min_clearance() < 0.0

    def _success_condition(self, goal_dist, progress, min_clearance):
        if self.env_cfg.dynamic_obstacles:
            return (
                goal_dist < 0.35
                and progress > 0.94
                and self.min_clearance_episode > 0.18
            )

        return (
            goal_dist < 0.35
            and progress > 0.94
            and self.min_clearance_episode > 0.18
        )

    # ============================================================
    # Observation
    # ============================================================
    def _obs(self):
        ref0 = self.reference.horizon(self.t, self.mpc_cfg.horizon)[0]

        scale_state = np.array(
            [5, 5, 3, 3, 3, 2, 1, 1, 3.14, 4, 4, 4],
            dtype=float
        )

        err = (self.state - ref0) / scale_state
        state = self.state / scale_state

        hover = self.p.m * self.p.g
        last_u = self.last_u / np.array([
            hover,
            self.p.tau_phi_max,
            self.p.tau_theta_max,
            self.p.tau_psi_max
        ])

        extra = np.array([
            self.reference.progress(self.state),
            self._goal_dist() / 8.0,
            np.clip(self._min_clearance(), -1.0, 2.0) / 2.0,
            1.0 if self.env_cfg.dynamic_obstacles else 0.0,
        ], dtype=float)

        return np.concatenate([err, state, last_u, extra]).astype(np.float32)

    # ============================================================
    # Gym API
    # ============================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self._reset_obstacles()

        if self.eval_mode or not self.env_cfg.random_init:
            self.state = np.zeros(12, dtype=np.float64)
        else:
            self.state = np.zeros(12, dtype=np.float64)
            self.state[0:3] += self.rng.normal(0.0, [0.03, 0.03, 0.02])
            self.state[3:6] += self.rng.normal(0.0, 0.02, size=3)

        self.last_u = np.array([self.p.m * self.p.g, 0.0, 0.0, 0.0], dtype=np.float64)
        self.mpc.last_u = None

        # PPO outputs MPC weights. If the action changes sharply from one step
        # to the next, the MPC objective changes sharply and the trajectory can
        # become wavy. Keep a smoothed action state to stabilize weight updates.
        self.last_action = np.zeros(self.action_dim, dtype=np.float64)
        self.last_action_delta = 0.0

        self.prev_goal_dist = self._goal_dist()
        self.prev_progress = self.reference.progress(self.state)
        self.min_clearance_episode = 9.99

        return self._obs(), {}

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float64).reshape(-1)
        raw_action = np.clip(raw_action, -1.0, 1.0)

        if raw_action.shape[0] != self.action_dim:
            raise ValueError(f'Expected action_dim={self.action_dim}, got {raw_action.shape[0]}')

        old_action = self.last_action.copy()

        action_smooth_alpha = 0.60
        smooth_action = action_smooth_alpha * old_action + (1.0 - action_smooth_alpha) * raw_action
        action_delta = float(np.linalg.norm(smooth_action - old_action))

        self.last_action = smooth_action.copy()
        self.last_action_delta = action_delta

        self.mpc.set_weights_from_action(smooth_action)

        # 2. MPC solve
        ref = self.reference.horizon(self.t, self.mpc_cfg.horizon)
        u, pred, mpc_cost, ok = self.mpc.solve(
            self.state,
            ref,
            self.static_obstacles,
            self.dynamic_obstacles,
            self.t
        )

        if not ok:
            status = getattr(self.mpc, 'last_status', {})
            reward = -120.0
            self.t += 1

            goal_dist = self._goal_dist()
            progress = self.reference.progress(self.state)
            min_clearance = self._min_clearance()
            self.min_clearance_episode = min(self.min_clearance_episode, min_clearance)

            reward -= 4.0 * goal_dist
            if min_clearance < 0.18:
                reward -= 45.0 * (0.18 - min_clearance)

            info = {
                'mpc_action': u,
                'mpc_ok': False,
                'mpc_failed': True,
                'mpc_cost': float(mpc_cost),
                'mpc_init_cost': float(status.get('init_cost', np.nan)),
                'mpc_final_cost': float(status.get('final_cost', np.nan)),
                'mpc_nit': int(status.get('nit', 0)),
                'mpc_message': str(status.get('message', '')),
                'success': False,
                'termination_reason': 'mpc_failed',
                'goal_dist': float(goal_dist),
                'path_progress': float(progress),
                'min_clearance': float(self.min_clearance_episode),
            }
            return self._obs(), float(reward), True, False, info

        # 3. Plant step
        self.state = self.plant.step(self.state, u)
        self.last_u = u.copy()
        self.t += 1

        # 4. Metrics
        goal_dist = self._goal_dist()
        progress = self.reference.progress(self.state)
        min_clearance = self._min_clearance()
        self.min_clearance_episode = min(self.min_clearance_episode, min_clearance)

        goal_progress = self.prev_goal_dist - goal_dist
        path_progress_delta = progress - self.prev_progress

        self.prev_goal_dist = goal_dist
        self.prev_progress = progress

        ref_now = self.reference.horizon(self.t, self.mpc_cfg.horizon)[0]

        pos_err = np.linalg.norm(self.state[0:3] - ref_now[0:3])
        vel_err = np.linalg.norm(self.state[3:6] - ref_now[3:6])
        att_cost = np.sum(self.state[6:12] ** 2)

        hover = self.p.m * self.p.g
        control_cost = ((u[0] - hover) / hover) ** 2 + np.sum(
            (u[1:] / np.array([
                self.p.tau_phi_max,
                self.p.tau_theta_max,
                self.p.tau_psi_max
            ])) ** 2
        )

        # ========================================================
        # Reward
        # ========================================================
        # Dense part: progress + tracking + effort
        reward = 12.0 * goal_progress + 5.0 * path_progress_delta
        reward += 0.8 * np.exp(-1.5 * pos_err)
        reward -= 0.16 * pos_err ** 2
        reward -= 0.04 * vel_err ** 2
        reward -= 0.01 * att_cost
        reward -= 0.015 * control_cost

        # Penalize rapid MPC-weight changes. This directly addresses the
        # oscillatory PPO-MPC trajectory issue.
        reward -= 0.18 * self.last_action_delta ** 2

        # Terminal precision shaping:
        # Dense signal for smaller goal_dist before terminal success.
        final_phase = np.clip((progress - 0.70) / 0.25, 0.0, 1.0)
        reward += final_phase * (10.0 * np.exp(-4.0 * goal_dist) - 5.0 * goal_dist)

        # Extra pressure when the path is almost finished.
        if progress > 0.90:
            reward -= 8.0 * max(goal_dist - 0.25, 0.0)

        # Safety shaping:
        # Encourage a larger safety margin, not merely "no collision".
        # Use both current clearance and episode-level minimum clearance so PPO
        # cannot get a high return by briefly passing too close to obstacles.
        reward += 2.0 * np.clip(min_clearance, 0.0, 0.45)
        reward += 1.0 * np.clip(self.min_clearance_episode, 0.0, 0.45)

        if min_clearance < 0.45:
            reward -= 22.0 * (0.45 - min_clearance) ** 2

        if min_clearance < 0.24:
            reward -= 36.0 * (0.24 - min_clearance)

        if self.min_clearance_episode < 0.16:
            reward -= 50.0 * (0.16 - self.min_clearance_episode)

        if min_clearance < 0.06:
            reward -= 90.0 * (0.06 - min_clearance)

        # MPC failure is handled immediately after solve.

        # ========================================================
        # Termination
        # ========================================================
        terminated = False
        reason = 'running'
        success = self._success_condition(goal_dist, progress, min_clearance)

        if self._collision():
            reward -= 220.0
            terminated = True
            reason = 'collision'

        elif (
            self.state[2] < -0.10
            or self.state[2] > 5.0
            or abs(self.state[0]) > 7.0
            or abs(self.state[1]) > 7.0
        ):
            reward -= 220.0
            terminated = True
            reason = 'out_of_bounds'

        elif self.t >= self.env_cfg.max_steps:
            # Final bonuses/penalties only at episode end; keeps reward curve interpretable.
            if success:
                # Terminal precision bonus:
                # Smaller final goal_dist gets a clearly larger reward, so PPO
                # has motivation to continue improving after merely entering
                # the old success radius.
                reward += 150.0 + 120.0 * np.exp(-5.0 * goal_dist)

                # Episode-level safety bonus:
                # Encourage the entire trajectory to keep a larger clearance.
                reward += 80.0 * np.clip(self.min_clearance_episode, 0.0, 0.45)

                reason = 'success'
            else:
                # Timeout should distinguish "almost successful but safe"
                # from "far away or unsafe".
                # Stronger terminal-distance penalty.  This reduces the
                # learned tendency to stop around goal_dist = 0.56~0.59.
                reward -= 18.0 * goal_dist

                if goal_dist > 0.35:
                    reward -= 30.0 * (goal_dist - 0.35)

                if self.min_clearance_episode < 0.16:
                    reward -= 60.0 * (0.16 - self.min_clearance_episode)

                if progress < 0.85:
                    reward -= 20.0 * (0.85 - progress)

                reason = 'timeout'

            terminated = True

        status = getattr(self.mpc, 'last_status', {})
        info = {
            'mpc_action': u,
            'mpc_ok': bool(ok),
            'mpc_failed': False,
            'mpc_cost': float(mpc_cost),
            'mpc_init_cost': float(status.get('init_cost', np.nan)),
            'mpc_final_cost': float(status.get('final_cost', np.nan)),
            'mpc_nit': int(status.get('nit', 0)),
            'mpc_message': str(status.get('message', '')),
            'success': bool(success),
            'termination_reason': reason,
            'goal_dist': float(goal_dist),
            'path_progress': float(progress),
            'min_clearance': float(self.min_clearance_episode),
            'action_delta': float(self.last_action_delta),
        }

        return self._obs(), float(reward), terminated, False, info
