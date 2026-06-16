#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train PPO-MPC on randomized balanced curve3d scenarios.

Run:
REFERENCE_KIND=curve3d DYNAMIC_OBSTACLES=1 TOTAL_TIMESTEPS=400000 \
python scripts/train_ppo_randomized_curve3d.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from drl_mpc_uav_tracking.envs import UAVMPCTuningEnv


MAX_STEPS = int(os.environ.get("MAX_STEPS", "220"))
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "400000"))
SEED = int(os.environ.get("SEED", "0"))
REFERENCE_KIND = os.environ.get("REFERENCE_KIND", "curve3d")
DYNAMIC_OBSTACLES = os.environ.get("DYNAMIC_OBSTACLES", "1") == "1"
N_ENVS = int(os.environ.get("N_ENVS", "1"))


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, print_freq=2048, window=50, verbose=1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.window = window
        self.episodes = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.episodes.append({
                    "r": float(ep.get("r", 0.0)),
                    "l": int(ep.get("l", 0)),
                    "success": bool(info.get("success", False)),
                    "reason": info.get("termination_reason", "unknown"),
                    "goal_dist": float(info.get("goal_dist", np.nan)),
                    "progress": float(info.get("path_progress", np.nan)),
                    "clearance": float(info.get("min_clearance", np.nan)),
                    "action_delta": float(info.get("action_delta", np.nan)),
                    "mpc_failed": bool(info.get("mpc_failed", False)),
                    "mpc_nit": float(info.get("mpc_nit", np.nan)),
                })

        if self.n_calls % self.print_freq == 0 and self.episodes:
            recent = self.episodes[-self.window:]
            print(
                f"[recent {len(recent)} eps] "
                f"mean_r={np.mean([e['r'] for e in recent]):.2f}, "
                f"success={np.mean([e['success'] for e in recent]):.1%}, "
                f"timeout={np.mean([e['reason']=='timeout' for e in recent]):.1%}, "
                f"collision={np.mean([e['reason']=='collision' for e in recent]):.1%}, "
                f"out={np.mean([e['reason']=='out_of_bounds' for e in recent]):.1%}, "
                f"mpc_failed={np.mean([e['mpc_failed'] for e in recent]):.1%}, "
                f"goal_dist={np.nanmean([e['goal_dist'] for e in recent]):.3f}, "
                f"progress={np.nanmean([e['progress'] for e in recent]):.3f}, "
                f"clearance={np.nanmean([e['clearance'] for e in recent]):.3f}, "
                f"action_delta={np.nanmean([e['action_delta'] for e in recent]):.4f}, "
                f"nit={np.nanmean([e['mpc_nit'] for e in recent]):.1f}"
            )
        return True


def make_env(seed: int, log_dir: Path, rank: int = 0):
    def _init():
        env = UAVMPCTuningEnv(
            max_steps=MAX_STEPS,
            reference_kind=REFERENCE_KIND,
            dynamic_obstacles=DYNAMIC_OBSTACLES,
            model_mismatch=True,
            disturbance=True,
            eval_mode=False,
            seed=seed + rank,
            action_dim=8,
            randomize_obstacles=True,
        )
        return Monitor(
            env,
            filename=str(log_dir / f"monitor_seed{seed + rank}.csv"),
            info_keywords=(
                "success",
                "termination_reason",
                "goal_dist",
                "path_progress",
                "min_clearance",
                "action_delta",
                "mpc_ok",
                "mpc_failed",
                "mpc_cost",
                "mpc_init_cost",
                "mpc_final_cost",
                "mpc_nit",
            ),
        )
    return _init


def main():
    run_id = datetime.now().strftime("ppo_randomized_curve3d_%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "scripts" / "runs" / run_id
    log_dir = out_dir / "logs"
    model_dir = out_dir / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    import drl_mpc_uav_tracking.envs as envs_module
    print("loaded envs.py =", envs_module.__file__)

    env = DummyVecEnv([make_env(SEED, log_dir, rank=i) for i in range(N_ENVS)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1.2e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=8,
        gamma=0.985,
        gae_lambda=0.95,
        clip_range=0.10,
        ent_coef=0.004,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(out_dir / "tb"),
        seed=SEED,
        device="cpu",
    )

    callbacks = [
        EpisodeStatsCallback(print_freq=2048, window=50),
        CheckpointCallback(
            save_freq=10000,
            save_path=str(model_dir),
            name_prefix="ppo_randomized_curve3d_ckpt",
        ),
    ]

    print("=" * 80)
    print("Output dir:", out_dir)
    print("MAX_STEPS:", MAX_STEPS)
    print("TOTAL_TIMESTEPS:", TOTAL_TIMESTEPS)
    print("REFERENCE_KIND:", REFERENCE_KIND)
    print("DYNAMIC_OBSTACLES:", DYNAMIC_OBSTACLES)
    print("N_ENVS:", N_ENVS)
    print("SEED:", SEED)
    print("=" * 80)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    final_path = model_dir / "ppo_randomized_curve3d_final"
    model.save(str(final_path))
    print("Training finished. Model saved to:", str(final_path) + ".zip")


if __name__ == "__main__":
    main()
