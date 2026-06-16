#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation study for PPO-MPC.

Methods:
- Baseline MPC: fixed hand-tuned MPC weights.
- Zero-action MPC: action = zeros, i.e. PPO mapping midpoint.
- Random-action MPC: random PPO-like action, to show random weight tuning is not enough.
- PPO-MPC: trained PPO policy.

Run example:
MODEL_PATH=/path/to/ppo_model.zip \
DYNAMIC_OBSTACLES=1 REFERENCE_KIND=curve3d SEEDS=0,1,2,3,4 \
python scripts/ablation_ppo_mpc.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from drl_mpc_uav_tracking.envs import UAVMPCTuningEnv


MODEL_PATH = os.environ.get("MODEL_PATH", "").strip()
MAX_STEPS = int(os.environ.get("MAX_STEPS", "220"))
SEEDS = [int(s.strip()) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",") if s.strip()]
REFERENCE_KIND = os.environ.get("REFERENCE_KIND", "curve3d")
DYNAMIC_OBSTACLES = os.environ.get("DYNAMIC_OBSTACLES", "1") == "1"
MODEL_MISMATCH = os.environ.get("MODEL_MISMATCH", "1") == "1"
DISTURBANCE = os.environ.get("DISTURBANCE", "1") == "1"
ACTION_DIM = int(os.environ.get("ACTION_DIM", "8"))
METHODS = [s.strip().lower() for s in os.environ.get("METHODS", "baseline,zero,random,ppo").split(",") if s.strip()]
RANDOM_ACTION_MODE = os.environ.get("RANDOM_ACTION_MODE", "constant").lower()  # constant or step
RANDOM_ACTION_SEED_OFFSET = int(os.environ.get("RANDOM_ACTION_SEED_OFFSET", "10000"))
MAKE_PLOTS = os.environ.get("MAKE_PLOTS", "1") == "1"
PLOT_SEED = int(os.environ.get("PLOT_SEED", str(SEEDS[0])))

BASELINE_Q_POS = float(os.environ.get("BASELINE_Q_POS", "11.5"))
BASELINE_Q_VEL = float(os.environ.get("BASELINE_Q_VEL", "1.9"))
BASELINE_Q_ATT = float(os.environ.get("BASELINE_Q_ATT", "0.425"))
BASELINE_R_THRUST = float(os.environ.get("BASELINE_R_THRUST", "0.26"))
BASELINE_R_TORQUE = float(os.environ.get("BASELINE_R_TORQUE", "0.26"))
BASELINE_BETA_TRACK = float(os.environ.get("BASELINE_BETA_TRACK", "1.55"))
BASELINE_BETA_STATIC = float(os.environ.get("BASELINE_BETA_STATIC", "15.0"))
BASELINE_BETA_DYNAMIC = float(os.environ.get("BASELINE_BETA_DYNAMIC", "15.0"))

METHOD_DISPLAY = {
    "baseline": "Baseline MPC",
    "zero": "Zero-action MPC",
    "random": "Random-action MPC",
    "ppo": "PPO-MPC",
}


def make_env(seed: int, eval_mode: bool = True) -> UAVMPCTuningEnv:
    return UAVMPCTuningEnv(
        max_steps=MAX_STEPS,
        reference_kind=REFERENCE_KIND,
        dynamic_obstacles=DYNAMIC_OBSTACLES,
        model_mismatch=MODEL_MISMATCH,
        disturbance=DISTURBANCE,
        eval_mode=eval_mode,
        seed=seed,
        action_dim=ACTION_DIM,
    )


def set_baseline_fixed_weights(env: UAVMPCTuningEnv) -> None:
    env.mpc_cfg.q_pos = BASELINE_Q_POS
    env.mpc_cfg.q_vel = BASELINE_Q_VEL
    env.mpc_cfg.q_att = BASELINE_Q_ATT
    env.mpc_cfg.r_thrust = BASELINE_R_THRUST
    env.mpc_cfg.r_torque = BASELINE_R_TORQUE
    env.mpc_cfg.beta_track = BASELINE_BETA_TRACK
    env.mpc_cfg.beta_static = BASELINE_BETA_STATIC
    env.mpc_cfg.beta_dynamic = BASELINE_BETA_DYNAMIC


def is_out_of_bounds(env: UAVMPCTuningEnv) -> bool:
    return bool(env.state[2] < -0.10 or env.state[2] > 5.0 or abs(env.state[0]) > 7.0 or abs(env.state[1]) > 7.0)


def get_mpc_nit(env: UAVMPCTuningEnv) -> int:
    status = getattr(env.mpc, "last_status", {}) or {}
    return int(status.get("nit", 0))


def get_obstacle_radius(ob, fallback=0.25) -> float:
    return float(getattr(ob, "radius", fallback))


def get_reference_curve(env: UAVMPCTuningEnv, length: int | None = None) -> np.ndarray:
    if length is None:
        length = MAX_STEPS + 1
    return np.asarray([env.reference.horizon(t, env.mpc_cfg.horizon)[0][:3] for t in range(length)], dtype=float)


def compute_tracking_rmse(traj: np.ndarray, env_for_ref: UAVMPCTuningEnv) -> float:
    if traj is None or len(traj) == 0:
        return np.nan
    ref = get_reference_curve(env_for_ref, len(traj))
    err = np.asarray(traj[:, :3], dtype=float) - ref[:len(traj), :3]
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def compute_axis_rmse(traj: np.ndarray, env_for_ref: UAVMPCTuningEnv):
    if traj is None or len(traj) == 0:
        return np.nan, np.nan, np.nan
    ref = get_reference_curve(env_for_ref, len(traj))
    err = np.asarray(traj[:, :3], dtype=float) - ref[:len(traj), :3]
    rmse_xyz = np.sqrt(np.mean(err ** 2, axis=0))
    return tuple(float(v) for v in rmse_xyz)


def compute_total_path_length(traj: np.ndarray) -> float:
    if traj is None or len(traj) < 2:
        return 0.0
    diff = np.diff(np.asarray(traj[:, :3], dtype=float), axis=0)
    return float(np.sum(np.linalg.norm(diff, axis=1)))


def get_dynamic_obstacle_positions(env: UAVMPCTuningEnv, steps=None):
    if steps is None:
        steps = np.arange(0, MAX_STEPS + 1)
    infos = []
    for i, ob in enumerate(env.dynamic_obstacles):
        pts = []
        for t in steps:
            p = np.asarray(ob.position(int(t), env.p.dt), dtype=float).reshape(-1)
            if p.shape[0] == 2:
                p = np.array([p[0], p[1], 0.0], dtype=float)
            pts.append(p[:3])
        infos.append({"name": str(getattr(ob, "name", f"dyn_{i}")), "radius": get_obstacle_radius(ob), "positions": np.asarray(pts), "steps": np.asarray(steps)})
    return infos


def compute_dynamic_clearance_series(traj: np.ndarray, env_for_plot: UAVMPCTuningEnv):
    if len(env_for_plot.dynamic_obstacles) == 0:
        return None
    clearances = []
    for k, p_uav in enumerate(traj):
        vals = []
        for ob in env_for_plot.dynamic_obstacles:
            p_obs = np.asarray(ob.position(k, env_for_plot.p.dt), dtype=float).reshape(-1)
            if p_obs.shape[0] == 2:
                p_obs = np.array([p_obs[0], p_obs[1], 0.0], dtype=float)
            dist = np.linalg.norm(np.asarray(p_uav[:3], dtype=float) - p_obs[:3])
            vals.append(dist - (get_obstacle_radius(ob) + env_for_plot.p.radius + env_for_plot.mpc_cfg.safe_margin))
        clearances.append(min(vals))
    return np.asarray(clearances, dtype=float)


def run_fixed_mpc(seed: int, method: str, rng=None):
    env = make_env(seed, eval_mode=True)
    env.reset(seed=seed)
    env.min_clearance_episode = 9.99

    if method == "baseline":
        set_baseline_fixed_weights(env)
    elif method == "zero":
        env.mpc.set_weights_from_action(np.zeros(ACTION_DIM, dtype=np.float64))
    elif method == "random":
        if rng is None:
            rng = np.random.default_rng(seed + RANDOM_ACTION_SEED_OFFSET)
        if RANDOM_ACTION_MODE == "constant":
            env.mpc.set_weights_from_action(rng.uniform(-1.0, 1.0, size=ACTION_DIM))
    else:
        raise ValueError(f"Unknown fixed method: {method}")

    traj, rows = [], []
    for step in range(MAX_STEPS):
        if method == "random" and RANDOM_ACTION_MODE == "step":
            env.mpc.set_weights_from_action(rng.uniform(-1.0, 1.0, size=ACTION_DIM))

        ref = env.reference.horizon(env.t, env.mpc_cfg.horizon)
        u, pred, cost, ok = env.mpc.solve(env.state, ref, env.static_obstacles, env.dynamic_obstacles, env.t)

        if ok:
            env.state = env.plant.step(env.state, u)
            env.last_u = u.copy()
            env.t += 1

        pos = env.state[:3].copy()
        traj.append(pos)
        goal_dist = env._goal_dist()
        path_progress = env.reference.progress(env.state)
        current_clearance = env._min_clearance()
        env.min_clearance_episode = min(env.min_clearance_episode, current_clearance)
        success = False if not ok else env._success_condition(goal_dist, path_progress, env.min_clearance_episode)

        reason, terminated = "running", False
        if not ok:
            reason, terminated = "mpc_failed", True
        elif env._collision():
            reason, terminated = "collision", True
        elif is_out_of_bounds(env):
            reason, terminated = "out_of_bounds", True
        elif env.t >= MAX_STEPS:
            reason, terminated = ("success" if success else "timeout"), True

        status = getattr(env.mpc, "last_status", {}) or {}
        rows.append({
            "step": step + 1, "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "success": bool(success), "termination_reason": reason,
            "goal_dist": float(goal_dist), "path_progress": float(path_progress),
            "min_clearance": float(env.min_clearance_episode), "current_clearance": float(current_clearance),
            "mpc_ok": bool(ok), "mpc_failed": bool(not ok), "mpc_cost": float(cost),
            "mpc_init_cost": float(status.get("init_cost", np.nan)),
            "mpc_final_cost": float(status.get("final_cost", np.nan)),
            "mpc_nit": get_mpc_nit(env),
        })
        if terminated:
            break
    return np.asarray(traj, dtype=float), pd.DataFrame(rows), env


def run_ppo_mpc(seed: int, model: PPO):
    env = make_env(seed, eval_mode=True)
    obs, _ = env.reset(seed=seed)
    traj, rows = [], []
    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        pos = env.state[:3].copy()
        traj.append(pos)
        rows.append({
            "step": step + 1, "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "reward": float(reward), "success": bool(info.get("success", False)),
            "termination_reason": info.get("termination_reason", "running"),
            "goal_dist": float(info.get("goal_dist", np.nan)),
            "path_progress": float(info.get("path_progress", np.nan)),
            "min_clearance": float(info.get("min_clearance", np.nan)),
            "current_clearance": float(info.get("current_clearance", np.nan)),
            "action_delta": float(info.get("action_delta", np.nan)),
            "mpc_ok": bool(info.get("mpc_ok", False)), "mpc_failed": bool(info.get("mpc_failed", False)),
            "mpc_cost": float(info.get("mpc_cost", np.nan)), "mpc_nit": int(info.get("mpc_nit", 0)),
        })
        if terminated or truncated:
            break
    return np.asarray(traj, dtype=float), pd.DataFrame(rows), env


def summarize_one(method_name, seed, log, traj, env_for_ref, dyn_clearance=None):
    final = log.iloc[-1]
    reason = str(final["termination_reason"])
    rmse_3d = compute_tracking_rmse(traj, env_for_ref)
    rmse_x, rmse_y, rmse_z = compute_axis_rmse(traj, env_for_ref)
    row = {
        "method": method_name, "seed": seed, "steps": int(len(log)),
        "success": bool(final["success"]), "termination_reason": reason,
        "collision": reason == "collision", "out_of_bounds": reason == "out_of_bounds",
        "timeout": reason == "timeout", "terminal_mpc_failed": reason == "mpc_failed",
        "goal_dist": float(final["goal_dist"]), "path_progress": float(final["path_progress"]),
        "rmse_3d": rmse_3d, "rmse_x": rmse_x, "rmse_y": rmse_y, "rmse_z": rmse_z,
        "path_length": compute_total_path_length(traj),
        "min_clearance": float(final["min_clearance"]),
        "mpc_failed_rate": float(np.mean(log["mpc_failed"].astype(float))),
        "mpc_ok_rate": float(np.mean(log["mpc_ok"].astype(float))),
        "mean_mpc_cost": float(np.nanmean(log["mpc_cost"])),
        "mean_mpc_nit": float(np.nanmean(log["mpc_nit"])),
    }
    if dyn_clearance is not None and len(dyn_clearance) > 0:
        row["min_dynamic_clearance"] = float(np.nanmin(dyn_clearance))
        row["mean_dynamic_clearance"] = float(np.nanmean(dyn_clearance))
    else:
        row["min_dynamic_clearance"] = np.nan
        row["mean_dynamic_clearance"] = np.nan
    return row


def build_summary(results_df):
    summary = results_df.groupby("method").agg(
        episodes=("success", "count"), success_rate=("success", "mean"),
        mean_rmse_3d=("rmse_3d", "mean"), std_rmse_3d=("rmse_3d", "std"),
        mean_goal_dist=("goal_dist", "mean"), std_goal_dist=("goal_dist", "std"),
        mean_min_clearance=("min_clearance", "mean"), std_min_clearance=("min_clearance", "std"),
        mean_min_dynamic_clearance=("min_dynamic_clearance", "mean"), std_min_dynamic_clearance=("min_dynamic_clearance", "std"),
        mean_dynamic_clearance=("mean_dynamic_clearance", "mean"),
        collision_rate=("collision", "mean"), out_of_bounds_rate=("out_of_bounds", "mean"), timeout_rate=("timeout", "mean"),
        mpc_failed_rate=("mpc_failed_rate", "mean"), mean_mpc_cost=("mean_mpc_cost", "mean"),
        mean_mpc_nit=("mean_mpc_nit", "mean"), mean_path_length=("path_length", "mean"), std_path_length=("path_length", "std"),
    ).reset_index()
    order = {METHOD_DISPLAY.get(m, m): i for i, m in enumerate(METHODS)}
    summary["order"] = summary["method"].map(order).fillna(999)
    summary = summary.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    for col in ["success_rate", "collision_rate", "out_of_bounds_rate", "timeout_rate", "mpc_failed_rate"]:
        summary[col + "_percent"] = 100.0 * summary[col]
    return summary


def mean_std_format(mean, std, digits=3):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_paper_table(summary_df):
    rows = []
    for _, r in summary_df.iterrows():
        rows.append({
            "Method": r["method"],
            "Success Rate (%) ↑": f'{r["success_rate_percent"]:.1f}',
            "RMSE (m) ↓": mean_std_format(r["mean_rmse_3d"], r["std_rmse_3d"]),
            "Goal Dist. (m) ↓": mean_std_format(r["mean_goal_dist"], r["std_goal_dist"]),
            "Min Clearance (m) ↑": mean_std_format(r["mean_min_clearance"], r["std_min_clearance"]),
            "Min Dynamic Clearance (m) ↑": mean_std_format(r["mean_min_dynamic_clearance"], r["std_min_dynamic_clearance"]),
            "Collision Rate (%) ↓": f'{r["collision_rate_percent"]:.1f}',
            "MPC Failed Rate (%) ↓": f'{r["mpc_failed_rate_percent"]:.1f}',
            "Mean MPC Iter. ↓": f'{r["mean_mpc_nit"]:.2f}',
            "Path Length (m) ↓": mean_std_format(r["mean_path_length"], r["std_path_length"]),
        })
    return pd.DataFrame(rows)


def plot_xy_all(seed, traj_dict, env_for_plot, out_dir):
    fig, ax = plt.subplots(figsize=(11, 9))
    ref_curve = get_reference_curve(env_for_plot, MAX_STEPS + 1)
    ax.plot(ref_curve[:, 0], ref_curve[:, 1], linestyle="--", linewidth=2.0, label="Reference")
    for method_name, traj in traj_dict.items():
        if len(traj) > 0:
            ax.plot(traj[:, 0], traj[:, 1], linewidth=2.2, label=method_name)
    ax.scatter([0], [0], marker="o", s=80, label="Start")
    target = env_for_plot.reference.target
    ax.scatter([target[0]], [target[1]], marker="*", s=180, label="Target")
    for ob in env_for_plot.static_obstacles:
        circle = plt.Circle((ob.x, ob.y), ob.radius, fill=False, linestyle="-.", linewidth=1.8)
        ax.add_patch(circle)
    if DYNAMIC_OBSTACLES:
        for info in get_dynamic_obstacle_positions(env_for_plot, np.arange(0, MAX_STEPS + 1)):
            pos = info["positions"]
            ax.plot(pos[:, 0], pos[:, 1], linestyle=":", linewidth=2.0, label=f"{info['name']} path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Ablation XY Trajectory Comparison, seed={seed}")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"ablation_xy_seed{seed}.png", dpi=260)
    plt.close(fig)


def plot_dynamic_clearance_all(seed, dyn_clearance_dict, out_dir):
    if not dyn_clearance_dict:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for method_name, c in dyn_clearance_dict.items():
        if c is not None and len(c) > 0:
            ax.plot(np.arange(len(c)), c, linewidth=2.2, label=method_name)
    ax.axhline(0.0, linestyle="--", linewidth=1.8, label="safety boundary")
    ax.set_xlabel("step")
    ax.set_ylabel("dynamic obstacle clearance")
    ax.set_title(f"Ablation Dynamic Obstacle Clearance, seed={seed}")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"ablation_dynamic_clearance_seed{seed}.png", dpi=240)
    plt.close(fig)


def main():
    if "ppo" in METHODS and not MODEL_PATH:
        raise RuntimeError("METHODS includes ppo, so please specify MODEL_PATH=/path/to/model.zip")
    model = None
    if "ppo" in METHODS:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = PPO.load(str(model_path), device="cpu")

    tag = "dynamic" if DYNAMIC_OBSTACLES else "static"
    out_dir = PROJECT_ROOT / "scripts" / "runs" / datetime.now().strftime(f"{tag}_ablation_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PPO-MPC ablation study")
    print("Output dir       :", out_dir)
    print("Model path       :", MODEL_PATH if MODEL_PATH else "(not used)")
    print("Methods          :", METHODS)
    print("Seeds            :", SEEDS)
    print("Reference kind   :", REFERENCE_KIND)
    print("Dynamic obstacles:", DYNAMIC_OBSTACLES)
    print("Action dim       :", ACTION_DIM)
    print("Random action    :", RANDOM_ACTION_MODE)
    print("=" * 90)

    all_rows = []
    representative_traj, representative_dyn, representative_env = {}, {}, None
    for seed in SEEDS:
        print(f"\n===== seed={seed} =====")
        rng_random = np.random.default_rng(seed + RANDOM_ACTION_SEED_OFFSET)
        for method in METHODS:
            if method not in METHOD_DISPLAY:
                raise ValueError(f"Unknown method: {method}. Available: {list(METHOD_DISPLAY)}")
            display_name = METHOD_DISPLAY[method]
            if method == "ppo":
                traj, log, env = run_ppo_mpc(seed, model)
            else:
                traj, log, env = run_fixed_mpc(seed, method=method, rng=rng_random)
            dyn_clearance = compute_dynamic_clearance_series(traj, env) if DYNAMIC_OBSTACLES else None

            safe_method = method.replace("-", "_")
            log.to_csv(out_dir / f"{safe_method}_step_log_seed{seed}.csv", index=False)
            pd.DataFrame(traj, columns=["x", "y", "z"]).to_csv(out_dir / f"{safe_method}_traj_seed{seed}.csv", index=False)
            if dyn_clearance is not None:
                pd.DataFrame({"step": np.arange(len(dyn_clearance)), "dynamic_clearance": dyn_clearance}).to_csv(
                    out_dir / f"{safe_method}_dynamic_clearance_seed{seed}.csv", index=False
                )

            all_rows.append(summarize_one(display_name, seed, log, traj, env, dyn_clearance))
            final = log.iloc[-1]
            min_dyn = float(np.nanmin(dyn_clearance)) if dyn_clearance is not None else np.nan
            print(f"{display_name}: success={bool(final['success'])}, reason={final['termination_reason']}, "
                  f"goal={float(final['goal_dist']):.3f}, rmse={compute_tracking_rmse(traj, env):.3f}, "
                  f"min_clearance={float(final['min_clearance']):.3f}, min_dyn={min_dyn:.3f}, "
                  f"mpc_failed_rate={float(np.mean(log['mpc_failed'].astype(float))):.1%}")

            if MAKE_PLOTS and seed == PLOT_SEED:
                representative_traj[display_name] = traj
                representative_dyn[display_name] = dyn_clearance
                representative_env = env

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_dir / "ablation_results.csv", index=False)
    summary_df = build_summary(results_df)
    summary_df.to_csv(out_dir / "ablation_summary.csv", index=False)
    paper_table = build_paper_table(summary_df)
    paper_table.to_csv(out_dir / "ablation_paper_table.csv", index=False)

    if MAKE_PLOTS and representative_env is not None:
        plot_xy_all(PLOT_SEED, representative_traj, representative_env, out_dir)
        if DYNAMIC_OBSTACLES:
            plot_dynamic_clearance_all(PLOT_SEED, representative_dyn, out_dir)

    print("\n================ ablation_paper_table.csv ================")
    print(paper_table)
    print("\nSaved to:")
    print(out_dir)


if __name__ == "__main__":
    main()
