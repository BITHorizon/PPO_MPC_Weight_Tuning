#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified comparison: PPO-MPC vs Baseline MPC.

Main outputs:
- compare_results.csv
- summary.csv
- paper_table.csv
- trajectory / clearance figures for each seed

Visualization note:
- Obstacles are plotted with their physical radii only.
- Clearance metrics are still computed with obstacle radius + UAV radius + safety margin.

Run examples:
cd ~/Projects/ppo_uav_rlmpc_project

# Dynamic obstacle evaluation
MODEL_PATH=/path/to/ppo_model.zip \
DYNAMIC_OBSTACLES=1 SEEDS=0,1,2,3,4,5,6,7,8,9 \
python scripts/compare_ppo_mpc_vs_baseline_unified.py

# Static obstacle evaluation
MODEL_PATH=/path/to/ppo_model.zip \
DYNAMIC_OBSTACLES=0 SEEDS=0,1,2,3,4,5,6,7,8,9 \
python scripts/compare_ppo_mpc_vs_baseline_unified.py
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

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# ============================================================
# Config from environment variables
# ============================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "").strip()
MAX_STEPS = int(os.environ.get("MAX_STEPS", "220"))
SEEDS = [
    int(s.strip())
    for s in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")
    if s.strip()
]

REFERENCE_KIND = os.environ.get("REFERENCE_KIND", "curve3d")
DYNAMIC_OBSTACLES = os.environ.get("DYNAMIC_OBSTACLES", "1") == "1"
MODEL_MISMATCH = os.environ.get("MODEL_MISMATCH", "1") == "1"
DISTURBANCE = os.environ.get("DISTURBANCE", "1") == "1"
ACTION_DIM = int(os.environ.get("ACTION_DIM", "8"))

CLEARANCE_THRESHOLD = float(os.environ.get("CLEARANCE_THRESHOLD", "0.0"))
SNAPSHOT_INTERVAL = int(os.environ.get("SNAPSHOT_INTERVAL", "90"))
SHOW_SNAPSHOT_TEXT = os.environ.get("SHOW_SNAPSHOT_TEXT", "0") == "1"

# Optional hand-tuned baseline weights.
# If not provided, use the midpoint of the PPO action mapping.
BASELINE_Q_POS = float(os.environ.get("BASELINE_Q_POS", "11.5"))
BASELINE_Q_VEL = float(os.environ.get("BASELINE_Q_VEL", "1.9"))
BASELINE_Q_ATT = float(os.environ.get("BASELINE_Q_ATT", "0.425"))
BASELINE_R_THRUST = float(os.environ.get("BASELINE_R_THRUST", "0.26"))
BASELINE_R_TORQUE = float(os.environ.get("BASELINE_R_TORQUE", "0.26"))
BASELINE_BETA_TRACK = float(os.environ.get("BASELINE_BETA_TRACK", "1.55"))
BASELINE_BETA_STATIC = float(os.environ.get("BASELINE_BETA_STATIC", "15.0"))
BASELINE_BETA_DYNAMIC = float(os.environ.get("BASELINE_BETA_DYNAMIC", "15.0"))


# ============================================================
# Environment and metrics
# ============================================================
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
    """Fixed MPC baseline.

    By default these are the midpoint values of the PPO action-to-weight mapping.
    You can override them using environment variables for a stronger hand-tuned baseline.
    """
    env.mpc_cfg.q_pos = BASELINE_Q_POS
    env.mpc_cfg.q_vel = BASELINE_Q_VEL
    env.mpc_cfg.q_att = BASELINE_Q_ATT
    env.mpc_cfg.r_thrust = BASELINE_R_THRUST
    env.mpc_cfg.r_torque = BASELINE_R_TORQUE
    env.mpc_cfg.beta_track = BASELINE_BETA_TRACK
    env.mpc_cfg.beta_static = BASELINE_BETA_STATIC
    env.mpc_cfg.beta_dynamic = BASELINE_BETA_DYNAMIC


def is_out_of_bounds(env: UAVMPCTuningEnv) -> bool:
    return bool(
        env.state[2] < -0.10
        or env.state[2] > 5.0
        or abs(env.state[0]) > 7.0
        or abs(env.state[1]) > 7.0
    )


def get_mpc_nit(env: UAVMPCTuningEnv) -> int:
    status = getattr(env.mpc, "last_status", {}) or {}
    return int(status.get("nit", 0))


def get_obstacle_radius(ob, fallback=0.25) -> float:
    return float(getattr(ob, "radius", fallback))


def get_reference_curve(env: UAVMPCTuningEnv, length: int | None = None) -> np.ndarray:
    if length is None:
        length = MAX_STEPS + 1
    return np.asarray([
        env.reference.horizon(t, env.mpc_cfg.horizon)[0][:3]
        for t in range(length)
    ], dtype=float)


def compute_tracking_rmse(traj: np.ndarray, env_for_ref: UAVMPCTuningEnv) -> float:
    """3D position RMSE relative to the time-synchronized reference."""
    if traj is None or len(traj) == 0:
        return np.nan
    ref = get_reference_curve(env_for_ref, len(traj))
    err = np.asarray(traj[:, :3], dtype=float) - ref[:len(traj), :3]
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def compute_axis_rmse(traj: np.ndarray, env_for_ref: UAVMPCTuningEnv) -> tuple[float, float, float]:
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
        infos.append({
            "name": str(getattr(ob, "name", f"dyn_{i}")),
            "radius": get_obstacle_radius(ob),
            "positions": np.asarray(pts, dtype=float),
            "steps": np.asarray(steps, dtype=int),
        })
    return infos


def compute_dynamic_clearance_series(traj: np.ndarray, env_for_plot: UAVMPCTuningEnv):
    """Time-synchronized dynamic-obstacle clearance.

    clearance = ||p_uav(k) - p_obs(k)|| - (r_obs + r_uav + safe_margin)
    """
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
            clearance = dist - (
                get_obstacle_radius(ob)
                + env_for_plot.p.radius
                + env_for_plot.mpc_cfg.safe_margin
            )
            vals.append(clearance)
        clearances.append(min(vals))
    return np.asarray(clearances, dtype=float)


# ============================================================
# Run methods
# ============================================================
def run_baseline_mpc(seed: int):
    env = make_env(seed, eval_mode=True)
    env.reset(seed=seed)
    set_baseline_fixed_weights(env)

    traj, rows = [], []
    env.min_clearance_episode = 9.99

    for step in range(MAX_STEPS):
        ref = env.reference.horizon(env.t, env.mpc_cfg.horizon)
        u, pred, cost, ok = env.mpc.solve(
            env.state,
            ref,
            env.static_obstacles,
            env.dynamic_obstacles,
            env.t,
        )

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

        success = False if not ok else env._success_condition(
            goal_dist,
            path_progress,
            env.min_clearance_episode,
        )

        reason, terminated = "running", False
        if not ok:
            reason, terminated = "mpc_failed", True
        elif env._collision():
            reason, terminated = "collision", True
        elif is_out_of_bounds(env):
            reason, terminated = "out_of_bounds", True
        elif env.t >= MAX_STEPS:
            reason, terminated = ("success" if success else "timeout"), True

        rows.append({
            "step": step + 1,
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "success": bool(success),
            "termination_reason": reason,
            "goal_dist": float(goal_dist),
            "path_progress": float(path_progress),
            "min_clearance": float(env.min_clearance_episode),
            "current_clearance": float(current_clearance),
            "mpc_ok": bool(ok),
            "mpc_failed": bool(not ok),
            "mpc_cost": float(cost),
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
            "step": step + 1,
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "reward": float(reward),
            "success": bool(info.get("success", False)),
            "termination_reason": info.get("termination_reason", "running"),
            "goal_dist": float(info.get("goal_dist", np.nan)),
            "path_progress": float(info.get("path_progress", np.nan)),
            "min_clearance": float(info.get("min_clearance", np.nan)),
            "current_clearance": float(info.get("current_clearance", np.nan)),
            "action_delta": float(info.get("action_delta", np.nan)),
            "mpc_ok": bool(info.get("mpc_ok", False)),
            "mpc_failed": bool(info.get("mpc_failed", False)),
            "mpc_cost": float(info.get("mpc_cost", np.nan)),
            "mpc_nit": int(info.get("mpc_nit", 0)),
        })

        if terminated or truncated:
            break

    return np.asarray(traj, dtype=float), pd.DataFrame(rows), env


# ============================================================
# Summaries
# ============================================================
def summarize_one(
    method: str,
    seed: int,
    log: pd.DataFrame,
    traj: np.ndarray,
    env_for_ref: UAVMPCTuningEnv,
    dyn_clearance=None,
) -> dict:
    final = log.iloc[-1]
    reason = str(final["termination_reason"])

    rmse_3d = compute_tracking_rmse(traj, env_for_ref)
    rmse_x, rmse_y, rmse_z = compute_axis_rmse(traj, env_for_ref)

    row = {
        "method": method,
        "seed": seed,
        "steps": int(len(log)),
        "success": bool(final["success"]),
        "termination_reason": reason,
        "collision": reason == "collision",
        "out_of_bounds": reason == "out_of_bounds",
        "timeout": reason == "timeout",
        "terminal_mpc_failed": reason == "mpc_failed",
        "goal_dist": float(final["goal_dist"]),
        "path_progress": float(final["path_progress"]),
        "rmse_3d": rmse_3d,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_z": rmse_z,
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


def mean_std_format(mean, std, digits=3):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_summary(compare_df: pd.DataFrame) -> pd.DataFrame:
    summary = compare_df.groupby("method").agg(
        episodes=("success", "count"),
        success_rate=("success", "mean"),
        mean_rmse_3d=("rmse_3d", "mean"),
        std_rmse_3d=("rmse_3d", "std"),
        mean_goal_dist=("goal_dist", "mean"),
        std_goal_dist=("goal_dist", "std"),
        mean_min_clearance=("min_clearance", "mean"),
        std_min_clearance=("min_clearance", "std"),
        mean_min_dynamic_clearance=("min_dynamic_clearance", "mean"),
        std_min_dynamic_clearance=("min_dynamic_clearance", "std"),
        mean_dynamic_clearance=("mean_dynamic_clearance", "mean"),
        collision_rate=("collision", "mean"),
        out_of_bounds_rate=("out_of_bounds", "mean"),
        timeout_rate=("timeout", "mean"),
        mpc_failed_rate=("mpc_failed_rate", "mean"),
        mean_mpc_cost=("mean_mpc_cost", "mean"),
        mean_mpc_nit=("mean_mpc_nit", "mean"),
        mean_path_length=("path_length", "mean"),
        std_path_length=("path_length", "std"),
    ).reset_index()

    for col in [
        "success_rate",
        "collision_rate",
        "out_of_bounds_rate",
        "timeout_rate",
        "mpc_failed_rate",
    ]:
        summary[col + "_percent"] = 100.0 * summary[col]

    return summary


def build_paper_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary_df.iterrows():
        rows.append({
            "Method": r["method"],
            "Success Rate (%) ↑": f'{r["success_rate_percent"]:.1f}',
            "RMSE (m) ↓": mean_std_format(r["mean_rmse_3d"], r["std_rmse_3d"]),
            "Goal Dist. (m) ↓": mean_std_format(r["mean_goal_dist"], r["std_goal_dist"]),
            "Min Clearance (m) ↑": mean_std_format(r["mean_min_clearance"], r["std_min_clearance"]),
            "Min Dynamic Clearance (m) ↑": mean_std_format(
                r["mean_min_dynamic_clearance"],
                r["std_min_dynamic_clearance"],
            ),
            "Collision Rate (%) ↓": f'{r["collision_rate_percent"]:.1f}',
            "MPC Failed Rate (%) ↓": f'{r["mpc_failed_rate_percent"]:.1f}',
            "Mean MPC Iter. ↓": f'{r["mean_mpc_nit"]:.2f}',
        })
    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================
def plot_static_cylinder_3d(ax, x0, y0, radius, height, z0=0.0, alpha=0.08):
    theta = np.linspace(0, 2 * np.pi, 60)
    z = np.linspace(z0, height, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = x0 + radius * np.cos(theta_grid)
    y_grid = y0 + radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, linewidth=0, shade=True)
    ax.plot(
        x0 + radius * np.cos(theta),
        y0 + radius * np.sin(theta),
        np.ones_like(theta) * height,
        linewidth=1.0,
        alpha=0.8,
    )


def plot_sphere_3d(ax, center, radius, alpha=0.06):
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0, alpha=alpha, shade=True)


def plot_dynamic_obstacles_xy(ax, dyn_infos, snapshot_interval=50):
    for info in dyn_infos:
        pos, steps, radius, name = (
            info["positions"],
            info["steps"],
            info["radius"],
            info["name"],
        )
        if len(pos) == 0:
            continue

        ax.plot(pos[:, 0], pos[:, 1], linestyle="--", linewidth=2.0, label=f"{name} path")

        if len(pos) >= 2:
            start, end = pos[0], pos[-1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            ax.annotate(
                "",
                xy=(start[0] + 0.28 * dx, start[1] + 0.28 * dy),
                xytext=(start[0] + 0.12 * dx, start[1] + 0.12 * dy),
                arrowprops=dict(arrowstyle="->", lw=1.8),
            )

        for k in range(0, len(pos), max(1, snapshot_interval)):
            circle = plt.Circle(
                (pos[k, 0], pos[k, 1]),
                radius,
                fill=False,
                linestyle=":",
                linewidth=1.3,
                alpha=0.75,
            )
            ax.add_patch(circle)
            if SHOW_SNAPSHOT_TEXT:
                ax.text(pos[k, 0], pos[k, 1], f"{steps[k]}", fontsize=8, ha="center", va="center")


def plot_dynamic_obstacles_3d(ax, dyn_infos, snapshot_interval=50):
    for info in dyn_infos:
        pos, steps, radius, name = (
            info["positions"],
            info["steps"],
            info["radius"],
            info["name"],
        )
        if len(pos) == 0:
            continue

        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], linestyle="--", linewidth=2.0, label=f"{name} path")

        for k in range(0, len(pos), max(1, snapshot_interval)):
            plot_sphere_3d(ax, pos[k], radius, alpha=0.12)
            if SHOW_SNAPSHOT_TEXT:
                ax.text(pos[k, 0], pos[k, 1], pos[k, 2] + radius + 0.05, f"{steps[k]}", fontsize=8)


def plot_3d_trajectory(seed: int, base_traj, ppo_traj, env_for_plot, out_dir: Path):
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    ref_curve = get_reference_curve(env_for_plot, MAX_STEPS + 1)
    ax.plot(ref_curve[:, 0], ref_curve[:, 1], ref_curve[:, 2], linestyle="--", linewidth=2.0, label="Reference")

    if len(base_traj) > 0:
        ax.plot(base_traj[:, 0], base_traj[:, 1], base_traj[:, 2], linewidth=2.4, label="Baseline MPC")
    if len(ppo_traj) > 0:
        ax.plot(ppo_traj[:, 0], ppo_traj[:, 1], ppo_traj[:, 2], linewidth=2.8, label="PPO-MPC")

    ax.scatter([0], [0], [0], marker="o", s=90, label="Start")
    target = env_for_plot.reference.target
    ax.scatter([target[0]], [target[1]], [target[2]], marker="*", s=200, label="Target")

    for ob in env_for_plot.static_obstacles:
        # Plot only the physical obstacle size for cleaner paper figures.
        # Clearance metrics still include UAV radius and safety margin.
        r_plot = ob.radius
        plot_static_cylinder_3d(ax, ob.x, ob.y, r_plot, ob.height)
        if SHOW_SNAPSHOT_TEXT:
            ax.text(ob.x, ob.y, ob.height + 0.12, ob.name)

    dyn_infos = get_dynamic_obstacle_positions(env_for_plot, np.arange(0, MAX_STEPS + 1))
    if DYNAMIC_OBSTACLES:
        plot_dynamic_obstacles_3d(ax, dyn_infos, SNAPSHOT_INTERVAL)

    all_pts = [ref_curve]
    if len(base_traj) > 0:
        all_pts.append(base_traj)
    if len(ppo_traj) > 0:
        all_pts.append(ppo_traj)
    for info in dyn_infos:
        all_pts.append(info["positions"])

    all_pts = np.vstack(all_pts)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    title_mode = "Dynamic" if DYNAMIC_OBSTACLES else "Static"
    ax.set_title(f"{title_mode} Obstacle 3D Trajectory Comparison, seed={seed}")

    ax.set_xlim(min(-0.5, np.min(all_pts[:, 0]) - 0.5), max(5.5, np.max(all_pts[:, 0]) + 0.5))
    ax.set_ylim(min(-0.5, np.min(all_pts[:, 1]) - 0.5), max(5.5, np.max(all_pts[:, 1]) + 0.5))
    ax.set_zlim(0.0, max(3.5, np.max(all_pts[:, 2]) + 0.5))
    ax.view_init(elev=28, azim=-55)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / f"trajectory_3d_seed{seed}.png", dpi=260)
    plt.close(fig)


def plot_xy_trajectory(seed: int, base_traj, ppo_traj, env_for_plot, out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 9))

    ref_curve = get_reference_curve(env_for_plot, MAX_STEPS + 1)
    ax.plot(ref_curve[:, 0], ref_curve[:, 1], linestyle="--", linewidth=2.0, label="Reference")

    if len(base_traj) > 0:
        ax.plot(base_traj[:, 0], base_traj[:, 1], linewidth=2.4, label="Baseline MPC")
    if len(ppo_traj) > 0:
        ax.plot(ppo_traj[:, 0], ppo_traj[:, 1], linewidth=2.8, label="PPO-MPC")

    ax.scatter([0], [0], marker="o", s=90, label="Start")
    target = env_for_plot.reference.target
    ax.scatter([target[0]], [target[1]], marker="*", s=200, label="Target")

    for ob in env_for_plot.static_obstacles:
        # Plot only the physical obstacle size for cleaner paper figures.
        # Clearance metrics still include UAV radius and safety margin.
        r_plot = ob.radius
        circle = plt.Circle((ob.x, ob.y), r_plot, fill=False, linestyle="-.", linewidth=2.0)
        ax.add_patch(circle)
        if SHOW_SNAPSHOT_TEXT:
            ax.text(ob.x, ob.y, ob.name, ha="center", va="center")

    dyn_infos = get_dynamic_obstacle_positions(env_for_plot, np.arange(0, MAX_STEPS + 1))
    if DYNAMIC_OBSTACLES:
        plot_dynamic_obstacles_xy(ax, dyn_infos, SNAPSHOT_INTERVAL)

    title_mode = "Dynamic" if DYNAMIC_OBSTACLES else "Static"
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title_mode} Obstacle XY Trajectory Comparison, seed={seed}")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"trajectory_xy_seed{seed}.png", dpi=260)
    plt.close(fig)


def plot_min_clearance(seed: int, base_log: pd.DataFrame, ppo_log: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(base_log["step"], base_log["min_clearance"], linewidth=2.4, label="Baseline MPC")
    ax.plot(ppo_log["step"], ppo_log["min_clearance"], linewidth=2.6, label="PPO-MPC")
    ax.axhline(CLEARANCE_THRESHOLD, linestyle="--", linewidth=2.0, label=f"safety boundary {CLEARANCE_THRESHOLD:.2f}")
    ax.set_xlabel("step")
    ax.set_ylabel("minimum clearance")
    ax.set_title(f"Overall Minimum Clearance, seed={seed}")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"minimum_clearance_seed{seed}.png", dpi=240)
    plt.close(fig)


def plot_dynamic_obstacle_clearance(seed: int, base_traj, ppo_traj, env_for_plot, out_dir: Path):
    base_clearance = compute_dynamic_clearance_series(base_traj, env_for_plot)
    ppo_clearance = compute_dynamic_clearance_series(ppo_traj, env_for_plot)

    if base_clearance is None or ppo_clearance is None:
        return None, None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(len(base_clearance)), base_clearance, linewidth=2.4, label="Baseline MPC")
    ax.plot(np.arange(len(ppo_clearance)), ppo_clearance, linewidth=2.6, label="PPO-MPC")
    ax.axhline(0.0, linestyle="--", linewidth=2.0, label="safety boundary")
    ax.set_xlabel("step")
    ax.set_ylabel("dynamic obstacle clearance")
    ax.set_title(f"Dynamic Obstacle Clearance, seed={seed}")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"dynamic_obstacle_clearance_seed{seed}.png", dpi=240)
    plt.close(fig)

    return base_clearance, ppo_clearance


def plot_tracking_error(seed: int, base_traj, ppo_traj, env_for_plot, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

    if len(base_traj) > 0:
        ref_base = get_reference_curve(env_for_plot, len(base_traj))
        e_base = np.linalg.norm(base_traj[:, :3] - ref_base[:, :3], axis=1)
        ax.plot(np.arange(len(e_base)), e_base, linewidth=2.4, label="Baseline MPC")

    if len(ppo_traj) > 0:
        ref_ppo = get_reference_curve(env_for_plot, len(ppo_traj))
        e_ppo = np.linalg.norm(ppo_traj[:, :3] - ref_ppo[:, :3], axis=1)
        ax.plot(np.arange(len(e_ppo)), e_ppo, linewidth=2.6, label="PPO-MPC")

    ax.set_xlabel("step")
    ax.set_ylabel("position tracking error")
    ax.set_title(f"Tracking Error, seed={seed}")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"tracking_error_seed{seed}.png", dpi=240)
    plt.close(fig)



def plot_tracking_focused_3d(seed: int, base_traj, ppo_traj, env_for_plot, out_dir: Path):
    
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    ref_curve = get_reference_curve(env_for_plot, MAX_STEPS + 1)

    # Reference: black dashed curve, similar to many UAV tracking papers.
    ax.plot(
        ref_curve[:, 0],
        ref_curve[:, 1],
        ref_curve[:, 2],
        linestyle="--",
        linewidth=2.4,
        color="black",
        label="Reference",
    )

    if len(base_traj) > 0:
        ax.plot(
            base_traj[:, 0],
            base_traj[:, 1],
            base_traj[:, 2],
            linewidth=2.3,
            label="Baseline MPC",
        )

    if len(ppo_traj) > 0:
        ax.plot(
            ppo_traj[:, 0],
            ppo_traj[:, 1],
            ppo_traj[:, 2],
            linewidth=3.2,
            label="PPO-MPC",
        )

    # Draw static obstacles with low opacity so they do not dominate the figure.
    for ob in env_for_plot.static_obstacles:
        # Plot only the physical obstacle size for cleaner paper figures.
        # Clearance metrics still include UAV radius and safety margin.
        r_plot = ob.radius
        plot_static_cylinder_3d(ax, ob.x, ob.y, r_plot, ob.height, alpha=0.07)

    # Draw dynamic obstacles only as paths to avoid overcrowding.
    if len(env_for_plot.dynamic_obstacles) > 0:
        dyn_infos = get_dynamic_obstacle_positions(
            env_for_plot,
            np.arange(0, MAX_STEPS + 1),
        )
        for info in dyn_infos:
            pos = info["positions"]
            ax.plot(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                linestyle=":",
                linewidth=2.2,
                label=f"{info['name']} path",
            )

    ax.scatter([0], [0], [0], marker="o", s=90, label="Start")
    target = env_for_plot.reference.target
    ax.scatter(
        [target[0]],
        [target[1]],
        [target[2]],
        marker="*",
        s=220,
        label="Target",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"3D Trajectory Tracking and Obstacle Avoidance, seed={seed}")

    # Fixed ranges make figures from different seeds visually comparable.
    ax.set_xlim(0.0, 5.4)
    ax.set_ylim(0.0, 5.4)
    ax.set_zlim(0.0, 4.0)

    # A slightly lower elevation helps show trajectory tracking in x-y and altitude together.
    ax.view_init(elev=24, azim=-55)

    ax.grid(True)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_dir / f"tracking_focused_3d_seed{seed}.png", dpi=320)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    if not MODEL_PATH:
        raise RuntimeError("Please specify PPO model path using MODEL_PATH=/path/to/model.zip")

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    tag = "dynamic" if DYNAMIC_OBSTACLES else "static"
    out_dir = PROJECT_ROOT / "scripts" / "runs" / datetime.now().strftime(f"{tag}_compare_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Unified PPO-MPC vs Baseline MPC comparison")
    print("Output dir       :", out_dir)
    print("Model path       :", model_path)
    print("Seeds            :", SEEDS)
    print("Max steps        :", MAX_STEPS)
    print("Reference kind   :", REFERENCE_KIND)
    print("Dynamic obstacles:", DYNAMIC_OBSTACLES)
    print("Model mismatch   :", MODEL_MISMATCH)
    print("Disturbance      :", DISTURBANCE)
    print("Baseline weights :")
    print("  q_pos/q_vel/q_att      =", BASELINE_Q_POS, BASELINE_Q_VEL, BASELINE_Q_ATT)
    print("  r_thrust/r_torque      =", BASELINE_R_THRUST, BASELINE_R_TORQUE)
    print("  beta_track/static/dyn  =", BASELINE_BETA_TRACK, BASELINE_BETA_STATIC, BASELINE_BETA_DYNAMIC)
    print("=" * 90)

    model = PPO.load(str(model_path), device="cpu")

    all_rows, all_dyn_rows = [], []

    for seed in SEEDS:
        print(f"\n===== seed={seed} =====")

        base_traj, base_log, base_env = run_baseline_mpc(seed)
        ppo_traj, ppo_log, ppo_env = run_ppo_mpc(seed, model)

        # Save raw logs and trajectories
        base_log.to_csv(out_dir / f"baseline_step_log_seed{seed}.csv", index=False)
        ppo_log.to_csv(out_dir / f"ppo_step_log_seed{seed}.csv", index=False)
        pd.DataFrame(base_traj, columns=["x", "y", "z"]).to_csv(out_dir / f"baseline_traj_seed{seed}.csv", index=False)
        pd.DataFrame(ppo_traj, columns=["x", "y", "z"]).to_csv(out_dir / f"ppo_traj_seed{seed}.csv", index=False)

        # Dynamic obstacle data and clearance
        base_dyn_clearance, ppo_dyn_clearance = None, None
        if DYNAMIC_OBSTACLES:
            dyn_infos = get_dynamic_obstacle_positions(ppo_env, np.arange(0, MAX_STEPS + 1))
            for info in dyn_infos:
                df_dyn = pd.DataFrame(info["positions"], columns=["x", "y", "z"])
                df_dyn.insert(0, "step", info["steps"])
                safe_name = info["name"].replace(" ", "_").replace("-", "_")
                df_dyn.to_csv(out_dir / f"dynamic_obstacle_{safe_name}_seed{seed}.csv", index=False)

            base_dyn_clearance, ppo_dyn_clearance = plot_dynamic_obstacle_clearance(
                seed,
                base_traj,
                ppo_traj,
                ppo_env,
                out_dir,
            )

            max_len = max(len(base_dyn_clearance), len(ppo_dyn_clearance))
            dyn_df = pd.DataFrame({
                "step": np.arange(max_len),
                "baseline_dynamic_clearance": np.nan,
                "ppo_dynamic_clearance": np.nan,
            })
            dyn_df.loc[:len(base_dyn_clearance) - 1, "baseline_dynamic_clearance"] = base_dyn_clearance
            dyn_df.loc[:len(ppo_dyn_clearance) - 1, "ppo_dynamic_clearance"] = ppo_dyn_clearance
            dyn_df.to_csv(out_dir / f"dynamic_obstacle_clearance_seed{seed}.csv", index=False)

            all_dyn_rows.extend([
                {
                    "seed": seed,
                    "method": "Baseline MPC",
                    "min_dynamic_clearance": float(np.nanmin(base_dyn_clearance)),
                    "mean_dynamic_clearance": float(np.nanmean(base_dyn_clearance)),
                },
                {
                    "seed": seed,
                    "method": "PPO-MPC",
                    "min_dynamic_clearance": float(np.nanmin(ppo_dyn_clearance)),
                    "mean_dynamic_clearance": float(np.nanmean(ppo_dyn_clearance)),
                },
            ])

        all_rows.append(summarize_one("Baseline MPC", seed, base_log, base_traj, base_env, base_dyn_clearance))
        all_rows.append(summarize_one("PPO-MPC", seed, ppo_log, ppo_traj, ppo_env, ppo_dyn_clearance))

        # Plots
        plot_3d_trajectory(seed, base_traj, ppo_traj, ppo_env, out_dir)
        plot_tracking_focused_3d(seed, base_traj, ppo_traj, ppo_env, out_dir)
        plot_xy_trajectory(seed, base_traj, ppo_traj, ppo_env, out_dir)
        plot_min_clearance(seed, base_log, ppo_log, out_dir)
        plot_tracking_error(seed, base_traj, ppo_traj, ppo_env, out_dir)

        print("Baseline final:")
        print(base_log.iloc[-1][[
            "success",
            "termination_reason",
            "goal_dist",
            "path_progress",
            "min_clearance",
            "mpc_failed",
        ]])
        print("Baseline RMSE:", compute_tracking_rmse(base_traj, base_env))

        print("PPO-MPC final:")
        print(ppo_log.iloc[-1][[
            "success",
            "termination_reason",
            "goal_dist",
            "path_progress",
            "min_clearance",
            "mpc_failed",
        ]])
        print("PPO-MPC RMSE:", compute_tracking_rmse(ppo_traj, ppo_env))

        if DYNAMIC_OBSTACLES:
            print("Baseline min dynamic clearance:", float(np.nanmin(base_dyn_clearance)))
            print("PPO-MPC min dynamic clearance:", float(np.nanmin(ppo_dyn_clearance)))

    compare_df = pd.DataFrame(all_rows)
    compare_df.to_csv(out_dir / "compare_results.csv", index=False)

    summary_df = build_summary(compare_df)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    paper_table = build_paper_table(summary_df)
    paper_table.to_csv(out_dir / "paper_table.csv", index=False)

    if len(all_dyn_rows) > 0:
        dyn_summary = pd.DataFrame(all_dyn_rows)
        dyn_summary.to_csv(out_dir / "dynamic_obstacle_clearance_summary.csv", index=False)

    print("\n================ paper_table.csv ================")
    print(paper_table)

    print("\nSaved to:")
    print(out_dir)


if __name__ == "__main__":
    main()
