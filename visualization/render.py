"""Static and animated visualization of the environment and replays."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle

from envs.city_defense_env import EpisodeReplay
from envs.fixed_map import FixedMap


def _setup_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f6f6f0")
    for spine in ax.spines.values():
        spine.set_color("#888")


def plot_map(
    fmap: FixedMap,
    ax: Optional[plt.Axes] = None,
    show_entry_points: bool = True,
    show_deployment_nodes: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 7))
    _setup_axes(ax)

    # City
    city = Circle(fmap.city_center, fmap.city_radius,
                  facecolor="#ffe0b3", edgecolor="#e48f19", lw=2, alpha=0.85, zorder=1)
    ax.add_patch(city)
    ax.plot(*fmap.city_center, marker="*", color="#e48f19", markersize=16, zorder=2)

    # Roads
    for u, v in fmap.edges:
        a = fmap.node_xy(u)
        b = fmap.node_xy(v)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#888", lw=1.5, zorder=2)

    # Nodes
    xs, ys = fmap.node_positions[:, 0], fmap.node_positions[:, 1]
    ax.scatter(xs, ys, s=16, color="#555", zorder=3)

    if show_deployment_nodes:
        dep_xs = fmap.node_positions[fmap.deployment_nodes, 0]
        dep_ys = fmap.node_positions[fmap.deployment_nodes, 1]
        ax.scatter(dep_xs, dep_ys, s=36, facecolor="none",
                   edgecolor="#2266dd", lw=1.2, zorder=4,
                   label="deployment nodes")

    if show_entry_points:
        ax.scatter(fmap.entry_points[:, 0], fmap.entry_points[:, 1],
                   s=60, marker="^", color="#aa0000", zorder=5,
                   label="entry points")

    if title:
        ax.set_title(title)

    if own_fig:
        ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_episode_static(
    fmap: FixedMap,
    replay: EpisodeReplay,
    intercept_radius: float,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    plot_map(fmap, ax=ax, title=title or "Episode replay (static)")

    # Drone trajectories.
    drone_positions = np.stack(replay.drone_positions, axis=0)  # (T, n_drones, 2)
    n_drones = drone_positions.shape[1]
    drone_alive_arr = np.stack(replay.drone_alive, axis=0)       # (T, n_drones) bool

    # Reached / destroyed timestamps.
    reached_t = {e.drone_id: e.t for e in replay.events if e.kind == "reached_city"}
    destroyed_t = {e.drone_id: e.t for e in replay.events if e.kind == "destroyed"}
    launched_t = {e.drone_id: e.t for e in replay.events if e.kind == "launched"}

    for i in range(n_drones):
        # clip the trajectory at the drone's resolution time.
        t_end = len(drone_positions)
        if i in reached_t:
            t_end = min(t_end, reached_t[i] + 1)
        if i in destroyed_t:
            t_end = min(t_end, destroyed_t[i] + 1)
        start = launched_t.get(i, 0)
        if t_end <= start:
            continue
        traj = drone_positions[start:t_end, i]
        ax.plot(traj[:, 0], traj[:, 1], color="#cc2222", lw=1.2, alpha=0.6, zorder=6)

        if i in reached_t:
            p = drone_positions[reached_t[i], i]
            ax.plot(*p, marker="s", color="#cc2222", markersize=8, zorder=7)
        elif i in destroyed_t:
            p = drone_positions[destroyed_t[i], i]
            ax.plot(*p, marker="x", color="#222", markersize=9, zorder=7)

    # Defender final positions + intercept circle.
    final_def = replay.defender_positions[-1]
    for p in final_def:
        ax.add_patch(Circle(p, intercept_radius, facecolor="#2266dd",
                            edgecolor="#2266dd", alpha=0.12, zorder=4))
        ax.plot(*p, marker="o", color="#2266dd", markersize=9, zorder=6)

    # Summary.
    n_reached = replay.n_reached
    n_destroyed = replay.n_destroyed
    ax.text(
        0.02, 0.02,
        f"reached: {n_reached}    destroyed: {n_destroyed}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.9),
    )
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def animate_episode(
    fmap: FixedMap,
    replay: EpisodeReplay,
    intercept_radius: float,
    save_path: Optional[str] = None,
    fps: int = 15,
    title: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    plot_map(fmap, ax=ax, title=title or "Episode animation")

    drone_positions = np.stack(replay.drone_positions, axis=0)  # (T, n_drones, 2)
    drone_alive_arr = np.stack(replay.drone_alive, axis=0)
    defender_positions = np.stack(replay.defender_positions, axis=0)
    T, n_drones, _ = drone_positions.shape
    n_def = defender_positions.shape[1]

    # Precompute resolution timestamps.
    destroyed_t = {e.drone_id: e.t for e in replay.events if e.kind == "destroyed"}
    reached_t = {e.drone_id: e.t for e in replay.events if e.kind == "reached_city"}

    # Artists.
    drone_points = ax.scatter(drone_positions[0, :, 0], drone_positions[0, :, 1],
                              s=32, color="#cc2222", zorder=6)
    defender_circles = []
    defender_points = ax.scatter(defender_positions[0, :, 0], defender_positions[0, :, 1],
                                 s=80, color="#2266dd", zorder=7, marker="o")
    for k in range(n_def):
        c = Circle(defender_positions[0, k], intercept_radius,
                   facecolor="#2266dd", edgecolor="#2266dd", alpha=0.10, zorder=4)
        ax.add_patch(c)
        defender_circles.append(c)

    # Lightweight trail: we keep last ~20 frames per drone.
    trail_len = 20
    trail_lines = []
    for _ in range(n_drones):
        ln, = ax.plot([], [], color="#cc2222", lw=1.0, alpha=0.5, zorder=5)
        trail_lines.append(ln)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                         bbox=dict(facecolor="white", edgecolor="#888", alpha=0.9))

    def update(frame_idx: int):
        # drones
        pos = drone_positions[frame_idx]
        alive = drone_alive_arr[frame_idx]
        colors = []
        sizes = []
        for i in range(n_drones):
            if i in destroyed_t and frame_idx >= destroyed_t[i]:
                colors.append((0.15, 0.15, 0.15, 0.9))
                sizes.append(34)
            elif i in reached_t and frame_idx >= reached_t[i]:
                colors.append((0.8, 0.5, 0.1, 0.9))
                sizes.append(40)
            elif alive[i]:
                colors.append((0.8, 0.13, 0.13, 1.0))
                sizes.append(30)
            else:
                # not yet launched
                colors.append((0.8, 0.13, 0.13, 0.25))
                sizes.append(12)
        drone_points.set_offsets(pos)
        drone_points.set_color(colors)
        drone_points.set_sizes(sizes)

        # trails
        for i in range(n_drones):
            start = max(0, frame_idx - trail_len)
            xs = drone_positions[start:frame_idx + 1, i, 0]
            ys = drone_positions[start:frame_idx + 1, i, 1]
            trail_lines[i].set_data(xs, ys)

        # defenders
        defender_points.set_offsets(defender_positions[frame_idx])
        for k, c in enumerate(defender_circles):
            c.center = tuple(defender_positions[frame_idx, k])

        reached_so_far = sum(1 for i, t in reached_t.items() if t <= frame_idx)
        destroyed_so_far = sum(1 for i, t in destroyed_t.items() if t <= frame_idx)
        time_text.set_text(
            f"t={frame_idx}   reached: {reached_so_far}   destroyed: {destroyed_so_far}"
        )
        return [drone_points, defender_points, time_text, *trail_lines, *defender_circles]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=False, repeat=False,
    )
    if save_path:
        ext = save_path.rsplit(".", 1)[-1].lower()
        if ext == "gif":
            anim.save(save_path, writer=animation.PillowWriter(fps=fps))
        elif ext == "mp4":
            anim.save(save_path, writer=animation.FFMpegWriter(fps=fps))
        else:
            raise ValueError(f"Unsupported animation extension: {ext}")
    return anim


def plot_training_curve(
    csv_path: str,
    save_path: Optional[str] = None,
    title: str = "Training curve",
):
    """Plot a training log with columns step, mean_ep_reward, mean_destroyed, mean_reached."""
    import csv
    rows = []
    with open(csv_path) as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            rows.append([float(x) for x in row])
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")
    arr = np.array(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(arr[:, 0], arr[:, 1])
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("mean episode reward")
    axes[0].set_title("reward")
    axes[1].plot(arr[:, 0], arr[:, 2], label="destroyed")
    axes[1].plot(arr[:, 0], arr[:, 3], label="reached")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("drones")
    axes[1].legend()
    axes[1].set_title("per-episode counts")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
