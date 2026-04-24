"""3-way round-robin self-play with a small opponent pool.

Per iteration, we train deployment -> tactical -> attacker for a short
budget each, holding the other two fixed. Past snapshots are kept in a
per-agent FIFO pool; when an env needs an opponent we sample a random
snapshot path from the pool (scripted fallback if the pool is empty).
Sampling a mix of recent opponents — rather than only the latest — is
the standard fix for cycling between trained agents in alternating
self-play.

Each run lands under ``outputs/runs/<run_name>/``:
    models/   run_name_{agent}_iter{it}.zip
    plots/    per-agent reward CSV + PNG curves
    tb/       TensorBoard loss logs (one subdir per agent)
    replays/  final.gif rendered with the last-iter trained triple
    config.json  captured args for reproducibility

Usage:
  python training/self_play_3way.py --run_name my_first_run --iterations 4
  python training/self_play_3way.py   # auto-names sp3_YYYYMMDD_HHMMSS

View loss curves:
  tensorboard --logdir outputs/runs/<run_name>/tb
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from envs import (
    AttackerPlannerEnv,
    DeploymentEnv,
    EnvConfig,
    TacticalDefenderEnv,
)
from envs.wrappers import (
    decode_plan,
    flatten_attacker_obs,
    flatten_deployment_obs,
    flatten_tactical_obs,
)
from training.common import EpisodeStatsCallback, make_vec_env


# --- Adapters: saved .zip -> callable opponent fn ---------------------- #

def _load_tactical_fn(path: str):
    model = PPO.load(path)

    def fn(tac_obs, cfg):
        obs_vec = flatten_tactical_obs(tac_obs, cfg)
        action, _ = model.predict(obs_vec, deterministic=True)
        return action
    return fn


def _load_attacker_fn(path: str):
    model = PPO.load(path)

    def fn(atk_obs, cfg):
        obs_vec = flatten_attacker_obs(atk_obs, cfg)
        action, _ = model.predict(obs_vec, deterministic=True)
        return decode_plan(np.asarray(action), cfg)
    return fn


def _load_deployment_fn(path: str):
    model = PPO.load(path)

    def fn(dep_obs, cfg, fmap):
        obs_vec = flatten_deployment_obs(dep_obs, cfg, fmap.n_nodes)
        action, _ = model.predict(obs_vec, deterministic=True)
        return np.asarray(action, dtype=np.int64)
    return fn


_LOADERS = {
    "tactical": _load_tactical_fn,
    "attacker": _load_attacker_fn,
    "deployment": _load_deployment_fn,
}


class StageProgressCallback(BaseCallback):
    """Print `<done> / <total>` every ``print_frac`` of the stage budget."""

    def __init__(self, label: str, total: int, print_frac: float = 0.25):
        super().__init__()
        self.label = label
        self.total = total
        self.step_bucket = max(1, int(total * print_frac))
        self._start = 0
        self._last = 0

    def _on_training_start(self) -> None:
        self._start = self.num_timesteps
        self._last = 0

    def _on_step(self) -> bool:
        done = self.num_timesteps - self._start
        if done - self._last >= self.step_bucket:
            pct = 100.0 * done / max(1, self.total)
            print(f"    [{self.label}] {done:,} / {self.total:,}  ({pct:.0f}%)",
                  flush=True)
            self._last = done
        return True


# --- Opponent pool ----------------------------------------------------- #

class OpponentPool:
    """FIFO of recent snapshot paths per agent kind."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._paths: dict[str, Deque[str]] = {
            k: deque(maxlen=max_size)
            for k in ("tactical", "attacker", "deployment")
        }

    def add(self, kind: str, path: str) -> None:
        self._paths[kind].append(path)

    def sample_fn(self, kind: str, rng: random.Random):
        bucket = self._paths[kind]
        if not bucket:
            return None
        return _LOADERS[kind](rng.choice(list(bucket)))

    def summary(self) -> str:
        parts = [f"{k}={len(v)}" for k, v in self._paths.items()]
        return ", ".join(parts)


# --- Env factories ----------------------------------------------------- #

def _deployment_env_fn(cfg, pool, seed_base):
    def make_fn(i):
        worker_rng = random.Random(seed_base + i * 7919)
        def _thunk():
            env = DeploymentEnv(
                cfg=cfg,
                attacker_fn=pool.sample_fn("attacker", worker_rng),
                tactical_fn=pool.sample_fn("tactical", worker_rng),
                seed=seed_base + i,
            )
            return Monitor(env)
        return _thunk
    return make_fn


def _tactical_env_fn(cfg, pool, seed_base):
    def make_fn(i):
        worker_rng = random.Random(seed_base + i * 7919)
        def _thunk():
            env = TacticalDefenderEnv(
                cfg=cfg,
                attacker_fn=pool.sample_fn("attacker", worker_rng),
                deployment_fn=pool.sample_fn("deployment", worker_rng),
                seed=seed_base + i,
            )
            return Monitor(env)
        return _thunk
    return make_fn


def _attacker_env_fn(cfg, pool, seed_base):
    def make_fn(i):
        worker_rng = random.Random(seed_base + i * 7919)
        def _thunk():
            env = AttackerPlannerEnv(
                cfg=cfg,
                deployment_fn=pool.sample_fn("deployment", worker_rng),
                tactical_fn=pool.sample_fn("tactical", worker_rng),
                seed=seed_base + i,
            )
            return Monitor(env)
        return _thunk
    return make_fn


def _parse_arch(arch_str: str) -> list[int]:
    """Parse '256,256' or '512,512,512' into [256, 256] etc."""
    parts = [p.strip() for p in arch_str.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"empty policy_arch: {arch_str!r}")
    return [int(p) for p in parts]


# --- Training stage helper -------------------------------------------- #

def _train_stage(model, vec_env, timesteps, ppo_kwargs, net_arch,
                 tb_log, log_name, callback, verbose):
    """Create the PPO model on first call, then reuse + set_env thereafter."""
    if model is None:
        model = PPO("MlpPolicy", vec_env, verbose=verbose,
                    policy_kwargs=dict(net_arch=list(net_arch)),
                    tensorboard_log=tb_log,
                    **ppo_kwargs)
        reset_ts = True
    else:
        model.set_env(vec_env)
        reset_ts = False
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        tb_log_name=log_name,
        reset_num_timesteps=reset_ts,
    )
    return model


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


# --- Post-training artefacts ------------------------------------------ #

def _render_reward_png(csv_path: str, png_path: str, title: str,
                       iter_boundaries=()) -> None:
    """Render a PNG showing cumulative reward across all iterations.

    ``iter_boundaries`` is a list of step counts where each iteration
    ended; we draw vertical dashed lines there so you can see where
    one round of round-robin ended and the next began.
    """
    if not os.path.exists(csv_path):
        return
    import csv as _csv
    rows = []
    with open(csv_path) as f:
        r = _csv.reader(f)
        next(r, None)  # header
        for row in r:
            rows.append([float(x) for x in row])
    if not rows:
        print(f"  [skip] {csv_path} has no rows yet (log_every too large?)")
        return
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
    for b in iter_boundaries:
        for ax in axes:
            ax.axvline(x=b, color="gray", linestyle="--",
                       alpha=0.35, linewidth=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_gif(run_dir: str, snapshot_paths: dict, episodes: int,
                seed: int, gif_name: str) -> Optional[str]:
    """Spawn evaluation/eval.py --save_gif with the given snapshots."""
    eval_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evaluation", "eval.py",
    )
    out_dir = os.path.join(run_dir, "replays")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, eval_script,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--save_gif",
        "--gif_name", gif_name,
        "--out_dir", out_dir,
        "--attacker_model", snapshot_paths["attacker"],
        "--deployment_model", snapshot_paths["deployment"],
        "--tactical_model", snapshot_paths["tactical"],
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"    [warn] gif render failed: {e.stderr[-200:] if e.stderr else e}")
        return None
    return os.path.join(out_dir, gif_name)


def _render_all_reward_pngs(plots_dir: str, reward_logs: dict, run_name: str,
                             boundaries: dict) -> None:
    for agent in ("deployment", "tactical", "attacker"):
        png = os.path.join(plots_dir, f"{run_name}_{agent}_reward.png")
        _render_reward_png(reward_logs[agent], png,
                           title=f"{run_name} :: {agent}",
                           iter_boundaries=boundaries.get(agent, ()))


# --- Main loop --------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=None,
                        help="Run identifier prefix. A unique "
                             "_YYYYMMDD_HHMMSS_NNNN suffix is always appended "
                             "so runs never overwrite each other. "
                             "If omitted, the prefix defaults to 'sp3'.")
    parser.add_argument("--runs_root", default="outputs/runs")
    # Smaller per-stage budgets + more iterations => faster rotation,
    # closer to simultaneous multi-agent RL. Tactical command learning is
    # the hardest stage, so it gets a larger default budget than the
    # one-shot deployment / attacker stages.
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--deployment_steps", type=int, default=1_024)
    parser.add_argument("--tactical_steps", type=int, default=204_800)
    parser.add_argument("--attacker_steps", type=int, default=512)
    parser.add_argument("--pool_size", type=int, default=6)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppo_verbose", type=int, default=0,
                        help="PPO verbose (0 silent, 1 prints SB3 rollout tables).")
    parser.add_argument("--subproc", action="store_true",
                        help="Use SubprocVecEnv (real multi-core rollouts). "
                             "Recommended on multi-core CPUs; harmless to keep off "
                             "for quick tests.")
    parser.add_argument("--policy_arch", default="128,128",
                        help="Comma-separated MLP widths, e.g. '256,256' or "
                             "'512,512,512'. Bigger nets let the GPU pull its "
                             "weight but risk overfitting for this small obs space.")
    # Per-stage PPO rollout/batch sizes. Bigger n_steps => more data per
    # update => lower-variance gradient; bigger batch => bigger GPU matmul.
    parser.add_argument("--deployment_n_steps", type=int, default=64)
    parser.add_argument("--deployment_batch", type=int, default=64)
    parser.add_argument("--tactical_n_steps", type=int, default=256)
    parser.add_argument("--tactical_batch", type=int, default=256)
    parser.add_argument("--attacker_n_steps", type=int, default=64)
    parser.add_argument("--attacker_batch", type=int, default=64)
    parser.add_argument("--no_animation", action="store_true",
                        help="Skip rendering the final gif.")
    parser.add_argument("--gif_episodes", type=int, default=1)
    parser.add_argument("--gif_every", type=int, default=4,
                        help="Render a gif every N iterations (and always on "
                             "the final iteration). With many short iterations "
                             "you don't want a gif per rotation.")
    args = parser.parse_args()

    suffix = (
        _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_{random.SystemRandom().randint(0, 9999):04d}"
    )
    prefix = args.run_name or "sp3"
    run_name = f"{prefix}_{suffix}"
    run_dir = os.path.join(args.runs_root, run_name)
    models_dir = os.path.join(run_dir, "models")
    plots_dir = os.path.join(run_dir, "plots")
    tb_dir = os.path.join(run_dir, "tb")
    for d in (models_dir, plots_dir, tb_dir):
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(run_dir, f"{run_name}_config.json"), "w") as f:
        json.dump({**vars(args), "run_name": run_name}, f, indent=2)

    print(f"\n=== run: {run_name} ===")
    print(f"artefacts -> {run_dir}\n")

    cfg = EnvConfig()
    net_arch = _parse_arch(args.policy_arch)
    pool = OpponentPool(max_size=args.pool_size)

    # One reward-log callback per agent, reused across iterations so the CSV
    # grows monotonically. Sampling frequency scales with per-iteration budget.
    reward_logs = {
        agent: os.path.join(plots_dir, f"{run_name}_{agent}.csv")
        for agent in ("deployment", "tactical", "attacker")
    }
    callbacks = {
        "deployment": EpisodeStatsCallback(reward_logs["deployment"], log_every=256),
        "tactical": EpisodeStatsCallback(reward_logs["tactical"], log_every=4096),
        "attacker": EpisodeStatsCallback(reward_logs["attacker"], log_every=256),
    }

    dep_model = tac_model = atk_model = None
    # Step count at the end of each iteration, per agent. Used to draw
    # vertical separators on the cumulative reward PNG.
    iter_boundaries: dict[str, list[int]] = {
        "deployment": [], "tactical": [], "attacker": []
    }

    # PPO kwargs from CLI. ent_coef stays fixed per stage (not a knob the
    # user is currently tuning).
    dep_ppo = dict(n_steps=args.deployment_n_steps,
                   batch_size=args.deployment_batch, ent_coef=0.02)
    tac_ppo = dict(n_steps=args.tactical_n_steps,
                   batch_size=args.tactical_batch, ent_coef=0.01)
    atk_ppo = dict(n_steps=args.attacker_n_steps,
                   batch_size=args.attacker_batch, ent_coef=0.01)

    total_steps = args.iterations * (args.deployment_steps
                                      + args.tactical_steps + args.attacker_steps)
    print(f"iterations={args.iterations}  n_envs={args.n_envs}  "
          f"pool_size={args.pool_size}  subproc={args.subproc}")
    print(f"policy_arch={net_arch}")
    print(f"per-iter budget: dep={args.deployment_steps}  "
          f"tac={args.tactical_steps}  atk={args.attacker_steps}")
    print(f"total PPO steps: {total_steps:,}\n")

    run_t0 = time.time()
    final_paths: dict[str, str] = {}

    for it in range(args.iterations):
        it_t0 = time.time()
        print(f"--- iteration {it + 1}/{args.iterations} "
              f"(pool: {pool.summary()}) ---")

        for stage_name, timesteps, ppo_kwargs, env_fn_factory, seed_base in [
            ("deployment", args.deployment_steps, dep_ppo,
             _deployment_env_fn, 3000 + it * 101 + args.seed),
            ("tactical", args.tactical_steps, tac_ppo,
             _tactical_env_fn, 4000 + it * 101 + args.seed),
            ("attacker", args.attacker_steps, atk_ppo,
             _attacker_env_fn, 5000 + it * 101 + args.seed),
        ]:
            print(f"  [{stage_name}] {timesteps:,} steps ...")
            stage_t0 = time.time()
            vec = make_vec_env(env_fn_factory(cfg, pool, seed_base),
                               args.n_envs, use_subproc=args.subproc)
            current_model = {
                "deployment": dep_model,
                "tactical": tac_model,
                "attacker": atk_model,
            }[stage_name]
            cb = CallbackList([
                StageProgressCallback(stage_name, timesteps),
                callbacks[stage_name],
            ])
            current_model = _train_stage(
                current_model, vec, timesteps, ppo_kwargs, net_arch,
                tb_log=tb_dir, log_name=f"{run_name}_{stage_name}",
                callback=cb,
                verbose=args.ppo_verbose,
            )
            if stage_name == "deployment":
                dep_model = current_model
            elif stage_name == "tactical":
                tac_model = current_model
            else:
                atk_model = current_model
            iter_boundaries[stage_name].append(int(current_model.num_timesteps))

            path = os.path.join(
                models_dir, f"{run_name}_{stage_name}_iter{it}"
            )
            current_model.save(path)
            zip_path = path + ".zip"
            pool.add(stage_name, zip_path)
            final_paths[stage_name] = zip_path
            print(f"    saved -> {os.path.relpath(zip_path)}  "
                  f"({_fmt_dur(time.time() - stage_t0)})")

        # Per-iteration artefacts: PNG reward curves and a sample gif.
        # Writing these each iteration means interrupted runs still have
        # plots + animations for whatever iters completed.
        print(f"  rendering artefacts for iter {it + 1} ...")
        _render_all_reward_pngs(plots_dir, reward_logs, run_name,
                                 boundaries=iter_boundaries)
        is_last_iter = (it + 1) == args.iterations
        should_gif = (not args.no_animation
                      and ((it + 1) % args.gif_every == 0 or is_last_iter))
        if should_gif:
            gif_path = _render_gif(
                run_dir, dict(final_paths),
                episodes=args.gif_episodes,
                seed=args.seed + 1000 * (it + 1),
                gif_name=f"{run_name}_iter{it}.gif",
            )
            if gif_path:
                print(f"    gif -> {os.path.relpath(gif_path)}")

        print(f"  iter {it + 1} done in {_fmt_dur(time.time() - it_t0)}\n")

    print(f"training complete in {_fmt_dur(time.time() - run_t0)}\n")
    print(f"=== run '{run_name}' done ===")
    print(f"snapshots : {os.path.relpath(models_dir)}")
    print(f"reward png: {os.path.relpath(plots_dir)}")
    print(f"tb losses : tensorboard --logdir {os.path.relpath(tb_dir)}")
    if not args.no_animation:
        print(f"animation : {os.path.relpath(os.path.join(run_dir, 'replays'))}")


if __name__ == "__main__":
    main()
