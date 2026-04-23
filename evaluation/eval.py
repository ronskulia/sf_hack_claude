"""Evaluate combinations of scripted and trained agents.

Examples:
    # scripted vs scripted
    python evaluation/eval.py --episodes 20
    # trained tactical vs scripted attacker/deployment
    python evaluation/eval.py --tactical_model outputs/models/tactical_defender \
                              --episodes 20
    # trained attacker vs scripted defense
    python evaluation/eval.py --attacker_model outputs/models/attacker_planner
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents import (
    ScriptedAttacker,
    ScriptedDefenderDeployment,
    ScriptedDefenderTactical,
)
from envs import AttackPlan, CityDefenseEnv, EnvConfig
from envs.wrappers import (
    decode_plan,
    flatten_attacker_obs,
    flatten_deployment_obs,
    flatten_tactical_obs,
)
from visualization import animate_episode, plot_episode_static


def _load_sb3(path: str):
    from stable_baselines3 import PPO
    return PPO.load(path)


def build_attacker(cfg: EnvConfig, model_path: Optional[str], seed: int) -> Callable:
    if model_path is None:
        base = ScriptedAttacker(cfg, seed=seed)
        return lambda obs: base.plan(obs)
    model = _load_sb3(model_path)

    def fn(obs):
        vec = flatten_attacker_obs(obs, cfg)
        action, _ = model.predict(vec, deterministic=True)
        return decode_plan(action, cfg)
    return fn


def build_deployment(cfg: EnvConfig, fmap, model_path: Optional[str]) -> Callable:
    if model_path is None:
        base = ScriptedDefenderDeployment(cfg, fmap)
        return lambda obs: base.deploy(obs)
    model = _load_sb3(model_path)

    def fn(obs):
        vec = flatten_deployment_obs(obs, cfg, fmap.n_nodes)
        action, _ = model.predict(vec, deterministic=True)
        return np.asarray(action, dtype=np.int64)
    return fn


def build_tactical(cfg: EnvConfig, fmap, model_path: Optional[str]) -> Callable:
    if model_path is None:
        base = ScriptedDefenderTactical(cfg, fmap)
        return lambda obs: base.act(obs)
    model = _load_sb3(model_path)

    def fn(obs):
        vec = flatten_tactical_obs(obs, cfg)
        action, _ = model.predict(vec, deterministic=True)
        return np.asarray(action, dtype=np.int64)
    return fn


def run_episodes(env: CityDefenseEnv, atk_fn, dep_fn, tac_fn, n_episodes: int, seed: int):
    results = []
    replays = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        plan = atk_fn(env.attacker_observation())
        env.commit_attacker_plan(plan)
        env.commit_deployment(dep_fn(env.deployment_observation()))
        while not env.done:
            env.tactical_step(tac_fn(env.tactical_observation()))
        results.append(env.summary())
        replays.append(env.replay)
    return results, replays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attacker_model", default=None)
    parser.add_argument("--deployment_model", default=None)
    parser.add_argument("--tactical_model", default=None)
    parser.add_argument("--save_first_replay", action="store_true")
    parser.add_argument("--save_gif", action="store_true",
                        help="Save an animated gif of the first episode.")
    parser.add_argument("--gif_name", default="eval_ep0_anim.gif")
    parser.add_argument("--out_dir", default="outputs/replays")
    args = parser.parse_args()

    cfg = EnvConfig()
    env = CityDefenseEnv(cfg)

    atk = build_attacker(cfg, args.attacker_model, seed=args.seed)
    dep = build_deployment(cfg, env.map, args.deployment_model)
    tac = build_tactical(cfg, env.map, args.tactical_model)

    results, replays = run_episodes(env, atk, dep, tac, args.episodes, seed=args.seed)

    reached = np.array([r["reached"] for r in results])
    destroyed = np.array([r["destroyed"] for r in results])

    print("=== Evaluation summary ===")
    print(f"Attacker: {args.attacker_model or 'scripted'}")
    print(f"Deployment: {args.deployment_model or 'scripted'}")
    print(f"Tactical: {args.tactical_model or 'scripted'}")
    print(f"episodes: {args.episodes}")
    print(f"reached    mean={reached.mean():.2f}  std={reached.std():.2f}")
    print(f"destroyed  mean={destroyed.mean():.2f}  std={destroyed.std():.2f}")
    print(f"defender_win_rate: {float((destroyed > reached).mean()):.2%}")

    if (args.save_first_replay or args.save_gif) and replays:
        os.makedirs(args.out_dir, exist_ok=True)
        title_parts = []
        title_parts.append(f"atk={'trained' if args.attacker_model else 'scripted'}")
        title_parts.append(f"def={'trained' if args.tactical_model else 'scripted'}")
        title = "Eval episode 0 — " + ", ".join(title_parts)
        if args.save_first_replay:
            out = os.path.join(args.out_dir, "eval_ep0.png")
            fig = plot_episode_static(env.map, replays[0], cfg.defender_intercept_radius,
                                       save_path=out, title=title)
            plt.close(fig)
            print(f"First-episode static replay saved to {out}")
        if args.save_gif:
            out_gif = os.path.join(args.out_dir, args.gif_name)
            _ = animate_episode(env.map, replays[0], cfg.defender_intercept_radius,
                                save_path=out_gif, fps=15, title=title)
            plt.close("all")
            print(f"First-episode gif saved to {out_gif}")


if __name__ == "__main__":
    main()
