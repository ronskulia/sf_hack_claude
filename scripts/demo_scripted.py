"""Run scripted attacker vs scripted defender and save plots + replay.

Usage:
    python scripts/demo_scripted.py --episodes 3 --gif
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# Make package imports work when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import (
    ScriptedAttacker,
    ScriptedDefenderDeployment,
    ScriptedDefenderTactical,
)
from envs import CityDefenseEnv, EnvConfig
from visualization import animate_episode, plot_episode_static, plot_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/replays")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    cfg = EnvConfig()
    env = CityDefenseEnv(cfg)

    # Save a map overview first.
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(1, 1, 1)
    plot_map(env.map, ax=ax, title="Fixed map (city + roads + entry points)")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig("outputs/plots/map_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote outputs/plots/map_overview.png")

    atk = ScriptedAttacker(cfg, seed=args.seed)
    dep = ScriptedDefenderDeployment(cfg, env.map)
    tac = ScriptedDefenderTactical(cfg, env.map)

    all_summaries = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        plan = atk.plan(env.attacker_observation())
        env.commit_attacker_plan(plan)
        env.commit_deployment(dep.deploy(env.deployment_observation()))
        t0 = time.time()
        while not env.done:
            a = tac.act(env.tactical_observation())
            env.tactical_step(a)
        dt = time.time() - t0
        summary = env.summary()
        print(f"ep {ep}: {summary}  [{dt:.2f}s]")
        all_summaries.append(summary)

        # Static replay plot
        out_png = os.path.join(args.out_dir, f"ep{ep:02d}_static.png")
        fig = plot_episode_static(
            env.map, env.replay, cfg.defender_intercept_radius,
            save_path=out_png,
            title=f"Episode {ep} (scripted vs scripted)",
        )
        plt.close(fig)
        print(f"  wrote {out_png}")

        if args.gif:
            out_gif = os.path.join(args.out_dir, f"ep{ep:02d}_anim.gif")
            anim = animate_episode(
                env.map, env.replay, cfg.defender_intercept_radius,
                save_path=out_gif, fps=15,
                title=f"Episode {ep} (scripted vs scripted)",
            )
            plt.close("all")
            print(f"  wrote {out_gif}")

    # Aggregate
    if all_summaries:
        mean_r = np.mean([s["reached"] for s in all_summaries])
        mean_d = np.mean([s["destroyed"] for s in all_summaries])
        print(f"\nOverall: mean reached={mean_r:.2f}  mean destroyed={mean_d:.2f}")


if __name__ == "__main__":
    main()
