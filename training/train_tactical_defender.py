"""Train the tactical defender with PPO.

Uses scripted attacker + scripted deployment as opponents.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs import EnvConfig, TacticalDefenderEnv
from training.common import EpisodeStatsCallback, load_config, make_vec_env


def make_env_fn(cfg: EnvConfig, seed: int):
    def _thunk():
        env = TacticalDefenderEnv(cfg=cfg, seed=seed)
        env = Monitor(env)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--out", default="outputs/models/tactical_defender")
    parser.add_argument("--log", default="outputs/plots/tactical_training.csv")
    args = parser.parse_args()

    conf = load_config(args.config)
    env_cfg = EnvConfig(
        T=conf["env"]["T"],
        dt=conf["env"]["dt"],
        n_drones=conf["env"]["n_drones"],
        n_defenders=conf["env"]["n_defenders"],
        n_entry_points=conf["env"]["n_entry_points"],
        n_route_templates=conf["env"]["n_route_templates"],
        drone_max_speed=conf["env"]["drone_max_speed"],
        drone_max_accel=conf["env"]["drone_max_accel"],
        defender_road_speed=conf["env"]["defender_road_speed"],
        defender_intercept_radius=conf["env"]["defender_intercept_radius"],
        tactical_decision_interval=conf["env"]["tactical_decision_interval"],
        city_center=tuple(conf["env"]["city_center"]),
        city_radius=conf["env"]["city_radius"],
        defender_visible_prob=conf["env"]["defender_visible_prob"],
        map_ring_radii=(tuple(conf["env"]["map_ring_radii"])
                        if conf["env"].get("map_ring_radii") else None),
        map_nodes_per_ring=(tuple(conf["env"]["map_nodes_per_ring"])
                            if conf["env"].get("map_nodes_per_ring") else None),
    )
    n_envs = args.n_envs or conf["training"]["tactical"]["n_envs"]
    timesteps = args.timesteps or conf["training"]["tactical"]["total_timesteps"]
    lr = conf["training"]["tactical"]["lr"]

    vec_env = make_vec_env(lambda i: make_env_fn(env_cfg, seed=1000 + i), n_envs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lr,
        n_steps=256,
        batch_size=256,
        n_epochs=6,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    cb = EpisodeStatsCallback(args.log, log_every=4096)
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip")
    print(f"Training log written to {args.log}")


if __name__ == "__main__":
    main()
