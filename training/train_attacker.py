"""Train the attacker planner with PPO.

Each RL step commits a full attack plan, runs the episode with the
scripted (or trained) defender, and returns the episode's attacker reward.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs import AttackerPlannerEnv, EnvConfig
from training.common import EpisodeStatsCallback, load_config, make_vec_env


def _defender_tactical_fn_from_model(model_path: str, cfg: EnvConfig):
    """Return a callable(tac_obs_dict, cfg) -> action using a trained SB3 model."""
    from stable_baselines3 import PPO as _PPO
    from envs.wrappers import flatten_tactical_obs
    model = _PPO.load(model_path)

    def fn(tac_obs, cfg_):
        obs_vec = flatten_tactical_obs(tac_obs, cfg_)
        action, _ = model.predict(obs_vec, deterministic=True)
        return action
    return fn


def make_env_fn(cfg: EnvConfig, seed: int, tactical_model: str | None):
    def _thunk():
        tactical_fn = None
        if tactical_model:
            tactical_fn = _defender_tactical_fn_from_model(tactical_model, cfg)
        env = AttackerPlannerEnv(cfg=cfg, seed=seed, tactical_fn=tactical_fn)
        env = Monitor(env)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--tactical_model", default=None,
                        help="Path to a trained tactical defender .zip; uses scripted otherwise.")
    parser.add_argument("--out", default="outputs/models/attacker_planner")
    parser.add_argument("--log", default="outputs/plots/attacker_training.csv")
    args = parser.parse_args()

    conf = load_config(args.config)
    env_cfg = EnvConfig(
        T=conf["env"]["T"],
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
    n_envs = args.n_envs or conf["training"]["attacker"]["n_envs"]
    timesteps = args.timesteps or conf["training"]["attacker"]["total_timesteps"]
    lr = conf["training"]["attacker"]["lr"]

    vec_env = make_vec_env(
        lambda i: make_env_fn(env_cfg, seed=2000 + i,
                              tactical_model=args.tactical_model),
        n_envs,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lr,
        n_steps=64,
        batch_size=64,
        n_epochs=6,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        verbose=1,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    cb = EpisodeStatsCallback(args.log, log_every=256)
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip")
    print(f"Training log written to {args.log}")


if __name__ == "__main__":
    main()
