"""Very small alternating self-play loop.

For each iteration:
  1. train tactical defender for ``tactical_steps`` steps vs current attacker;
  2. train attacker for ``attacker_steps`` steps vs current tactical defender.

This is a prototype-grade loop. No opponent checkpoint buffer: the latest
snapshot is used. It's here to demonstrate the idea, not as a serious
training regime.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs import AttackerPlannerEnv, EnvConfig, TacticalDefenderEnv
from envs.wrappers import flatten_attacker_obs, flatten_tactical_obs
from training.common import make_vec_env


def _tactical_fn_from_model(model: PPO):
    def fn(tac_obs, cfg_):
        obs_vec = flatten_tactical_obs(tac_obs, cfg_)
        action, _ = model.predict(obs_vec, deterministic=True)
        return action
    return fn


def _attacker_fn_from_model(model: PPO):
    from envs.wrappers import decode_plan

    def fn(atk_obs, cfg_):
        obs_vec = flatten_attacker_obs(atk_obs, cfg_)
        action, _ = model.predict(obs_vec, deterministic=True)
        return decode_plan(action, cfg_)
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--tactical_steps", type=int, default=50_000)
    parser.add_argument("--attacker_steps", type=int, default=15_000)
    parser.add_argument("--tac_out", default="outputs/models/sp_tactical")
    parser.add_argument("--atk_out", default="outputs/models/sp_attacker")
    args = parser.parse_args()

    cfg = EnvConfig()

    # Start with scripted opponents; after each training stage replace with latest model.
    tactical_model: PPO | None = None
    attacker_model: PPO | None = None

    for it in range(args.iterations):
        print(f"=== self-play iteration {it} ===")

        # 1) Train tactical defender.
        def make_tac(i):
            def _thunk():
                atk_fn = _attacker_fn_from_model(attacker_model) if attacker_model else None
                env = TacticalDefenderEnv(cfg=cfg, seed=4000 + it * 37 + i,
                                          attacker_fn=atk_fn)
                return Monitor(env)
            return _thunk
        vec_env = make_vec_env(make_tac, n_envs=4)
        if tactical_model is None:
            tactical_model = PPO("MlpPolicy", vec_env, verbose=0,
                                  policy_kwargs=dict(net_arch=[128, 128]),
                                  n_steps=256, batch_size=256)
        else:
            tactical_model.set_env(vec_env)
        tactical_model.learn(total_timesteps=args.tactical_steps)
        tactical_model.save(f"{args.tac_out}_iter{it}")

        # 2) Train attacker.
        def make_atk(i):
            def _thunk():
                tac_fn = _tactical_fn_from_model(tactical_model)
                env = AttackerPlannerEnv(cfg=cfg, seed=5000 + it * 37 + i,
                                         tactical_fn=tac_fn)
                return Monitor(env)
            return _thunk
        vec_env = make_vec_env(make_atk, n_envs=4)
        if attacker_model is None:
            attacker_model = PPO("MlpPolicy", vec_env, verbose=0,
                                  policy_kwargs=dict(net_arch=[128, 128]),
                                  n_steps=64, batch_size=64)
        else:
            attacker_model.set_env(vec_env)
        attacker_model.learn(total_timesteps=args.attacker_steps)
        attacker_model.save(f"{args.atk_out}_iter{it}")

    print("Self-play done.")


if __name__ == "__main__":
    main()
