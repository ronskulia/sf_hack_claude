"""Common utilities used by the train_* scripts."""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_vec_env(make_env_fn: Callable, n_envs: int, use_subproc: bool = False):
    if use_subproc and n_envs > 1:
        return SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])
    return DummyVecEnv([make_env_fn(i) for i in range(n_envs)])


class EpisodeStatsCallback(BaseCallback):
    """Logs mean reached/destroyed counts from the env's info dict.

    Writes a CSV at log_path: step, mean_ep_reward, mean_destroyed, mean_reached.
    """

    def __init__(self, log_path: str, log_every: int = 2048):
        super().__init__()
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        # Reset file with header.
        with open(self.log_path, "w") as f:
            f.write("step,mean_ep_reward,mean_destroyed,mean_reached\n")
        self.log_every = log_every
        self._buf_reward = []
        self._buf_destroyed = []
        self._buf_reached = []
        self._last_log = 0

    def _on_step(self) -> bool:
        # Monitor wrapper injects {"episode": {"r": total_r, "l": length}} on
        # terminal steps. Fall back to the raw reward/info keys otherwise.
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for i, d in enumerate(dones):
            if d:
                info = infos[i] if i < len(infos) else {}
                ep = info.get("episode")
                if ep is not None and "r" in ep:
                    self._buf_reward.append(float(ep["r"]))
                else:
                    self._buf_reward.append(float(rewards[i]))
                dest = info.get("destroyed", info.get("n_destroyed"))
                reached = info.get("reached", info.get("n_reached"))
                if dest is not None:
                    self._buf_destroyed.append(float(dest))
                if reached is not None:
                    self._buf_reached.append(float(reached))
        if self.num_timesteps - self._last_log >= self.log_every and self._buf_reward:
            mean_r = float(np.mean(self._buf_reward))
            mean_d = float(np.mean(self._buf_destroyed)) if self._buf_destroyed else float("nan")
            mean_c = float(np.mean(self._buf_reached)) if self._buf_reached else float("nan")
            with open(self.log_path, "a") as f:
                f.write(f"{self.num_timesteps},{mean_r:.4f},{mean_d:.4f},{mean_c:.4f}\n")
            self._last_log = self.num_timesteps
            self._buf_reward.clear()
            self._buf_destroyed.clear()
            self._buf_reached.clear()
        return True
