"""Gymnasium-compatible wrappers around CityDefenseEnv for SB3 training.

Three wrappers, one per policy:

 - TacticalDefenderEnv:   per-tactical-step RL env (many steps per episode)
 - AttackerPlannerEnv:    one-shot RL env (1 step per episode, full plan committed)
 - DeploymentEnv:         one-shot RL env (1 step per episode, placement chosen)

Each wrapper internally owns a ``CityDefenseEnv`` and configures the other
two policies via scripted baselines or loaded SB3 models.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .city_defense_env import AttackPlan, CityDefenseEnv, EnvConfig


# ---------------- helpers for plan / deployment encoding ---------------- #

def plan_action_size(cfg: EnvConfig) -> np.ndarray:
    """For a MultiDiscrete with one component per (drone, field)."""
    # launch bins coarsened to avoid T*10 = 1800 discrete options.
    # We use 12 launch bins that map to [0, T*0.7] uniformly.
    launch_bins = 12
    return np.array(
        [launch_bins, cfg.n_entry_points, cfg.n_route_templates, cfg.n_speed_modes]
        * cfg.n_drones,
        dtype=np.int64,
    )


def decode_plan(action: np.ndarray, cfg: EnvConfig) -> AttackPlan:
    launch_bins = 12
    max_launch = int(cfg.T * 0.7)
    arr = np.asarray(action, dtype=np.int64).reshape(cfg.n_drones, 4)
    launch_time = (arr[:, 0] * (max_launch / launch_bins)).astype(np.int64)
    entry_point_id = arr[:, 1].astype(np.int64)
    route_template_id = arr[:, 2].astype(np.int64)
    speed_mode = arr[:, 3].astype(np.int64)
    return AttackPlan(
        launch_time=launch_time,
        entry_point_id=entry_point_id,
        route_template_id=route_template_id,
        speed_mode=speed_mode,
    )


def flatten_attacker_obs(raw_obs: Dict[str, Any], cfg: EnvConfig) -> np.ndarray:
    """Flatten the attacker observation into a single float32 vector."""
    parts = [
        raw_obs["city_center"].ravel(),
        np.array([raw_obs["city_radius"]], dtype=np.float32),
        raw_obs["entry_points"].ravel(),
        raw_obs["defender_positions"].ravel(),
        raw_obs["defender_visibility"].ravel(),
    ]
    return np.concatenate(parts).astype(np.float32)


def attacker_obs_dim(cfg: EnvConfig) -> int:
    # 2 + 1 + 2*E + 2*D + D
    return 2 + 1 + 2 * cfg.n_entry_points + 2 * cfg.n_defenders + cfg.n_defenders


def flatten_deployment_obs(raw_obs: Dict[str, Any], cfg: EnvConfig, n_nodes: int) -> np.ndarray:
    parts = [
        raw_obs["city_center"].ravel(),
        np.array([raw_obs["city_radius"]], dtype=np.float32),
        raw_obs["node_positions"].ravel(),
    ]
    return np.concatenate(parts).astype(np.float32)


def deployment_obs_dim(cfg: EnvConfig, n_nodes: int) -> int:
    return 2 + 1 + 2 * n_nodes


def flatten_tactical_obs(raw_obs: Dict[str, Any], cfg: EnvConfig) -> np.ndarray:
    """Flatten tactical defender observation into a single float32 vector."""
    t_norm = np.array([float(raw_obs["t"]) / float(cfg.T)], dtype=np.float32)
    parts = [
        t_norm,
        raw_obs["drones_pos"].ravel(),
        raw_obs["drones_vel"].ravel(),
        raw_obs["drones_alive"].ravel(),
        raw_obs["drones_launched"].ravel(),
        raw_obs["defender_pos"].ravel(),
        raw_obs["city_center"].ravel(),
    ]
    return np.concatenate(parts).astype(np.float32)


def tactical_obs_dim(cfg: EnvConfig) -> int:
    # t (1) + drones (2+2+1+1)*D + defenders (2)*F + city (2)
    return 1 + 6 * cfg.n_drones + 2 * cfg.n_defenders + 2


# --------------------- Tactical defender env --------------------------- #

class TacticalDefenderEnv(gym.Env):
    """Gym env for the tactical defender.

    attacker_fn:   callable(attacker_obs, cfg) -> AttackPlan
    deployment_fn: callable(deployment_obs, cfg, fmap) -> np.ndarray (n_defenders,)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        attacker_fn: Optional[Callable] = None,
        deployment_fn: Optional[Callable] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.engine = CityDefenseEnv(self.cfg)
        self._seed = seed

        # Default: scripted attacker + scripted deployment.
        if attacker_fn is None:
            from agents.scripted_attacker import ScriptedAttacker
            self._attacker = ScriptedAttacker(self.cfg, seed=seed)
            attacker_fn = lambda obs, cfg: self._attacker.plan(obs)
        if deployment_fn is None:
            from agents.scripted_defender_deployment import ScriptedDefenderDeployment
            self._deployment = ScriptedDefenderDeployment(self.cfg, self.engine.map)
            deployment_fn = lambda obs, cfg, fmap: self._deployment.deploy(obs)
        self._attacker_fn = attacker_fn
        self._deployment_fn = deployment_fn

        self.action_space = spaces.MultiDiscrete(
            [self.cfg.tactical_candidate_neighbors] * self.cfg.n_defenders
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(tactical_obs_dim(self.cfg),), dtype=np.float32
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None:
            seed = self._seed
            self._seed += 1
        self.engine.reset(seed=seed)
        atk_obs = self.engine.attacker_observation()
        plan = self._attacker_fn(atk_obs, self.cfg)
        self.engine.commit_attacker_plan(plan)
        dep_obs = self.engine.deployment_observation()
        placements = self._deployment_fn(dep_obs, self.cfg, self.engine.map)
        self.engine.commit_deployment(placements)
        obs = flatten_tactical_obs(self.engine.tactical_observation(), self.cfg)
        return obs, {}

    def step(self, action):
        obs_raw, reward, done, info = self.engine.tactical_step(np.asarray(action))
        obs = flatten_tactical_obs(obs_raw, self.cfg)
        truncated = False
        return obs, float(reward), bool(done), truncated, info


# ---------------------- Attacker planner env --------------------------- #

class AttackerPlannerEnv(gym.Env):
    """Gym env for the one-shot attacker planner.

    Each `step` commits a full attack plan, runs the episode with the
    provided defender policies, and returns the aggregated attacker reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        deployment_fn: Optional[Callable] = None,
        tactical_fn: Optional[Callable] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.engine = CityDefenseEnv(self.cfg)
        self._seed = seed

        if deployment_fn is None:
            from agents.scripted_defender_deployment import ScriptedDefenderDeployment
            self._deployment = ScriptedDefenderDeployment(self.cfg, self.engine.map)
            deployment_fn = lambda obs, cfg, fmap: self._deployment.deploy(obs)
        if tactical_fn is None:
            from agents.scripted_defender_tactical import ScriptedDefenderTactical
            self._tactical = ScriptedDefenderTactical(self.cfg, self.engine.map)
            tactical_fn = lambda obs, cfg: self._tactical.act(obs)
        self._deployment_fn = deployment_fn
        self._tactical_fn = tactical_fn

        self.action_space = spaces.MultiDiscrete(plan_action_size(self.cfg))
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(attacker_obs_dim(self.cfg),), dtype=np.float32
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None:
            seed = self._seed
            self._seed += 1
        self.engine.reset(seed=seed)
        atk_obs_raw = self.engine.attacker_observation()
        self._current_atk_obs = atk_obs_raw
        obs = flatten_attacker_obs(atk_obs_raw, self.cfg)
        return obs, {}

    def step(self, action):
        plan = decode_plan(np.asarray(action), self.cfg)
        self.engine.commit_attacker_plan(plan)
        dep_obs = self.engine.deployment_observation()
        placements = self._deployment_fn(dep_obs, self.cfg, self.engine.map)
        self.engine.commit_deployment(placements)
        while not self.engine.done:
            tac_obs = self.engine.tactical_observation()
            a = self._tactical_fn(tac_obs, self.cfg)
            self.engine.tactical_step(np.asarray(a))
        reward = self.engine.rewards()["attacker"]
        info = self.engine.summary()
        # Episode terminates immediately; return zero observation for next reset.
        obs = flatten_attacker_obs(self._current_atk_obs, self.cfg)
        return obs, float(reward), True, False, info


# ----------------------- Deployment env -------------------------------- #

class DeploymentEnv(gym.Env):
    """Gym env for the one-shot defender deployment policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        attacker_fn: Optional[Callable] = None,
        tactical_fn: Optional[Callable] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.engine = CityDefenseEnv(self.cfg)
        self._seed = seed

        if attacker_fn is None:
            from agents.scripted_attacker import ScriptedAttacker
            self._attacker = ScriptedAttacker(self.cfg, seed=seed)
            attacker_fn = lambda obs, cfg: self._attacker.plan(obs)
        if tactical_fn is None:
            from agents.scripted_defender_tactical import ScriptedDefenderTactical
            self._tactical = ScriptedDefenderTactical(self.cfg, self.engine.map)
            tactical_fn = lambda obs, cfg: self._tactical.act(obs)
        self._attacker_fn = attacker_fn
        self._tactical_fn = tactical_fn

        self.n_deployment_nodes = len(self.engine.map.deployment_nodes)
        self.action_space = spaces.MultiDiscrete(
            [self.n_deployment_nodes] * self.cfg.n_defenders
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0,
            shape=(deployment_obs_dim(self.cfg, self.engine.map.n_nodes),),
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None:
            seed = self._seed
            self._seed += 1
        self.engine.reset(seed=seed)
        atk_obs = self.engine.attacker_observation()
        plan = self._attacker_fn(atk_obs, self.cfg)
        self.engine.commit_attacker_plan(plan)
        dep_obs_raw = self.engine.deployment_observation()
        self._current_dep_obs = dep_obs_raw
        obs = flatten_deployment_obs(dep_obs_raw, self.cfg, self.engine.map.n_nodes)
        return obs, {}

    def step(self, action):
        placements = np.asarray(action, dtype=np.int64)
        self.engine.commit_deployment(placements)
        while not self.engine.done:
            tac_obs = self.engine.tactical_observation()
            a = self._tactical_fn(tac_obs, self.cfg)
            self.engine.tactical_step(np.asarray(a))
        reward = self.engine.rewards()["defender"]
        info = self.engine.summary()
        obs = flatten_deployment_obs(self._current_dep_obs, self.cfg, self.engine.map.n_nodes)
        return obs, float(reward), True, False, info
