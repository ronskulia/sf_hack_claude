"""A scripted attacker that picks plausible plans with some diversity.

Strategy:
 - spread launch times across the first ~1/3 of the episode so the fleet
   arrives in staggered waves
 - pick the entry points that face the city (all of ours do, since they're
   on the boundary, so we randomize among them)
 - bias route templates toward "direct" but include flank/late-turn/etc.
 - choose speed modes mostly "normal", occasionally "fast" to punish bad defense
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.city_defense_env import AttackPlan, EnvConfig


class ScriptedAttacker:
    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def plan(self, obs: Dict[str, Any]) -> AttackPlan:
        cfg = self.cfg
        n = cfg.n_drones
        # Spread launch times across the first 60% of the horizon.
        latest_launch = max(1, int(cfg.T * 0.6))
        launch_time = self.rng.integers(0, latest_launch, size=n)

        # Pick entry points uniformly.
        entry_point_id = self.rng.integers(0, cfg.n_entry_points, size=n)

        # Route templates: bias toward varied but useful templates.
        # weights: direct, left_flank, right_flank, wide_loop, late_turn, feint
        weights = np.array([0.25, 0.2, 0.2, 0.1, 0.15, 0.1])
        weights = weights[: cfg.n_route_templates]
        weights = weights / weights.sum()
        route_template_id = self.rng.choice(cfg.n_route_templates, size=n, p=weights)

        # Speed mostly normal.
        speed_mode = self.rng.choice(cfg.n_speed_modes, size=n, p=[0.2, 0.6, 0.2])

        return AttackPlan(
            launch_time=launch_time.astype(np.int64),
            entry_point_id=entry_point_id.astype(np.int64),
            route_template_id=route_template_id.astype(np.int64),
            speed_mode=speed_mode.astype(np.int64),
        )
