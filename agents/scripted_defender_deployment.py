"""Scripted deployment: place defenders at high-degree nodes near the city.

Heuristic:
 - rank deployment_nodes by (graph degree) * (1 / distance_to_city).
 - pick the top ones, with a minimum angular separation so defenders
   cover different sectors around the city.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from envs.city_defense_env import EnvConfig
from envs.fixed_map import FixedMap


class ScriptedDefenderDeployment:
    def __init__(self, cfg: EnvConfig, fmap: FixedMap):
        self.cfg = cfg
        self.map = fmap

    def deploy(self, obs: Dict[str, Any]) -> np.ndarray:
        fmap = self.map
        cc = np.array(self.cfg.city_center, dtype=np.float64)
        deployment_nodes: List[int] = fmap.deployment_nodes
        scores = []
        angles = []
        for n in deployment_nodes:
            p = fmap.node_xy(n)
            d = float(np.linalg.norm(p - cc)) + 1e-6
            deg = fmap.graph.degree(n)
            scores.append(deg * (1.0 / d))
            angles.append(np.arctan2(p[1] - cc[1], p[0] - cc[0]))
        scores = np.array(scores)
        angles = np.array(angles)

        # Greedy pick with angular separation constraint.
        picked: List[int] = []
        picked_angles: List[float] = []
        min_sep = 2 * np.pi / (self.cfg.n_defenders + 1)
        order = np.argsort(-scores)  # highest score first
        for idx in order:
            if len(picked) >= self.cfg.n_defenders:
                break
            ang = float(angles[idx])
            ok = True
            for a in picked_angles:
                # circular distance
                diff = abs((ang - a + np.pi) % (2 * np.pi) - np.pi)
                if diff < min_sep:
                    ok = False
                    break
            if ok:
                picked.append(int(idx))
                picked_angles.append(ang)

        # Fall back: fill remaining by score if separation constraint was too strict.
        if len(picked) < self.cfg.n_defenders:
            for idx in order:
                if len(picked) >= self.cfg.n_defenders:
                    break
                if int(idx) not in picked:
                    picked.append(int(idx))

        return np.array(picked[: self.cfg.n_defenders], dtype=np.int64)
