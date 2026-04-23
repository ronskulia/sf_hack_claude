"""Scripted tactical defender: greedy nearest-threat interception on roads.

At each decision:
 1. Predict a short-horizon future position for each live drone (pos + vel*h).
 2. Assign defenders to the closest unassigned drone (Hungarian-free greedy).
 3. For each defender, pick the candidate neighbor that most reduces
    straight-line distance to its assigned drone's predicted position.
 4. If no live drones, drift toward city access nodes (stand ready).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from envs.city_defense_env import EnvConfig
from envs.fixed_map import FixedMap


class ScriptedDefenderTactical:
    def __init__(self, cfg: EnvConfig, fmap: FixedMap, lookahead: float = 6.0):
        self.cfg = cfg
        self.map = fmap
        self.lookahead = lookahead  # seconds
        # Precompute neighbor lists same way the env does.
        K = cfg.tactical_candidate_neighbors
        self.neighbor_lists = np.zeros((fmap.n_nodes, K), dtype=np.int64)
        for i in range(fmap.n_nodes):
            nbrs = fmap.neighbors(i)
            options = [i] + nbrs[: K - 1]
            while len(options) < K:
                options.append(i)
            self.neighbor_lists[i] = options

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        cfg = self.cfg
        n_def = cfg.n_defenders

        alive_mask = obs["drones_alive"].astype(bool)
        drone_pos = obs["drones_pos"]
        drone_vel = obs["drones_vel"]

        def_pos = obs["defender_pos"]
        def_node = obs["defender_node"]

        if not np.any(alive_mask):
            # No threats — drift each defender toward the nearest city-access node.
            actions = np.zeros(n_def, dtype=np.int64)
            for k in range(n_def):
                best_slot, best_d = 0, np.inf
                cur_pos = def_pos[k]
                target_pts = [self.map.node_xy(n) for n in self.map.city_access_nodes] or [
                    np.array(cfg.city_center)
                ]
                target = min(target_pts, key=lambda p: float(np.linalg.norm(p - cur_pos)))
                for slot in range(cfg.tactical_candidate_neighbors):
                    nxt = int(self.neighbor_lists[def_node[k], slot])
                    nxt_xy = self.map.node_xy(nxt)
                    d = float(np.linalg.norm(nxt_xy - target))
                    if d < best_d:
                        best_d = d
                        best_slot = slot
                actions[k] = best_slot
            return actions

        # Project drones forward for assignment.
        future = drone_pos + drone_vel * self.lookahead
        actions = np.zeros(n_def, dtype=np.int64)

        alive_idx = np.where(alive_mask)[0]
        # Greedy assignment: sort defender-drone pairs by distance ascending.
        pairs = []
        for k in range(n_def):
            for j in alive_idx:
                d = float(np.linalg.norm(def_pos[k] - future[j]))
                pairs.append((d, k, int(j)))
        pairs.sort()
        assigned_def: set = set()
        assigned_drone: set = set()
        assignment: Dict[int, int] = {}
        for d, k, j in pairs:
            if k in assigned_def or j in assigned_drone:
                continue
            assignment[k] = j
            assigned_def.add(k)
            assigned_drone.add(j)
            if len(assigned_def) == n_def:
                break
        # Any remaining defenders assign to the closest threat (duplicates allowed).
        for k in range(n_def):
            if k in assigned_def:
                continue
            best_j = min(alive_idx, key=lambda j: float(np.linalg.norm(def_pos[k] - future[j])))
            assignment[k] = int(best_j)

        # Pick best candidate neighbor for each defender.
        for k in range(n_def):
            target_point = future[assignment[k]]
            best_slot, best_d = 0, np.inf
            for slot in range(cfg.tactical_candidate_neighbors):
                nxt = int(self.neighbor_lists[def_node[k], slot])
                nxt_xy = self.map.node_xy(nxt)
                d = float(np.linalg.norm(nxt_xy - target_point))
                if d < best_d:
                    best_d = d
                    best_slot = slot
            actions[k] = best_slot

        return actions
