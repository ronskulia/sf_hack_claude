"""Core multi-agent environment: drones vs road-based defenders around a city.

This module provides a low-level engine (``CityDefenseEnv``) with explicit
methods for each policy's decision point:

    reset(seed)                          -> initial state for attacker/deployment
    attacker_observation()               -> obs dict for attacker
    commit_attacker_plan(plan)           -> commits attack plan
    deployment_observation()             -> obs dict for deployment policy
    commit_deployment(placements)        -> seeds defender positions
    tactical_observation()               -> obs dict for tactical defender
    tactical_step(tactical_action)       -> advances the sim until next tactical
                                             decision (or episode end)

That low-level engine is the single source of truth. Thin gymnasium
wrappers around it (see ``envs.wrappers``) expose each policy as a
standard Gym env usable with Stable-Baselines3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dynamics import (
    DefenderState,
    DroneState,
    set_defender_target,
    step_defender,
    step_drone,
)
from .fixed_map import FixedMap, build_fixed_map
from .route_templates import (
    ROUTE_TEMPLATES,
    SPEED_MODE_MULTIPLIERS,
    sample_route_waypoints,
)


# -------------------- Config ------------------------------------ #

@dataclass
class EnvConfig:
    T: int = 180
    dt: float = 1.0
    n_drones: int = 10
    n_defenders: int = 5
    n_entry_points: int = 8
    n_route_templates: int = len(ROUTE_TEMPLATES)
    n_speed_modes: int = 3

    drone_max_speed: float = 0.020
    drone_max_accel: float = 0.010
    defender_road_speed: float = 0.012
    defender_intercept_radius: float = 0.035

    city_center: Tuple[float, float] = (0.5, 0.5)
    city_radius: float = 0.126
    # Optional override for the road ring layout. If None, build_fixed_map
    # derives ring radii from city_radius (default/compact map). Provide a
    # 3-tuple of absolute distances to get a more spacious layout.
    map_ring_radii: Optional[Tuple[float, float, float]] = None
    map_nodes_per_ring: Optional[Tuple[int, int, int]] = None

    tactical_decision_interval: int = 4
    tactical_candidate_neighbors: int = 6  # max neighbors considered per step
    # rewards
    r_reach_city_attacker: float = 10.0
    r_drone_destroyed_attacker: float = -5.0
    r_drone_destroyed_defender: float = 5.0
    r_reach_city_defender: float = -10.0
    defender_move_penalty: float = 0.0

    # partial observability for attacker
    defender_visible_prob: float = 0.5


# -------------------- Container dataclasses --------------------- #

@dataclass
class AttackPlan:
    """Per-drone structured attack plan."""
    launch_time: np.ndarray        # (n_drones,) int
    entry_point_id: np.ndarray     # (n_drones,) int
    route_template_id: np.ndarray  # (n_drones,) int
    speed_mode: np.ndarray         # (n_drones,) int  (0=slow, 1=normal, 2=fast)


@dataclass
class EpisodeEvent:
    t: int
    kind: str   # "destroyed" | "reached_city" | "launched"
    drone_id: int
    position: Tuple[float, float]


@dataclass
class EpisodeReplay:
    """Tight container used for visualization and eval."""
    drone_positions: List[np.ndarray] = field(default_factory=list)   # each (n_drones, 2)
    drone_alive: List[np.ndarray] = field(default_factory=list)       # each (n_drones,) bool
    defender_positions: List[np.ndarray] = field(default_factory=list)  # each (n_defenders, 2)
    drone_waypoints: Optional[List[np.ndarray]] = None                # per-drone full waypoints
    events: List[EpisodeEvent] = field(default_factory=list)
    attacker_plan: Optional[AttackPlan] = None
    # final outcome summary (filled at done)
    n_reached: int = 0
    n_destroyed: int = 0
    n_unreleased: int = 0


# --------------------------- Env --------------------------------- #

class CityDefenseEnv:
    """Low-level engine used by wrappers and by scripted evaluation.

    The engine deliberately exposes three commit points (attacker plan,
    defender deployment, tactical steps) instead of a single step() API,
    because the three policies act at different cadences.
    """

    def __init__(self, config: Optional[EnvConfig] = None, fmap: Optional[FixedMap] = None):
        self.cfg = config or EnvConfig()
        self.map: FixedMap = fmap or build_fixed_map(
            n_entry_points=self.cfg.n_entry_points,
            city_center=self.cfg.city_center,
            city_radius=self.cfg.city_radius,
            ring_radii=self.cfg.map_ring_radii,
            nodes_per_ring=self.cfg.map_nodes_per_ring,
        )
        self.rng = np.random.default_rng(0)
        self._precompute_candidate_lists()
        self.reset(seed=0)

    # -- setup helpers --------------------------------------------------- #

    def _precompute_candidate_lists(self) -> None:
        """For each node, store up to K closest neighbors, padded with the
        node itself. Used as tactical action space."""
        K = self.cfg.tactical_candidate_neighbors
        self.neighbor_lists: np.ndarray = np.zeros((self.map.n_nodes, K), dtype=np.int64)
        self.neighbor_mask: np.ndarray = np.zeros((self.map.n_nodes, K), dtype=np.float32)
        for i in range(self.map.n_nodes):
            nbrs = self.map.neighbors(i)
            # Always include "stay" as slot 0.
            options = [i] + nbrs[: K - 1]
            # pad
            while len(options) < K:
                options.append(i)
            self.neighbor_lists[i] = options
            for k, _ in enumerate(options):
                self.neighbor_mask[i, k] = 1.0

    # -- state ----------------------------------------------------------- #

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t: int = 0
        self.done: bool = False
        self.attacker_committed: bool = False
        self.deployment_committed: bool = False
        self.drones: List[DroneState] = []
        self.defenders: List[DefenderState] = []
        self.replay: EpisodeReplay = EpisodeReplay()
        self._defender_visibility_mask: Optional[np.ndarray] = None
        self._rewards: Dict[str, float] = {"attacker": 0.0, "defender": 0.0}
        self._pending_defender_deployment: Optional[np.ndarray] = None
        self._events_this_tactical: List[EpisodeEvent] = []

    # -- attacker -------------------------------------------------------- #

    def attacker_observation(self) -> Dict[str, Any]:
        """Obs for the attacker planner. Returns dict of arrays/scalars."""
        # Defender visibility is sampled once here, locked in for the episode.
        if self._defender_visibility_mask is None:
            self._defender_visibility_mask = (
                self.rng.random(self.cfg.n_defenders) < self.cfg.defender_visible_prob
            ).astype(np.float32)
        # For defender positions, use predicted deployment (which is not yet
        # known to us) — we fall back to a prior: a rough average of city
        # access node positions. Attacker sees visible defenders' positions
        # once they are actually deployed; otherwise a masked default.
        if self.deployment_committed:
            def_pos = np.array([d.position for d in self.defenders], dtype=np.float32)
        else:
            default = np.mean([self.map.node_xy(n) for n in self.map.deployment_nodes], axis=0)
            def_pos = np.tile(default.astype(np.float32), (self.cfg.n_defenders, 1))
        masked_def_pos = def_pos.copy()
        for i, visible in enumerate(self._defender_visibility_mask):
            if visible < 0.5:
                masked_def_pos[i] = -1.0  # sentinel for "masked/unseen"

        obs = {
            "city_center": np.array(self.cfg.city_center, dtype=np.float32),
            "city_radius": np.float32(self.cfg.city_radius),
            "node_positions": self.map.node_positions.astype(np.float32),
            "entry_points": self.map.entry_points.astype(np.float32),
            "defender_positions": masked_def_pos,
            "defender_visibility": self._defender_visibility_mask.copy(),
        }
        return obs

    def commit_attacker_plan(self, plan: AttackPlan) -> None:
        assert not self.attacker_committed, "Attacker plan already committed"
        # Instantiate 10 drones following the committed plan.
        self.drones = []
        n_entries = self.map.n_entry_points
        for i in range(self.cfg.n_drones):
            launch = int(np.clip(plan.launch_time[i], 0, self.cfg.T - 1))
            epid = int(plan.entry_point_id[i]) % n_entries
            rtid = int(plan.route_template_id[i]) % self.cfg.n_route_templates
            smid = int(plan.speed_mode[i]) % self.cfg.n_speed_modes
            entry = self.map.entry_points[epid]
            wp = sample_route_waypoints(rtid, entry, np.array(self.cfg.city_center))
            speed_mult = float(SPEED_MODE_MULTIPLIERS[smid])
            d = DroneState(
                drone_id=i,
                waypoints=wp,
                launch_time=launch,
                max_speed=self.cfg.drone_max_speed * speed_mult,
                max_accel=self.cfg.drone_max_accel * max(1.0, speed_mult),
                position=wp[0].copy(),
                velocity=np.zeros(2),
            )
            self.drones.append(d)
        self.replay.attacker_plan = plan
        self.replay.drone_waypoints = [d.waypoints.copy() for d in self.drones]
        self.attacker_committed = True

    # -- deployment ------------------------------------------------------ #

    def deployment_observation(self) -> Dict[str, Any]:
        return {
            "city_center": np.array(self.cfg.city_center, dtype=np.float32),
            "city_radius": np.float32(self.cfg.city_radius),
            "node_positions": self.map.node_positions.astype(np.float32),
            "deployment_nodes": np.asarray(self.map.deployment_nodes, dtype=np.int64),
        }

    def commit_deployment(self, node_indices: np.ndarray) -> None:
        """node_indices: (n_defenders,) indices into self.map.deployment_nodes."""
        assert self.attacker_committed, "Attacker must commit first (observation snap)"
        assert not self.deployment_committed
        self.defenders = []
        for k in range(self.cfg.n_defenders):
            idx = int(node_indices[k]) % len(self.map.deployment_nodes)
            node = self.map.deployment_nodes[idx]
            pos = self.map.node_xy(node).copy()
            dfn = DefenderState(
                defender_id=k,
                current_node=node,
                target_node=node,
                position=pos,
                edge_progress=0.0,
                speed=self.cfg.defender_road_speed,
                intercept_radius=self.cfg.defender_intercept_radius,
            )
            self.defenders.append(dfn)
        self.deployment_committed = True
        # Record initial frame.
        self._record_frame()

    # -- tactical -------------------------------------------------------- #

    def tactical_observation(self) -> Dict[str, Any]:
        drones_pos = np.zeros((self.cfg.n_drones, 2), dtype=np.float32)
        drones_vel = np.zeros((self.cfg.n_drones, 2), dtype=np.float32)
        drones_alive = np.zeros(self.cfg.n_drones, dtype=np.float32)
        drones_launched = np.zeros(self.cfg.n_drones, dtype=np.float32)
        for i, d in enumerate(self.drones):
            drones_pos[i] = d.position
            drones_vel[i] = d.velocity
            drones_alive[i] = 1.0 if d.alive else 0.0
            drones_launched[i] = 1.0 if d.launched else 0.0

        def_pos = np.zeros((self.cfg.n_defenders, 2), dtype=np.float32)
        def_node = np.zeros(self.cfg.n_defenders, dtype=np.int64)
        def_target = np.zeros(self.cfg.n_defenders, dtype=np.int64)
        for k, dfn in enumerate(self.defenders):
            def_pos[k] = dfn.position
            def_node[k] = dfn.current_node
            def_target[k] = dfn.target_node

        # Per-defender candidate action list (legal next nodes).
        candidates = np.zeros(
            (self.cfg.n_defenders, self.cfg.tactical_candidate_neighbors), dtype=np.int64
        )
        for k, dfn in enumerate(self.defenders):
            candidates[k] = self.neighbor_lists[dfn.current_node]

        return {
            "t": np.int64(self.t),
            "T": np.int64(self.cfg.T),
            "drones_pos": drones_pos,
            "drones_vel": drones_vel,
            "drones_alive": drones_alive,
            "drones_launched": drones_launched,
            "defender_pos": def_pos,
            "defender_node": def_node,
            "defender_target": def_target,
            "defender_candidates": candidates,
            "city_center": np.array(self.cfg.city_center, dtype=np.float32),
            "city_radius": np.float32(self.cfg.city_radius),
            "node_positions": self.map.node_positions.astype(np.float32),
        }

    def tactical_step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply tactical decisions then advance the sim by ``tactical_decision_interval``
        physical timesteps (or until episode end).

        ``action`` is an int array of length n_defenders: slot index into the
        candidate neighbor list for each defender.
        Returns:
            (next_tactical_obs, defender_reward_chunk, done, info)
        """
        assert self.deployment_committed, "Must commit deployment first"
        assert not self.done

        # Apply tactical decisions.
        for k, dfn in enumerate(self.defenders):
            slot = int(action[k]) % self.cfg.tactical_candidate_neighbors
            target_node = int(self.neighbor_lists[dfn.current_node, slot])
            set_defender_target(dfn, self.map, target_node)

        # Advance physical steps.
        interval = self.cfg.tactical_decision_interval
        defender_chunk_reward = 0.0
        self._events_this_tactical = []
        for _ in range(interval):
            if self.done:
                break
            self._physical_step()
            defender_chunk_reward += self._step_rewards["defender"]
            # slight movement penalty
            if self.cfg.defender_move_penalty != 0.0:
                moving = sum(1 for dfn in self.defenders if dfn.current_node != dfn.target_node)
                defender_chunk_reward -= self.cfg.defender_move_penalty * moving

        info = {
            "events": list(self._events_this_tactical),
            "t": self.t,
            "n_reached": self.replay.n_reached,
            "n_destroyed": self.replay.n_destroyed,
        }
        obs = self.tactical_observation()
        return obs, float(defender_chunk_reward), self.done, info

    # -- core physical step --------------------------------------------- #

    def _physical_step(self) -> None:
        """Advance one dt: drones move, defenders move, check interceptions & city entry."""
        if self.done:
            return
        self._step_rewards = {"attacker": 0.0, "defender": 0.0}
        # drones move
        city_center = np.array(self.cfg.city_center, dtype=np.float64)
        for d in self.drones:
            prev_launched = d.launched
            prev_reached = d.reached_city
            step_drone(d, self.t, self.cfg.dt, city_center, self.cfg.city_radius)
            if not prev_launched and d.launched:
                self._events_this_tactical.append(EpisodeEvent(
                    t=self.t, kind="launched", drone_id=d.drone_id,
                    position=(float(d.position[0]), float(d.position[1])),
                ))
            if not prev_reached and d.reached_city:
                self._step_rewards["attacker"] += self.cfg.r_reach_city_attacker
                self._step_rewards["defender"] += self.cfg.r_reach_city_defender
                self.replay.n_reached += 1
                self._events_this_tactical.append(EpisodeEvent(
                    t=self.t, kind="reached_city", drone_id=d.drone_id,
                    position=(float(d.position[0]), float(d.position[1])),
                ))

        # defenders move
        for dfn in self.defenders:
            step_defender(dfn, self.map, self.cfg.dt)

        # interception checks
        for d in self.drones:
            if not d.alive or d.destroyed or d.reached_city:
                continue
            for dfn in self.defenders:
                if np.linalg.norm(d.position - dfn.position) <= dfn.intercept_radius:
                    d.destroyed = True
                    d.alive = False
                    d.destroyed_at = self.t
                    self.replay.n_destroyed += 1
                    self._step_rewards["attacker"] += self.cfg.r_drone_destroyed_attacker
                    self._step_rewards["defender"] += self.cfg.r_drone_destroyed_defender
                    self._events_this_tactical.append(EpisodeEvent(
                        t=self.t, kind="destroyed", drone_id=d.drone_id,
                        position=(float(d.position[0]), float(d.position[1])),
                    ))
                    break

        self._rewards["attacker"] += self._step_rewards["attacker"]
        self._rewards["defender"] += self._step_rewards["defender"]
        self.replay.events.extend(self._events_this_tactical[-len(self._events_this_tactical):])

        self._record_frame()
        self.t += 1
        if self.t >= self.cfg.T:
            self.done = True
        # Also end early if every drone resolved.
        if all(d.destroyed or d.reached_city or (not d.launched and d.launch_time >= self.cfg.T)
               for d in self.drones):
            # everyone resolved or unlaunchable
            unresolved = sum(1 for d in self.drones
                             if not d.destroyed and not d.reached_city and d.launched)
            active_waiting = any(not d.launched and d.launch_time >= self.t for d in self.drones)
            if unresolved == 0 and not active_waiting:
                self.done = True
        if self.done:
            self.replay.n_unreleased = sum(
                1 for d in self.drones if not d.launched and not d.destroyed and not d.reached_city
            )

    def _record_frame(self) -> None:
        dpos = np.array([d.position for d in self.drones], dtype=np.float32)
        dalive = np.array([d.alive for d in self.drones], dtype=bool)
        fpos = np.array([f.position for f in self.defenders], dtype=np.float32)
        self.replay.drone_positions.append(dpos)
        self.replay.drone_alive.append(dalive)
        self.replay.defender_positions.append(fpos)

    # -- totals / utilities --------------------------------------------- #

    def rewards(self) -> Dict[str, float]:
        return dict(self._rewards)

    def summary(self) -> Dict[str, Any]:
        reached = sum(1 for d in self.drones if d.reached_city)
        destroyed = sum(1 for d in self.drones if d.destroyed)
        unreleased = sum(1 for d in self.drones
                         if not d.launched and not d.destroyed and not d.reached_city)
        return {
            "reached": reached,
            "destroyed": destroyed,
            "unreleased": unreleased,
            "attacker_reward": self._rewards["attacker"],
            "defender_reward": self._rewards["defender"],
            "t": self.t,
            "done": self.done,
        }
