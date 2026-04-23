"""Continuous drone dynamics and defender-on-graph dynamics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .fixed_map import FixedMap


# ------------------------------- Drones ------------------------------------ #

@dataclass
class DroneState:
    drone_id: int
    waypoints: np.ndarray        # (K, 2)
    launch_time: int
    max_speed: float
    max_accel: float
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    wp_idx: int = 1              # index of the waypoint currently being tracked
    alive: bool = False          # becomes True at launch_time
    destroyed: bool = False
    reached_city: bool = False
    launched: bool = False
    destroyed_at: Optional[int] = None
    reached_at: Optional[int] = None


def step_drone(d: DroneState, t: int, dt: float, city_center: np.ndarray,
               city_radius: float) -> None:
    """Advance one drone one step. Idempotent if drone is not active."""
    if d.destroyed or d.reached_city:
        return
    if not d.launched:
        if t >= d.launch_time:
            d.launched = True
            d.alive = True
            d.position = d.waypoints[0].copy()
            d.velocity = np.zeros(2)
            d.wp_idx = 1
        else:
            return

    if d.wp_idx >= len(d.waypoints):
        target = d.waypoints[-1]
    else:
        target = d.waypoints[d.wp_idx]

    # Desired velocity points toward the current waypoint with a small
    # look-ahead effect for smoother curves.
    to_target = target - d.position
    dist = float(np.linalg.norm(to_target))
    if dist < 1e-6:
        d.wp_idx = min(d.wp_idx + 1, len(d.waypoints) - 1)
        to_target = d.waypoints[d.wp_idx] - d.position
        dist = float(np.linalg.norm(to_target))
        if dist < 1e-6:
            return

    desired_dir = to_target / dist
    desired_v = desired_dir * d.max_speed

    dv = desired_v - d.velocity
    dv_norm = float(np.linalg.norm(dv))
    max_dv = d.max_accel * dt
    if dv_norm > max_dv:
        dv = dv * (max_dv / dv_norm)
    d.velocity = d.velocity + dv

    # Clamp speed just in case.
    v_norm = float(np.linalg.norm(d.velocity))
    if v_norm > d.max_speed:
        d.velocity = d.velocity * (d.max_speed / v_norm)

    d.position = d.position + d.velocity * dt
    # Keep in map.
    d.position = np.clip(d.position, 0.0, 1.0)

    # Waypoint arrival radius scales a bit with step size.
    arrive_r = max(0.02, d.max_speed * dt * 1.5)
    if np.linalg.norm(target - d.position) < arrive_r:
        d.wp_idx = min(d.wp_idx + 1, len(d.waypoints))

    # Check city entry.
    if np.linalg.norm(d.position - city_center) <= city_radius:
        d.reached_city = True
        d.alive = False
        d.reached_at = t


# ------------------------------ Defenders --------------------------------- #

@dataclass
class DefenderState:
    defender_id: int
    current_node: int
    target_node: int
    position: np.ndarray         # continuous 2D
    edge_progress: float = 0.0   # [0, edge_length] along current (cur_node -> target_node)
    speed: float = 0.012
    intercept_radius: float = 0.035


def step_defender(dfn: DefenderState, fmap: FixedMap, dt: float) -> None:
    """Advance a defender along its current edge. When it reaches a node,
    it waits there (velocity drops to 0) until the tactical policy picks a
    new target_node."""
    if dfn.current_node == dfn.target_node:
        dfn.position = fmap.node_xy(dfn.current_node).copy()
        dfn.edge_progress = 0.0
        return

    if not fmap.graph.has_edge(dfn.current_node, dfn.target_node):
        # Invalid target; stay put.
        dfn.target_node = dfn.current_node
        dfn.position = fmap.node_xy(dfn.current_node).copy()
        dfn.edge_progress = 0.0
        return

    length = fmap.edge_length(dfn.current_node, dfn.target_node)
    max_speed = fmap.edge_max_speed(dfn.current_node, dfn.target_node)
    speed = min(dfn.speed, max_speed)
    dfn.edge_progress += speed * dt
    if dfn.edge_progress >= length:
        dfn.current_node = dfn.target_node
        dfn.position = fmap.node_xy(dfn.current_node).copy()
        dfn.edge_progress = 0.0
    else:
        frac = dfn.edge_progress / length
        a = fmap.node_xy(dfn.current_node)
        b = fmap.node_xy(dfn.target_node)
        dfn.position = a + frac * (b - a)


def set_defender_target(dfn: DefenderState, fmap: FixedMap, target_node: int) -> None:
    """Set a new target node. If the defender is mid-edge, we only accept
    adjacent nodes. If we're at a node, any neighbor is fine; non-neighbors
    are redirected via the next hop of the shortest path."""
    if dfn.current_node == dfn.target_node:
        # At an intersection: accept any target, route via shortest path.
        if target_node == dfn.current_node:
            return
        if fmap.graph.has_edge(dfn.current_node, target_node):
            dfn.target_node = target_node
            dfn.edge_progress = 0.0
        else:
            path = fmap.shortest_path(dfn.current_node, target_node)
            if len(path) >= 2:
                dfn.target_node = path[1]
                dfn.edge_progress = 0.0
    else:
        # Mid-edge: the only legal target changes are
        # continue to target_node, or reverse to current_node.
        if target_node == dfn.current_node:
            # reverse along the edge
            length = fmap.edge_length(dfn.current_node, dfn.target_node)
            dfn.current_node, dfn.target_node = dfn.target_node, dfn.current_node
            dfn.edge_progress = max(0.0, length - dfn.edge_progress)
        # else: ignore, keep current plan (will be re-decided at next node).
