"""Small library of route templates for the attacker planner.

Each template turns (entry_point, city_center) into a list of 2D waypoints.
A drone then follows these waypoints via simple point-mass dynamics.
Keeping the attacker action discrete over a small library makes training
tractable while still producing visually distinct attacks.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


Waypoints = np.ndarray  # (K, 2)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _perp(v: np.ndarray) -> np.ndarray:
    """Right-hand perpendicular of a 2D vector."""
    return np.array([v[1], -v[0]], dtype=np.float64)


def _direct(entry: np.ndarray, city: np.ndarray) -> Waypoints:
    mid = entry + 0.5 * (city - entry)
    return np.stack([entry, mid, city])


def _flank(entry: np.ndarray, city: np.ndarray, side: float) -> Waypoints:
    """Curved flank approach. side=+1 right-flank, -1 left-flank."""
    direction = _unit(city - entry)
    perp = _perp(direction) * side
    mid1 = entry + 0.33 * (city - entry) + 0.18 * perp
    mid2 = entry + 0.66 * (city - entry) + 0.10 * perp
    return np.stack([entry, mid1, mid2, city])


def _wide_loop(entry: np.ndarray, city: np.ndarray) -> Waypoints:
    """Wide circling approach that arrives from a tangential direction."""
    direction = _unit(city - entry)
    perp = _perp(direction)
    p1 = entry + 0.25 * (city - entry) + 0.28 * perp
    p2 = entry + 0.55 * (city - entry) + 0.32 * perp
    p3 = entry + 0.85 * (city - entry) + 0.12 * perp
    return np.stack([entry, p1, p2, p3, city])


def _late_turn(entry: np.ndarray, city: np.ndarray) -> Waypoints:
    """Fly straight, then turn sharply into the city near the end."""
    direction = _unit(city - entry)
    perp = _perp(direction)
    p1 = entry + 0.7 * (city - entry) + 0.02 * perp
    p2 = entry + 0.85 * (city - entry) - 0.18 * perp
    return np.stack([entry, p1, p2, city])


def _feint(entry: np.ndarray, city: np.ndarray) -> Waypoints:
    """Head toward one side, then swing back across to the city."""
    direction = _unit(city - entry)
    perp = _perp(direction)
    p1 = entry + 0.35 * (city - entry) + 0.22 * perp
    p2 = entry + 0.55 * (city - entry) - 0.18 * perp
    p3 = entry + 0.8 * (city - entry) - 0.05 * perp
    return np.stack([entry, p1, p2, p3, city])


ROUTE_TEMPLATES: List[Tuple[str, Callable[[np.ndarray, np.ndarray], Waypoints]]] = [
    ("direct",      lambda e, c: _direct(e, c)),
    ("left_flank",  lambda e, c: _flank(e, c, side=-1.0)),
    ("right_flank", lambda e, c: _flank(e, c, side=+1.0)),
    ("wide_loop",   lambda e, c: _wide_loop(e, c)),
    ("late_turn",   lambda e, c: _late_turn(e, c)),
    ("feint",       lambda e, c: _feint(e, c)),
]


def sample_route_waypoints(template_id: int, entry: np.ndarray, city: np.ndarray) -> Waypoints:
    template_id = int(template_id) % len(ROUTE_TEMPLATES)
    name, fn = ROUTE_TEMPLATES[template_id]
    wp = fn(np.asarray(entry, dtype=np.float64), np.asarray(city, dtype=np.float64))
    # Clip to map bounds for safety.
    wp = np.clip(wp, 0.01, 0.99)
    return wp


def route_template_names() -> List[str]:
    return [name for name, _ in ROUTE_TEMPLATES]


SPEED_MODE_MULTIPLIERS = np.array([0.6, 1.0, 1.3], dtype=np.float64)  # slow, normal, fast


if __name__ == "__main__":
    entry = np.array([0.02, 0.5])
    city = np.array([0.5, 0.5])
    for tid, (name, _) in enumerate(ROUTE_TEMPLATES):
        wp = sample_route_waypoints(tid, entry, city)
        print(f"{tid} {name}: {wp.shape} first={wp[0]} last={wp[-1]}")
