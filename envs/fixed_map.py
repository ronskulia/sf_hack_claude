"""Fixed 2D map: central city + road network + boundary entry points.

Everything lives in the unit square [0,1]x[0,1]. The map is hand-designed
so that behavior is reproducible and visually convincing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import networkx as nx
import numpy as np


@dataclass
class FixedMap:
    city_center: Tuple[float, float]
    city_radius: float
    graph: nx.Graph
    node_positions: np.ndarray         # (N, 2) float
    edges: List[Tuple[int, int]]
    entry_points: np.ndarray           # (E, 2) float, on map boundary
    deployment_nodes: List[int]        # subset of nodes allowed for initial deployment
    city_access_nodes: List[int] = field(default_factory=list)  # road nodes touching the city

    @property
    def n_nodes(self) -> int:
        return self.node_positions.shape[0]

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_entry_points(self) -> int:
        return self.entry_points.shape[0]

    def node_xy(self, node: int) -> np.ndarray:
        return self.node_positions[node]

    def neighbors(self, node: int) -> List[int]:
        return list(self.graph.neighbors(node))

    def edge_length(self, u: int, v: int) -> float:
        return float(self.graph[u][v]["length"])

    def edge_max_speed(self, u: int, v: int) -> float:
        return float(self.graph[u][v]["max_speed"])

    def shortest_path(self, src: int, dst: int) -> List[int]:
        return nx.shortest_path(self.graph, src, dst, weight="length")


def _add_edge(g: nx.Graph, u: int, v: int, pos: np.ndarray, max_speed: float = 0.012):
    length = float(np.linalg.norm(pos[u] - pos[v]))
    g.add_edge(u, v, length=length, max_speed=max_speed)


def build_fixed_map(
    n_entry_points: int = 8,
    city_center: Tuple[float, float] = (0.5, 0.5),
    city_radius: float = 0.126,
) -> FixedMap:
    """Build a deterministic map: a central city with a ring road, two inner
    rings, radial spokes, and a few cross-links to create alternative routes.
    ~30 nodes and ~50 edges.
    """
    cx, cy = city_center

    # Build nodes in concentric rings around the city.
    ring_radii = [city_radius * 1.15,   # inner ring close to city
                  city_radius * 1.9,    # mid ring
                  city_radius * 2.9]    # outer ring (still inside map)
    nodes_per_ring = [8, 12, 8]

    positions: List[Tuple[float, float]] = []
    ring_node_indices: List[List[int]] = []

    for r, n in zip(ring_radii, nodes_per_ring):
        ring_idx = []
        # offset each ring slightly so nodes don't all lie on the same spokes
        offset = 0.0 if len(ring_node_indices) == 0 else math.pi / n
        for k in range(n):
            angle = offset + 2 * math.pi * k / n
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            x = float(np.clip(x, 0.03, 0.97))
            y = float(np.clip(y, 0.03, 0.97))
            ring_idx.append(len(positions))
            positions.append((x, y))
        ring_node_indices.append(ring_idx)

    pos_arr = np.array(positions, dtype=np.float64)

    g = nx.Graph()
    for i, p in enumerate(positions):
        g.add_node(i, pos=p)

    # Ring edges
    for ring in ring_node_indices:
        n = len(ring)
        for k in range(n):
            u = ring[k]
            v = ring[(k + 1) % n]
            _add_edge(g, u, v, pos_arr, max_speed=0.012)

    # Radial / spoke edges connecting each ring to the next
    # Connect each node of inner ring to its ~nearest node on the next ring.
    for inner, outer in zip(ring_node_indices, ring_node_indices[1:]):
        for i in inner:
            # nearest 2 nodes on the outer ring
            dists = [float(np.linalg.norm(pos_arr[i] - pos_arr[j])) for j in outer]
            order = np.argsort(dists)
            for k in order[:2]:
                j = outer[k]
                if not g.has_edge(i, j):
                    _add_edge(g, i, j, pos_arr, max_speed=0.014)

    # A handful of diagonal cross-links among the mid ring for extra variety.
    mid = ring_node_indices[1]
    for k in range(0, len(mid), 3):
        u = mid[k]
        v = mid[(k + len(mid) // 2) % len(mid)]
        if not g.has_edge(u, v):
            _add_edge(g, u, v, pos_arr, max_speed=0.010)

    # Entry points on map boundary. Evenly spaced around the perimeter,
    # placed exactly on the border of the unit square.
    entry_points = []
    for k in range(n_entry_points):
        theta = 2 * math.pi * k / n_entry_points + math.pi / n_entry_points
        dx, dy = math.cos(theta), math.sin(theta)
        # project ray from city center onto the unit square border
        t_candidates = []
        if dx > 0:
            t_candidates.append((1.0 - cx) / dx)
        elif dx < 0:
            t_candidates.append((0.0 - cx) / dx)
        if dy > 0:
            t_candidates.append((1.0 - cy) / dy)
        elif dy < 0:
            t_candidates.append((0.0 - cy) / dy)
        t = min(t for t in t_candidates if t > 0)
        ex = cx + t * dx
        ey = cy + t * dy
        # pull slightly inside so drones can move before plot clipping
        ex = float(np.clip(ex, 0.01, 0.99))
        ey = float(np.clip(ey, 0.01, 0.99))
        entry_points.append((ex, ey))

    entry_points = np.array(entry_points, dtype=np.float64)

    # deployment nodes = inner + mid ring (closer to city so defenders matter)
    deployment_nodes = sorted(ring_node_indices[0] + ring_node_indices[1])

    # city access nodes = any node inside city_radius * 1.3
    city_access_nodes = [i for i, p in enumerate(positions)
                         if np.linalg.norm(np.array(p) - np.array(city_center)) <= city_radius * 1.3]

    edges = list(g.edges())
    return FixedMap(
        city_center=city_center,
        city_radius=city_radius,
        graph=g,
        node_positions=pos_arr,
        edges=edges,
        entry_points=entry_points,
        deployment_nodes=deployment_nodes,
        city_access_nodes=city_access_nodes,
    )


if __name__ == "__main__":
    m = build_fixed_map()
    print(f"Map built: {m.n_nodes} nodes, {m.n_edges} edges, {m.n_entry_points} entries")
    print(f"Deployment nodes: {len(m.deployment_nodes)} candidates")
