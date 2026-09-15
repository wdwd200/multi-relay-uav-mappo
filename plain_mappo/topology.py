"""Frozen-information topology features shared by P3 and P4 Actors.

The environment remains the source of all dynamics and communication.  This
module consumes only its documented read-only global-state interface and calls
the existing :func:`relay_env.communication.link_metrics` implementation for
logical-hop capacity.  It deliberately excludes privileged Critic-only fields
such as waypoints, cruise speeds, step/time, and outage counters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from relay_env.communication import link_metrics


TOPOLOGY_NODE_DIM = 6
TOPOLOGY_EDGE_DIM = 7
GRAPH_HIDDEN_DIM = 32
CAPACITY_NORMALIZER_BPS = 60_000_000.0


def topology_feature_shapes(num_relays: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``([K+2, 6], [K+1, 7])`` for the fixed logical chain."""
    if num_relays < 1:
        raise ValueError("topology requires at least one relay")
    return (num_relays + 2, TOPOLOGY_NODE_DIM), (num_relays + 1, TOPOLOGY_EDGE_DIM)


def build_topology_features(env: Any, state: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized nodes and directed logical-hop edges from pre-action state.

    Node order is ``[H, R1, ..., RK, L]``.  Edge order is the fixed logical
    communication chain, with deltas always ``downstream - upstream``.  Passing
    an already-read ``state`` lets rollout collection use exactly the same
    action-predecessor state for the Critic flattening and topology features.
    """
    global_state = env.get_global_state() if state is None else state
    num_relays = int(env.num_relays)
    node_shape, edge_shape = topology_feature_shapes(num_relays)
    relays = tuple(global_state["relays"])
    if len(relays) != num_relays:
        raise ValueError("global-state relay count does not match environment")
    physical_nodes = (global_state["H"],) + relays + (global_state["L"],)
    config = env.config
    nodes = np.empty(node_shape, dtype=np.float32)
    for index, node in enumerate(physical_nodes):
        position = tuple(float(value) for value in node["position"])
        velocity = tuple(float(value) for value in node["velocity"])
        nodes[index] = (position[0] / config.map_x_m, position[1] / config.map_y_m,
                        (position[2] - 200.0) / 100.0,
                        velocity[0] / 20.0, velocity[1] / 20.0, velocity[2] / 6.0)
    edges = np.empty(edge_shape, dtype=np.float32)
    communication = config.comm
    for index, (upstream, downstream) in enumerate(zip(physical_nodes[:-1], physical_nodes[1:])):
        up_position, down_position = upstream["position"], downstream["position"]
        up_velocity, down_velocity = upstream["velocity"], downstream["velocity"]
        capacity = link_metrics(up_position, down_position, communication.reference_gain,
                                communication.path_loss_exponent, communication.tx_power_w,
                                communication.noise_density_w_hz, communication.bandwidth_hz)["capacity_bps"]
        edges[index] = ((down_position[0] - up_position[0]) / 2000.0,
                        (down_position[1] - up_position[1]) / 2000.0,
                        (down_position[2] - up_position[2]) / 200.0,
                        (down_velocity[0] - up_velocity[0]) / 40.0,
                        (down_velocity[1] - up_velocity[1]) / 40.0,
                        (down_velocity[2] - up_velocity[2]) / 12.0,
                        min(float(capacity) / CAPACITY_NORMALIZER_BPS, 1.0))
    if not (np.isfinite(nodes).all() and np.isfinite(edges).all()):
        raise FloatingPointError("non-finite topology feature")
    return nodes, edges
