"""CPU rollout storage and boundary-safe team GAE for Plain MAPPO."""

from __future__ import annotations

from typing import Any

import numpy as np


class RolloutBuffer:
    """Fixed `[time, env, ...]` CPU buffer; advantages are team-time values."""

    def __init__(self, rollout_length: int, num_envs: int, num_relays: int, obs_dim: int,
                 state_dim: int, action_dim: int, topology_node_dim: int | None = None,
                 topology_edge_dim: int | None = None) -> None:
        self.rollout_length, self.num_envs = int(rollout_length), int(num_envs)
        self.num_relays, self.obs_dim, self.state_dim, self.action_dim = int(num_relays), int(obs_dim), int(state_dim), int(action_dim)
        shape = (self.rollout_length, self.num_envs)
        self.local_obs = np.zeros(shape + (self.num_relays, self.obs_dim), dtype=np.float32)
        if (topology_node_dim is None) != (topology_edge_dim is None):
            raise ValueError("topology node and edge dimensions must be both present or both absent")
        self.topology_node_dim = None if topology_node_dim is None else int(topology_node_dim)
        self.topology_edge_dim = None if topology_edge_dim is None else int(topology_edge_dim)
        self.topology_nodes = (None if self.topology_node_dim is None else
                               np.zeros(shape + (self.num_relays + 2, self.topology_node_dim), dtype=np.float32))
        self.topology_edges = (None if self.topology_edge_dim is None else
                               np.zeros(shape + (self.num_relays + 1, self.topology_edge_dim), dtype=np.float32))
        self.global_state = np.zeros(shape + (self.state_dim,), dtype=np.float32)
        self.latent_z = np.zeros(shape + (self.num_relays, self.action_dim), dtype=np.float32)
        self.action_u = np.zeros_like(self.latent_z)
        self.old_log_prob = np.zeros(shape + (self.num_relays,), dtype=np.float32)
        self.value = np.zeros(shape, dtype=np.float32)
        self.next_value = np.zeros(shape, dtype=np.float32)
        self.reward = np.zeros(shape, dtype=np.float32)
        self.terminated = np.zeros(shape, dtype=np.bool_)
        self.truncated = np.zeros(shape, dtype=np.bool_)
        self.bootstrap_mask = np.zeros(shape, dtype=np.float32)
        self.trace_mask = np.zeros(shape, dtype=np.float32)
        self.advantage = np.zeros(shape, dtype=np.float32)
        self.return_target = np.zeros(shape, dtype=np.float32)
        self.position = 0

    def add(self, *, local_obs: np.ndarray, global_state: np.ndarray, latent_z: np.ndarray,
            action_u: np.ndarray, old_log_prob: np.ndarray, value: np.ndarray,
            next_value: np.ndarray, reward: np.ndarray, terminated: np.ndarray,
            truncated: np.ndarray, topology_nodes: np.ndarray | None = None,
            topology_edges: np.ndarray | None = None) -> None:
        if self.position >= self.rollout_length:
            raise RuntimeError("rollout buffer is already full")
        index = self.position
        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "local_obs": (self.local_obs[index], local_obs), "global_state": (self.global_state[index], global_state),
            "latent_z": (self.latent_z[index], latent_z), "action_u": (self.action_u[index], action_u),
            "old_log_prob": (self.old_log_prob[index], old_log_prob), "value": (self.value[index], value),
            "next_value": (self.next_value[index], next_value), "reward": (self.reward[index], reward),
            "terminated": (self.terminated[index], terminated), "truncated": (self.truncated[index], truncated),
        }
        for name, (destination, source) in arrays.items():
            source_array = np.asarray(source)
            if destination.shape != source_array.shape:
                raise ValueError(f"{name}: expected {destination.shape}, got {source_array.shape}")
            destination[...] = source_array
        if self.topology_nodes is None:
            if topology_nodes is not None or topology_edges is not None:
                raise ValueError("plain/Role rollout buffer does not store topology")
        else:
            if topology_nodes is None or topology_edges is None:
                raise ValueError("topology rollout requires action-time node and edge features")
            for name, destination, source in (("topology_nodes", self.topology_nodes[index], topology_nodes),
                                              ("topology_edges", self.topology_edges[index], topology_edges)):
                source_array = np.asarray(source)
                if destination.shape != source_array.shape:
                    raise ValueError(f"{name}: expected {destination.shape}, got {source_array.shape}")
                destination[...] = source_array
        self.bootstrap_mask[index] = (~self.terminated[index]).astype(np.float32)
        self.trace_mask[index] = (~(self.terminated[index] | self.truncated[index])).astype(np.float32)
        self.position += 1

    @property
    def full(self) -> bool:
        return self.position == self.rollout_length

    def compute_gae(self, gamma: float, gae_lambda: float) -> np.ndarray:
        if not self.full:
            raise RuntimeError("cannot compute GAE from a partial rollout")
        deltas = self.reward + gamma * self.bootstrap_mask * self.next_value - self.value
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for index in range(self.rollout_length - 1, -1, -1):
            gae = deltas[index] + gamma * gae_lambda * self.trace_mask[index] * gae
            self.advantage[index] = gae
        self.return_target[...] = self.advantage + self.value
        return self.advantage.copy()

    def normalize_advantages(self) -> tuple[float, float]:
        if not self.full:
            raise RuntimeError("cannot normalize a partial rollout")
        mean, std = float(self.advantage.mean()), float(self.advantage.std())
        self.advantage[...] = (self.advantage - mean) / (std + 1e-8)
        return mean, std

    def flatten(self, field: str) -> np.ndarray:
        value = getattr(self, field)
        return value.reshape((self.rollout_length * self.num_envs,) + value.shape[2:])

    def mini_batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        names = ("local_obs", "global_state", "latent_z", "old_log_prob", "advantage", "return_target")
        result = {name: self.flatten(name)[indices] for name in names}
        if self.topology_nodes is not None:
            result["topology_nodes"] = self.flatten("topology_nodes")[indices]
            result["topology_edges"] = self.flatten("topology_edges")[indices]
        return result

    def assert_finite(self) -> bool:
        arrays = (self.local_obs, self.global_state, self.latent_z, self.action_u, self.old_log_prob,
                  self.value, self.next_value, self.reward, self.advantage, self.return_target)
        optional_arrays = () if self.topology_nodes is None else (self.topology_nodes, self.topology_edges)
        return all(np.isfinite(array).all() for array in arrays + optional_arrays)
