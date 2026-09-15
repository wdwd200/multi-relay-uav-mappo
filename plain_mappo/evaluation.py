"""Independent deterministic Actor evaluation; it never updates training state."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from relay_env import RelayEnv

from .config import MappoConfig
from .metrics import EpisodeMetrics, safety_priority_key, summarize_episodes
from .networks import SharedActor
from .topology import build_topology_features


def evaluate_actor(actor: SharedActor, config: MappoConfig, seeds: Iterable[int]) -> tuple[dict, tuple[float, ...], list[dict]]:
    """Run `tanh(mean)` episodes in separate seeded environments, without sampling."""
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("evaluation requires at least one seed")
    device = torch.device(config.device)
    was_training = actor.training
    actor.eval()
    episodes: list[dict] = []
    with torch.no_grad():
        for seed in seed_list:
            env = RelayEnv(config.num_relays)
            observation, _ = env.reset(seed=int(seed))
            metrics = EpisodeMetrics(env.config)
            terminated = truncated = False
            reason: str | None = None
            while not (terminated or truncated):
                before_state = env.get_global_state()
                tensor_obs = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=device).unsqueeze(0)
                topology_nodes = topology_edges = None
                if actor.uses_topology:
                    nodes, edges = build_topology_features(env, before_state)
                    topology_nodes = torch.as_tensor(nodes, device=device).unsqueeze(0)
                    topology_edges = torch.as_tensor(edges, device=device).unsqueeze(0)
                action_u = actor.deterministic_actions(tensor_obs, topology_nodes, topology_edges).squeeze(0).cpu().numpy()
                observation, rewards, terminated, truncated, info = env.step(action_u.tolist())
                state = env.get_global_state()
                metrics.add_step(reward=float(rewards[0]), info=info, previous_state=before_state, state=state, action_u=action_u)
                reason = info["termination_reason"]
            episodes.append(metrics.as_dict(terminated=terminated, truncated=truncated, reason=reason))
    if was_training:
        actor.train()
    summary = summarize_episodes(episodes)
    return summary, safety_priority_key(summary), episodes


def evaluate_random_policy(config: MappoConfig, seeds: Iterable[int], action_seed_offset: int = 50_000) -> tuple[dict, tuple[float, ...], list[dict]]:
    """Evaluate an independent uniform random-action baseline on fixed seeds."""
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("evaluation requires at least one seed")
    episodes: list[dict] = []
    for seed in seed_list:
        env = RelayEnv(config.num_relays)
        observation, _ = env.reset(seed=int(seed))
        del observation
        action_rng = np.random.default_rng(action_seed_offset + int(seed))
        metrics = EpisodeMetrics(env.config)
        terminated = truncated = False
        reason: str | None = None
        while not (terminated or truncated):
            before_state = env.get_global_state()
            action_u = action_rng.uniform(-1.0, 1.0, size=(config.num_relays, config.action_dim)).astype(np.float32)
            _, rewards, terminated, truncated, info = env.step(action_u.tolist())
            state = env.get_global_state()
            metrics.add_step(reward=float(rewards[0]), info=info, previous_state=before_state, state=state, action_u=action_u)
            reason = info["termination_reason"]
        episodes.append(metrics.as_dict(terminated=terminated, truncated=truncated, reason=reason))
    summary = summarize_episodes(episodes)
    return summary, safety_priority_key(summary), episodes
