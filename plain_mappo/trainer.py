"""Synchronous 8-environment Plain MAPPO collection, update, logs, and resume."""

from __future__ import annotations

import csv
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from relay_env import RelayEnv

from .checkpoint import atomic_torch_save, capture_rng_state, load_checkpoint, restore_rng_state
from .config import MappoConfig
from .evaluation import evaluate_actor
from .experiment_protocol import (LEGACY_PROTOCOL_VERSION, load_seed_manifest,
                                  protocol_fields)
from .metrics import EpisodeMetrics
from .networks import CentralizedCritic, SharedActor
from .normalization import RunningMeanStd
from .rollout_buffer import RolloutBuffer
from .state import flatten_global_state
from .topology import build_topology_features


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def explained_variance(return_target: np.ndarray, value_prediction: np.ndarray) -> float | None:
    """Population explained variance, explicitly undefined for zero target variance."""
    target = np.asarray(return_target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(value_prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError("return_target and value_prediction must have the same shape")
    target_variance = float(np.var(target))
    if target_variance <= 1e-12:
        return None
    value = 1.0 - float(np.var(target - prediction)) / target_variance
    if not math.isfinite(value):
        raise FloatingPointError("non-finite explained variance")
    return value


def gradient_clip_fraction(pre_clip_norms: list[float], max_grad_norm: float) -> float:
    """Fraction of mini-batches whose pre-clip norm exceeds the fixed bound."""
    if not pre_clip_norms:
        raise ValueError("pre_clip_norms cannot be empty")
    return float(np.mean(np.asarray(pre_clip_norms, dtype=np.float64) > float(max_grad_norm)))


class MappoTrainer:
    """Owns shared models and eight synchronous training environments."""

    train_fields = ("total_env_steps", "update", "completed_episodes", "completed_episode_return_mean",
                    "completed_episode_length_mean", "actor_policy_loss", "critic_loss", "entropy", "approx_kl",
                    "clip_fraction", "actor_grad_norm", "critic_grad_norm", "advantage_raw_mean", "advantage_raw_std",
                    "value_target_mean", "value_target_std", "value_prediction_mean_pre", "value_prediction_std_pre",
                    "explained_variance_pre", "actor_grad_norm_max", "critic_grad_norm_max",
                    "actor_grad_clip_fraction", "critic_grad_clip_fraction",
                    "action_mean", "action_std", "action_saturation_ratio", "raw_log_std_min", "raw_log_std_mean",
                    "raw_log_std_max", "clamped_log_std_min", "clamped_log_std_mean", "clamped_log_std_max",
                    "raw_log_std_gt_max_ratio", "finite")
    eval_fields = ("update", "seeds", "score", "episodes", "normal_completion_count", "collision_count",
                   "boundary_count", "persistent_outage_count", "speed_accel_violations", "outage_step_ratio",
                   "rate_satisfaction_ratio", "mean_e2e_rate_mbps", "summary_json")

    def __init__(self, config: MappoConfig) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(config.device)
        self.protocol_is_legacy = config.protocol_version == LEGACY_PROTOCOL_VERSION
        # A fixed global seed keeps third-party incidental use reproducible,
        # while model construction/action noise/minibatches use isolated
        # Stage-4.1 streams below.
        set_all_seeds(config.base_seed if self.protocol_is_legacy else config.run_seed)
        self.actor = self._construct_actor()
        self.critic = self._construct_critic()
        self._initialize_protocol_rng_streams()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.normalizer = RunningMeanStd((config.global_state_dim,), config.normalizer_epsilon, config.normalizer_clip)
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.eval_checkpoint_dir = self.output_dir / "eval_checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.json").write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.envs = [RelayEnv(config.num_relays) for _ in range(config.num_envs)]
        # Each entry is the next never-used episode index for that environment.
        # A reset consumes it immediately, so a checkpoint at an update boundary
        # can restart with fresh seeds without saving environment internals.
        self.next_episode_indices = [0 for _ in self.envs]
        self.current_obs: list[np.ndarray] = []
        self.episode_returns = np.zeros(config.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
        self.total_env_steps, self.update_count = 0, 0
        self.best_score: tuple[float, ...] | None = None
        self.best_summary: dict[str, Any] | None = None
        self.loaded_checkpoint_protocol = config.protocol_version
        self._reset_training_envs()

    def _construct_with_torch_seed(self, seed: int, factory: Any) -> Any:
        """Construct a module without allowing its shape to consume global RNG."""
        devices = [] if self.device.type != "cuda" else [self.device.index or 0]
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            return factory().to(self.device)

    def _construct_actor(self) -> SharedActor:
        def factory() -> SharedActor:
            c = self.config
            return SharedActor(c.local_obs_dim, c.action_dim, c.actor_hidden_sizes,
                               c.log_std_min, c.log_std_max, c.actor_log_std_mode,
                               c.state_independent_log_std_init, c.actor_variant,
                               c.topology_node_dim, c.topology_edge_dim, c.graph_hidden_dim)
        if self.protocol_is_legacy:
            return factory().to(self.device)
        assert self.config.actor_init_seed is not None
        return self._construct_with_torch_seed(self.config.actor_init_seed, factory)

    def _construct_critic(self) -> CentralizedCritic:
        def factory() -> CentralizedCritic:
            return CentralizedCritic(self.config.global_state_dim, self.config.critic_hidden_sizes)
        if self.protocol_is_legacy:
            return factory().to(self.device)
        assert self.config.critic_init_seed is not None
        return self._construct_with_torch_seed(self.config.critic_init_seed, factory)

    def _initialize_protocol_rng_streams(self) -> None:
        self.action_noise_generator: torch.Generator | None = None
        self.minibatch_rng: np.random.Generator | None = None
        if self.protocol_is_legacy:
            return
        assert self.config.action_noise_seed is not None and self.config.minibatch_seed is not None
        self.action_noise_generator = torch.Generator(device=self.device.type)
        self.action_noise_generator.manual_seed(int(self.config.action_noise_seed))
        self.minibatch_rng = np.random.Generator(np.random.PCG64(int(self.config.minibatch_seed)))

    def _protocol_rng_state(self) -> dict[str, Any] | None:
        if self.protocol_is_legacy:
            return None
        assert self.action_noise_generator is not None and self.minibatch_rng is not None
        return {"action_noise_generator": self.action_noise_generator.get_state(),
                "minibatch_bit_generator": copy.deepcopy(self.minibatch_rng.bit_generator.state)}

    def _restore_protocol_rng_state(self, state: dict[str, Any] | None) -> None:
        if self.protocol_is_legacy:
            return
        if state is None:
            raise ValueError("Stage-4.1 checkpoint is missing independent RNG stream state")
        assert self.action_noise_generator is not None and self.minibatch_rng is not None
        self.action_noise_generator.set_state(state["action_noise_generator"])
        self.minibatch_rng.bit_generator.state = state["minibatch_bit_generator"]

    def _seed_for(self, env_id: int) -> int:
        base = self.config.base_seed if self.protocol_is_legacy else self.config.train_env_seed_base
        assert base is not None
        return int(base) + env_id + self.next_episode_indices[env_id] * self.config.num_envs

    def _reset_one_env(self, env_id: int) -> None:
        seed = self._seed_for(env_id)
        self.next_episode_indices[env_id] += 1
        observation, _ = self.envs[env_id].reset(seed=seed)
        self.current_obs[env_id] = np.asarray(observation, dtype=np.float32)
        self.episode_returns[env_id], self.episode_lengths[env_id] = 0.0, 0

    def _reset_training_envs(self) -> None:
        self.current_obs = [np.empty((self.config.num_relays, self.config.local_obs_dim), dtype=np.float32) for _ in self.envs]
        for env_id in range(len(self.envs)):
            self._reset_one_env(env_id)

    def _new_buffer(self) -> RolloutBuffer:
        c = self.config
        topology_dimensions = ((c.topology_node_dim, c.topology_edge_dim) if self.actor.uses_topology else (None, None))
        return RolloutBuffer(c.rollout_length, c.num_envs, c.num_relays, c.local_obs_dim, c.global_state_dim,
                             c.action_dim, *topology_dimensions)

    def _periodic_validation_seeds(self) -> list[int]:
        """Use the locked validation split for new formal-protocol runs.

        Legacy artifacts preserve their historical five-seed monitoring path;
        their selection is repaired later by the independent Stage-4.1 grid.
        """
        if not self.protocol_is_legacy and self.config.validation_seed_manifest:
            manifest_path = Path(self.config.validation_seed_manifest)
            if manifest_path.exists():
                manifest, _ = load_seed_manifest(manifest_path)
                return [int(seed) for seed in manifest["splits"]["validation"]["seeds"]]
        return list(range(10_000, 10_000 + self.config.periodic_eval_episodes))

    def collect_rollout(self) -> tuple[RolloutBuffer, list[dict[str, float]]]:
        """Collect exactly 128 steps per environment, preserving terminal next values."""
        c, buffer = self.config, self._new_buffer()
        completed: list[dict[str, float]] = []
        for _ in range(c.rollout_length):
            local_obs = np.stack(self.current_obs).astype(np.float32)
            current_states = [env.get_global_state() for env in self.envs]
            raw_state = np.stack([flatten_global_state(state, c.num_relays) for state in current_states])
            topology_nodes = topology_edges = None
            if self.actor.uses_topology:
                topology = [build_topology_features(env, state) for env, state in zip(self.envs, current_states)]
                topology_nodes = np.stack([item[0] for item in topology]).astype(np.float32)
                topology_edges = np.stack([item[1] for item in topology]).astype(np.float32)
            self.normalizer.update(raw_state)  # training environments only
            normalized_state = self.normalizer.normalize(raw_state)
            with torch.no_grad():
                tensor_obs = torch.as_tensor(local_obs, device=self.device)
                tensor_state = torch.as_tensor(normalized_state, device=self.device)
                tensor_nodes = None if topology_nodes is None else torch.as_tensor(topology_nodes, device=self.device)
                tensor_edges = None if topology_edges is None else torch.as_tensor(topology_edges, device=self.device)
                latent_z, action_u, old_log_prob, _ = self.actor.sample_actions(
                    tensor_obs, tensor_nodes, tensor_edges, generator=self.action_noise_generator)
                values = self.critic(tensor_state)
            z_np, action_np = latent_z.cpu().numpy(), action_u.cpu().numpy()
            log_prob_np, value_np = old_log_prob.cpu().numpy(), values.cpu().numpy()
            next_obs: list[np.ndarray] = []
            next_value = np.zeros(c.num_envs, dtype=np.float32)
            rewards = np.zeros(c.num_envs, dtype=np.float32)
            terminated = np.zeros(c.num_envs, dtype=np.bool_)
            truncated = np.zeros(c.num_envs, dtype=np.bool_)
            for env_id, env in enumerate(self.envs):
                before_state = env.get_global_state()
                observation, reward_tuple, did_terminate, did_truncate, info = env.step(action_np[env_id].tolist())
                terminal_state = flatten_global_state(env.get_global_state(), c.num_relays)
                with torch.no_grad():
                    next_value[env_id] = self.critic(torch.as_tensor(self.normalizer.normalize(terminal_state[None, :]), device=self.device)).item()
                reward = float(reward_tuple[0])
                if not all(math.isclose(reward, float(value), rel_tol=0.0, abs_tol=0.0) for value in reward_tuple):
                    raise RuntimeError("RelayEnv must return one shared team reward")
                rewards[env_id], terminated[env_id], truncated[env_id] = reward, did_terminate, did_truncate
                self.episode_returns[env_id] += reward
                self.episode_lengths[env_id] += 1
                if did_terminate or did_truncate:
                    completed.append({"return": float(self.episode_returns[env_id]), "length": float(self.episode_lengths[env_id])})
                    self._reset_one_env(env_id)
                    next_obs.append(self.current_obs[env_id])
                else:
                    next_obs.append(np.asarray(observation, dtype=np.float32))
                # `before_state` is deliberately read before action only; all physical diagnostics stay in evaluation.
                del before_state, info
            buffer.add(local_obs=local_obs, global_state=normalized_state, latent_z=z_np, action_u=action_np,
                       old_log_prob=log_prob_np, value=value_np, next_value=next_value, reward=rewards,
                       terminated=terminated, truncated=truncated, topology_nodes=topology_nodes,
                       topology_edges=topology_edges)
            self.current_obs = next_obs
            self.total_env_steps += c.num_envs
        if not buffer.assert_finite():
            raise FloatingPointError("non-finite rollout data")
        return buffer, completed

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        c = self.config
        advantage_raw_mean, advantage_raw_std = buffer.normalize_advantages() if buffer.compute_gae(c.gamma, c.gae_lambda) is not None else (0.0, 0.0)
        with torch.no_grad():
            full_global_state = torch.as_tensor(buffer.flatten("global_state"), dtype=torch.float32, device=self.device)
            pre_value_prediction = self.critic(full_global_state).cpu().numpy()
        full_return_target = buffer.flatten("return_target")
        diagnostics: dict[str, float | None] = {
            "value_target_mean": float(np.mean(full_return_target)),
            "value_target_std": float(np.std(full_return_target)),
            "value_prediction_mean_pre": float(np.mean(pre_value_prediction)),
            "value_prediction_std_pre": float(np.std(pre_value_prediction)),
            "explained_variance_pre": explained_variance(full_return_target, pre_value_prediction),
        }
        policy_losses: list[float] = []; critic_losses: list[float] = []; entropies: list[float] = []
        approx_kls: list[float] = []; clip_fractions: list[float] = []; actor_norms: list[float] = []; critic_norms: list[float] = []
        for _ in range(c.ppo_epochs):
            permutation = (np.random.permutation(c.team_time_samples) if self.minibatch_rng is None
                           else self.minibatch_rng.permutation(c.team_time_samples))
            for indices in permutation.reshape(-1, c.mini_batch_size):
                batch = buffer.mini_batch(indices)
                local_obs = torch.as_tensor(batch["local_obs"], dtype=torch.float32, device=self.device)
                global_state = torch.as_tensor(batch["global_state"], dtype=torch.float32, device=self.device)
                latent_z = torch.as_tensor(batch["latent_z"], dtype=torch.float32, device=self.device)
                old_log_prob = torch.as_tensor(batch["old_log_prob"], dtype=torch.float32, device=self.device)
                advantage = torch.as_tensor(batch["advantage"], dtype=torch.float32, device=self.device).unsqueeze(-1)
                return_target = torch.as_tensor(batch["return_target"], dtype=torch.float32, device=self.device)
                topology_nodes = None if "topology_nodes" not in batch else torch.as_tensor(batch["topology_nodes"], dtype=torch.float32, device=self.device)
                topology_edges = None if "topology_edges" not in batch else torch.as_tensor(batch["topology_edges"], dtype=torch.float32, device=self.device)
                new_log_prob, entropy = self.actor.log_prob_entropy(local_obs, latent_z, topology_nodes, topology_edges)  # reuses buffer's old z; never resamples.
                ratio = torch.exp(new_log_prob - old_log_prob)
                surrogate_a, surrogate_b = ratio * advantage, ratio.clamp(1.0 - c.clip_epsilon, 1.0 + c.clip_epsilon) * advantage
                policy_loss = -torch.minimum(surrogate_a, surrogate_b).mean()
                actor_loss = policy_loss - c.entropy_coef * entropy.mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_norm = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm).item())
                self.actor_optimizer.step()
                prediction = self.critic(global_state)
                critic_loss = c.value_loss_coef * torch.nn.functional.mse_loss(prediction, return_target)
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_norm = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm).item())
                self.critic_optimizer.step()
                policy_losses.append(float(policy_loss.detach().cpu())); critic_losses.append(float(critic_loss.detach().cpu()))
                entropies.append(float(entropy.mean().detach().cpu())); approx_kls.append(float((old_log_prob - new_log_prob).mean().detach().cpu()))
                clip_fractions.append(float((torch.abs(ratio - 1.0) > c.clip_epsilon).float().mean().detach().cpu()))
                actor_norms.append(actor_norm); critic_norms.append(critic_norm)
        finite = all(torch.isfinite(parameter).all() for model in (self.actor, self.critic) for parameter in model.parameters())
        if not finite:
            raise FloatingPointError("non-finite actor or critic parameter")
        # These post-update values use the rollout's local observations.  In
        # the state-independent mode ``raw_log_std`` is the shared unbounded
        # coordinate and ``clamped_log_std`` is its smooth bounded image.
        # Legacy mode retains the historical raw-head/hard-clamp semantics.
        with torch.no_grad():
            rollout_obs = torch.as_tensor(buffer.flatten("local_obs"), dtype=torch.float32, device=self.device)
            rollout_nodes = None if buffer.topology_nodes is None else torch.as_tensor(buffer.flatten("topology_nodes"), dtype=torch.float32, device=self.device)
            rollout_edges = None if buffer.topology_edges is None else torch.as_tensor(buffer.flatten("topology_edges"), dtype=torch.float32, device=self.device)
            # A view of nn.Parameter can retain requires_grad even in a
            # no_grad block.  These are CSV diagnostics only, never a loss.
            raw_log_std = self.actor.raw_parameters(rollout_obs, rollout_nodes, rollout_edges)[1].detach()
            clamped_log_std = self.actor.effective_log_std(raw_log_std)
        if not (torch.isfinite(raw_log_std).all() and torch.isfinite(clamped_log_std).all()):
            raise FloatingPointError("non-finite actor log_std diagnostic")
        raw_exceeds_hard_max = (raw_log_std > c.log_std_max).float().mean() if self.actor.log_std_mode == "state_dependent_clamp" else torch.zeros((), device=self.device)
        result: dict[str, float | None] = {"actor_policy_loss": float(np.mean(policy_losses)), "critic_loss": float(np.mean(critic_losses)),
                "entropy": float(np.mean(entropies)), "approx_kl": float(np.mean(approx_kls)), "clip_fraction": float(np.mean(clip_fractions)),
                "actor_grad_norm": float(np.mean(actor_norms)), "critic_grad_norm": float(np.mean(critic_norms)),
                "actor_grad_norm_max": float(np.max(actor_norms)), "critic_grad_norm_max": float(np.max(critic_norms)),
                "actor_grad_clip_fraction": gradient_clip_fraction(actor_norms, c.max_grad_norm),
                "critic_grad_clip_fraction": gradient_clip_fraction(critic_norms, c.max_grad_norm),
                "advantage_raw_mean": advantage_raw_mean, "advantage_raw_std": advantage_raw_std,
                "action_mean": float(buffer.action_u.mean()), "action_std": float(buffer.action_u.std()),
                "action_saturation_ratio": float(np.mean(np.abs(buffer.action_u) > 0.95)),
                "raw_log_std_min": float(raw_log_std.min().cpu()), "raw_log_std_mean": float(raw_log_std.mean().cpu()),
                "raw_log_std_max": float(raw_log_std.max().cpu()),
                "clamped_log_std_min": float(clamped_log_std.min().cpu()), "clamped_log_std_mean": float(clamped_log_std.mean().cpu()),
                "clamped_log_std_max": float(clamped_log_std.max().cpu()),
                "raw_log_std_gt_max_ratio": float(raw_exceeds_hard_max.cpu()), "finite": float(finite)}
        result.update(diagnostics)
        finite_diagnostics = all(value is None or math.isfinite(float(value)) for value in result.values())
        if not finite_diagnostics:
            raise FloatingPointError("non-finite training diagnostic")
        return result

    def _checkpoint_payload(self) -> dict[str, Any]:
        protocol_metadata = (protocol_fields(self.config.run_seed) if not self.protocol_is_legacy else {
            "protocol_version": LEGACY_PROTOCOL_VERSION, "run_seed": self.config.run_seed})
        return {"actor_state": self.actor.state_dict(), "critic_state": self.critic.state_dict(),
                "actor_optimizer_state": self.actor_optimizer.state_dict(), "critic_optimizer_state": self.critic_optimizer.state_dict(),
                "normalizer_state": self.normalizer.state_dict(), "update": self.update_count, "total_env_steps": self.total_env_steps,
                "config": self.config.to_dict(), "rng_state": capture_rng_state(),
                "protocol_rng_state": self._protocol_rng_state(), "protocol_metadata": {
                    **protocol_metadata,
                    "resume_mode": "fresh_episode_at_update_boundary", "exact_environment_resume": False},
                "best_score": self.best_score, "best_summary": self.best_summary,
                "next_episode_indices": list(self.next_episode_indices),
                "resume_mode": "fresh_episode_at_update_boundary", "exact_environment_resume": False}

    def save_latest(self) -> Path:
        path = self.checkpoint_dir / "latest.pt"
        atomic_torch_save(self._checkpoint_payload(), path)
        return path

    def save_evaluation_actor(self, seeds: list[int]) -> Path:
        """Save a compact, independently re-evaluable Actor at an eval boundary."""
        path = self.eval_checkpoint_dir / f"actor_update_{self.update_count:04d}.pt"
        atomic_torch_save({"actor_state": self.actor.state_dict(), "config": self.config.to_dict(),
                           "update": self.update_count, "evaluation_seeds": list(seeds),
                           "checkpoint_type": "periodic_evaluation_actor",
                           "global_state_contract": "23 + 6*K; K=4 is 47"}, path)
        return path

    def maybe_save_best(self, summary: dict[str, Any], score: tuple[float, ...]) -> bool:
        if self.best_score is not None and score >= self.best_score:
            return False
        self.best_score, self.best_summary = tuple(float(value) for value in score), summary
        atomic_torch_save(self._checkpoint_payload(), self.checkpoint_dir / "best.pt")
        return True

    def export_actor_final(self) -> Path:
        best_path = self.checkpoint_dir / "best.pt"
        source = load_checkpoint(best_path, self.config.device) if best_path.exists() else self._checkpoint_payload()
        path = self.checkpoint_dir / "actor_final.pt"
        atomic_torch_save({"actor_state": source["actor_state"], "config": source["config"],
                           "global_state_contract": "23 + 6*K; K=4 is 47", "selected_from": "best.pt" if best_path.exists() else "latest"}, path)
        return path

    def load(self, checkpoint_path: str | Path) -> None:
        payload = load_checkpoint(Path(checkpoint_path), self.config.device)
        saved_config = MappoConfig.from_dict(payload["config"])
        protocol_fields_to_ignore = ("protocol_version", "run_seed", "actor_init_seed", "critic_init_seed",
                                     "action_noise_seed", "minibatch_seed", "train_env_seed_base",
                                     "validation_seed_manifest", "final_test_seed_manifest")
        def resume_comparison(config: MappoConfig, *, ignore_protocol: bool) -> dict[str, Any]:
            payload = config.to_dict()
            if ignore_protocol:
                for field in protocol_fields_to_ignore:
                    payload.pop(field, None)
            return payload
        ignore_protocol = saved_config.protocol_version == LEGACY_PROTOCOL_VERSION or self.protocol_is_legacy
        if (resume_comparison(saved_config, ignore_protocol=ignore_protocol)
                != resume_comparison(self.config, ignore_protocol=ignore_protocol)):
            raise ValueError("resume configuration does not match checkpoint")
        self.actor.load_state_dict(payload["actor_state"]); self.critic.load_state_dict(payload["critic_state"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state"]); self.critic_optimizer.load_state_dict(payload["critic_optimizer_state"])
        self.normalizer.load_state_dict(payload["normalizer_state"])
        self.update_count, self.total_env_steps = int(payload["update"]), int(payload["total_env_steps"])
        self.best_score = tuple(payload["best_score"]) if payload.get("best_score") is not None else None
        self.best_summary = payload.get("best_summary")
        restore_rng_state(payload["rng_state"])
        self.loaded_checkpoint_protocol = payload.get("protocol_metadata", {}).get(
            "protocol_version", LEGACY_PROTOCOL_VERSION)
        if saved_config.protocol_version == LEGACY_PROTOCOL_VERSION:
            # Historic payloads have no independent streams.  They remain
            # deployable/resumable under their documented legacy semantics.
            self.protocol_is_legacy = True
            self.action_noise_generator = None
            self.minibatch_rng = None
            self.loaded_checkpoint_protocol = LEGACY_PROTOCOL_VERSION
        self._restore_protocol_rng_state(payload.get("protocol_rng_state"))
        # Stage 3 deliberately does not checkpoint environment internals or a
        # partial rollout.  Recreate eight fresh seeded environments only
        # after restoring the training RNG/model state.
        self.envs = [RelayEnv(self.config.num_relays) for _ in range(self.config.num_envs)]
        saved_progress = payload.get("next_episode_indices")
        if saved_progress is None:
            # Compatibility for old checkpoints that omitted seed progress.  It
            # cannot recreate their lost in-progress episodes, so skip initial
            # seed 0 rather than silently reuse the earliest seeds.
            self.next_episode_indices = [1 for _ in self.envs]
        else:
            self.next_episode_indices = [int(value) for value in saved_progress]
            if len(self.next_episode_indices) != self.config.num_envs or any(value < 0 for value in self.next_episode_indices):
                raise ValueError("invalid next_episode_indices in checkpoint")
        self.episode_returns = np.zeros(self.config.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(self.config.num_envs, dtype=np.int64)
        self._reset_training_envs()

    def _append_csv(self, filename: str, fields: tuple[str, ...], row: dict[str, Any]) -> None:
        path, new_file = self.output_dir / filename, not (self.output_dir / filename).exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in fields})

    def _write_train_log(self, update_stats: dict[str, float], completed: list[dict[str, float]]) -> None:
        returns, lengths = [item["return"] for item in completed], [item["length"] for item in completed]
        row: dict[str, Any] = {"total_env_steps": self.total_env_steps, "update": self.update_count,
                               "completed_episodes": len(completed), "completed_episode_return_mean": float(np.mean(returns)) if returns else "",
                               "completed_episode_length_mean": float(np.mean(lengths)) if lengths else "", **update_stats}
        self._append_csv("train.csv", self.train_fields, row)

    def run_evaluation(self, seeds: list[int]) -> tuple[dict[str, Any], tuple[float, ...], bool]:
        summary, score, _ = evaluate_actor(self.actor, self.config, seeds)
        self.save_evaluation_actor(seeds)
        is_best = self.maybe_save_best(summary, score)
        row = {"update": self.update_count, "seeds": ",".join(map(str, seeds)), "score": json.dumps(score),
               "summary_json": json.dumps(summary, ensure_ascii=False), **summary}
        self._append_csv("eval.csv", self.eval_fields, row)
        return summary, score, is_best

    def train(self, updates: int, *, final_evaluation: bool = True) -> list[dict[str, Any]]:
        if updates < 1:
            raise ValueError("updates must be positive")
        records: list[dict[str, Any]] = []
        for _ in range(updates):
            buffer, completed = self.collect_rollout()
            update_stats = self.update(buffer)
            self.update_count += 1
            self._write_train_log(update_stats, completed)
            self.save_latest()
            record: dict[str, Any] = {"update": self.update_count, **update_stats}
            if self.update_count % self.config.eval_interval_updates == 0:
                summary, score, is_best = self.run_evaluation(self._periodic_validation_seeds())
                record.update({"evaluation": summary, "score": score, "best": is_best})
                self.save_latest()  # latest includes the current historical-best record.
            records.append(record)
        if final_evaluation and not any("evaluation" in record for record in records):
            summary, score, is_best = self.run_evaluation(self._periodic_validation_seeds())
            records[-1].update({"evaluation": summary, "score": score, "best": is_best})
            self.save_latest()
        self.export_actor_final()
        return records
