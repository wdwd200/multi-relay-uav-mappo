"""Centralized, serializable configuration for the Plain MAPPO baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .experiment_protocol import (LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION,
                                  protocol_fields)


@dataclass
class MappoConfig:
    """All Stage 3 learning constants; environment physics remain elsewhere."""

    num_relays: int = 4
    local_obs_dim: int = 26
    action_dim: int = 3
    actor_hidden_sizes: tuple[int, int] = (128, 128)
    critic_hidden_sizes: tuple[int, int] = (128, 128)
    # ``plain`` preserves the Stage-3 actor architecture/state_dict.  Stage 4
    # uses the two explicit Role ablations without changing the environment's
    # 26-dimensional observation or the centralized Critic.
    actor_variant: str = "plain"
    # P3/P4 frozen topology/graph contracts.  They are serialized with new
    # checkpoints while old P0--P2 checkpoints safely receive these defaults.
    topology_node_dim: int = 6
    topology_edge_dim: int = 7
    graph_hidden_dim: int = 32
    # Keep the legacy default so that the four completed training runs and
    # their checkpoints remain independently loadable.  New controlled Actor
    # distribution experiments opt in explicitly.
    actor_log_std_mode: str = "state_dependent_clamp"
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    state_independent_log_std_init: float = -1.5
    num_envs: int = 8
    rollout_length: int = 128
    mini_batch_size: int = 256
    ppo_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalizer_clip: float = 10.0
    normalizer_epsilon: float = 1e-4
    base_seed: int = 2026
    # Stage 4.1 separates all process random domains.  ``from_dict`` marks
    # checkpoints that predate these fields as ``legacy_protocol``.
    protocol_version: str = PROTOCOL_VERSION
    run_seed: int = 2026
    actor_init_seed: int | None = None
    critic_init_seed: int | None = None
    action_noise_seed: int | None = None
    minibatch_seed: int | None = None
    train_env_seed_base: int | None = None
    validation_seed_manifest: str = ""
    final_test_seed_manifest: str = ""
    resume_mode: str = "fresh_episode_at_update_boundary"
    exact_environment_resume: bool = False
    eval_interval_updates: int = 50
    periodic_eval_episodes: int = 5
    acceptance_eval_episodes: int = 20
    smoke_updates: int = 10
    full_updates: int = 1000
    device: str = "cpu"
    output_dir: str = "artifacts/plain_mappo"

    def __post_init__(self) -> None:
        if self.protocol_version != LEGACY_PROTOCOL_VERSION:
            fields = protocol_fields(self.run_seed)
            for name, value in fields.items():
                # All Stage-4.1 domain seeds are derived, never user-tuned.
                # This also makes ``dataclasses.replace(config, run_seed=...)``
                # safe: copied seed fields cannot remain tied to the old run.
                setattr(self, name, value)

    @property
    def global_state_dim(self) -> int:
        """47 for K=4: 23 fixed scalar fields plus 6 per relay."""
        return 23 + 6 * self.num_relays

    @property
    def team_time_samples(self) -> int:
        return self.num_envs * self.rollout_length

    def validate(self) -> None:
        if self.num_relays != 4:
            raise ValueError("Stage 3 training is fixed to K=4; K=3/5 are inference-only")
        if self.global_state_dim != 47:
            raise AssertionError("K=4 global-state contract must resolve to 47 dimensions")
        if self.team_time_samples % self.mini_batch_size:
            raise ValueError("mini_batch_size must divide rollout_length * num_envs")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        if self.actor_log_std_mode not in {"state_dependent_clamp", "state_independent_tanh"}:
            raise ValueError("actor_log_std_mode must be state_dependent_clamp or state_independent_tanh")
        if self.actor_variant not in {"plain", "role_info", "role_head", "topology_info", "graph"}:
            raise ValueError("actor_variant must be plain, role_info, role_head, topology_info, or graph")
        if self.actor_variant != "plain" and self.actor_log_std_mode != "state_independent_tanh":
            raise ValueError("non-plain Actor variants require state_independent_tanh log_std")
        if (self.topology_node_dim, self.topology_edge_dim, self.graph_hidden_dim) != (6, 7, 32):
            raise ValueError("Stage 4 topology/graph dimensions are frozen to node=6, edge=7, hidden=32")
        if self.actor_log_std_mode == "state_independent_tanh" and not (
            self.log_std_min < self.state_independent_log_std_init < self.log_std_max
        ):
            raise ValueError("state_independent_log_std_init must lie strictly inside the configured log_std bounds")
        if self.resume_mode != "fresh_episode_at_update_boundary" or self.exact_environment_resume:
            raise ValueError("only fresh_episode_at_update_boundary / exact_environment_resume=false is supported")
        if self.protocol_version == LEGACY_PROTOCOL_VERSION:
            return
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported experiment protocol version")
        expected = protocol_fields(self.run_seed)
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise ValueError(f"{name} must match the frozen protocol derivation")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["global_state_dim"] = self.global_state_dim
        data["team_time_samples"] = self.team_time_samples
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MappoConfig":
        accepted = {name for name in cls.__dataclass_fields__}
        values = {name: value for name, value in data.items() if name in accepted}
        if "protocol_version" not in data:
            # Do not pretend a pre-Stage-4.1 artifact used independent streams.
            values["protocol_version"] = LEGACY_PROTOCOL_VERSION
            values.setdefault("run_seed", int(data.get("base_seed", 2026)))
        for name in ("actor_hidden_sizes", "critic_hidden_sizes"):
            if name in values:
                values[name] = tuple(values[name])
        config = cls(**values)
        config.validate()
        return config
