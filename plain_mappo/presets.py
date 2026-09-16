"""Frozen, atomic configuration presets for authorized formal experiments."""

from __future__ import annotations

from typing import Any

from .experiment_protocol import PREEXPERIMENT_PROTOCOL_VERSION


STAGE4_FORMAL_PRESET = "stage4-formal"
STAGE4_PREEXPERIMENT_PRESET = "stage4.2-preexperiment"

# These are deliberately centralized.  A formal P0--P4 comparison may vary
# only Actor variant, its structurally required parameters, and output path.
STAGE4_FORMAL_FIELDS: dict[str, Any] = {
    "num_relays": 4,
    "local_obs_dim": 26,
    "action_dim": 3,
    "actor_hidden_sizes": (128, 128),
    "critic_hidden_sizes": (128, 128),
    "actor_log_std_mode": "state_independent_tanh",
    "log_std_min": -4.0,
    "log_std_max": 0.0,
    "state_independent_log_std_init": -1.5,
    "num_envs": 8,
    "rollout_length": 128,
    "mini_batch_size": 256,
    "ppo_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "actor_lr": 1e-4,
    "critic_lr": 3e-4,
    "entropy_coef": 0.0,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalizer_clip": 10.0,
    "normalizer_epsilon": 1e-4,
    "base_seed": 2026,
    "resume_mode": "fresh_episode_at_update_boundary",
    "exact_environment_resume": False,
    "eval_interval_updates": 50,
    # Retained for historical config parity; Stage-4.1 formal periodic
    # evaluation obtains its locked 20 seeds from the manifest.
    "periodic_eval_episodes": 5,
    "acceptance_eval_episodes": 20,
    "smoke_updates": 10,
    "full_updates": 1000,
    "device": "cpu",
    "topology_node_dim": 6,
    "topology_edge_dim": 7,
    "graph_hidden_dim": 32,
}


def stage4_formal_overrides(*, actor_variant: str, run_seed: int, output_dir: str) -> dict[str, Any]:
    """Return the complete common configuration for an authorized P0--P4 run."""
    return {**STAGE4_FORMAL_FIELDS, "actor_variant": actor_variant, "run_seed": int(run_seed),
            "actor_init_seed": None, "critic_init_seed": None, "action_noise_seed": None,
            "minibatch_seed": None, "train_env_seed_base": None,
            "validation_seed_manifest": "", "final_test_seed_manifest": "", "output_dir": output_dir}


# Deliberately separate from ``STAGE4_FORMAL_FIELDS``.  These are authorized
# 200-update diagnostic pilots only and must not silently change a later
# P0--P4 formal comparison.
STAGE4_PREEXPERIMENT_FIELDS: dict[str, Any] = {
    **STAGE4_FORMAL_FIELDS,
    "full_updates": 200,
    "eval_interval_updates": 50,
    "periodic_eval_episodes": 20,
}


def stage4_preexperiment_overrides(*, candidate: str, run_seed: int, output_dir: str,
                                   protocol_sha256: str, value_normalization: bool,
                                   gamma: float, gae_lambda: float,
                                   evaluation_episode_log_path: str) -> dict[str, Any]:
    """Return the frozen Stage-4.2 P0 value/horizon pre-experiment config."""
    return {
        **STAGE4_PREEXPERIMENT_FIELDS,
        "protocol_version": PREEXPERIMENT_PROTOCOL_VERSION,
        "actor_variant": "plain", "run_seed": int(run_seed),
        "actor_init_seed": None, "critic_init_seed": None, "action_noise_seed": None,
        "minibatch_seed": None, "train_env_seed_base": None,
        "validation_seed_manifest": "", "final_test_seed_manifest": "",
        "value_normalization": bool(value_normalization), "gamma": float(gamma),
        "gae_lambda": float(gae_lambda), "output_dir": output_dir,
        "preexperiment_candidate": candidate,
        "preexperiment_protocol_sha256": protocol_sha256,
        "evaluation_episode_log_path": evaluation_episode_log_path,
    }
