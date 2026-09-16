"""Immutable Stage-4.3 P3/P4 engineering-gate helpers.

These helpers deliberately construct the existing Stage-4 formal P3/P4
implementations.  They do not alter the environment, PPO objective, or formal
preset, and they are not a formal-training entrypoint.
"""

from __future__ import annotations

from typing import Any

from .config import MappoConfig
from .networks import SharedActor
from .presets import stage4_formal_overrides


ENGINEERING_GATE_PROTOCOL_VERSION = "stage4.3-p3-p4-engineering-gate-v1"
ENGINEERING_GATE_VARIANTS = ("topology_info", "graph")
SMOKE_RUN_SEED = 2026
SMOKE_UPDATES = 10
SAMPLES_PER_UPDATE = 8 * 128


def stage43_smoke_config(*, actor_variant: str, output_dir: str) -> MappoConfig:
    """Return the frozen V0H0 formal configuration for one 10-update smoke.

    The value treatment, horizon, seed derivation, and all shared optimizer
    fields come from ``stage4-formal``.  Only Actor variant and local output
    directory differ, exactly as they would in a later fair comparison.
    """
    if actor_variant not in ENGINEERING_GATE_VARIANTS:
        raise ValueError(f"Stage-4.3 supports only {ENGINEERING_GATE_VARIANTS}")
    config = MappoConfig(**stage4_formal_overrides(
        actor_variant=actor_variant, run_seed=SMOKE_RUN_SEED, output_dir=output_dir))
    config.validate()
    if (config.value_normalization, config.gamma, config.gae_lambda) != (False, 0.99, 0.95):
        raise AssertionError("Stage-4.3 smoke must use frozen V0H0")
    return config


def actor_parameter_counts() -> dict[str, int]:
    """Return P3/P4 counts from the existing frozen Actor implementations."""
    result: dict[str, int] = {}
    for variant in ENGINEERING_GATE_VARIANTS:
        config = stage43_smoke_config(actor_variant=variant, output_dir="artifacts/stage4-3-parameter-count")
        actor = SharedActor(config.local_obs_dim, config.action_dim, config.actor_hidden_sizes,
                            config.log_std_min, config.log_std_max, config.actor_log_std_mode,
                            config.state_independent_log_std_init, config.actor_variant,
                            config.topology_node_dim, config.topology_edge_dim, config.graph_hidden_dim)
        result[variant] = sum(parameter.numel() for parameter in actor.parameters())
    return result


def implementation_audit() -> dict[str, Any]:
    """Describe the already-implemented, frozen P3/P4 contracts."""
    counts = actor_parameter_counts()
    larger = max(counts.values())
    difference = abs(counts["topology_info"] - counts["graph"])
    return {
        "protocol_version": ENGINEERING_GATE_PROTOCOL_VERSION,
        "scope": "engineering gate and 10-update smoke only",
        "frozen_scientific_contract_changed": False,
        "formal_training_executed": False,
        "final_test_executed": False,
        "common_v0h0": {"value_normalization": False, "gamma": 0.99, "gae_lambda": 0.95},
        "shared_training_contract": {
            "local_observation_dim": 26,
            "centralized_critic_input_dim_k4": 47,
            "actor_lr": 1e-4,
            "critic_lr": 3e-4,
            "entropy_coef": 0.0,
            "rollout_length": 128,
            "num_envs": 8,
            "mini_batch_size": 256,
            "ppo_epochs": 10,
            "state_independent_log_std": True,
        },
        "P3": {
            "actor_variant": "topology_info",
            "actor_input_dim_k4": 97,
            "topology_node_dim": 6,
            "topology_edge_dim": 7,
            "topology_structure": "ordered logical chain H-R1-...-RK-L; shared block repeated for each relay action",
            "environment_observation_changed": False,
            "critic_privileged_information_added": False,
        },
        "P4": {
            "actor_variant": "graph",
            "actor_input_dim_k4": 58,
            "topology_node_dim": 6,
            "topology_edge_dim": 7,
            "graph_hidden_dim": 32,
            "topology_structure": "ordered chain with K shared bidirectional message-passing rounds",
            "parameters_shared_across_nodes_edges_and_rounds": True,
            "uses_relay_own_node_embedding": True,
            "environment_observation_changed": False,
            "critic_privileged_information_added": False,
        },
        "parameter_fairness": {
            "p3_actor_parameters": counts["topology_info"],
            "p4_actor_parameters": counts["graph"],
            "absolute_difference": difference,
            "relative_difference_of_larger": difference / larger,
            "maximum_allowed_relative_difference": 0.10,
            "passes": difference / larger <= 0.10,
        },
    }
