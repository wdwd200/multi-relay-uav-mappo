"""Stage 4.5 P3/P4 diagnostics and gated paired 2027--2028 training.

``--diagnose`` is validation-only: it reads the completed Stage-4.4 records,
loads the two selected local Actor checkpoints supplied through
``--checkpoint-root``, and performs deterministic Graph ablations only on the
locked validation split.  ``--run`` is enabled only after that diagnostic
records ``GRAPH_INFO_CONFIRMED``.  It then runs P3 followed by P4 for each
paired seed 2027 and 2028 under the unchanged ``stage4-formal`` V0H0 preset.
No mode is permitted to use final-test seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from plain_mappo.checkpoint import load_checkpoint
from plain_mappo.config import MappoConfig
from plain_mappo.engineering_gate import ENGINEERING_GATE_VARIANTS, actor_parameter_counts
from plain_mappo.experiment_protocol import (FINAL_TEST_SEEDS, VALIDATION_SEEDS, canonical_json_bytes,
                                              sha256_file, sha256_json_file, sha256_json_payload)
from plain_mappo.metrics import EpisodeMetrics, safety_priority_key, summarize_episodes
from plain_mappo.networks import SharedActor
from plain_mappo.presets import STAGE4_FORMAL_FIELDS, stage4_formal_overrides
from plain_mappo.topology import build_topology_features
from plain_mappo.trainer import MappoTrainer
from relay_env import EnvironmentConfig, RelayEnv


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "stage4-5-p3-p4-diagnostics-multiseed"
STAGE44 = ROOT / "artifacts" / "stage4-4-p3-p4-seed2026"
DIAGNOSTIC_PROTOCOL = OUTPUT / "diagnostic_protocol.json"
DIAGNOSTIC_REPORT = OUTPUT / "diagnostic_report.json"
DIAGNOSTIC_REPORT_MD = OUTPUT / "diagnostic_report.md"
TRAJECTORY_CSV = OUTPUT / "validation_performance_trajectory.csv"
GRADIENT_CSV = OUTPUT / "gradient_clipping_windows.csv"
FAILURES_CSV = OUTPUT / "failure_reason_migration.csv"
GRAPH_ACTIVATIONS = OUTPUT / "graph_activation_diagnostics.json"
GRAPH_SENSITIVITY = OUTPUT / "graph_sensitivity.json"
RELEASE_RECORD = OUTPUT / "stage4_4_release.json"
MULTI_PROTOCOL = OUTPUT / "multiseed_protocol.json"
MULTI_RUN_MANIFEST = OUTPUT / "multiseed_run_manifest.json"
MULTI_EPISODES = OUTPUT / "multiseed_validation_episodes.jsonl"
MULTI_CHECKPOINT_MANIFEST = OUTPUT / "multiseed_checkpoint_manifest.json"
MULTI_SELECTED = OUTPUT / "multiseed_selected_checkpoints.json"
MULTI_SUMMARY = OUTPUT / "multiseed_summary.csv"
MULTI_PAIRED = OUTPUT / "multiseed_paired_differences.csv"
MULTI_AUDIT = OUTPUT / "multiseed_audit.json"
MULTI_AUDIT_MD = OUTPUT / "multiseed_audit.md"

VARIANTS = (("P3", "topology_info"), ("P4", "graph"))
NEW_SEEDS = (2027, 2028)
BASELINE_SEED = 2026
UPDATES = 1000
SAMPLES_PER_UPDATE = 8 * 128
EVAL_UPDATES = tuple(range(50, UPDATES + 1, 50))
DIAGNOSTIC_PROTOCOL_VERSION = "stage4.5-p3-p4-graph-diagnostics-v1"
MULTISEED_PROTOCOL_VERSION = "stage4.5-p3-p4-multiseed-v1"
STAGE44_RELEASE_URL = "https://github.com/wdwd200/multi-relay-uav-mappo/releases/tag/stage4.4-p3-p4-seed2026-selected-checkpoints"
STAGE44_RELEASE_ASSET_URL = "https://github.com/wdwd200/multi-relay-uav-mappo/releases/download/stage4.4-p3-p4-seed2026-selected-checkpoints/stage4.4-p3-p4-seed2026-selected-checkpoints.zip"
STAGE44_RELEASE_ZIP_SHA256 = "e599aac92f04084f3b887187d14b1f7f3b841404cabb959f89bf5fcb146d5316"
METADATA_FIELDS = {"protocol_version", "protocol_sha256", "variant", "actor_variant", "run_seed", "update",
                   "split", "seed", "checkpoint_relative_path"}
GRAPH_THRESHOLDS = {
    "message_vector_variance_min": 1e-10,
    "node_embedding_variance_min": 1e-10,
    "mean_abs_action_delta_min": 1e-6,
    "trace_abs_error_max": 1e-6,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _canonical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json_bytes(payload).decode("utf-8"))


def _stage44_protocol_hash() -> str:
    return sha256_json_file(STAGE44 / "protocol.json")


def _stage44_selected() -> dict[str, Any]:
    selected = _read_json(STAGE44 / "selected_checkpoints.json")
    if selected.get("final_test_executed") is not False or len(selected.get("selected", [])) != 2:
        raise ValueError("Stage-4.4 selected-checkpoint manifest is invalid")
    return selected


def _diagnostic_payload() -> dict[str, Any]:
    stage44_protocol = _read_json(STAGE44 / "protocol.json")
    if stage44_protocol.get("final_test", {}).get("executed") is not False:
        raise ValueError("Stage-4.4 source must not have executed final test")
    payload = {
        "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _run_git("rev-parse", "HEAD"),
        "dirty_worktree_at_lock": bool(_run_git("status", "--porcelain")),
        "stage44": {
            "protocol_sha256": _stage44_protocol_hash(),
            "selected_checkpoints_sha256": sha256_json_file(STAGE44 / "selected_checkpoints.json"),
            "checkpoint_manifest_sha256": sha256_json_file(STAGE44 / "checkpoint_manifest.json"),
            "validation_episode_rows": 800,
        },
        "locked_validation": {
            "seed_manifest_sha256": sha256_json_file(ROOT / "artifacts/stage4-1-protocol-repair/seed_manifest.json"),
            "seeds": list(VALIDATION_SEEDS), "updates": list(EVAL_UPDATES),
        },
        "final_test": {"executed": False, "permitted": False, "seeds": list(FINAL_TEST_SEEDS)},
        "frozen_contract": {
            "environment": "Environment v1.1", "reward": "unchanged", "local_observation_dim": 26,
            "centralized_critic_dim": 47, "ppo_gae": "unchanged stage4-formal V0H0",
            "value_normalization": False, "gamma": 0.99, "gae_lambda": 0.95,
        },
        "trajectory": {"source": "Stage-4.4 committed eval.csv and validation_episodes.jsonl", "checkpoints": 20},
        "gradient_windows": ["last_50_updates", "last_100_updates", "all_updates"],
        "graph_probe": {
            "selected_variant": "P4", "checkpoint_selection": "Stage-4.4 selected update only",
            "message_ablation": "replace every message aggregate with zeros while retaining shared update layers",
            "edge_perturbation": "cyclically roll the ordered chain edge features by one hop",
            "action_response": "compare deterministic actions at identical baseline validation states",
            "thresholds": GRAPH_THRESHOLDS,
        },
        "release": {"url": STAGE44_RELEASE_URL, "asset_url": STAGE44_RELEASE_ASSET_URL,
                    "zip_sha256": STAGE44_RELEASE_ZIP_SHA256},
    }
    return _canonical_payload(payload)


def _prepare_diagnostic_protocol() -> tuple[dict[str, Any], str]:
    proposed = _diagnostic_payload()
    if proposed["dirty_worktree_at_lock"]:
        raise RuntimeError("Stage-4.5 diagnostic protocol must be locked from a clean worktree")
    if DIAGNOSTIC_PROTOCOL.exists():
        existing = _read_json(DIAGNOSTIC_PROTOCOL)
        left = {key: value for key, value in existing.items() if key != "generated_at_utc"}
        right = {key: value for key, value in proposed.items() if key != "generated_at_utc"}
        if left != right:
            raise ValueError("Stage-4.5 diagnostic protocol is locked and differs from current contract")
        return existing, sha256_json_payload(existing)
    _write_json(DIAGNOSTIC_PROTOCOL, proposed)
    return proposed, sha256_json_payload(proposed)


def _run_dir(label: str, run_seed: int) -> Path:
    return OUTPUT / "runs" / label / f"seed-{run_seed}"


def _config_for(label: str, actor_variant: str, run_seed: int) -> MappoConfig:
    if (label, actor_variant) not in VARIANTS or run_seed not in NEW_SEEDS:
        raise ValueError("Stage-4.5 trains only P3/P4 paired seeds 2027 and 2028")
    config = MappoConfig(**stage4_formal_overrides(actor_variant=actor_variant, run_seed=run_seed,
                                                    output_dir=str(_run_dir(label, run_seed))))
    config.validate()
    if (config.value_normalization, config.gamma, config.gae_lambda, config.full_updates) != (False, 0.99, 0.95, UPDATES):
        raise AssertionError("Stage-4.5 requires unchanged stage4-formal V0H0 configuration")
    return config


def _load_selected_actor(checkpoint_root: Path, variant: str) -> tuple[SharedActor, MappoConfig, dict[str, Any]]:
    selected = _stage44_selected()
    record = next((item for item in selected["selected"] if item["variant"] == variant), None)
    if record is None:
        raise ValueError(f"Stage-4.4 has no selected {variant} checkpoint")
    path = checkpoint_root / Path(record["checkpoint_relative_path"])
    if not path.is_file():
        raise FileNotFoundError(f"selected {variant} checkpoint is missing: {path}")
    if sha256_file(path) != record["checkpoint_sha256"]:
        raise ValueError(f"selected {variant} checkpoint SHA-256 does not match manifest")
    payload = load_checkpoint(path, "cpu")
    config = MappoConfig.from_dict(payload["config"])
    if config.actor_variant != record["actor_variant"]:
        raise ValueError(f"selected {variant} checkpoint config has wrong Actor variant")
    actor = SharedActor(config.local_obs_dim, config.action_dim, config.actor_hidden_sizes,
                        config.log_std_min, config.log_std_max, config.actor_log_std_mode,
                        config.state_independent_log_std_init, config.actor_variant,
                        config.topology_node_dim, config.topology_edge_dim, config.graph_hidden_dim)
    actor.load_state_dict(payload["actor_state"])
    actor.eval()
    return actor, config, record


def _graph_trace(actor: SharedActor, local_obs: torch.Tensor, topology_nodes: torch.Tensor,
                 topology_edges: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Reproduce the frozen Graph message calculation without changing Actor behavior."""
    if actor.actor_variant != "graph":
        raise ValueError("graph trace requires the P4 Graph Actor")
    nodes, edges = actor._topology_tensors(local_obs, topology_nodes, topology_edges)
    embedding = actor.node_encoder(nodes)
    reverse_edges = torch.cat((-edges[..., :6], edges[..., 6:]), dim=-1)
    forward_edges = actor.edge_encoder(edges)
    reverse_edges_encoded = actor.edge_encoder(reverse_edges)
    relays = int(local_obs.shape[-2])
    traces: list[dict[str, torch.Tensor]] = []
    for _ in range(relays):
        forward = actor.message_mlp(torch.cat((embedding[..., :-1, :], forward_edges), dim=-1))
        reverse = actor.message_mlp(torch.cat((embedding[..., 1:, :], reverse_edges_encoded), dim=-1))
        zero = torch.zeros_like(forward[..., :1, :])
        aggregate = torch.cat((zero, forward), dim=-2) + torch.cat((reverse, zero), dim=-2)
        degree = torch.ones((relays + 2,), device=embedding.device, dtype=embedding.dtype)
        degree[1:-1] = 2.0
        view_shape = (1,) * (embedding.ndim - 2) + (relays + 2, 1)
        traces.append({"forward": forward, "reverse": reverse, "node_before": embedding})
        embedding = actor.update_mlp(torch.cat((embedding, aggregate / degree.view(view_shape)), dim=-1))
    return embedding, traces


def _graph_embeddings_without_messages(actor: SharedActor, local_obs: torch.Tensor, topology_nodes: torch.Tensor,
                                       topology_edges: torch.Tensor) -> torch.Tensor:
    nodes, _ = actor._topology_tensors(local_obs, topology_nodes, topology_edges)
    embedding = actor.node_encoder(nodes)
    relays = int(local_obs.shape[-2])
    for _ in range(relays):
        embedding = actor.update_mlp(torch.cat((embedding, torch.zeros_like(embedding)), dim=-1))
    return embedding


def _actions_from_embeddings(actor: SharedActor, local_obs: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    effective = torch.cat((local_obs, embedding[..., 1:-1, :]), dim=-1)
    return torch.tanh(actor._mean_from_hidden(actor.backbone(effective), local_obs))


def _evaluate_graph_mode(actor: SharedActor, config: MappoConfig, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if tuple(VALIDATION_SEEDS) == tuple(FINAL_TEST_SEEDS):
        raise AssertionError("validation/final split contract unexpectedly overlaps")
    episodes: list[dict[str, Any]] = []
    with torch.no_grad():
        for seed in VALIDATION_SEEDS:
            env = RelayEnv(config.num_relays)
            observation, _ = env.reset(seed=int(seed))
            metrics = EpisodeMetrics(env.config)
            terminated = truncated = False
            reason: str | None = None
            while not (terminated or truncated):
                before = env.get_global_state()
                obs = torch.as_tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
                nodes_np, edges_np = build_topology_features(env, before)
                nodes = torch.as_tensor(nodes_np).unsqueeze(0)
                edges = torch.as_tensor(edges_np).unsqueeze(0)
                if mode == "baseline":
                    action = actor.deterministic_actions(obs, nodes, edges)
                elif mode == "message_blocked":
                    action = _actions_from_embeddings(actor, obs, _graph_embeddings_without_messages(actor, obs, nodes, edges))
                elif mode == "edge_rolled":
                    action = actor.deterministic_actions(obs, nodes, torch.roll(edges, shifts=1, dims=-2))
                else:
                    raise ValueError(f"unsupported graph mode: {mode}")
                action_np = action.squeeze(0).cpu().numpy()
                observation, rewards, terminated, truncated, info = env.step(action_np.tolist())
                metrics.add_step(reward=float(rewards[0]), info=info, previous_state=before,
                                 state=env.get_global_state(), action_u=action_np)
                reason = info["termination_reason"]
            episodes.append(metrics.as_dict(terminated=terminated, truncated=truncated, reason=reason))
    return summarize_episodes(episodes), episodes


def _graph_state_probe(actor: SharedActor, config: MappoConfig) -> dict[str, Any]:
    """Collect Graph activations and ablation action responses on baseline states only."""
    message_samples: list[list[np.ndarray]] = [[] for _ in range(config.num_relays)]
    embedding_samples: list[np.ndarray] = []
    action_samples: list[np.ndarray] = []
    blocked_deltas: list[np.ndarray] = []
    edge_deltas: list[np.ndarray] = []
    trace_error = 0.0
    with torch.no_grad():
        for seed in VALIDATION_SEEDS:
            env = RelayEnv(config.num_relays)
            observation, _ = env.reset(seed=int(seed))
            terminated = truncated = False
            while not (terminated or truncated):
                state = env.get_global_state()
                obs = torch.as_tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
                nodes_np, edges_np = build_topology_features(env, state)
                nodes = torch.as_tensor(nodes_np).unsqueeze(0)
                edges = torch.as_tensor(edges_np).unsqueeze(0)
                traced_embedding, trace = _graph_trace(actor, obs, nodes, edges)
                direct_embedding = actor.graph_embeddings(obs, nodes, edges)
                trace_error = max(trace_error, float((traced_embedding - direct_embedding).abs().max().item()))
                baseline = actor.deterministic_actions(obs, nodes, edges)
                blocked = _actions_from_embeddings(actor, obs, _graph_embeddings_without_messages(actor, obs, nodes, edges))
                rolled = actor.deterministic_actions(obs, nodes, torch.roll(edges, shifts=1, dims=-2))
                embedding_samples.append(traced_embedding.squeeze(0).cpu().numpy())
                action_samples.append(baseline.squeeze(0).cpu().numpy())
                blocked_deltas.append((baseline - blocked).abs().squeeze(0).cpu().numpy())
                edge_deltas.append((baseline - rolled).abs().squeeze(0).cpu().numpy())
                for round_index, item in enumerate(trace):
                    message_samples[round_index].append(torch.cat((item["forward"], item["reverse"]), dim=-2).squeeze(0).cpu().numpy())
                observation, _, terminated, truncated, _ = env.step(baseline.squeeze(0).cpu().numpy().tolist())
    embeddings = np.asarray(embedding_samples)
    actions = np.asarray(action_samples)
    blocked = np.asarray(blocked_deltas)
    rolled = np.asarray(edge_deltas)
    return {
        "sampled_validation_states": int(actions.shape[0]),
        "trace_max_abs_error": trace_error,
        "node_embedding_variance_by_node_dim": np.var(embeddings, axis=0).tolist(),
        "node_embedding_variance_mean": float(np.var(embeddings, axis=0).mean()),
        "message_vector_variance_by_round": [float(np.var(np.asarray(samples), axis=0).mean())
                                               for samples in message_samples],
        "action_variance_by_relay_dim": np.var(actions, axis=0).tolist(),
        "action_variance_mean": float(np.var(actions, axis=0).mean()),
        "message_blocked_action_delta_mean_by_relay_dim": np.mean(blocked, axis=0).tolist(),
        "message_blocked_action_delta_mean": float(np.mean(blocked)),
        "message_blocked_action_delta_max": float(np.max(blocked)),
        "edge_rolled_action_delta_mean_by_relay_dim": np.mean(rolled, axis=0).tolist(),
        "edge_rolled_action_delta_mean": float(np.mean(rolled)),
        "edge_rolled_action_delta_max": float(np.max(rolled)),
    }


def _write_existing_validation_diagnostics() -> dict[str, Any]:
    trajectory: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for label, _ in VARIANTS:
        run_dir = STAGE44 / "runs" / label / "seed-2026"
        eval_rows = _read_csv(run_dir / "eval.csv")
        train_rows = _read_csv(run_dir / "train.csv")
        if len(eval_rows) != 20 or len(train_rows) != UPDATES:
            raise ValueError(f"Stage-4.4 {label} logs are incomplete")
        for row in eval_rows:
            summary = json.loads(row["summary_json"])
            trajectory.append({"variant": label, "run_seed": BASELINE_SEED, "update": int(row["update"]),
                               **{key: summary[key] for key in ("mean_return", "mean_e2e_rate_mbps",
                                   "rate_satisfaction_ratio", "outage_step_ratio", "normal_completion_count",
                                   "persistent_outage_count", "collision_count", "boundary_count",
                                   "speed_accel_violations")}})
            failures.append({"variant": label, "run_seed": BASELINE_SEED, "update": int(row["update"]),
                             "normal": int(summary["normal_completion_count"]),
                             "persistent_outage": int(summary["persistent_outage_count"]),
                             "collision": int(summary["collision_count"]), "boundary": int(summary["boundary_count"]),
                             "hard_speed_accel_violations": int(summary["speed_accel_violations"])})
        for window_name, count in (("last_50_updates", 50), ("last_100_updates", 100), ("all_updates", UPDATES)):
            window = train_rows[-count:]
            for kind, field in (("actor", "actor_grad_clip_fraction"), ("critic", "critic_grad_clip_fraction")):
                values = [float(row[field]) for row in window]
                gradients.append({"variant": label, "run_seed": BASELINE_SEED, "window": window_name,
                                  "gradient": kind, "updates": len(values), "mean": statistics.fmean(values),
                                  "std": statistics.pstdev(values), "median": statistics.median(values),
                                  "minimum": min(values), "maximum": max(values),
                                  "near_one_fraction": sum(value >= 0.99 for value in values) / len(values)})
    _write_csv(TRAJECTORY_CSV, trajectory, list(trajectory[0]))
    _write_csv(GRADIENT_CSV, gradients, list(gradients[0]))
    _write_csv(FAILURES_CSV, failures, list(failures[0]))
    return {"trajectory_rows": len(trajectory), "gradient_rows": len(gradients), "failure_rows": len(failures)}


def diagnose(checkpoint_root: Path) -> dict[str, Any]:
    protocol, protocol_hash = _prepare_diagnostic_protocol()
    if protocol["final_test"]["executed"] or protocol["final_test"]["permitted"]:
        raise AssertionError("Stage-4.5 diagnostic protocol must forbid final test")
    existing = _write_existing_validation_diagnostics()
    actor, config, selected = _load_selected_actor(checkpoint_root, "P4")
    activations = _graph_state_probe(actor, config)
    outcomes = {mode: _evaluate_graph_mode(actor, config, mode)[0]
                for mode in ("baseline", "message_blocked", "edge_rolled")}
    threshold = protocol["graph_probe"]["thresholds"]
    message_variance = activations["message_vector_variance_by_round"]
    checks = {
        "trace_matches_frozen_graph_implementation": activations["trace_max_abs_error"] <= threshold["trace_abs_error_max"],
        "message_vectors_not_collapsed": min(message_variance) > threshold["message_vector_variance_min"],
        "node_embeddings_not_collapsed": activations["node_embedding_variance_mean"] > threshold["node_embedding_variance_min"],
        "actions_respond_to_message_block": activations["message_blocked_action_delta_mean"] > threshold["mean_abs_action_delta_min"],
        "actions_respond_to_edge_perturbation": activations["edge_rolled_action_delta_mean"] > threshold["mean_abs_action_delta_min"],
        "all_probe_outputs_finite": all(math.isfinite(float(value)) for value in (
            activations["trace_max_abs_error"], activations["node_embedding_variance_mean"],
            activations["action_variance_mean"], activations["message_blocked_action_delta_mean"],
            activations["edge_rolled_action_delta_mean"], *message_variance)),
    }
    verdict = "GRAPH_INFO_CONFIRMED" if all(checks.values()) else "BLOCK_MULTISEED_TRAINING"
    report = {
        "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
        "stage44_selected_p4": {"update": selected["selected_update"], "sha256": selected["checkpoint_sha256"]},
        "existing_validation": existing, "graph_activations": activations, "sensitivity_validation": outcomes,
        "checks": checks, "verdict": verdict, "final_test_executed": False,
        "minimal_fix_if_blocked": (
            "Preserve Stage-4.4 evidence unchanged; create a new protocol, repair only the failing Graph message/edge pathway, "
            "add a regression test for the failed signal, then retrain every compared P3/P4 seed from update 0."
            if verdict != "GRAPH_INFO_CONFIRMED" else None),
    }
    _write_json(GRAPH_ACTIVATIONS, {"protocol_sha256": protocol_hash, **activations})
    _write_json(GRAPH_SENSITIVITY, {"protocol_sha256": protocol_hash, "outcomes": outcomes, "checks": checks})
    _write_json(RELEASE_RECORD, protocol["release"])
    _write_json(DIAGNOSTIC_REPORT, report)
    lines = ["# Stage 4.5 P3/P4 diagnostic", "", f"Verdict: **{verdict}**.", "",
             "The probes use only the locked validation seeds; final-test was not executed.", "",
             f"P4 selected update: {selected['selected_update']}.",
             f"Message-block mean action delta: {activations['message_blocked_action_delta_mean']:.9g}.",
             f"Edge-roll mean action delta: {activations['edge_rolled_action_delta_mean']:.9g}."]
    if verdict != "GRAPH_INFO_CONFIRMED":
        lines.extend(("", report["minimal_fix_if_blocked"]))
    DIAGNOSTIC_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return report


def _prepare_multiseed_protocol() -> tuple[dict[str, Any], str]:
    diagnostic = _read_json(DIAGNOSTIC_REPORT)
    if diagnostic.get("verdict") != "GRAPH_INFO_CONFIRMED":
        raise RuntimeError("Stage-4.5 graph gate is not healthy; multi-seed training is prohibited")
    p3_seed_2027 = _config_for("P3", "topology_info", 2027)
    payload = {
        "protocol_version": MULTISEED_PROTOCOL_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _run_git("rev-parse", "HEAD"), "dirty_worktree_at_lock": bool(_run_git("status", "--porcelain")),
        "diagnostic_protocol_sha256": sha256_json_file(DIAGNOSTIC_PROTOCOL),
        "diagnostic_verdict": diagnostic["verdict"], "paired_seeds": list((BASELINE_SEED, *NEW_SEEDS)),
        "new_training_seeds": list(NEW_SEEDS), "variants": [{"label": label, "actor_variant": variant}
                                                             for label, variant in VARIANTS],
        "stage4_formal_common_config": {key: value for key, value in p3_seed_2027.to_dict().items()
                                         if key not in {"actor_variant", "output_dir", "run_seed", "actor_init_seed",
                                                        "critic_init_seed", "action_noise_seed", "minibatch_seed",
                                                        "train_env_seed_base", "train_env_seed_base"}},
        "independent_rng_seeds": {str(seed): {label: {name: getattr(_config_for(label, variant, seed), name)
                                                         for name in ("actor_init_seed", "critic_init_seed", "action_noise_seed",
                                                                      "minibatch_seed", "train_env_seed_base")}
                                                for label, variant in VARIANTS} for seed in NEW_SEEDS},
        "validation": {"seed_manifest_sha256": sha256_json_file(ROOT / "artifacts/stage4-1-protocol-repair/seed_manifest.json"),
                       "seeds": list(VALIDATION_SEEDS), "updates": list(EVAL_UPDATES), "episodes_per_run": 400},
        "training_budget": {"new_runs": 4, "updates_per_run": UPDATES, "samples_per_run": UPDATES * SAMPLES_PER_UPDATE,
                            "new_total_samples": 4 * UPDATES * SAMPLES_PER_UPDATE},
        "final_test": {"executed": False, "permitted": False},
        "selection_rule": "safety_priority_key: hard violations, normal completion, collision+boundary, persistent outage, outage ratio, rate satisfaction, mean e2e; earlier update only on exact tie",
    }
    proposed = _canonical_payload(payload)
    if proposed["dirty_worktree_at_lock"]:
        raise RuntimeError("Stage-4.5 multi-seed protocol must be locked from a clean worktree")
    if MULTI_PROTOCOL.exists():
        existing = _read_json(MULTI_PROTOCOL)
        left = {key: value for key, value in existing.items() if key != "generated_at_utc"}
        right = {key: value for key, value in proposed.items() if key != "generated_at_utc"}
        if left != right:
            raise ValueError("Stage-4.5 multi-seed protocol is locked and differs from current contract")
        return existing, sha256_json_payload(existing)
    _write_json(MULTI_PROTOCOL, proposed)
    return proposed, sha256_json_payload(proposed)


def _append_new_episodes(protocol_hash: str, label: str, actor_variant: str, run_seed: int,
                         update: int, seeds: list[int], episodes: list[dict[str, Any]]) -> None:
    if tuple(seeds) != VALIDATION_SEEDS:
        raise ValueError("Stage-4.5 recorder refuses non-validation seeds")
    checkpoint = _run_dir(label, run_seed) / "eval_checkpoints" / f"actor_update_{update:04d}.pt"
    with MULTI_EPISODES.open("a", encoding="utf-8", newline="\n") as handle:
        for seed, episode in zip(seeds, episodes):
            row = {"protocol_version": MULTISEED_PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
                   "variant": label, "actor_variant": actor_variant, "run_seed": run_seed, "update": update,
                   "split": "validation", "seed": int(seed),
                   "checkpoint_relative_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"), **episode}
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def _equal_state(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal_state(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_equal_state(a, b) for a, b in zip(left, right))
    return left == right


def _checkpoint_reload_check(config: MappoConfig, trainer: MappoTrainer) -> dict[str, bool]:
    latest = Path(config.output_dir) / "checkpoints" / "latest.pt"
    payload = load_checkpoint(latest, config.device)
    restored = MappoTrainer(config)
    restored.load(latest)
    rng = payload["protocol_rng_state"]
    return {
        "latest_exists": latest.is_file(), "update": restored.update_count == UPDATES == int(payload["update"]),
        "total_env_steps": restored.total_env_steps == UPDATES * SAMPLES_PER_UPDATE == int(payload["total_env_steps"]),
        "actor": _equal_state(trainer.actor.state_dict(), restored.actor.state_dict()),
        "critic": _equal_state(trainer.critic.state_dict(), restored.critic.state_dict()),
        "actor_optimizer": _equal_state(payload["actor_optimizer_state"], restored.actor_optimizer.state_dict()),
        "critic_optimizer": _equal_state(payload["critic_optimizer_state"], restored.critic_optimizer.state_dict()),
        "normalizer": _equal_state(payload["normalizer_state"], restored.normalizer.state_dict()),
        "value_normalizer_v0": payload["value_normalizer_state"] is None and restored.value_normalizer is None,
        "protocol_rng": (torch.equal(restored.action_noise_generator.get_state(), rng["action_noise_generator"])
                         and _equal_state(restored.minibatch_rng.bit_generator.state, rng["minibatch_bit_generator"])),
        "fresh_boundary_resume": (payload["resume_mode"] == "fresh_episode_at_update_boundary"
                                  and payload["exact_environment_resume"] is False),
    }


def _train_one(label: str, actor_variant: str, run_seed: int, protocol_hash: str) -> dict[str, Any]:
    config = _config_for(label, actor_variant, run_seed)
    trainer = MappoTrainer(config)
    trainer.evaluation_episode_recorder = lambda update, seeds, summary, score, episodes: _append_new_episodes(
        protocol_hash, label, actor_variant, run_seed, update, seeds, episodes)
    records = trainer.train(UPDATES, final_evaluation=False, export_final_actor=True)
    if len(records) != UPDATES or trainer.update_count != UPDATES or trainer.total_env_steps != UPDATES * SAMPLES_PER_UPDATE:
        raise RuntimeError(f"{label} seed-{run_seed} did not complete the exact Stage-4.5 budget")
    reload_check = _checkpoint_reload_check(config, trainer)
    if not all(reload_check.values()):
        raise RuntimeError(f"{label} seed-{run_seed} checkpoint reload failed: {reload_check}")
    _write_json(_run_dir(label, run_seed) / "stage4_5_complete.json", {
        "protocol_sha256": protocol_hash, "label": label, "run_seed": run_seed, "updates": UPDATES,
        "training_samples": UPDATES * SAMPLES_PER_UPDATE, "checkpoint_reload": reload_check,
        "final_test_executed": False,
    })
    return {"status": "complete", "label": label, "actor_variant": actor_variant, "run_seed": run_seed,
            "updates": UPDATES, "training_samples": UPDATES * SAMPLES_PER_UPDATE,
            "checkpoint_reload": reload_check, "final_test_executed": False}


def _episode_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite_row(row: dict[str, str]) -> bool:
    for value in row.values():
        if value == "":
            continue
        try:
            if not math.isfinite(float(value)):
                return False
        except ValueError:
            continue
    return True


def _same_summary(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare JSON-reloaded metric summaries without binary-float noise."""
    if left.keys() != right.keys():
        return False
    for key in left:
        first, second = left[key], right[key]
        if isinstance(first, (int, float)) and isinstance(second, (int, float)):
            if not math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1e-12):
                return False
        elif first != second:
            return False
    return True


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from empty validation summaries")
    return min(rows, key=lambda row: (tuple(row["score"]), int(row["update"])))


def _validate_new_run(label: str, actor_variant: str, run_seed: int, protocol_hash: str,
                      episodes: list[dict[str, Any]], verify_binaries: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = _run_dir(label, run_seed)
    actual = MappoConfig.from_dict(_read_json(run_dir / "config.json")).to_dict()
    expected = _config_for(label, actor_variant, run_seed).to_dict()
    saved_output = str(actual.pop("output_dir")).replace("\\", "/")
    expected.pop("output_dir")
    suffix = str(run_dir.relative_to(ROOT)).replace("\\", "/")
    if not saved_output.endswith(suffix) or actual != expected:
        raise ValueError(f"{label} seed-{run_seed} config differs from frozen stage4-formal V0H0")
    train = _read_csv(run_dir / "train.csv")
    if len(train) != UPDATES or [int(row["update"]) for row in train] != list(range(1, UPDATES + 1)):
        raise ValueError(f"{label} seed-{run_seed} train log is not continuous 1..1000")
    if not all(_finite_row(row) and float(row["finite"]) == 1.0 for row in train):
        raise ValueError(f"{label} seed-{run_seed} train log contains non-finite data")
    if sum(float(row["hard_speed_accel_violations"]) for row in train) != 0.0:
        raise ValueError(f"{label} seed-{run_seed} has hard speed/acceleration violations")
    evaluation = _read_csv(run_dir / "eval.csv")
    if len(evaluation) != len(EVAL_UPDATES) or tuple(int(row["update"]) for row in evaluation) != EVAL_UPDATES:
        raise ValueError(f"{label} seed-{run_seed} does not have exactly 20 evaluation summaries")
    rows = [row for row in episodes if row["variant"] == label and int(row["run_seed"]) == run_seed]
    if len(rows) != 400 or len({(int(row["update"]), int(row["seed"])) for row in rows}) != 400:
        raise ValueError(f"{label} seed-{run_seed} does not have complete raw validation coverage")
    summaries: list[dict[str, Any]] = []
    for row in evaluation:
        update = int(row["update"])
        if tuple(int(seed) for seed in row["seeds"].split(",")) != VALIDATION_SEEDS:
            raise ValueError(f"{label} seed-{run_seed} used a non-validation split")
        expected_summary = json.loads(row["summary_json"])
        raw = [{key: value for key, value in item.items() if key not in METADATA_FIELDS}
               for item in rows if int(item["update"]) == update]
        recomputed = summarize_episodes(raw)
        if not _same_summary(recomputed, expected_summary):
            raise ValueError(f"{label} seed-{run_seed} summary at update {update} does not recompute")
        score = tuple(float(value) for value in json.loads(row["score"]))
        if tuple(float(value) for value in safety_priority_key(recomputed)) != score:
            raise ValueError(f"{label} seed-{run_seed} selection key drifted")
        summaries.append({"update": update, "summary": recomputed, "score": score})
    paths = [run_dir / "eval_checkpoints" / f"actor_update_{update:04d}.pt" for update in EVAL_UPDATES]
    if verify_binaries and not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"{label} seed-{run_seed} periodic Actor checkpoint missing")
    selected = _select(summaries)
    last = train[-1]
    report = {"label": label, "actor_variant": actor_variant, "run_seed": run_seed, "train_rows": UPDATES,
              "validation_summaries": 20, "validation_episodes": 400, "selected_update": selected["update"],
              "selected_summary": selected["summary"], "selected_score": list(selected["score"]),
              "checkpoint_relative_path": str(paths[EVAL_UPDATES.index(selected["update"])].relative_to(ROOT)).replace("\\", "/"),
              "checkpoint_reload": _read_json(run_dir / "stage4_5_complete.json")["checkpoint_reload"],
              "diagnostics": {key: float(last[key]) for key in ("explained_variance_pre", "actor_grad_clip_fraction",
                  "critic_grad_clip_fraction", "action_saturation_ratio", "clamped_log_std_min", "clamped_log_std_max")}}
    return report, paths + [run_dir / "checkpoints" / name for name in ("latest.pt", "best.pt", "actor_final.pt")]


def _selected_2026_reports() -> list[dict[str, Any]]:
    selected = _stage44_selected()
    return [{"label": item["variant"], "actor_variant": item["actor_variant"], "run_seed": BASELINE_SEED,
             "selected_update": item["selected_update"], "selected_summary": item["validation_summary"],
             "selected_score": item["validation_score"], "checkpoint_relative_path": item["checkpoint_relative_path"],
             "checkpoint_sha256": item["checkpoint_sha256"]} for item in selected["selected"]]


def _write_multiseed_artifacts(protocol_hash: str, *, write: bool, verify_binaries: bool) -> dict[str, Any]:
    episodes = _episode_rows(MULTI_EPISODES)
    if len(episodes) != 1600:
        raise ValueError("Stage-4.5 requires exactly 1,600 new raw validation episodes")
    reports: list[dict[str, Any]] = []
    binary_paths: list[tuple[str, int, str, Path]] = []
    for run_seed in NEW_SEEDS:
        for label, variant in VARIANTS:
            report, paths = _validate_new_run(label, variant, run_seed, protocol_hash, episodes, verify_binaries)
            reports.append(report)
            for index, path in enumerate(paths):
                binary_paths.append((label, run_seed, "periodic_validation_actor" if index < 20 else "training_checkpoint", path))
    entries: list[dict[str, Any]] = []
    if verify_binaries:
        for label, seed, category, path in binary_paths:
            if not path.is_file():
                raise FileNotFoundError(f"missing Stage-4.5 checkpoint: {path}")
            entries.append({"variant": label, "run_seed": seed, "category": category,
                            "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(path)})
    else:
        entries = list(_read_json(MULTI_CHECKPOINT_MANIFEST).get("entries", []))
        expected = {str(path.relative_to(ROOT)).replace("\\", "/"): (label, seed, category)
                    for label, seed, category, path in binary_paths}
        if len(entries) != len(expected) or {item.get("relative_path") for item in entries} != set(expected):
            raise ValueError("Stage-4.5 checkpoint metadata is incomplete")
        for item in entries:
            if expected[item["relative_path"]] != (item.get("variant"), item.get("run_seed"), item.get("category")):
                raise ValueError("Stage-4.5 checkpoint metadata does not match expected paths")
            if not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64:
                raise ValueError("Stage-4.5 checkpoint metadata has invalid SHA-256")
    checkpoint_manifest = {"protocol_version": MULTISEED_PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
                           "hash_scheme": "sha256-file-v1", "entries": entries, "final_test_executed": False}
    if write:
        _write_json(MULTI_CHECKPOINT_MANIFEST, checkpoint_manifest)
    elif _read_json(MULTI_CHECKPOINT_MANIFEST) != checkpoint_manifest:
        raise ValueError("Stage-4.5 checkpoint manifest drifted")
    hashes = {item["relative_path"]: item["sha256"] for item in entries}
    selected: list[dict[str, Any]] = []
    for report in reports:
        selected.append({"variant": report["label"], "actor_variant": report["actor_variant"], "run_seed": report["run_seed"],
                         "selected_update": report["selected_update"], "checkpoint_relative_path": report["checkpoint_relative_path"],
                         "checkpoint_sha256": hashes[report["checkpoint_relative_path"]], "validation_score": report["selected_score"],
                         "validation_summary": report["selected_summary"], "final_test_used": False})
    selected_manifest = {"protocol_version": MULTISEED_PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
                         "selected": selected, "final_test_executed": False}
    if write:
        _write_json(MULTI_SELECTED, selected_manifest)
    elif _read_json(MULTI_SELECTED) != selected_manifest:
        raise ValueError("Stage-4.5 selected checkpoint manifest drifted")
    all_reports = _selected_2026_reports() + reports
    fields = ("normal_completion_count", "persistent_outage_count", "collision_count", "boundary_count",
              "outage_step_ratio", "rate_satisfaction_ratio", "mean_e2e_rate_mbps")
    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for label, _ in VARIANTS:
        by_label = sorted((report for report in all_reports if report["label"] == label), key=lambda item: item["run_seed"])
        if [item["run_seed"] for item in by_label] != [BASELINE_SEED, *NEW_SEEDS]:
            raise ValueError(f"{label} lacks one of the three paired seeds")
        for field in fields:
            values = [float(item["selected_summary"][field]) for item in by_label]
            summary_rows.append({"variant": label, "metric": field, "n": len(values), "mean": statistics.fmean(values),
                                 "std": statistics.stdev(values), "median": statistics.median(values)})
    wins = {"P3": 0, "P4": 0, "tie": 0}
    for seed in (BASELINE_SEED, *NEW_SEEDS):
        p3 = next(item for item in all_reports if item["label"] == "P3" and item["run_seed"] == seed)
        p4 = next(item for item in all_reports if item["label"] == "P4" and item["run_seed"] == seed)
        key3, key4 = tuple(p3["selected_score"]), tuple(p4["selected_score"])
        winner = "P3" if key3 < key4 else "P4" if key4 < key3 else "tie"
        wins[winner] += 1
        row: dict[str, Any] = {"run_seed": seed, "winner_by_safety_priority_key": winner,
                               "p3_selected_update": p3["selected_update"], "p4_selected_update": p4["selected_update"]}
        for field in fields:
            row[f"p3_{field}"] = p3["selected_summary"][field]
            row[f"p4_{field}"] = p4["selected_summary"][field]
            row[f"p4_minus_p3_{field}"] = float(p4["selected_summary"][field]) - float(p3["selected_summary"][field])
        paired_rows.append(row)
    if write:
        _write_csv(MULTI_SUMMARY, summary_rows, list(summary_rows[0]))
        _write_csv(MULTI_PAIRED, paired_rows, list(paired_rows[0]))
    elif _read_csv(MULTI_SUMMARY) != [{key: str(value) for key, value in row.items()} for row in summary_rows]:
        raise ValueError("Stage-4.5 aggregate summary CSV drifted")
    audit = {"protocol_version": MULTISEED_PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
             "result_status": "THREE-SEED P3/P4 VALIDATION-ONLY COMPARISON", "final_test_executed": False,
             "new_training_samples": 4 * UPDATES * SAMPLES_PER_UPDATE, "new_validation_episodes": 1600,
             "runs": reports, "aggregates": summary_rows, "paired_differences": paired_rows, "win_counts": wins,
             "interpretation": "P3/P4-only paired three-seed comparison; no final-test or P0/P1/P2 retraining."}
    if write:
        _write_json(MULTI_AUDIT, audit)
        lines = ["# Stage 4.5 P3/P4 paired multi-seed audit", "", "Final-test was not executed.", "",
                 f"Safety-priority win counts: P3={wins['P3']}, P4={wins['P4']}, tie={wins['tie']}.", "",
                 "This is a P3/P4-only validation comparison; it does not authorize P0/P1/P2 training."]
        MULTI_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    elif _read_json(MULTI_AUDIT) != audit:
        raise ValueError("Stage-4.5 multi-seed audit drifted")
    return audit


def run_multiseed() -> dict[str, Any]:
    if not DIAGNOSTIC_REPORT.is_file():
        raise RuntimeError("run --diagnose first; multi-seed training is gated on its committed verdict")
    diagnostic = _read_json(DIAGNOSTIC_REPORT)
    if diagnostic.get("verdict") != "GRAPH_INFO_CONFIRMED":
        raise RuntimeError("Graph diagnostic blocked multi-seed training")
    protocol, protocol_hash = _prepare_multiseed_protocol()
    del protocol
    if MULTI_CHECKPOINT_MANIFEST.exists() or MULTI_SELECTED.exists() or MULTI_AUDIT.exists():
        raise RuntimeError("Stage-4.5 multi-seed artifacts already finalized; use --verify")
    prior = _read_json(MULTI_RUN_MANIFEST) if MULTI_RUN_MANIFEST.exists() else {"runs": {}}
    if prior.get("protocol_sha256") not in {None, protocol_hash}:
        raise ValueError("prior Stage-4.5 run manifest belongs to another protocol")
    runs: dict[str, Any] = dict(prior.get("runs", {}))
    _write_json(MULTI_RUN_MANIFEST, {"protocol_version": MULTISEED_PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
                                     "runs": runs, "final_test_executed": False})
    for run_seed in NEW_SEEDS:
        for label, variant in VARIANTS:
            key = f"{label}-seed-{run_seed}"
            if runs.get(key, {}).get("status") == "complete":
                continue
            if _run_dir(label, run_seed).exists():
                raise RuntimeError(f"refusing to resume or overwrite interrupted {key}; preserve evidence and restart under a new protocol")
            runs[key] = {"status": "running", "label": label, "actor_variant": variant, "run_seed": run_seed}
            _write_json(MULTI_RUN_MANIFEST, {"protocol_version": MULTISEED_PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
                                             "runs": runs, "final_test_executed": False})
            runs[key] = _train_one(label, variant, run_seed, protocol_hash)
            _write_json(MULTI_RUN_MANIFEST, {"protocol_version": MULTISEED_PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
                                             "runs": runs, "final_test_executed": False})
    audit = _write_multiseed_artifacts(protocol_hash, write=True, verify_binaries=True)
    return {"protocol_sha256": protocol_hash, "runs_completed": len(audit["runs"]),
            "new_training_samples": audit["new_training_samples"], "final_test_executed": False}


def verify() -> dict[str, Any]:
    diagnostic_required = (DIAGNOSTIC_PROTOCOL, DIAGNOSTIC_REPORT, DIAGNOSTIC_REPORT_MD, TRAJECTORY_CSV, GRADIENT_CSV,
                           FAILURES_CSV, GRAPH_ACTIVATIONS, GRAPH_SENSITIVITY, RELEASE_RECORD)
    missing = [str(path.relative_to(ROOT)) for path in diagnostic_required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage-4.5 diagnostic artifacts missing: {missing}")
    protocol = _read_json(DIAGNOSTIC_PROTOCOL)
    report = _read_json(DIAGNOSTIC_REPORT)
    if protocol.get("protocol_version") != DIAGNOSTIC_PROTOCOL_VERSION or protocol.get("final_test", {}).get("executed") is not False:
        raise ValueError("Stage-4.5 diagnostic protocol invalid")
    if report.get("protocol_sha256") != sha256_json_payload(protocol):
        raise ValueError("Stage-4.5 diagnostic report protocol hash drifted")
    if report.get("verdict") != "GRAPH_INFO_CONFIRMED":
        return {"diagnostic_verdict": report.get("verdict"), "multiseed_training": "BLOCKED", "final_test_executed": False}
    required = (MULTI_PROTOCOL, MULTI_RUN_MANIFEST, MULTI_EPISODES, MULTI_CHECKPOINT_MANIFEST, MULTI_SELECTED,
                MULTI_SUMMARY, MULTI_PAIRED, MULTI_AUDIT, MULTI_AUDIT_MD)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage-4.5 multi-seed artifacts missing: {missing}")
    multi_protocol = _read_json(MULTI_PROTOCOL)
    if multi_protocol.get("final_test", {}).get("executed") is not False or multi_protocol.get("diagnostic_verdict") != "GRAPH_INFO_CONFIRMED":
        raise ValueError("Stage-4.5 multi-seed protocol invalid")
    protocol_hash = sha256_json_payload(multi_protocol)
    run_manifest = _read_json(MULTI_RUN_MANIFEST)
    if run_manifest.get("protocol_sha256") != protocol_hash or len(run_manifest.get("runs", {})) != 4:
        raise ValueError("Stage-4.5 multi-seed run manifest incomplete")
    if any(item.get("status") != "complete" for item in run_manifest["runs"].values()):
        raise ValueError("Stage-4.5 multi-seed run incomplete")
    paths = [path for seed in NEW_SEEDS for label, _ in VARIANTS for path in (
        *[_run_dir(label, seed) / "eval_checkpoints" / f"actor_update_{update:04d}.pt" for update in EVAL_UPDATES],
        *[_run_dir(label, seed) / "checkpoints" / name for name in ("latest.pt", "best.pt", "actor_final.pt")])]
    present = [path for path in paths if path.is_file()]
    if present and len(present) != len(paths):
        raise FileNotFoundError("Stage-4.5 checkpoint set partially present")
    audit = _write_multiseed_artifacts(protocol_hash, write=False, verify_binaries=len(present) == len(paths))
    return {"diagnostic_verdict": report["verdict"], "runs_completed": len(audit["runs"]),
            "new_validation_episodes": audit["new_validation_episodes"], "final_test_executed": False,
            "checkpoint_binary_verification": "PASS" if len(present) == len(paths) else "SKIPPED_MISSING_LOCAL_CHECKPOINTS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--diagnose", action="store_true", help="Run Stage-4.5 validation-only diagnostics.")
    group.add_argument("--run", action="store_true", help="Run gated P3/P4 seeds 2027 and 2028 training.")
    group.add_argument("--verify", action="store_true", help="Read-only verify completed Stage-4.5 artifacts.")
    parser.add_argument("--checkpoint-root", type=Path, help="Root that contains the local Stage-4.4 selected checkpoints.")
    args = parser.parse_args()
    if args.diagnose:
        if args.checkpoint_root is None:
            parser.error("--diagnose requires --checkpoint-root; checkpoints are intentionally not committed")
        result = diagnose(args.checkpoint_root)
    elif args.run:
        result = run_multiseed()
    else:
        result = verify()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
