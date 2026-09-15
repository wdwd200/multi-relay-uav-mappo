"""Read-only Stage 4.1 checkpoint-grid evaluation.

Validation evaluation may process a numeric grid of periodic Actor checkpoints.
The test split is intentionally restricted to exactly one already-selected
checkpoint, preventing post-hoc model selection on final-test outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from plain_mappo.config import MappoConfig
from plain_mappo.evaluation import evaluate_actor
from plain_mappo.experiment_protocol import (CANONICAL_JSON_HASH_SCHEME, checkpoint_update,
                                              load_seed_manifest, sha256_file, split_seeds)
from plain_mappo.metrics import safety_priority_key
from plain_mappo.networks import SharedActor


GRID_FIELDS = ("variant", "run_seed", "protocol_status", "update", "checkpoint_path", "checkpoint_sha256",
               "checkpoint_size_bytes", "validation_seed_set", "seed_manifest_sha256", "seed_manifest_hash_scheme", "episodes", "score",
               "normal_completion_count", "collision_count", "boundary_count", "persistent_outage_count",
               "speed_accel_violations", "outage_step_ratio", "rate_satisfaction_ratio", "mean_e2e_rate_mbps",
               "mean_return", "config_summary", "deterministic_action")
FINAL_FIELDS = tuple(field.replace("validation_seed_set", "final_test_seed_set") for field in GRID_FIELDS)


def load_actor_checkpoint(path: Path, device: str) -> tuple[SharedActor, MappoConfig, dict[str, Any]]:
    """Load a lightweight periodic Actor checkpoint without mutating it."""
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    config = MappoConfig.from_dict(payload["config"])
    actor = SharedActor(config.local_obs_dim, config.action_dim, config.actor_hidden_sizes,
                        config.log_std_min, config.log_std_max, config.actor_log_std_mode,
                        config.state_independent_log_std_init, config.actor_variant,
                        config.topology_node_dim, config.topology_edge_dim,
                        config.graph_hidden_dim).to(device)
    actor.load_state_dict(payload["actor_state"])
    actor.eval()
    return actor, config, payload


def checkpoint_paths(run_dir: Path, checkpoint_glob: str) -> list[Path]:
    paths = list(run_dir.glob(checkpoint_glob))
    if not paths:
        raise FileNotFoundError(f"no checkpoints under {run_dir} matching {checkpoint_glob!r}")
    return sorted(paths, key=checkpoint_update)


def config_summary(config: MappoConfig) -> dict[str, Any]:
    return {key: config.to_dict()[key] for key in (
        "actor_variant", "num_relays", "local_obs_dim", "global_state_dim", "actor_hidden_sizes",
        "critic_hidden_sizes", "actor_log_std_mode", "log_std_min", "log_std_max", "gamma",
        "gae_lambda", "clip_epsilon", "actor_lr", "critic_lr", "entropy_coef", "protocol_version",
        "run_seed", "resume_mode", "exact_environment_resume")}


def evaluate_checkpoint(path: Path, seeds: Iterable[int], *, split: str, device: str,
                        seed_manifest_sha256: str, run_seed_override: int | None = None,
                        seed_manifest_hash_scheme: str = CANONICAL_JSON_HASH_SCHEME) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actor, config, payload = load_actor_checkpoint(path, device)
    seed_list = [int(seed) for seed in seeds]
    summary, score, episodes = evaluate_actor(actor, config, seed_list)
    update = checkpoint_update(path)
    digest = sha256_file(path)
    run_seed = int(run_seed_override if run_seed_override is not None else config.run_seed)
    base = {
        "variant": config.actor_variant,
        "run_seed": run_seed,
        "protocol_status": config.protocol_version,
        "update": update,
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "checkpoint_size_bytes": path.stat().st_size,
        "seed_manifest_sha256": seed_manifest_sha256,
        "seed_manifest_hash_scheme": seed_manifest_hash_scheme,
        "episodes": len(episodes),
        "score": list(score),
        "config_summary": config_summary(config),
        "deterministic_action": "tanh(mean)",
        "checkpoint_type": payload.get("checkpoint_type", "periodic_evaluation_actor"),
    }
    row = {**base, **summary}
    episode_rows = [{**base, "split": split, "seed": seed, **episode}
                    for seed, episode in zip(seed_list, episodes)]
    return row, episode_rows


def select_checkpoint_from_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select by frozen score; equal scores use the earlier periodic update."""
    if not rows:
        raise ValueError("validation selection requires at least one row")
    if any(row.get("split") == "test" for row in rows):
        raise ValueError("test rows are forbidden in checkpoint selection")
    def key(row: dict[str, Any]) -> tuple[float, ...]:
        score = tuple(float(value) for value in row["score"])
        return score + (float(row["update"]),)
    return min(rows, key=key)


def assert_episode_completeness(episodes: list[dict[str, Any]], checkpoint_hashes: Iterable[str],
                                seeds: Iterable[int]) -> None:
    """Require one finite raw Episode record for every checkpoint/seed pair."""
    expected = {(str(checkpoint_hash), int(seed)) for checkpoint_hash in checkpoint_hashes for seed in seeds}
    actual = {(str(row.get("checkpoint_sha256")), int(row.get("seed"))) for row in episodes}
    if len(episodes) != len(expected) or actual != expected:
        raise ValueError("raw Episode rows must equal checkpoint count × seed count without duplicates")
    for row in episodes:
        for field in ("return", "length", "mean_e2e_rate_mbps", "rate_satisfaction_ratio", "outage_step_ratio",
                      "min_separation_m", "max_xy_speed_mps", "max_xy_accel_mps2", "action_saturation_ratio",
                      "movement_distance_m"):
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError(f"Episode row has non-finite required metric {field}")


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else str(value)


def _append_csv(path: Path, fields: tuple[str, ...], row: dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({field: _json_cell(row.get(field, "")) for field in fields})


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def run_grid(*, run_dir: Path, checkpoint_glob: str, seed_manifest: Path, split: str,
             output_dir: Path, device: str, run_seed: int | None = None,
             updates: set[int] | None = None) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    manifest, manifest_hash = load_seed_manifest(seed_manifest)
    seeds = split_seeds(manifest, split)
    paths = checkpoint_paths(run_dir, checkpoint_glob)
    if updates is not None:
        paths = [path for path in paths if checkpoint_update(path) in updates]
        if not paths:
            raise ValueError("--updates selected no checkpoint files")
    if split == "test" and len(paths) != 1:
        raise ValueError("test split accepts exactly one previously selected checkpoint; grid selection is validation-only")
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / ("validation_grid.csv" if split == "validation" else "final_test_summary.csv")
    episode_path = output_dir / ("validation_episodes.jsonl" if split == "validation" else "final_test_episodes.jsonl")
    rows: list[dict[str, Any]] = []
    for path in paths:
        row, episodes = evaluate_checkpoint(path, seeds, split=split, device=device,
                                            seed_manifest_sha256=manifest_hash, run_seed_override=run_seed,
                                            seed_manifest_hash_scheme=manifest["manifest_hash_scheme"])
        row["split"] = split
        row["validation_seed_set" if split == "validation" else "final_test_seed_set"] = list(seeds)
        _append_csv(grid_path, GRID_FIELDS if split == "validation" else FINAL_FIELDS, row)
        _append_jsonl(episode_path, episodes)
        rows.append(row)
    result = {"split": split, "rows": len(rows), "episodes": sum(int(row["episodes"]) for row in rows),
            "grid_path": str(grid_path), "episode_path": str(episode_path), "seed_manifest_sha256": manifest_hash,
            "seed_manifest_hash_scheme": manifest["manifest_hash_scheme"],
            "run_dir": str(run_dir), "checkpoint_glob": checkpoint_glob, "device": device,
            "started_at_utc": started_at, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    _append_jsonl(output_dir / "evaluation_invocations.jsonl", [result])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Stage 4.1 deterministic checkpoint-grid evaluator.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-glob", required=True, help="Glob relative to --run-dir, normally eval_checkpoints/actor_update_*.pt")
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-seed", type=int, help="Historical run seed recorded in output; old checkpoints are legacy_protocol.")
    parser.add_argument("--updates", help="Optional comma-separated periodic updates for resumable bounded batches.")
    args = parser.parse_args()
    updates = None if args.updates is None else {int(value) for value in args.updates.split(",") if value.strip()}
    result = run_grid(run_dir=args.run_dir, checkpoint_glob=args.checkpoint_glob, seed_manifest=args.seed_manifest,
                      split=args.split, output_dir=args.output_dir, device=args.device, run_seed=args.run_seed,
                      updates=updates)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
