"""One-time, idempotent migration from raw-file JSON hashes to canonical JSON.

This utility is intentionally a metadata migration.  It never evaluates an
Actor, opens a training loop, or changes any Episode metric.  Checkpoint SHA-
256 values remain raw binary-file hashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import torch

from plain_mappo.experiment_protocol import (CANONICAL_JSON_HASH_SCHEME,
                                              canonical_json_bytes, load_seed_manifest,
                                              sha256_file, sha256_json_file,
                                              sha256_json_payload)
from plain_mappo.config import MappoConfig
from plain_mappo.metrics import summarize_episodes
from plain_mappo.networks import SharedActor
from reevaluate_checkpoint_grid import assert_episode_completeness


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "stage4-1-protocol-repair"
SEED_MANIFEST = OUTPUT / "seed_manifest.json"
CHECKPOINT_MANIFEST = OUTPUT / "checkpoint_manifest.json"
SELECTED = OUTPUT / "selected_checkpoints.json"
VALIDATION_GRID = OUTPUT / "validation_grid.csv"
VALIDATION_EPISODES = OUTPUT / "validation_episodes.jsonl"
FINAL_SUMMARY = OUTPUT / "final_test_summary.csv"
FINAL_EPISODES = OUTPUT / "final_test_episodes.jsonl"
AUDIT_JSON = OUTPUT / "protocol_audit.json"
MIGRATION = OUTPUT / "hash_migration.json"
INVOCATIONS = OUTPUT / "evaluation_invocations.jsonl"

OLD_RAW_HASHES = {
    "seed_manifest": "d2622a91c144aa7d32ccd5bbf0353694ba281a2044b7985a0290d4c9e7227acb",
    "checkpoint_manifest": "c7dcbb98493230c890758dbcb76295ce0b34f99a0d651e2b06aa0601f787a435",
    "selected_checkpoints": "ff21a6534bbe646fa356a9d255695b8c2742faa290c9dad31f6dfaf37cd1df31",
}
MUTABLE_EPISODE_REFERENCE_FIELDS = {"seed_manifest_sha256", "seed_manifest_hash_scheme"}
REFERENCE_DOCUMENTS = (
    OUTPUT / "protocol_audit.md",
    ROOT / "stage4_1_protocol_validation.md",
    ROOT / "22_阶段4.1_训练与评估协议修复交接.md",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                            for row in rows), encoding="utf-8", newline="\n")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _immutable_episode_digest(rows: list[dict[str, Any]]) -> str:
    projected = [{key: value for key, value in row.items() if key not in MUTABLE_EPISODE_REFERENCE_FIELDS}
                 for row in rows]
    return sha256_json_payload(projected)


def _repository_path(value: str) -> Path:
    # Historical JSON was written on Windows.  Preserve the string in the
    # artifact while making verification work after an LF-only clone elsewhere.
    return ROOT.joinpath(*PureWindowsPath(value).parts)


def _parse_summary_rows(path: Path) -> list[dict[str, Any]]:
    _, rows = _read_csv(path)
    parsed: list[dict[str, Any]] = []
    int_fields = {"episodes", "normal_completion_count", "collision_count", "boundary_count",
                  "persistent_outage_count", "speed_accel_violations"}
    float_fields = {"outage_step_ratio", "rate_satisfaction_ratio", "mean_e2e_rate_mbps", "mean_return"}
    for row in rows:
        value: dict[str, Any] = dict(row)
        for field in int_fields:
            value[field] = int(value[field])
        for field in float_fields:
            value[field] = float(value[field])
        parsed.append(value)
    return parsed


def _assert_summary_recompute(summary_rows: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> None:
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_hash.setdefault(str(episode["checkpoint_sha256"]), []).append(episode)
    for row in summary_rows:
        recomputed = summarize_episodes(by_hash[str(row["checkpoint_sha256"])])
        for field in ("episodes", "normal_completion_count", "collision_count", "boundary_count",
                      "persistent_outage_count", "speed_accel_violations"):
            if int(row[field]) != int(recomputed[field]):
                raise ValueError(f"raw Episode recomputation mismatch for {field}")
        for field in ("outage_step_ratio", "rate_satisfaction_ratio", "mean_e2e_rate_mbps"):
            if not math.isclose(float(row[field]), float(recomputed[field]), abs_tol=1e-12, rel_tol=0.0):
                raise ValueError(f"raw Episode recomputation mismatch for {field}")
        if not math.isclose(float(row["mean_return"]), float(recomputed["mean_return"]), abs_tol=1e-12, rel_tol=0.0):
            raise ValueError("raw Episode recomputation mismatch for mean_return")


def _check_historical_artifacts(*, verify_checkpoint_binaries: bool = True,
                                checkpoint_root: Path | None = None) -> dict[str, Any]:
    manifest, manifest_hash = load_seed_manifest(SEED_MANIFEST)
    checkpoint_manifest = _read_json(CHECKPOINT_MANIFEST)
    selected = _read_json(SELECTED)
    validation_rows = _parse_summary_rows(VALIDATION_GRID)
    validation_episodes = _read_jsonl(VALIDATION_EPISODES)
    final_rows = _parse_summary_rows(FINAL_SUMMARY)
    final_episodes = _read_jsonl(FINAL_EPISODES)
    checkpoint_records = [(label, record) for label, run in checkpoint_manifest["runs"].items()
                          for record in run["checkpoints"]]
    if len(checkpoint_records) != 60:
        raise ValueError("checkpoint manifest must contain exactly 60 periodic Actors")
    missing_checkpoints = 0
    for label, record in checkpoint_records:
        checkpoint = (_repository_path(record["path"]) if checkpoint_root is None
                      else checkpoint_root / label / "eval_checkpoints" / PureWindowsPath(record["path"]).name)
        if not checkpoint.is_file():
            missing_checkpoints += 1
            if verify_checkpoint_binaries:
                raise FileNotFoundError(f"binary checkpoint is unavailable: {record['path']}")
            continue
        if sha256_file(checkpoint) != record["sha256"]:
            raise ValueError(f"binary checkpoint hash mismatch: {record['path']}")
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint_payload["config"]
        if record.get("config_hash_scheme") != CANONICAL_JSON_HASH_SCHEME or record.get("config_sha256") != sha256_json_payload(config):
            raise ValueError(f"checkpoint config JSON hash mismatch: {record['path']}")
        # This is Release-only integration verification.  CI's metadata-only
        # mode intentionally never requires untracked .pt files.
        checkpoint_config = MappoConfig.from_dict(config)
        actor = SharedActor(checkpoint_config.local_obs_dim, checkpoint_config.action_dim,
                            checkpoint_config.actor_hidden_sizes, checkpoint_config.log_std_min,
                            checkpoint_config.log_std_max, checkpoint_config.actor_log_std_mode,
                            checkpoint_config.state_independent_log_std_init,
                            checkpoint_config.actor_variant, checkpoint_config.topology_node_dim,
                            checkpoint_config.topology_edge_dim, checkpoint_config.graph_hidden_dim)
        actor.load_state_dict(checkpoint_payload["actor_state"])
        if actor.uses_topology:
            raise ValueError(f"historical P0/P1/P2 checkpoint unexpectedly uses topology: {record['path']}")
    validation_hashes = [row["checkpoint_sha256"] for row in validation_rows]
    assert_episode_completeness(validation_episodes, validation_hashes, manifest["splits"]["validation"]["seeds"])
    selected_values = selected["selected"]
    expected_updates = {"P0": 1000, "P1": 800, "P2": 650}
    if {name: int(item["selected_update"]) for name, item in selected_values.items()} != expected_updates:
        raise ValueError("selected updates changed")
    selected_hashes = [item["checkpoint_sha256"] for item in selected_values.values()]
    assert_episode_completeness(final_episodes, selected_hashes, manifest["splits"]["test"]["seeds"])
    if len(validation_rows) != 60 or len(validation_episodes) != 1200 or len(final_rows) != 3 or len(final_episodes) != 150:
        raise ValueError("Stage 4.1 historical record counts changed")
    _assert_summary_recompute(validation_rows, validation_episodes)
    _assert_summary_recompute(final_rows, final_episodes)
    return {
        "seed_manifest_sha256": manifest_hash,
        "checkpoint_manifest_sha256": sha256_json_file(CHECKPOINT_MANIFEST),
        "selected_checkpoints_sha256": sha256_json_file(SELECTED),
        "validation_rows": len(validation_rows), "validation_episode_rows": len(validation_episodes),
        "final_rows": len(final_rows), "final_episode_rows": len(final_episodes),
        "validation_immutable_episode_payload_sha256": _immutable_episode_digest(validation_episodes),
        "final_immutable_episode_payload_sha256": _immutable_episode_digest(final_episodes),
        "checkpoint_binary_hashes": {record["path"]: record["sha256"] for _, record in checkpoint_records},
        "checkpoint_binary_verification": "PASS" if missing_checkpoints == 0 else "SKIPPED_MISSING_RELEASE_ASSETS",
    }


def _migrate_csv(path: Path, new_seed_hash: str) -> None:
    fields, rows = _read_csv(path)
    if "seed_manifest_hash_scheme" not in fields:
        fields.insert(fields.index("seed_manifest_sha256") + 1, "seed_manifest_hash_scheme")
    for row in rows:
        row["seed_manifest_sha256"] = new_seed_hash
        row["seed_manifest_hash_scheme"] = CANONICAL_JSON_HASH_SCHEME
    _write_csv(path, fields, rows)


def _migrate_episode_jsonl(path: Path, new_seed_hash: str) -> None:
    rows = _read_jsonl(path)
    for row in rows:
        row["seed_manifest_sha256"] = new_seed_hash
        row["seed_manifest_hash_scheme"] = CANONICAL_JSON_HASH_SCHEME
    _write_jsonl(path, rows)


def migrate() -> dict[str, Any]:
    existing = _read_json(MIGRATION) if MIGRATION.exists() else None
    if existing is None:
        actual_old = {"seed_manifest": sha256_file(SEED_MANIFEST),
                      "checkpoint_manifest": sha256_file(CHECKPOINT_MANIFEST),
                      "selected_checkpoints": sha256_file(SELECTED)}
        if actual_old != OLD_RAW_HASHES:
            raise ValueError("historical raw-file hashes do not match the approved Stage 4.1 values")

    pre_validation = _immutable_episode_digest(_read_jsonl(VALIDATION_EPISODES))
    pre_final = _immutable_episode_digest(_read_jsonl(FINAL_EPISODES))

    seed_manifest = _read_json(SEED_MANIFEST)
    seed_manifest["manifest_hash_scheme"] = CANONICAL_JSON_HASH_SCHEME
    _write_json(SEED_MANIFEST, seed_manifest)
    new_seed_hash = sha256_json_file(SEED_MANIFEST)

    checkpoint_manifest = _read_json(CHECKPOINT_MANIFEST)
    checkpoint_manifest["manifest_hash_scheme"] = CANONICAL_JSON_HASH_SCHEME
    for run in checkpoint_manifest["runs"].values():
        for record in run["checkpoints"]:
            config = torch.load(_repository_path(record["path"]), map_location="cpu", weights_only=False)["config"]
            record["config_sha256"] = sha256_json_payload(config)
            record["config_hash_scheme"] = CANONICAL_JSON_HASH_SCHEME
    _write_json(CHECKPOINT_MANIFEST, checkpoint_manifest)
    new_checkpoint_hash = sha256_json_file(CHECKPOINT_MANIFEST)

    selected = _read_json(SELECTED)
    selected["manifest_hash_scheme"] = CANONICAL_JSON_HASH_SCHEME
    for item in selected["selected"].values():
        item["validation_seed_manifest_sha256"] = new_seed_hash
        item["validation_seed_manifest_hash_scheme"] = CANONICAL_JSON_HASH_SCHEME
    _write_json(SELECTED, selected)
    new_selected_hash = sha256_json_file(SELECTED)

    _migrate_csv(VALIDATION_GRID, new_seed_hash)
    _migrate_episode_jsonl(VALIDATION_EPISODES, new_seed_hash)
    _migrate_csv(FINAL_SUMMARY, new_seed_hash)
    _migrate_episode_jsonl(FINAL_EPISODES, new_seed_hash)

    for path in (AUDIT_JSON, ROOT / "stage4_1_protocol_validation.json"):
        audit = _read_json(path)
        audit["manifest_hash_scheme"] = CANONICAL_JSON_HASH_SCHEME
        audit["seed_manifest_sha256"] = new_seed_hash
        audit["seed_manifest_hash_scheme"] = CANONICAL_JSON_HASH_SCHEME
        audit["checkpoint_manifest_sha256"] = new_checkpoint_hash
        audit["checkpoint_manifest_hash_scheme"] = CANONICAL_JSON_HASH_SCHEME
        audit["selected_checkpoints_sha256"] = new_selected_hash
        audit["selected_checkpoints_hash_scheme"] = CANONICAL_JSON_HASH_SCHEME
        _write_json(path, audit)

    for path in REFERENCE_DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        for old, new in ((OLD_RAW_HASHES["seed_manifest"], new_seed_hash),
                         (OLD_RAW_HASHES["checkpoint_manifest"], new_checkpoint_hash),
                         (OLD_RAW_HASHES["selected_checkpoints"], new_selected_hash)):
            text = text.replace(old, new)
        if "canonical-json-v1" not in text:
            text += "\n\n## Stage 4.1.1 hash migration\n\nManifest references use `canonical-json-v1`; binary checkpoint hashes remain raw SHA-256. Episode numeric payloads were not changed.\n"
        path.write_text(text, encoding="utf-8", newline="\n")

    invocations = _read_jsonl(INVOCATIONS)
    for row in invocations:
        row.setdefault("historical_manifest_hash_scheme", "raw-file-sha256")
    _write_jsonl(INVOCATIONS, invocations)

    verified = _check_historical_artifacts()
    if verified["validation_immutable_episode_payload_sha256"] != pre_validation or verified["final_immutable_episode_payload_sha256"] != pre_final:
        raise AssertionError("Episode payload changed during hash-only migration")
    migration = {
        "migration": "stage4.1.1-json-hash-hardening",
        "migration_timestamp_utc": existing.get("migration_timestamp_utc") if existing else datetime.now(timezone.utc).isoformat(),
        "old_hash_scheme": "raw-file-sha256 (Windows CRLF file bytes)",
        "new_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
        "old_raw_file_sha256": OLD_RAW_HASHES,
        "new_canonical_json_sha256": {key: verified[f"{key}_sha256"] for key in OLD_RAW_HASHES},
        "modified_files": [str(path.relative_to(ROOT)) for path in (
            SEED_MANIFEST, CHECKPOINT_MANIFEST, SELECTED, VALIDATION_GRID, VALIDATION_EPISODES,
            FINAL_SUMMARY, FINAL_EPISODES, AUDIT_JSON, OUTPUT / "protocol_audit.md",
            ROOT / "stage4_1_protocol_validation.json", ROOT / "stage4_1_protocol_validation.md",
            ROOT / "22_阶段4.1_训练与评估协议修复交接.md", INVOCATIONS)],
        "episode_numeric_payload_changed": False,
        "episode_immutable_payload_sha256": {
            "validation_pre_post": verified["validation_immutable_episode_payload_sha256"],
            "final_pre_post": verified["final_immutable_episode_payload_sha256"],
        },
        "checkpoint_binary_hashes_unchanged": True,
        "checkpoint_binary_hash_count": len(verified["checkpoint_binary_hashes"]),
        "validation_records": {"summaries": verified["validation_rows"], "episodes": verified["validation_episode_rows"]},
        "final_test_records": {"summaries": verified["final_rows"], "episodes": verified["final_episode_rows"]},
        "selected_updates": {"P0": 1000, "P1": 800, "P2": 650},
        "raw_episode_summary_recompute_matches": True,
    }
    _write_json(MIGRATION, migration)
    return migration


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Stage 4.1 JSON manifest references without re-evaluation.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = _check_historical_artifacts() if args.verify_only else migrate()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
