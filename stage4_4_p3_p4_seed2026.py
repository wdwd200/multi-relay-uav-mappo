"""Run or read-only verify the authorized paired Stage-4.4 P3/P4 seed-2026 pilot.

``--run`` is deliberately the only training mode: it executes one continuous
1,000-update P3 run followed by one continuous 1,000-update P4 run under the
unchanged ``stage4-formal`` V0H0 preset.  It never calls final-test seeds.
``--verify`` performs no training and writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plain_mappo.checkpoint import load_checkpoint
from plain_mappo.engineering_gate import ENGINEERING_GATE_VARIANTS, actor_parameter_counts
from plain_mappo.experiment_protocol import (FINAL_TEST_SEEDS, VALIDATION_SEEDS, canonical_json_bytes,
                                              sha256_file, sha256_json_file, sha256_json_payload)
from plain_mappo.metrics import safety_priority_key, summarize_episodes
from plain_mappo.presets import STAGE4_FORMAL_FIELDS, stage4_formal_overrides
from plain_mappo.trainer import MappoTrainer
from relay_env import EnvironmentConfig, RelayEnv


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "stage4-4-p3-p4-seed2026"
PROTOCOL = OUTPUT / "protocol.json"
RUN_MANIFEST = OUTPUT / "run_manifest.json"
EPISODES = OUTPUT / "validation_episodes.jsonl"
CHECKPOINT_MANIFEST = OUTPUT / "checkpoint_manifest.json"
SELECTED = OUTPUT / "selected_checkpoints.json"
TRAINING_AUDIT = OUTPUT / "training_audit.json"
TRAINING_AUDIT_MD = OUTPUT / "training_audit.md"
VARIANTS = (("P3", "topology_info"), ("P4", "graph"))
RUN_SEED = 2026
UPDATES = 1000
SAMPLES_PER_UPDATE = 8 * 128
EVAL_UPDATES = tuple(range(50, UPDATES + 1, 50))
PROTOCOL_VERSION = "stage4.4-p3-p4-seed2026-v1"
METADATA_FIELDS = {"protocol_version", "protocol_sha256", "variant", "actor_variant", "run_seed", "update",
                   "split", "seed", "checkpoint_relative_path"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _run_directory(label: str) -> Path:
    return OUTPUT / "runs" / label / f"seed-{RUN_SEED}"


def _config_for(label: str, actor_variant: str) -> Any:
    if label not in {"P3", "P4"} or actor_variant not in ENGINEERING_GATE_VARIANTS:
        raise ValueError("Stage-4.4 supports only P3 topology_info and P4 graph")
    from plain_mappo.config import MappoConfig
    result = MappoConfig(**stage4_formal_overrides(actor_variant=actor_variant, run_seed=RUN_SEED,
                                                    output_dir=str(_run_directory(label))))
    result.validate()
    if (result.value_normalization, result.gamma, result.gae_lambda, result.full_updates) != (False, 0.99, 0.95, 1000):
        raise AssertionError("Stage-4.4 requires frozen V0H0 and 1,000 updates")
    return result


def _protocol_contract(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "generated_at_utc"}


def _protocol_payload() -> dict[str, Any]:
    p3, p4 = (_config_for(label, variant) for label, variant in VARIANTS)
    common = p3.to_dict()
    for field in ("actor_variant", "output_dir"):
        common.pop(field)
    p4_common = p4.to_dict()
    for field in ("actor_variant", "output_dir"):
        p4_common.pop(field)
    if common != p4_common:
        raise AssertionError("P3/P4 changed a common stage4-formal field")
    environment = EnvironmentConfig()
    dirty = bool(_run_git("status", "--porcelain"))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _run_git("rev-parse", "HEAD"),
        "dirty_worktree_at_lock": dirty,
        "result_status": "SINGLE-SEED PILOT",
        "purpose": "paired P3/P4 learning-capability pilot; not a stable ranking claim",
        "variants": [{"label": label, "actor_variant": variant, "run_seed": RUN_SEED,
                      "updates": UPDATES, "samples": UPDATES * SAMPLES_PER_UPDATE,
                      "output_dir": str(_run_directory(label).relative_to(ROOT)).replace("\\", "/")}
                     for label, variant in VARIANTS],
        "stage4_formal_common_config": common,
        "stage4_formal_fields": STAGE4_FORMAL_FIELDS,
        "independent_rng_seeds": {label: {name: getattr(_config_for(label, variant), name) for name in (
            "actor_init_seed", "critic_init_seed", "action_noise_seed", "minibatch_seed", "train_env_seed_base")}
                                  for label, variant in VARIANTS},
        "parameter_counts": actor_parameter_counts(),
        "validation": {
            "seed_manifest_relative_path": "artifacts/stage4-1-protocol-repair/seed_manifest.json",
            "seed_manifest_sha256": sha256_json_file(ROOT / "artifacts/stage4-1-protocol-repair/seed_manifest.json"),
            "seeds": list(VALIDATION_SEEDS), "updates": list(EVAL_UPDATES), "episodes_per_variant": 400,
        },
        "training_budget": {"runs": 2, "updates_per_run": UPDATES, "samples_per_update": SAMPLES_PER_UPDATE,
                            "samples_per_run": UPDATES * SAMPLES_PER_UPDATE,
                            "total_samples": 2 * UPDATES * SAMPLES_PER_UPDATE},
        "environment_version": "Environment v1.1",
        "environment_config": asdict(environment),
        "reward_status": "frozen Reward configuration in Environment v1.1",
        "final_test": {"executed": False, "permitted": False, "seeds": list(FINAL_TEST_SEEDS)},
        "checkpoint_selection": {
            "rule": "safety_priority_key: hard violations, normal completion, collision+boundary, persistent outage, outage ratio, rate satisfaction, mean e2e; earlier update only on exact tie",
            "final_test_used": False,
        },
    }
    # JSON normalization makes a later read/verify compare identical on tuple
    # fields such as hidden sizes, independent of Python's in-memory types.
    return json.loads(canonical_json_bytes(payload).decode("utf-8"))


def _prepare_protocol() -> tuple[dict[str, Any], str]:
    proposed = _protocol_payload()
    if proposed["dirty_worktree_at_lock"]:
        raise RuntimeError("Stage-4.4 protocol must be locked from a clean worktree")
    if PROTOCOL.exists():
        existing = _read_json(PROTOCOL)
        if _protocol_contract(existing) != _protocol_contract(proposed):
            raise ValueError("Stage-4.4 protocol is locked and differs from the current contract")
        return existing, sha256_json_payload(existing)
    _write_json(PROTOCOL, proposed)
    return proposed, sha256_json_payload(proposed)


def _write_run_manifest(protocol_hash: str, runs: dict[str, Any], interrupted: list[dict[str, Any]]) -> None:
    _write_json(RUN_MANIFEST, {
        "protocol_version": PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
        "runs": runs, "interrupted_attempts": interrupted,
        "final_test_executed": False, "formal_training_executed": True,
    })


def _append_raw_episodes(path: Path, *, protocol_hash: str, label: str, actor_variant: str,
                         update: int, seeds: list[int], episodes: list[dict[str, Any]]) -> None:
    if tuple(seeds) != VALIDATION_SEEDS:
        raise ValueError("Stage-4.4 validation recorder refuses a non-validation seed split")
    checkpoint = _run_directory(label) / "eval_checkpoints" / f"actor_update_{update:04d}.pt"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for seed, episode in zip(seeds, episodes):
            row = {"protocol_version": PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
                   "variant": label, "actor_variant": actor_variant, "run_seed": RUN_SEED,
                   "update": int(update), "split": "validation", "seed": int(seed),
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


def _checkpoint_reload_check(config: Any, trainer: MappoTrainer) -> dict[str, bool]:
    latest = Path(config.output_dir) / "checkpoints" / "latest.pt"
    payload = load_checkpoint(latest, config.device)
    restored = MappoTrainer(config)
    restored.load(latest)
    protocol_rng = payload["protocol_rng_state"]
    return {
        "latest_exists": latest.is_file(),
        "update": restored.update_count == UPDATES == int(payload["update"]),
        "total_env_steps": restored.total_env_steps == UPDATES * SAMPLES_PER_UPDATE == int(payload["total_env_steps"]),
        "actor": _equal_state(trainer.actor.state_dict(), restored.actor.state_dict()),
        "critic": _equal_state(trainer.critic.state_dict(), restored.critic.state_dict()),
        "actor_optimizer": _equal_state(payload["actor_optimizer_state"], restored.actor_optimizer.state_dict()),
        "critic_optimizer": _equal_state(payload["critic_optimizer_state"], restored.critic_optimizer.state_dict()),
        "normalizer": _equal_state(payload["normalizer_state"], restored.normalizer.state_dict()),
        "value_normalizer_v0": payload["value_normalizer_state"] is None and restored.value_normalizer is None,
        "protocol_rng": (
            torch.equal(restored.action_noise_generator.get_state(), protocol_rng["action_noise_generator"])  # type: ignore[union-attr]
            and _equal_state(restored.minibatch_rng.bit_generator.state, protocol_rng["minibatch_bit_generator"])  # type: ignore[union-attr]
        ),
        "fresh_boundary_resume": (payload["resume_mode"] == "fresh_episode_at_update_boundary"
                                  and payload["exact_environment_resume"] is False),
    }


def _interrupt_existing_attempt(label: str, interrupted: list[dict[str, Any]]) -> None:
    run_dir = _run_directory(label)
    if not run_dir.exists():
        return
    complete = run_dir / "stage4_4_complete.json"
    if complete.is_file():
        return
    destination = OUTPUT / "interrupted" / label / f"seed-{RUN_SEED}-interrupted-{len(interrupted) + 1:02d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite interrupted evidence: {destination}")
    destination.mkdir()
    if EPISODES.exists():
        all_rows = _episode_rows()
        interrupted_rows = [row for row in all_rows if row.get("variant") == label]
        if interrupted_rows:
            evidence = destination / "validation_episodes_partial.jsonl"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                                        for row in interrupted_rows), encoding="utf-8", newline="\n")
            remaining = [row for row in all_rows if row.get("variant") != label]
            EPISODES.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                                         for row in remaining), encoding="utf-8", newline="\n")
    shutil.move(str(run_dir), str(destination / "run"))
    interrupted.append({"variant": label, "run_seed": RUN_SEED,
                        "reason": "previous continuous training attempt did not reach verified completion",
                        "preserved_relative_path": str(destination.relative_to(ROOT)).replace("\\", "/"),
                        "invalidated": True})


def _run_variant(label: str, actor_variant: str, protocol_hash: str) -> dict[str, Any]:
    config = _config_for(label, actor_variant)
    run_dir = Path(config.output_dir)
    raw_path = EPISODES
    trainer = MappoTrainer(config)
    trainer.evaluation_episode_recorder = lambda update, seeds, summary, score, episodes: _append_raw_episodes(
        raw_path, protocol_hash=protocol_hash, label=label, actor_variant=actor_variant,
        update=update, seeds=seeds, episodes=episodes)
    records = trainer.train(UPDATES, final_evaluation=False, export_final_actor=True)
    if trainer.update_count != UPDATES or trainer.total_env_steps != UPDATES * SAMPLES_PER_UPDATE or len(records) != UPDATES:
        raise RuntimeError(f"{label} did not complete its exact Stage-4.4 budget")
    reload_check = _checkpoint_reload_check(config, trainer)
    if not all(reload_check.values()):
        raise RuntimeError(f"{label} checkpoint reload failed: {reload_check}")
    _write_json(run_dir / "stage4_4_complete.json", {"protocol_sha256": protocol_hash, "label": label,
                                                       "updates": UPDATES, "training_samples": UPDATES * SAMPLES_PER_UPDATE,
                                                       "checkpoint_reload": reload_check,
                                                       "final_test_executed": False})
    return {"status": "complete", "actor_variant": actor_variant, "run_seed": RUN_SEED,
            "updates": UPDATES, "training_samples": UPDATES * SAMPLES_PER_UPDATE,
            "checkpoint_reload": reload_check, "final_test_executed": False}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _episode_rows() -> list[dict[str, Any]]:
    if not EPISODES.is_file():
        raise FileNotFoundError(f"missing validation episode log: {EPISODES}")
    return [json.loads(line) for line in EPISODES.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_finite_csv_row(row: dict[str, str]) -> bool:
    for value in row.values():
        if value == "":
            continue
        try:
            if not math.isfinite(float(value)):
                return False
        except ValueError:
            continue
    return True


def _near(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _select_validation_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen safety key, using update only as an exact-tie breaker."""
    if not summaries:
        raise ValueError("cannot select from an empty validation grid")
    return min(summaries, key=lambda item: (tuple(item["score"]), int(item["update"])))


def _validate_run(label: str, actor_variant: str, protocol_hash: str, all_episodes: list[dict[str, Any]],
                  *, verify_binaries: bool) -> dict[str, Any]:
    run_dir = _run_directory(label)
    from plain_mappo.config import MappoConfig
    config = MappoConfig.from_dict(_read_json(run_dir / "config.json")).to_dict()
    expected_config = _config_for(label, actor_variant).to_dict()
    # output_dir is intentionally saved as the producing worktree's absolute
    # path.  It cannot equal a verifier's path in a clean checkout, so compare
    # the complete frozen training contract independently and bind the saved
    # location to this run's required relative artifact directory.
    saved_output_dir = str(config.pop("output_dir")).replace("\\", "/")
    expected_config.pop("output_dir")
    required_suffix = str(run_dir.relative_to(ROOT)).replace("\\", "/")
    if not saved_output_dir.endswith(required_suffix) or config != expected_config:
        raise ValueError(f"{label} saved config differs from stage4-formal contract")
    train_rows = _load_csv(run_dir / "train.csv")
    if len(train_rows) != UPDATES or [int(row["update"]) for row in train_rows] != list(range(1, UPDATES + 1)):
        raise ValueError(f"{label} train.csv must contain contiguous updates 1..{UPDATES}")
    if not all(_is_finite_csv_row(row) and float(row["finite"]) == 1.0 for row in train_rows):
        raise ValueError(f"{label} train.csv contains non-finite diagnostics")
    if sum(float(row["hard_speed_accel_violations"]) for row in train_rows) != 0.0:
        raise ValueError(f"{label} has hard speed/acceleration violations")
    eval_rows = _load_csv(run_dir / "eval.csv")
    if len(eval_rows) != len(EVAL_UPDATES) or tuple(int(row["update"]) for row in eval_rows) != EVAL_UPDATES:
        raise ValueError(f"{label} eval.csv must contain exactly the 20 fixed validation updates")
    selected_rows = [row for row in all_episodes if row["variant"] == label]
    if len(selected_rows) != len(EVAL_UPDATES) * len(VALIDATION_SEEDS):
        raise ValueError(f"{label} does not have 400 raw validation episodes")
    seen = {(int(row["update"]), int(row["seed"])) for row in selected_rows}
    if len(seen) != 400 or {seed for _, seed in seen} != set(VALIDATION_SEEDS) or {update for update, _ in seen} != set(EVAL_UPDATES):
        raise ValueError(f"{label} raw validation coverage is incomplete or duplicated")
    summaries: list[dict[str, Any]] = []
    for row in eval_rows:
        update = int(row["update"])
        if tuple(int(value) for value in row["seeds"].split(",")) != VALIDATION_SEEDS:
            raise ValueError(f"{label} eval uses a non-validation seed")
        expected_summary = json.loads(row["summary_json"])
        episodes = [{key: value for key, value in raw.items() if key not in METADATA_FIELDS}
                    for raw in selected_rows if int(raw["update"]) == update]
        recomputed = summarize_episodes(episodes)
        if any(not _near(recomputed[key], expected_summary[key]) for key in recomputed):
            raise ValueError(f"{label} validation summary at update {update} does not recompute from raw episodes")
        expected_score = tuple(float(value) for value in json.loads(row["score"]))
        if tuple(float(value) for value in safety_priority_key(recomputed)) != expected_score:
            raise ValueError(f"{label} validation score at update {update} is not the frozen safety key")
        summaries.append({"update": update, "summary": recomputed, "score": expected_score})
    periodic_paths = [run_dir / "eval_checkpoints" / f"actor_update_{update:04d}.pt" for update in EVAL_UPDATES]
    if verify_binaries and not all(path.is_file() for path in periodic_paths):
        raise FileNotFoundError(f"{label} periodic Actor checkpoint is missing")
    winner = _select_validation_summary(summaries)
    last = train_rows[-1]
    return {"label": label, "actor_variant": actor_variant, "run_seed": RUN_SEED,
            "train_rows": len(train_rows), "validation_summaries": len(summaries),
            "validation_episodes": len(selected_rows), "selected_update": winner["update"],
            "selected_summary": winner["summary"], "selected_score": list(winner["score"]),
            "checkpoint_relative_path": str(periodic_paths[EVAL_UPDATES.index(winner["update"])].relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_reload": _read_json(run_dir / "stage4_4_complete.json")["checkpoint_reload"],
            "diagnostics": {key: float(last[key]) for key in (
                "explained_variance_pre", "actor_grad_clip_fraction", "critic_grad_clip_fraction",
                "action_saturation_ratio", "clamped_log_std_min", "clamped_log_std_max")}}


def _write_final_artifacts(protocol_hash: str, *, write: bool = True, verify_binaries: bool = True) -> dict[str, Any]:
    episodes = _episode_rows()
    if len(episodes) != 800:
        raise ValueError("Stage-4.4 requires exactly 800 raw validation Episodes")
    reports = [_validate_run(label, variant, protocol_hash, episodes, verify_binaries=verify_binaries)
               for label, variant in VARIANTS]
    entries: list[dict[str, Any]] = []
    expected_entries: list[tuple[str, str, Path]] = []
    for label, _ in VARIANTS:
        run_dir = _run_directory(label)
        periodic = [run_dir / "eval_checkpoints" / f"actor_update_{update:04d}.pt" for update in EVAL_UPDATES]
        training = [run_dir / "checkpoints" / name for name in ("latest.pt", "best.pt", "actor_final.pt")]
        for category, paths in (("periodic_validation_actor", periodic), ("training_checkpoint", training)):
            for path in paths:
                expected_entries.append((label, category, path))
                if verify_binaries:
                    if not path.is_file():
                        raise FileNotFoundError(f"missing {category}: {path}")
                    entries.append({"variant": label, "category": category,
                                    "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                    "sha256": sha256_file(path)})
    if not verify_binaries:
        existing = _read_json(CHECKPOINT_MANIFEST)
        entries = list(existing.get("entries", []))
        expected_by_path = {str(path.relative_to(ROOT)).replace("\\", "/"): (label, category)
                            for label, category, path in expected_entries}
        if (len(entries) != len(expected_entries)
                or {entry.get("relative_path") for entry in entries} != set(expected_by_path)
                or any(expected_by_path[entry["relative_path"]] != (entry.get("variant"), entry.get("category"))
                       or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
                       for entry in entries)):
            raise ValueError("Stage-4.4 checkpoint hash manifest metadata is incomplete")
    checkpoint_manifest = {"protocol_version": PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
                           "hash_scheme": "sha256-file-v1", "entries": entries,
                           "final_test_executed": False}
    if write:
        _write_json(CHECKPOINT_MANIFEST, checkpoint_manifest)
    elif _read_json(CHECKPOINT_MANIFEST) != checkpoint_manifest:
        raise ValueError("Stage-4.4 checkpoint hash manifest drifted")
    hash_by_path = {entry["relative_path"]: entry["sha256"] for entry in entries}
    selected: list[dict[str, Any]] = []
    for report in reports:
        relative = report["checkpoint_relative_path"]
        selected.append({"variant": report["label"], "actor_variant": report["actor_variant"], "run_seed": RUN_SEED,
                         "selected_update": report["selected_update"], "checkpoint_relative_path": relative,
                         "checkpoint_sha256": hash_by_path[relative], "validation_score": report["selected_score"],
                         "validation_summary": report["selected_summary"],
                         "validation_seed_manifest_sha256": _read_json(PROTOCOL)["validation"]["seed_manifest_sha256"],
                         "selection_rule": _read_json(PROTOCOL)["checkpoint_selection"]["rule"],
                         "final_test_used": False})
    selected_manifest = {"protocol_version": PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
                         "result_status": "SINGLE-SEED PILOT", "selected": selected,
                         "final_test_executed": False}
    if write:
        _write_json(SELECTED, selected_manifest)
    elif _read_json(SELECTED) != selected_manifest:
        raise ValueError("Stage-4.4 selected-checkpoint manifest drifted")
    audit = {"protocol_version": PROTOCOL_VERSION, "protocol_sha256": protocol_hash,
             "result_status": "SINGLE-SEED PILOT", "final_test_executed": False,
             "formal_training_executed": True, "total_training_samples": 2 * UPDATES * SAMPLES_PER_UPDATE,
             "validation_episodes": 800, "runs": reports,
             "interpretation": "Engineering-correct single-seed pilot only; no stable ranking or Graph-effectiveness claim."}
    if write:
        _write_json(TRAINING_AUDIT, audit)
    elif _read_json(TRAINING_AUDIT) != audit:
        raise ValueError("Stage-4.4 training audit drifted")
    lines = ["# Stage 4.4 P3/P4 seed-2026 training audit", "", "Result status: **SINGLE-SEED PILOT**.",
             "", "No final-test evaluation was executed. These runs do not establish a stable method ranking.", "",
             "| Variant | Selected update | Normal | Outage | Collision | Boundary | Mean e2e Mbps |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        summary = report["selected_summary"]
        lines.append(f"| {report['label']} | {report['selected_update']} | {summary['normal_completion_count']} | "
                     f"{summary['persistent_outage_count']} | {summary['collision_count']} | "
                     f"{summary['boundary_count']} | {summary['mean_e2e_rate_mbps']:.9f} |")
    lines.extend(("", "The selected checkpoints were chosen exclusively by the frozen safety-priority key on the locked validation split."))
    markdown = "\n".join(lines) + "\n"
    if write:
        TRAINING_AUDIT_MD.write_text(markdown, encoding="utf-8", newline="\n")
    elif TRAINING_AUDIT_MD.read_text(encoding="utf-8") != markdown:
        raise ValueError("Stage-4.4 training audit Markdown drifted")
    return audit


def run() -> dict[str, Any]:
    protocol, protocol_hash = _prepare_protocol()
    del protocol
    if CHECKPOINT_MANIFEST.exists() or SELECTED.exists() or TRAINING_AUDIT.exists():
        raise RuntimeError("Stage-4.4 final artifacts already exist; use --verify rather than rerunning")
    prior = _read_json(RUN_MANIFEST) if RUN_MANIFEST.exists() else {}
    if prior and prior.get("protocol_sha256") != protocol_hash:
        raise ValueError("Stage-4.4 interrupted run manifest belongs to a different protocol")
    runs: dict[str, Any] = dict(prior.get("runs", {}))
    interrupted: list[dict[str, Any]] = list(prior.get("interrupted_attempts", []))
    _write_run_manifest(protocol_hash, runs, interrupted)
    for label, variant in VARIANTS:
        _interrupt_existing_attempt(label, interrupted)
        _write_run_manifest(protocol_hash, runs, interrupted)
        if runs.get(label, {}).get("status") == "complete":
            continue
        runs[label] = {"status": "running", "actor_variant": variant, "run_seed": RUN_SEED}
        _write_run_manifest(protocol_hash, runs, interrupted)
        runs[label] = _run_variant(label, variant, protocol_hash)
        _write_run_manifest(protocol_hash, runs, interrupted)
    audit = _write_final_artifacts(protocol_hash, write=True)
    _write_run_manifest(protocol_hash, runs, interrupted)
    return {"protocol_sha256": protocol_hash, "p3": "COMPLETE", "p4": "COMPLETE",
            "validation_episodes": audit["validation_episodes"], "final_test_executed": False}


def verify() -> dict[str, Any]:
    required = (PROTOCOL, RUN_MANIFEST, EPISODES, CHECKPOINT_MANIFEST, SELECTED, TRAINING_AUDIT, TRAINING_AUDIT_MD)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage-4.4 artifacts missing: {missing}")
    protocol = _read_json(PROTOCOL)
    if protocol.get("protocol_version") != PROTOCOL_VERSION or protocol.get("final_test", {}).get("executed") is not False:
        raise ValueError("Stage-4.4 protocol is invalid")
    protocol_hash = sha256_json_payload(protocol)
    manifest = _read_json(RUN_MANIFEST)
    if manifest.get("protocol_sha256") != protocol_hash or set(manifest.get("runs", {})) != {"P3", "P4"}:
        raise ValueError("Stage-4.4 run manifest is incomplete")
    if any(manifest["runs"][label].get("status") != "complete" for label, _ in VARIANTS):
        raise ValueError("Stage-4.4 run did not complete")
    binary_paths = [path for label, _ in VARIANTS for path in (
        *[_run_directory(label) / "eval_checkpoints" / f"actor_update_{update:04d}.pt" for update in EVAL_UPDATES],
        *[_run_directory(label) / "checkpoints" / name for name in ("latest.pt", "best.pt", "actor_final.pt")],
    )]
    present = [path for path in binary_paths if path.is_file()]
    if present and len(present) != len(binary_paths):
        raise FileNotFoundError("Stage-4.4 checkpoint set is partially present; refusing ambiguous verification")
    verify_binaries = len(present) == len(binary_paths)
    audit = _write_final_artifacts(protocol_hash, write=False, verify_binaries=verify_binaries)
    return {"protocol_sha256": protocol_hash, "p3": "COMPLETE", "p4": "COMPLETE",
            "validation_episodes": audit["validation_episodes"], "final_test_executed": False,
            "checkpoint_binary_verification": "PASS" if verify_binaries else "SKIPPED_MISSING_LOCAL_CHECKPOINTS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run the authorized continuous P3 then P4 1,000-update pilot.")
    group.add_argument("--verify", action="store_true", help="Read-only verify completed Stage-4.4 artifacts.")
    args = parser.parse_args()
    result = run() if args.run else verify()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
