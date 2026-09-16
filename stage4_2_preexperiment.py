"""Execute or verify the locked Stage-4.2 value/horizon P0 pre-experiment.

This is intentionally not a formal P0--P4 training entrypoint.  It runs only
the immutable 6-candidate x 3-seed, 200-update diagnostic matrix and evaluates
only the locked Stage-4.1 validation split at updates 50/100/150/200.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

from plain_mappo.config import MappoConfig
from plain_mappo.experiment_protocol import (FINAL_TEST_SEEDS, PREEXPERIMENT_RUN_SEEDS,
                                              sha256_json_file)
from plain_mappo.preexperiment import (PREEXPERIMENT_CANDIDATES,
                                       STAGE4_1_CANONICAL_HASHES,
                                       validate_preexperiment_protocol,
                                       write_locked_preexperiment_protocol)
from plain_mappo.presets import stage4_preexperiment_overrides
from plain_mappo.trainer import MappoTrainer


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "stage4-2-preexperiment"
PROTOCOL = OUTPUT / "protocol.json"
RUN_MANIFEST = OUTPUT / "run_manifest.json"
GRID = OUTPUT / "value_horizon_grid.csv"
EPISODES = OUTPUT / "validation_episodes.jsonl"
PAIRED_SUMMARY = OUTPUT / "paired_summary.csv"
DECISION = OUTPUT / "decision.json"
DECISION_MD = OUTPUT / "decision.md"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def _matrix() -> list[dict[str, Any]]:
    return [{**candidate, "run_seed": run_seed, "updates": 200, "samples_per_update": 1024}
            for candidate in PREEXPERIMENT_CANDIDATES for run_seed in PREEXPERIMENT_RUN_SEEDS]


def _verify_stage4_1_hashes() -> None:
    actual = {
        "seed_manifest": sha256_json_file(ROOT / "artifacts/stage4-1-protocol-repair/seed_manifest.json"),
        "checkpoint_manifest": sha256_json_file(ROOT / "artifacts/stage4-1-protocol-repair/checkpoint_manifest.json"),
        "selected_checkpoints": sha256_json_file(ROOT / "artifacts/stage4-1-protocol-repair/selected_checkpoints.json"),
    }
    if actual != STAGE4_1_CANONICAL_HASHES:
        raise ValueError(f"Stage-4.1 canonical hashes changed: {actual}")


def _locked_manifest(protocol_hash: str) -> dict[str, Any]:
    return {"protocol_version": "stage4.2-preexperiment-v1", "protocol_sha256": protocol_hash,
            "matrix": _matrix(), "validation_updates": [50, 100, 150, 200],
            "final_test_executed": False, "total_runs": 18,
            "training_samples_per_run": 200 * 1024,
            "training_samples_total": 18 * 200 * 1024,
            "completed_runs": []}


def _prepare_protocol_and_manifest() -> tuple[dict[str, Any], str, dict[str, Any]]:
    _verify_stage4_1_hashes()
    protocol, protocol_hash, _ = write_locked_preexperiment_protocol(PROTOCOL)
    validate_preexperiment_protocol(protocol)
    expected = _locked_manifest(protocol_hash)
    if RUN_MANIFEST.exists():
        manifest = _read_json(RUN_MANIFEST)
        for field in ("protocol_version", "protocol_sha256", "matrix", "validation_updates", "final_test_executed",
                      "total_runs", "training_samples_per_run", "training_samples_total"):
            if manifest.get(field) != expected[field]:
                raise ValueError(f"run manifest locked field changed: {field}")
        if not isinstance(manifest.get("completed_runs"), list):
            raise ValueError("run manifest completed_runs is invalid")
        return protocol, protocol_hash, manifest
    _write_json(RUN_MANIFEST, expected)
    return protocol, protocol_hash, expected


def _run_dir(spec: dict[str, Any]) -> Path:
    return OUTPUT / "runs" / str(spec["candidate"]) / f"seed-{int(spec['run_seed'])}"


def _config_for(spec: dict[str, Any], protocol_hash: str) -> MappoConfig:
    output_dir = _run_dir(spec)
    return MappoConfig(**stage4_preexperiment_overrides(
        candidate=str(spec["candidate"]), run_seed=int(spec["run_seed"]), output_dir=str(output_dir),
        protocol_sha256=protocol_hash, value_normalization=bool(spec["value_normalization"]),
        gamma=float(spec["gamma"]), gae_lambda=float(spec["gae_lambda"]),
        evaluation_episode_log_path=str(EPISODES),
    ))


def _completed_key(spec: dict[str, Any]) -> tuple[str, int]:
    return str(spec["candidate"]), int(spec["run_seed"])


def _run_one(spec: dict[str, Any], protocol_hash: str) -> dict[str, Any]:
    config = _config_for(spec, protocol_hash)
    latest = Path(config.output_dir) / "checkpoints" / "latest.pt"
    existing_directory = Path(config.output_dir).exists()
    if existing_directory and not latest.is_file():
        raise FileExistsError(f"refusing to overwrite incomplete pre-experiment directory without checkpoint: {config.output_dir}")
    trainer = MappoTrainer(config)
    if latest.is_file():
        trainer.load(latest)
        train_path = Path(config.output_dir) / "train.csv"
        with train_path.open(newline="", encoding="utf-8") as handle:
            existing_updates = [int(row["update"]) for row in csv.DictReader(handle)]
        if existing_updates != list(range(1, trainer.update_count + 1)):
            raise ValueError("resume refused: train log does not exactly match the atomic checkpoint update boundary")
    if trainer.update_count > 200:
        raise RuntimeError(f"pre-experiment checkpoint exceeds its 200-update budget: {config.output_dir}")
    if trainer.update_count < 200:
        trainer.train(200 - trainer.update_count)
    if trainer.update_count != 200 or trainer.total_env_steps != 200 * 1024:
        raise RuntimeError("pre-experiment run did not complete exactly 200 updates / 204800 samples")
    with (Path(config.output_dir) / "eval.csv").open(newline="", encoding="utf-8") as handle:
        evaluation_updates = [int(row["update"]) for row in csv.DictReader(handle)]
    if evaluation_updates != [50, 100, 150, 200]:
        raise RuntimeError(f"validation did not occur only at 50/100/150/200: {evaluation_updates}")
    with (Path(config.output_dir) / "train.csv").open(newline="", encoding="utf-8") as handle:
        if [int(row["update"]) for row in csv.DictReader(handle)] != list(range(1, 201)):
            raise RuntimeError("training diagnostics do not contain exactly 200 contiguous updates")
    # A true reload smoke verifies the optional value normalizer along with
    # complete optimizer/RNG checkpoint state without running a new episode.
    restored = MappoTrainer(config)
    restored.load(latest)
    if restored.update_count != 200 or restored.total_env_steps != 200 * 1024:
        raise RuntimeError("checkpoint reload smoke did not restore the final boundary")
    return {"candidate": spec["candidate"], "run_seed": int(spec["run_seed"]), "status": "complete",
            "updates": trainer.update_count, "training_samples": trainer.total_env_steps,
            "checkpoint_reload_smoke": "PASS"}


def _read_eval_rows(spec: dict[str, Any], protocol_hash: str) -> list[dict[str, Any]]:
    config = _config_for(spec, protocol_hash)
    path = Path(config.output_dir) / "eval.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if [int(row["update"]) for row in raw_rows] != [50, 100, 150, 200]:
        raise ValueError(f"evaluation checkpoints are incomplete: {path}")
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        summary = json.loads(row["summary_json"])
        score = json.loads(row["score"])
        rows.append({"candidate": spec["candidate"], "value_normalization": bool(spec["value_normalization"]),
                     "gamma": float(spec["gamma"]), "gae_lambda": float(spec["gae_lambda"]),
                     "run_seed": int(spec["run_seed"]), "update": int(row["update"]),
                     "protocol_sha256": protocol_hash, "validation_seed_count": 20,
                     "safety_priority_key": json.dumps(score, separators=(",", ":")), **summary})
    return rows


def _read_train_rows(spec: dict[str, Any], protocol_hash: str) -> list[dict[str, Any]]:
    config = _config_for(spec, protocol_hash)
    with (Path(config.output_dir) / "train.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200 or [int(row["update"]) for row in rows] != list(range(1, 201)):
        raise ValueError(f"training log is incomplete for {spec['candidate']}/{spec['run_seed']}")
    return rows


def _float_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field, "") not in {"", None}]


def _mean_or_nan(values: list[float]) -> float:
    return float(mean(values)) if values else float("nan")


def _paired_rows(grid_rows: list[dict[str, Any]], protocol_hash: str) -> list[dict[str, Any]]:
    final = [row for row in grid_rows if int(row["update"]) == 200]
    rows: list[dict[str, Any]] = []
    for candidate in PREEXPERIMENT_CANDIDATES:
        candidate_rows = [row for row in final if row["candidate"] == candidate["candidate"]]
        if len(candidate_rows) != 3:
            raise ValueError(f"final validation rows missing for {candidate['candidate']}")
        train_rows = [row for spec in _matrix() if spec["candidate"] == candidate["candidate"]
                      for row in _read_train_rows(spec, protocol_hash)]
        tail = [row for row in train_rows if int(row["update"]) > 150]
        def stats(field: str) -> dict[str, float]:
            values = [float(row[field]) for row in candidate_rows]
            return {f"{field}_mean": float(mean(values)), f"{field}_median": float(median(values)),
                    f"{field}_std": float(pstdev(values))}
        critic_clip = _float_values(tail, "critic_grad_clip_fraction")
        explained = _float_values(tail, "explained_variance_pre")
        finite = all(float(row["finite"]) == 1.0 for row in train_rows)
        hard = sum(float(row["hard_speed_accel_violations"]) for row in train_rows)
        row: dict[str, Any] = {"candidate": candidate["candidate"], "value_normalization": candidate["value_normalization"],
                               "gamma": candidate["gamma"], "gae_lambda": candidate["gae_lambda"],
                               "paired_seeds": ",".join(map(str, PREEXPERIMENT_RUN_SEEDS)),
                               "finite_all_updates": finite, "hard_speed_accel_violations": hard,
                               "critic_grad_clip_fraction_last50_mean": _mean_or_nan(critic_clip),
                               "explained_variance_last50_mean": _mean_or_nan(explained),
                               "explained_variance_last50_std": float(pstdev(explained)) if len(explained) > 1 else 0.0}
        for field in ("normal_completion_count", "persistent_outage_count", "collision_count", "boundary_count",
                      "mean_e2e_rate_mbps", "outage_step_ratio", "rate_satisfaction_ratio", "speed_accel_violations"):
            row.update(stats(field))
        rows.append(row)
    return rows


def _safety_score(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in json.loads(row["safety_priority_key"]))


def _decision(grid_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], protocol_hash: str) -> dict[str, Any]:
    final = [row for row in grid_rows if int(row["update"]) == 200]
    baseline = {int(row["run_seed"]): row for row in final if row["candidate"] == "V0H0"}
    paired = {row["candidate"]: row for row in paired_rows}
    diagnostics: list[dict[str, Any]] = []
    recommendable: list[str] = []
    baseline_rate = float(paired["V0H0"]["mean_e2e_rate_mbps_mean"])
    for candidate in PREEXPERIMENT_CANDIDATES:
        name = candidate["candidate"]
        candidate_rows = {int(row["run_seed"]): row for row in final if row["candidate"] == name}
        comparisons = [(-1 if _safety_score(candidate_rows[seed]) < _safety_score(baseline[seed])
                        else 1 if _safety_score(candidate_rows[seed]) > _safety_score(baseline[seed]) else 0)
                       for seed in PREEXPERIMENT_RUN_SEEDS]
        summary = paired[name]
        healthy = bool(summary["finite_all_updates"]) and float(summary["hard_speed_accel_violations"]) == 0.0
        clip_improved = float(summary["critic_grad_clip_fraction_last50_mean"]) <= 0.90 * float(paired["V0H0"]["critic_grad_clip_fraction_last50_mean"])
        ev_improved = float(summary["explained_variance_last50_mean"]) >= float(paired["V0H0"]["explained_variance_last50_mean"])
        throughput_ok = float(summary["mean_e2e_rate_mbps_mean"]) >= 0.95 * baseline_rate
        seed_consistent = comparisons.count(-1) >= 2 and comparisons.count(1) == 0
        if name != "V0H0" and healthy and seed_consistent and (clip_improved or ev_improved) and throughput_ok:
            recommendable.append(name)
        diagnostics.append({"candidate": name, "seedwise_safety_comparison_vs_V0H0": comparisons,
                            "healthy": healthy, "critic_clip_improved": clip_improved,
                            "explained_variance_improved": ev_improved, "throughput_within_5pct": throughput_ok,
                            "seed_consistent": seed_consistent})
    if len(recommendable) == 1:
        recommendation, chosen = "RECOMMEND_CANDIDATE", recommendable[0]
        rationale = "The sole candidate met the pre-registered safety, health, throughput, and paired-seed consistency rules."
    elif not recommendable and all(item["seed_consistent"] is False for item in diagnostics if item["candidate"] != "V0H0"):
        recommendation, chosen = "INCONCLUSIVE", None
        rationale = "No non-baseline candidate showed a seed-consistent safety improvement under the pre-registered rule."
    else:
        recommendation, chosen = "INCONCLUSIVE", None
        rationale = "More than one candidate qualified or health/safety evidence did not identify a unique robust choice."
    return {"protocol_version": "stage4.2-preexperiment-v1", "protocol_sha256": protocol_hash,
            "recommendation": recommendation, "recommended_candidate": chosen,
            "selection_rule": "safety/finite first; Critic health second; throughput only after safety parity; final test unused",
            "rationale": rationale, "candidate_diagnostics": diagnostics,
            "final_test_executed": False,
            "fairness_notice": "Any later formal change must retrain P0-P4 under one shared configuration; it must not compare new P3/P4 runs against old P0/P1/P2 results."}


def _verify_episodes(protocol_hash: str) -> None:
    rows = [json.loads(line) for line in EPISODES.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = 18 * 4 * 20
    if len(rows) != expected:
        raise ValueError(f"validation Episode count changed: {len(rows)} vs {expected}")
    keys = {(row["candidate"], int(row["run_seed"]), int(row["update"]), int(row["seed"])) for row in rows}
    if len(keys) != expected or any(row["protocol_sha256"] != protocol_hash for row in rows):
        raise ValueError("validation Episode records are duplicated or have incorrect provenance")
    if any(row["split"] != "validation" or int(row["seed"]) in FINAL_TEST_SEEDS for row in rows):
        raise ValueError("pre-experiment attempted final-test evaluation")


def _write_outputs(protocol_hash: str) -> dict[str, Any]:
    grid_rows = [row for spec in _matrix() for row in _read_eval_rows(spec, protocol_hash)]
    grid_fields = ["candidate", "value_normalization", "gamma", "gae_lambda", "run_seed", "update", "protocol_sha256",
                   "validation_seed_count", "safety_priority_key", "episodes", "total_steps", "normal_completion_count",
                   "collision_count", "boundary_count", "persistent_outage_count", "speed_accel_violations",
                   "outage_step_ratio", "rate_satisfaction_ratio", "mean_e2e_rate_mbps", "mean_return",
                   "mean_acceleration_mps2", "action_saturation_ratio", "movement_distance_m", "min_link_margin_db",
                   "min_separation_m", "max_xy_speed_mps", "max_abs_z_speed_mps", "max_xy_accel_mps2", "max_abs_z_accel_mps2"]
    _write_csv(GRID, grid_fields, grid_rows)
    _verify_episodes(protocol_hash)
    paired_rows = _paired_rows(grid_rows, protocol_hash)
    paired_fields = list(paired_rows[0])
    _write_csv(PAIRED_SUMMARY, paired_fields, paired_rows)
    decision = _decision(grid_rows, paired_rows, protocol_hash)
    _write_json(DECISION, decision)
    markdown = "\n".join((
        "# Stage 4.2 value/horizon pre-experiment decision",
        "",
        f"Recommendation: `{decision['recommendation']}`",
        f"Recommended candidate: `{decision['recommended_candidate'] or 'none'}`",
        "",
        decision["rationale"],
        "",
        "Final test executed: `false`.",
        "",
        decision["fairness_notice"],
        "",
        "The CSV/JSON artifacts contain all 18 paired runs, 72 validation summaries, and 1,440 raw validation Episodes.",
    )) + "\n"
    DECISION_MD.write_text(markdown, encoding="utf-8", newline="\n")
    return decision


def run() -> dict[str, Any]:
    _, protocol_hash, manifest = _prepare_protocol_and_manifest()
    completed = {(str(row["candidate"]), int(row["run_seed"])) for row in manifest["completed_runs"]}
    for spec in _matrix():
        if _completed_key(spec) in completed:
            continue
        result = _run_one(spec, protocol_hash)
        manifest["completed_runs"].append(result)
        _write_json(RUN_MANIFEST, manifest)
    if len(manifest["completed_runs"]) != 18:
        raise RuntimeError("Stage-4.2 pre-experiment did not complete all 18 runs")
    decision = _write_outputs(protocol_hash)
    return {"runs_completed": len(manifest["completed_runs"]), "training_samples": 18 * 200 * 1024,
            "validation_episodes": 18 * 4 * 20, "final_test_executed": False,
            "recommendation": decision["recommendation"], "recommended_candidate": decision["recommended_candidate"]}


def verify() -> dict[str, Any]:
    _, protocol_hash, manifest = _prepare_protocol_and_manifest()
    if len(manifest["completed_runs"]) != 18:
        raise ValueError("run manifest does not record all 18 completed runs")
    decision = _write_outputs(protocol_hash)
    return {"runs_completed": 18, "validation_episodes": 18 * 4 * 20,
            "final_test_executed": False, "recommendation": decision["recommendation"]}


def recover_invalidated_attempt() -> dict[str, Any]:
    """Clear only invalid completion status after an externally interrupted run.

    This does not alter the locked protocol, candidate matrix, seeds, or
    budget.  It is intentionally explicit so incomplete/duplicate diagnostics
    can never be mistaken for a completed scientific run.
    """
    _, protocol_hash, manifest = _prepare_protocol_and_manifest()
    if any(_run_dir(spec).exists() for spec in _matrix()):
        raise RuntimeError("recovery requires partial run directories to be isolated before status can be cleared")
    invalidated = [dict(row) for row in manifest["completed_runs"]]
    manifest["completed_runs"] = []
    manifest.setdefault("interrupted_attempts", []).append({
        "reason": "execution host interruption produced duplicate diagnostics before completion verification",
        "invalidated_completion_records": invalidated,
        "protocol_sha256": protocol_hash,
        "final_test_executed": False,
    })
    _write_json(RUN_MANIFEST, manifest)
    return {"recovered_invalid_completion_records": len(invalidated), "protocol_sha256": protocol_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Validate existing complete artifacts without training.")
    parser.add_argument("--recover-invalidated-attempt", action="store_true",
                        help="Explicitly clear isolated, invalid completion status after a host interruption.")
    args = parser.parse_args()
    if args.verify_only and args.recover_invalidated_attempt:
        parser.error("--verify-only and --recover-invalidated-attempt are mutually exclusive")
    result = recover_invalidated_attempt() if args.recover_invalidated_attempt else (verify() if args.verify_only else run())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
