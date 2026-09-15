"""Stage 4.1 protocol-repair orchestration and audit reporting.

This utility has three explicit, order-sensitive stages: ``prepare`` locks the
seed/checkpoint manifests; ``select`` consumes validation-only grid results;
``finalize`` audits already-completed selected-checkpoint final tests.  It
never trains a 1000-update model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plain_mappo.experiment_protocol import (CANONICAL_JSON_HASH_SCHEME, FINAL_TEST_SEEDS,
                                              LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION, VALIDATION_SEEDS,
                                              checkpoint_update, load_seed_manifest, provenance_snapshot,
                                              sha256_file, sha256_json_file, sha256_json_payload,
                                              split_seeds, write_locked_seed_manifest)
from reevaluate_checkpoint_grid import assert_episode_completeness, select_checkpoint_from_validation


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "stage4-1-protocol-repair"
SEED_MANIFEST = OUTPUT / "seed_manifest.json"
CHECKPOINT_MANIFEST = OUTPUT / "checkpoint_manifest.json"
VALIDATION_GRID = OUTPUT / "validation_grid.csv"
VALIDATION_EPISODES = OUTPUT / "validation_episodes.jsonl"
SELECTED = OUTPUT / "selected_checkpoints.json"
FINAL_SUMMARY = OUTPUT / "final_test_summary.csv"
FINAL_EPISODES = OUTPUT / "final_test_episodes.jsonl"
AUDIT_JSON = OUTPUT / "protocol_audit.json"
AUDIT_MD = OUTPUT / "protocol_audit.md"

RUNS = {
    "P0": {"variant": "plain", "run_seed": 2026, "directory": ROOT / "artifacts" / "stage3-stateindependent-full"},
    "P1": {"variant": "role_info", "run_seed": 2026, "directory": ROOT / "artifacts" / "stage4-role-info-full"},
    "P2": {"variant": "role_head", "run_seed": 2026, "directory": ROOT / "artifacts" / "stage4-role-head-full"},
}
REQUIRED_UPDATES = tuple(range(50, 1001, 50))
MISSING_REQUIRED_DOCUMENTS = (
    "20_Stage4_Role收口与Graph启动交接.md",
    "00_论文当前总体方案与决策记录_阶段3收口版.md",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _git_provenance() -> dict[str, Any]:
    completed = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode:
        return {"repository": False, "commit": None, "dirty": None, "detail": completed.stderr.strip() or completed.stdout.strip()}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return {"repository": True, "commit": commit.stdout.strip() if not commit.returncode else None,
            "dirty": bool(completed.stdout.strip()), "status": completed.stdout.splitlines()}


def _checkpoint_manifest() -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for name, specification in RUNS.items():
        directory = specification["directory"]
        files = sorted((directory / "eval_checkpoints").glob("actor_update_*.pt"), key=checkpoint_update)
        updates = tuple(checkpoint_update(path) for path in files)
        if updates != REQUIRED_UPDATES:
            raise ValueError(f"{name} periodic checkpoint updates are not the required 50..1000 sequence: {updates}")
        records = []
        for path in files:
            payload = __import__("torch").load(path, map_location="cpu", weights_only=False)
            config = payload["config"]
            records.append({"update": checkpoint_update(path), "path": str(path.relative_to(ROOT)),
                            "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                            "actor_variant": config.get("actor_variant", "plain"),
                            "protocol_status": config.get("protocol_version", LEGACY_PROTOCOL_VERSION),
                            "config_sha256": sha256_json_payload(config),
                            "config_hash_scheme": CANONICAL_JSON_HASH_SCHEME})
        runs[name] = {"variant": specification["variant"], "historical_run_seed": specification["run_seed"],
                      "directory": str(directory.relative_to(ROOT)), "checkpoints": records}
    return {"protocol_version": PROTOCOL_VERSION, "manifest_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "runs": runs}


def prepare() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest, manifest_hash, created = write_locked_seed_manifest(SEED_MANIFEST)
    checkpoint_manifest = _checkpoint_manifest()
    _write_json(CHECKPOINT_MANIFEST, checkpoint_manifest)
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "prepared",
        "manifest_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_manifest": str(SEED_MANIFEST.relative_to(ROOT)),
        "seed_manifest_sha256": manifest_hash,
        "seed_manifest_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
        "seed_manifest_created_this_invocation": created,
        "checkpoint_manifest": str(CHECKPOINT_MANIFEST.relative_to(ROOT)),
        "checkpoint_manifest_sha256": sha256_json_file(CHECKPOINT_MANIFEST),
        "checkpoint_manifest_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
        "provenance": {**provenance_snapshot("cpu"), "git": _git_provenance()},
        "source_document_conflicts": {"missing_required_documents": list(MISSING_REQUIRED_DOCUMENTS),
                                      "resolution": "current source code and extant artifacts take precedence"},
        "frozen_scope": ["Environment v1.1", "Reward", "26-d local observation", "47-d Critic input",
                         "P0-P4 Actor architectures", "PPO/GAE mathematics", "safety-priority ranking"],
    }
    _write_json(AUDIT_JSON, audit)
    return {"seed_manifest": manifest, "seed_manifest_sha256": manifest_hash,
            "checkpoint_manifest": checkpoint_manifest, "audit": audit}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for key in ("run_seed", "update", "checkpoint_size_bytes", "episodes", "normal_completion_count", "collision_count",
                    "boundary_count", "persistent_outage_count", "speed_accel_violations"):
            if key in item and item[key] != "":
                item[key] = int(item[key])
        for key in ("outage_step_ratio", "rate_satisfaction_ratio", "mean_e2e_rate_mbps", "mean_return"):
            if key in item and item[key] != "":
                item[key] = float(item[key])
        for key in ("score", "config_summary", "validation_seed_set", "final_test_seed_set"):
            if key in item and item[key]:
                item[key] = json.loads(item[key])
        parsed.append(item)
    return parsed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def select() -> dict[str, Any]:
    if not VALIDATION_GRID.exists() or not VALIDATION_EPISODES.exists():
        raise FileNotFoundError("validation_grid.csv and validation_episodes.jsonl must exist before selection")
    if FINAL_SUMMARY.exists() or FINAL_EPISODES.exists():
        raise RuntimeError("final-test artifacts already exist; selection is locked and cannot be rerun")
    manifest, manifest_hash = load_seed_manifest(SEED_MANIFEST)
    validation_rows = _read_csv_rows(VALIDATION_GRID)
    validation_episodes = _read_jsonl(VALIDATION_EPISODES)
    expected_row_count = len(RUNS) * len(REQUIRED_UPDATES)
    if len(validation_rows) != expected_row_count:
        raise ValueError(f"expected {expected_row_count} validation rows, got {len(validation_rows)}")
    assert_episode_completeness(validation_episodes,
                                [row["checkpoint_sha256"] for row in validation_rows], VALIDATION_SEEDS)
    selected: dict[str, Any] = {}
    for label, specification in RUNS.items():
        rows = [row for row in validation_rows if row["variant"] == specification["variant"]]
        if len(rows) != len(REQUIRED_UPDATES):
            raise ValueError(f"{label} does not have exactly 20 validation checkpoints")
        if {int(row["update"]) for row in rows} != set(REQUIRED_UPDATES):
            raise ValueError(f"{label} validation updates are incomplete")
        if any(row["seed_manifest_sha256"] != manifest_hash
               or row.get("seed_manifest_hash_scheme") != CANONICAL_JSON_HASH_SCHEME for row in rows):
            raise ValueError(f"{label} validation rows use a different seed manifest")
        winner = select_checkpoint_from_validation([{**row, "split": "validation"} for row in rows])
        selected[label] = {
            "variant": winner["variant"], "run_seed": winner["run_seed"], "selected_update": winner["update"],
            "checkpoint_path": winner["checkpoint_path"], "checkpoint_sha256": winner["checkpoint_sha256"],
            "checkpoint_size_bytes": winner["checkpoint_size_bytes"], "validation_score": winner["score"],
            "selection_rule": "safety_priority_key; exact tie selects earlier update", "validation_seed_manifest": str(SEED_MANIFEST.relative_to(ROOT)),
            "validation_seed_manifest_sha256": manifest_hash,
            "validation_seed_manifest_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
            "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    payload = {"protocol_version": PROTOCOL_VERSION, "manifest_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
               "split_used_for_selection": "validation",
               "validation_seeds": list(split_seeds(manifest, "validation")), "selected": selected}
    _write_json(SELECTED, payload)
    return payload


def _selected_validation_episode_stats(selected: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_hashes = {entry["checkpoint_sha256"] for entry in selected.values()}
    for episode in episodes:
        if episode["checkpoint_sha256"] in selected_hashes:
            by_variant[episode["variant"]].append(episode)
    output: dict[str, Any] = {}
    for variant, values in by_variant.items():
        lengths = [int(item["length"]) for item in values]
        first_outages = [int(item["first_outage_step"]) for item in values if item.get("first_outage_step") is not None]
        boundaries = [int(item["terminal_step"]) for item in values if item.get("termination_reason") == "boundary_violation"]
        terminals = [int(item["terminal_step"]) for item in values if not bool(item.get("truncated"))]
        output[variant] = {
            "episodes": len(values), "episode_length_min_mean_max": [min(lengths), sum(lengths) / len(lengths), max(lengths)],
            "first_outage_step_min_mean_max": None if not first_outages else [min(first_outages), sum(first_outages) / len(first_outages), max(first_outages)],
            "boundary_termination_steps": boundaries,
            "terminal_penalty_steps": terminals,
        }
    return output


def gamma_horizon_analysis(selected: dict[str, Any]) -> dict[str, Any]:
    episodes = _read_jsonl(VALIDATION_EPISODES)
    gammas = (0.99, 0.995, 0.997, 0.999)
    table = []
    for gamma in gammas:
        trace = gamma * 0.95
        table.append({"gamma": gamma, "step_half_life": math.log(0.5) / math.log(gamma),
                      "seconds_half_life": 0.2 * math.log(0.5) / math.log(gamma),
                      "terminal_weight_at_100_seconds": gamma ** 500,
                      "gae_trace_half_life_steps": math.log(0.5) / math.log(trace),
                      "gae_trace_half_life_seconds": 0.2 * math.log(0.5) / math.log(trace)})
    observed = _selected_validation_episode_stats(selected, episodes)
    # At gamma=.99 a 100-s terminal signal has <1% direct weight.  This is a
    # concrete long-horizon credit risk, not proof that a different gamma wins.
    return {"dt_seconds": 0.2, "episode_horizon_steps": 500, "episode_horizon_seconds": 100.0,
            "gae_lambda": 0.95, "table": table, "selected_validation_episode_timing": observed,
            "recommendation": "PRE-EXPERIMENT REQUIRED",
            "rationale": "gamma=0.99 gives a 100-s terminal reward weight of %.8f and a %.2f-s half-life; observed terminal/outage timing must be tested under a separately authorized symmetric protocol before changing gamma." % (0.99 ** 500, table[0]["seconds_half_life"])}


def _historical_train_diagnostics() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, specification in RUNS.items():
        path = specification["directory"] / "train.csv"
        rows = _read_csv_rows(path)
        critic_norms = [float(row["critic_grad_norm"]) for row in rows if row.get("critic_grad_norm", "") != ""]
        result[label] = {"train_csv": str(path.relative_to(ROOT)), "updates": len(rows),
                         "critic_grad_norm_min_mean_max": None if not critic_norms else [min(critic_norms), sum(critic_norms) / len(critic_norms), max(critic_norms)],
                         "new_value_diagnostics_available": "value_target_mean" in rows[0] if rows else False}
    smoke_path = OUTPUT / "diagnostics-smoke" / "train.csv"
    if smoke_path.exists():
        smoke_rows = _read_csv_rows(smoke_path)
        requested = ("value_target_mean", "value_target_std", "value_prediction_mean_pre",
                     "value_prediction_std_pre", "explained_variance_pre", "actor_grad_norm",
                     "actor_grad_norm_max", "actor_grad_clip_fraction", "critic_grad_norm",
                     "critic_grad_norm_max", "critic_grad_clip_fraction", "finite")
        final = {key: (float(smoke_rows[-1][key]) if smoke_rows and smoke_rows[-1].get(key, "") != "" else None)
                 for key in requested}
        result["diagnostic_smoke"] = {"train_csv": str(smoke_path.relative_to(ROOT)), "updates": len(smoke_rows),
                                       "final": final}
    return result


def finalize() -> dict[str, Any]:
    required = (SEED_MANIFEST, CHECKPOINT_MANIFEST, VALIDATION_GRID, VALIDATION_EPISODES, SELECTED, FINAL_SUMMARY, FINAL_EPISODES)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cannot finalize; missing {missing}")
    manifest, manifest_hash = load_seed_manifest(SEED_MANIFEST)
    selected_payload = _json(SELECTED)
    selected = selected_payload["selected"]
    final_rows = _read_csv_rows(FINAL_SUMMARY)
    final_episodes = _read_jsonl(FINAL_EPISODES)
    if len(final_rows) != len(RUNS):
        raise ValueError("final-test rows must be exactly three selected checkpoints × 50 episodes")
    final_by_variant = {row["variant"]: row for row in final_rows}
    for entry in selected.values():
        row = final_by_variant.get(entry["variant"])
        if row is None or row["checkpoint_sha256"] != entry["checkpoint_sha256"]:
            raise ValueError("final test contains an unselected checkpoint")
        if (row["seed_manifest_sha256"] != manifest_hash
                or row.get("seed_manifest_hash_scheme") != CANONICAL_JSON_HASH_SCHEME):
            raise ValueError("final test used the wrong seed manifest")
    expected_hashes = {entry["checkpoint_sha256"] for entry in selected.values()}
    assert_episode_completeness(final_episodes, expected_hashes, FINAL_TEST_SEEDS)
    gamma = gamma_horizon_analysis(selected)
    critic = _historical_train_diagnostics()
    critic_recommendation = "PRE-EXPERIMENT REQUIRED"
    invocation_path = OUTPUT / "evaluation_invocations.jsonl"
    invocations = _read_jsonl(invocation_path) if invocation_path.exists() else []
    audit = {
        "protocol_version": PROTOCOL_VERSION, "manifest_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
        "stage": "finalized", "started_at_utc": _json(AUDIT_JSON).get("started_at_utc"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "seed_manifest_sha256": manifest_hash,
        "seed_manifest_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
        "checkpoint_manifest_sha256": sha256_json_file(CHECKPOINT_MANIFEST),
        "checkpoint_manifest_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
        "selected_checkpoints_sha256": sha256_json_file(SELECTED),
        "selected_checkpoints_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
        "validation_rows": len(_read_csv_rows(VALIDATION_GRID)), "validation_episode_rows": len(_read_jsonl(VALIDATION_EPISODES)),
        "final_rows": len(final_rows), "final_episode_rows": len(final_episodes), "provenance": {**provenance_snapshot("cpu"), "git": _git_provenance()},
        "source_document_conflicts": {"missing_required_documents": list(MISSING_REQUIRED_DOCUMENTS),
                                      "resolution": "current source code and extant artifacts take precedence"},
        "evaluation_execution": {"invocation_log": str(invocation_path.relative_to(ROOT)),
                                 "completed_bounded_invocations": len(invocations),
                                 "note": "P0 initial all-grid invocation reached the host 30-second limit after writing rows. Its two subsequently repeated P0 summaries/raw Episodes were mechanically deduplicated by checkpoint SHA-256 and seed; final integrity checks require exactly 60×20 unique validation rows."},
        "critic_grad_diagnostics": critic, "critic_recommendation": critic_recommendation,
        "gamma_horizon_analysis": gamma,
        "resume_semantics": {"resume_mode": "fresh_episode_at_update_boundary", "exact_environment_resume": False,
                             "statement": "weights, optimizers, normalizer, RNG and unused environment seed progress resume at a PPO update boundary; environment internals/current observations/partial episodes are not restored."},
        "formal_training_executed": False,
    }
    _write_json(AUDIT_JSON, audit)
    _write_json(ROOT / "stage4_1_protocol_validation.json", audit)
    markdown = _markdown(audit, selected, final_rows)
    (ROOT / "stage4_1_protocol_validation.md").write_text(markdown, encoding="utf-8")
    AUDIT_MD.write_text(markdown, encoding="utf-8")
    return audit


def _markdown(audit: dict[str, Any], selected: dict[str, Any], final_rows: list[dict[str, Any]]) -> str:
    lines = ["# Stage 4.1 训练与评估协议修复验证", "",
             "本报告记录 Stage 4.1 的真实协议修复、旧 checkpoint 独立重评和最终 test。没有执行任何新的 1000-update 正式训练。", "",
             "## Seed 与选择协议", "",
             f"- protocol: `{audit['protocol_version']}`；manifest hash scheme: `{audit['manifest_hash_scheme']}`；validation/test manifest SHA-256: `{audit['seed_manifest_sha256']}`。",
             f"- validation: 20 seeds `{VALIDATION_SEEDS[0]}..{VALIDATION_SEEDS[-1]}`；final test: 50 seeds `{FINAL_TEST_SEEDS[0]}..{FINAL_TEST_SEEDS[-1]}`；两者严格分离。",
             "- checkpoint 只按 validation 的 frozen safety-priority key 选择；完全同分选择较早 update；test 未参与选择。", "",
             "## Selected checkpoints", "", "| Variant | update | validation score | SHA-256 |", "|---|---:|---|---|"]
    for label in ("P0", "P1", "P2"):
        item = selected[label]
        lines.append(f"| {label} {item['variant']} | {item['selected_update']} | {item['validation_score']} | `{item['checkpoint_sha256']}` |")
    lines += ["", "## Independent 50-episode final test", "",
              "| Variant | normal completion | persistent outage | collision | boundary | hard violations | mean e2e Mbps | outage ratio | rate satisfaction |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(final_rows, key=lambda item: item["variant"]):
        lines.append(f"| {row['variant']} | {row['normal_completion_count']}/50 | {row['persistent_outage_count']} | {row['collision_count']} | {row['boundary_count']} | {row['speed_accel_violations']} | {row['mean_e2e_rate_mbps']:.6f} | {row['outage_step_ratio']:.6f} | {row['rate_satisfaction_ratio']:.6f} |")
    gamma = audit["gamma_horizon_analysis"]
    critic = audit["critic_grad_diagnostics"]
    lines += ["", "## Critic / gradient diagnostics", "",
              "| Historic run | critic pre-clip norm min / mean / max | New value diagnostics in historic log |",
              "|---|---|---|"]
    for label in ("P0", "P1", "P2"):
        values = critic[label]["critic_grad_norm_min_mean_max"]
        lines.append(f"| {label} | {values[0]:.6f} / {values[1]:.6f} / {values[2]:.6f} | {critic[label]['new_value_diagnostics_available']} |")
    smoke = critic.get("diagnostic_smoke", {}).get("final", {})
    lines += ["", "10-update diagnostic smoke final update: "
              f"value target mean/std={smoke.get('value_target_mean')}/{smoke.get('value_target_std')}; "
              f"prediction mean/std={smoke.get('value_prediction_mean_pre')}/{smoke.get('value_prediction_std_pre')}; "
              f"explained variance={smoke.get('explained_variance_pre')}; actor clip fraction={smoke.get('actor_grad_clip_fraction')}; "
              f"critic clip fraction={smoke.get('critic_grad_clip_fraction')}; finite={smoke.get('finite')}.",
              f"- Recommendation: `{audit['critic_recommendation']}`. The evidence is diagnostic only; no value normalization, value clipping or critic-LR change was enabled.",
              "", "## Gamma / horizon diagnosis", "",
              "| gamma | half-life steps | half-life seconds | terminal weight at 100 s | GAE trace half-life seconds |",
              "|---:|---:|---:|---:|---:|"]
    for item in gamma["table"]:
        lines.append(f"| {item['gamma']:.3f} | {item['step_half_life']:.3f} | {item['seconds_half_life']:.3f} | {item['terminal_weight_at_100_seconds']:.8f} | {item['gae_trace_half_life_seconds']:.3f} |")
    lines += [f"- Recommendation: `{gamma['recommendation']}`. {gamma['rationale']}",
              "", "## Resume semantics", "",
              "`resume_mode = fresh_episode_at_update_boundary`; `exact_environment_resume = false`. A resume restores model/optimizer/normalizer/RNG/unused seed progress but starts fresh episodes; it is not a bitwise-equivalent uninterrupted trajectory.",
              "", "## Audit caveat", "",
              "The task-specified `20_...` and `00_...` source documents were absent from this workspace. This audit used current source code plus actual artifacts and records that conflict explicitly.",
              "", "## Execution trace note", "",
              "`evaluation_invocations.jsonl` records completed bounded evaluation commands. The initial P0 all-grid command reached the host 30-second limit after writing output; the only duplicated P0 950/1000 records were mechanically deduplicated by checkpoint SHA-256 and seed. The final integrity audit verifies exactly 60×20 unique validation Episode rows and 3×50 final-test rows.",
              "", "## Status", "",
              "Stage 4.1 protocol repair = COMPLETE", "", "Engineering validation = PASS", "",
              "P0/P1/P2 retrospective status = SINGLE-SEED PILOT", "", "P3 formal training = NOT STARTED", "",
              "P4 formal training = NOT STARTED", "", "Multi-seed P0-P4 training = NOT AUTHORIZED", "", "Role+Graph = NOT STARTED", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4.1 protocol repair stages.")
    parser.add_argument("stage", choices=("prepare", "select", "finalize"))
    args = parser.parse_args()
    result = {"prepare": prepare, "select": select, "finalize": finalize}[args.stage]()
    print(json.dumps({"stage": args.stage, "completed": True, "result_keys": sorted(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
