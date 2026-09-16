"""Audit existing P3/P4 implementations and run their explicit 10-update smoke.

This is an engineering gate, not a performance experiment.  It never invokes
validation/final-test evaluation, never selects a checkpoint, and refuses to
run any update count other than the pre-declared two 10-update smokes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plain_mappo.checkpoint import load_checkpoint
from plain_mappo.engineering_gate import (ENGINEERING_GATE_PROTOCOL_VERSION,
                                          ENGINEERING_GATE_VARIANTS, SAMPLES_PER_UPDATE,
                                          SMOKE_RUN_SEED, SMOKE_UPDATES, implementation_audit,
                                          stage43_smoke_config)
from plain_mappo.topology import build_topology_features
from plain_mappo.trainer import MappoTrainer
from relay_env import RelayEnv


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "stage4-3-p3-p4-engineering-gate"
IMPLEMENTATION_AUDIT = OUTPUT / "implementation_audit.json"
IMPLEMENTATION_AUDIT_MD = OUTPUT / "implementation_audit.md"
PARAMETER_FAIRNESS = OUTPUT / "parameter_fairness.json"
SMOKE_MANIFEST = OUTPUT / "smoke_manifest.json"
SMOKE_SUMMARY = OUTPUT / "smoke_summary.csv"
PROTOCOL_AUDIT = OUTPUT / "protocol_audit.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _equal_state(left: Any, right: Any) -> bool:
    """Exact recursive comparison for serialized models, optimizers, and RNG."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal_state(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_equal_state(a, b) for a, b in zip(left, right))
    return left == right


def _parameters_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(before[name], after[name]) for name in before)


def _all_finite(values: dict[str, Any]) -> bool:
    return all(value is None or math.isfinite(float(value)) for value in values.values())


def _topology_action_check(trainer: MappoTrainer) -> dict[str, bool]:
    """Check the actual P3/P4 action-time tensors without evaluating a task."""
    env = RelayEnv(4)
    observation, _ = env.reset(seed=SMOKE_RUN_SEED)
    nodes, edges = build_topology_features(env, env.get_global_state())
    local = torch.as_tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
    node_tensor, edge_tensor = torch.as_tensor(nodes).unsqueeze(0), torch.as_tensor(edges).unsqueeze(0)
    with torch.no_grad():
        action = trainer.actor.deterministic_actions(local, node_tensor, edge_tensor)
        _, raw_log_std, log_std = trainer.actor.distribution_parameters(local, node_tensor, edge_tensor)
    return {
        "topology_nodes_finite": bool(np.isfinite(nodes).all()),
        "topology_edges_finite": bool(np.isfinite(edges).all()),
        "action_finite": bool(torch.isfinite(action).all()),
        "raw_log_std_finite": bool(torch.isfinite(raw_log_std).all()),
        "log_std_finite": bool(torch.isfinite(log_std).all()),
    }


def _reload_check(config: Any, trainer: MappoTrainer, checkpoint: Path) -> dict[str, bool]:
    payload = load_checkpoint(checkpoint, config.device)
    restored = MappoTrainer(config)
    restored.load(checkpoint)
    protocol_rng = payload["protocol_rng_state"]
    return {
        "checkpoint_exists": checkpoint.is_file(),
        "update_correct": restored.update_count == SMOKE_UPDATES == int(payload["update"]),
        "total_env_steps_correct": restored.total_env_steps == SMOKE_UPDATES * SAMPLES_PER_UPDATE == int(payload["total_env_steps"]),
        "actor_state_restored": _equal_state(trainer.actor.state_dict(), restored.actor.state_dict()),
        "critic_state_restored": _equal_state(trainer.critic.state_dict(), restored.critic.state_dict()),
        "actor_optimizer_restored": _equal_state(payload["actor_optimizer_state"], restored.actor_optimizer.state_dict()),
        "critic_optimizer_restored": _equal_state(payload["critic_optimizer_state"], restored.critic_optimizer.state_dict()),
        "normalizer_restored": _equal_state(payload["normalizer_state"], restored.normalizer.state_dict()),
        "value_normalizer_v0": payload["value_normalizer_state"] is None and restored.value_normalizer is None,
        "protocol_rng_restored": (
            torch.equal(restored.action_noise_generator.get_state(), protocol_rng["action_noise_generator"])  # type: ignore[union-attr]
            and _equal_state(restored.minibatch_rng.bit_generator.state, protocol_rng["minibatch_bit_generator"])  # type: ignore[union-attr]
        ),
        "fresh_boundary_resume": (
            payload["resume_mode"] == "fresh_episode_at_update_boundary"
            and payload["exact_environment_resume"] is False
            and payload["protocol_metadata"]["resume_mode"] == "fresh_episode_at_update_boundary"
            and payload["protocol_metadata"]["exact_environment_resume"] is False
        ),
    }


def _run_one_smoke(variant: str) -> dict[str, Any]:
    """Run one exact, no-evaluation ten-update P3/P4 engineering smoke."""
    with tempfile.TemporaryDirectory(prefix=f"stage4-3-{variant}-") as directory:
        config = stage43_smoke_config(actor_variant=variant, output_dir=directory)
        trainer = MappoTrainer(config)
        topology = _topology_action_check(trainer)
        actor_before = {name: parameter.detach().clone() for name, parameter in trainer.actor.state_dict().items()}
        critic_before = {name: parameter.detach().clone() for name, parameter in trainer.critic.state_dict().items()}
        records = trainer.train(SMOKE_UPDATES, final_evaluation=False, export_final_actor=False)
        if len(records) != SMOKE_UPDATES or trainer.update_count != SMOKE_UPDATES:
            raise RuntimeError("Stage-4.3 smoke did not complete exactly ten updates")
        finite = all(_all_finite(record) and bool(record.get("finite")) for record in records)
        diagnostics = {
            "actor_updated": _parameters_changed(actor_before, trainer.actor.state_dict()),
            "critic_updated": _parameters_changed(critic_before, trainer.critic.state_dict()),
            "diagnostics_finite": finite,
            "hard_speed_accel_violations": int(sum(float(record["hard_speed_accel_violations"]) for record in records)),
            "no_evaluation_performed": not (Path(directory) / "eval.csv").exists(),
            "no_actor_final_exported": not (Path(directory) / "checkpoints" / "actor_final.pt").exists(),
            **topology,
        }
        reload = _reload_check(config, trainer, Path(directory) / "checkpoints" / "latest.pt")
        checks = {**diagnostics, **reload}
        if (not all(value if isinstance(value, bool) else value == 0 for value in checks.values())
                or diagnostics["hard_speed_accel_violations"] != 0):
            raise RuntimeError(f"Stage-4.3 {variant} smoke failed engineering checks: {checks}")
        last = records[-1]
        return {
            "actor_variant": variant,
            "run_seed": SMOKE_RUN_SEED,
            "updates": SMOKE_UPDATES,
            "training_samples": SMOKE_UPDATES * SAMPLES_PER_UPDATE,
            "value_normalization": config.value_normalization,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "final_test_executed": False,
            "performance_evaluation_executed": False,
            "checkpoint_reload_pass": all(reload.values()),
            "actor_updated": diagnostics["actor_updated"],
            "critic_updated": diagnostics["critic_updated"],
            "diagnostics_finite": diagnostics["diagnostics_finite"],
            "hard_speed_accel_violations": diagnostics["hard_speed_accel_violations"],
            "topology_tensors_finite": topology["topology_nodes_finite"] and topology["topology_edges_finite"],
            "action_and_log_std_finite": (topology["action_finite"] and topology["raw_log_std_finite"]
                                           and topology["log_std_finite"]),
            "resume_mode": "fresh_episode_at_update_boundary",
            "exact_environment_resume": False,
            "actor_grad_norm_last": float(last["actor_grad_norm"]),
            "critic_grad_norm_last": float(last["critic_grad_norm"]),
            "actor_grad_clip_fraction_last": float(last["actor_grad_clip_fraction"]),
            "critic_grad_clip_fraction_last": float(last["critic_grad_clip_fraction"]),
            "approx_kl_last": float(last["approx_kl"]),
        }


def _write_static_audits() -> dict[str, Any]:
    audit = implementation_audit()
    fairness = audit["parameter_fairness"]
    _write_json(IMPLEMENTATION_AUDIT, audit)
    _write_json(PARAMETER_FAIRNESS, {
        "protocol_version": ENGINEERING_GATE_PROTOCOL_VERSION,
        "p3_actor_variant": "topology_info",
        "p4_actor_variant": "graph",
        **fairness,
    })
    markdown = "\n".join((
        "# Stage 4.3 P3/P4 implementation audit",
        "",
        "This is an engineering-contract audit only; it is not performance evidence.",
        "",
        f"- P3 (`topology_info`): 26 local-observation values plus fixed chain topology = {audit['P3']['actor_input_dim_k4']} inputs for K=4.",
        f"- P4 (`graph`): node=6, edge=7, hidden=32, shared chain message passing; Actor input = {audit['P4']['actor_input_dim_k4']} for K=4.",
        f"- P3 parameters: {fairness['p3_actor_parameters']}; P4 parameters: {fairness['p4_actor_parameters']}; relative gap: {fairness['relative_difference_of_larger']:.6f}.",
        "- Environment, Reward, 26-D observation, 47-D Critic, PPO/GAE, and `stage4-formal` are unchanged.",
        "- V0H0 remains fixed: value normalization disabled, gamma=0.99, lambda=0.95.",
        "",
    ))
    IMPLEMENTATION_AUDIT_MD.write_text(markdown, encoding="utf-8", newline="\n")
    return audit


def run_smokes() -> dict[str, Any]:
    """Write engineering reports after exactly one P3 and one P4 smoke."""
    audit = _write_static_audits()
    manifest = {
        "protocol_version": ENGINEERING_GATE_PROTOCOL_VERSION,
        "purpose": "P3/P4 engineering smoke; not an effectiveness comparison",
        "variants": list(ENGINEERING_GATE_VARIANTS),
        "paired_run_seed": SMOKE_RUN_SEED,
        "updates_per_variant": SMOKE_UPDATES,
        "samples_per_update": SAMPLES_PER_UPDATE,
        "samples_per_variant": SMOKE_UPDATES * SAMPLES_PER_UPDATE,
        "total_samples": len(ENGINEERING_GATE_VARIANTS) * SMOKE_UPDATES * SAMPLES_PER_UPDATE,
        "value_normalization": False,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "validation_executed": False,
        "final_test_executed": False,
        "checkpoint_selected": False,
        "formal_training_executed": False,
        "smoke_results": [],
    }
    _write_json(SMOKE_MANIFEST, manifest)
    results = [_run_one_smoke(variant) for variant in ENGINEERING_GATE_VARIANTS]
    manifest["smoke_results"] = results
    _write_json(SMOKE_MANIFEST, manifest)
    fields = tuple(results[0].keys())
    with SMOKE_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(results)
    _write_json(PROTOCOL_AUDIT, {
        "protocol_version": ENGINEERING_GATE_PROTOCOL_VERSION,
        "p3_engineering_contract_pass": True,
        "p4_engineering_contract_pass": True,
        "parameter_fairness_pass": audit["parameter_fairness"]["passes"],
        "smoke_pass": all(row["checkpoint_reload_pass"] and row["diagnostics_finite"] for row in results),
        "final_test_executed": False,
        "formal_training_executed": False,
        "scientific_claim": "No effectiveness or ranking claim is made from smoke data.",
    })
    return {"p3_smoke": "PASS", "p4_smoke": "PASS", "total_samples": manifest["total_samples"],
            "final_test_executed": False}


def verify() -> dict[str, Any]:
    """Read-only verification for committed Stage-4.3 non-binary reports."""
    required = (IMPLEMENTATION_AUDIT, IMPLEMENTATION_AUDIT_MD, PARAMETER_FAIRNESS, SMOKE_MANIFEST,
                SMOKE_SUMMARY, PROTOCOL_AUDIT)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage-4.3 reports missing: {missing}")
    audit, fairness, manifest, protocol = (_read_json(IMPLEMENTATION_AUDIT), _read_json(PARAMETER_FAIRNESS),
                                            _read_json(SMOKE_MANIFEST), _read_json(PROTOCOL_AUDIT))
    expected = implementation_audit()
    if audit != expected:
        raise ValueError("Stage-4.3 implementation audit drifted from source contract")
    if fairness != {"protocol_version": ENGINEERING_GATE_PROTOCOL_VERSION, "p3_actor_variant": "topology_info",
                    "p4_actor_variant": "graph", **expected["parameter_fairness"]}:
        raise ValueError("Stage-4.3 parameter fairness report drifted")
    if (manifest.get("variants") != list(ENGINEERING_GATE_VARIANTS)
            or manifest.get("updates_per_variant") != SMOKE_UPDATES
            or manifest.get("samples_per_update") != SAMPLES_PER_UPDATE
            or manifest.get("final_test_executed") is not False
            or len(manifest.get("smoke_results", [])) != 2):
        raise ValueError("Stage-4.3 smoke manifest violates its locked contract")
    if protocol.get("smoke_pass") is not True or protocol.get("final_test_executed") is not False:
        raise ValueError("Stage-4.3 protocol audit is incomplete")
    return {"p3_smoke": "PASS", "p4_smoke": "PASS", "final_test_executed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-smoke", action="store_true", help="Run the two exact 10-update engineering smokes.")
    group.add_argument("--verify", action="store_true", help="Read-only verify existing Stage-4.3 reports.")
    args = parser.parse_args()
    result = run_smokes() if args.run_smoke else verify()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
