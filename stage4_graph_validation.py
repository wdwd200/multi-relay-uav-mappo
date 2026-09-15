"""Reproducible implementation validation for Stage 4 P3/P4 topology ablations.

This script verifies only contracts, checkpoint compatibility, and the two
ten-update engineering smoke runs.  It never starts a formal 1000-update
training run and does not mutate the frozen environment.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plain_mappo.config import MappoConfig
from plain_mappo.evaluation import evaluate_actor
from plain_mappo.networks import SharedActor
from plain_mappo.topology import (GRAPH_HIDDEN_DIM, TOPOLOGY_EDGE_DIM,
                                  TOPOLOGY_NODE_DIM, build_topology_features,
                                  topology_feature_shapes)
from relay_env import RelayEnv


ROOT = Path(__file__).resolve().parent
SMOKES = {
    "topology_info": ROOT / "artifacts" / "stage4-topology-info-smoke",
    "graph": ROOT / "artifacts" / "stage4-graph-smoke",
}


def _load_actor(checkpoint: Path) -> tuple[SharedActor, MappoConfig]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = MappoConfig.from_dict(payload["config"])
    actor = SharedActor(config.local_obs_dim, config.action_dim, config.actor_hidden_sizes,
                        config.log_std_min, config.log_std_max, config.actor_log_std_mode,
                        config.state_independent_log_std_init, config.actor_variant,
                        config.topology_node_dim, config.topology_edge_dim,
                        config.graph_hidden_dim)
    actor.load_state_dict(payload["actor_state"])
    actor.eval()
    return actor, config


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _topology_contract() -> dict[str, Any]:
    shapes = {f"K{relays}": {"nodes": list(topology_feature_shapes(relays)[0]),
                               "edges": list(topology_feature_shapes(relays)[1])}
              for relays in (3, 4, 5)}
    env = RelayEnv(4)
    _, _ = env.reset(seed=2026)
    state = env.get_global_state()
    nodes, edges = build_topology_features(env, state)
    h = state["H"]
    expected_h = np.array((h["position"][0] / env.config.map_x_m,
                           h["position"][1] / env.config.map_y_m,
                           (h["position"][2] - 200.0) / 100.0,
                           h["velocity"][0] / 20.0,
                           h["velocity"][1] / 20.0,
                           h["velocity"][2] / 6.0), dtype=np.float32)
    exact_node_normalization = bool(np.allclose(nodes[0], expected_h, rtol=0.0, atol=1e-7))
    return {
        "node_dim": TOPOLOGY_NODE_DIM,
        "edge_dim": TOPOLOGY_EDGE_DIM,
        "graph_hidden_dim": GRAPH_HIDDEN_DIM,
        "shapes": shapes,
        "k4_nodes_shape": list(nodes.shape),
        "k4_edges_shape": list(edges.shape),
        "exact_h_node_normalization": exact_node_normalization,
        "all_features_finite": bool(np.isfinite(nodes).all() and np.isfinite(edges).all()),
        "prohibited_fields_excluded": ["waypoint", "cruise_speed", "step", "sim_time",
                                        "consecutive_outage_steps", "Role", "Agent ID", "chain index"],
        "pass": bool(shapes["K3"] == {"nodes": [5, 6], "edges": [4, 7]}
                     and shapes["K4"] == {"nodes": [6, 6], "edges": [5, 7]}
                     and shapes["K5"] == {"nodes": [7, 6], "edges": [6, 7]}
                     and exact_node_normalization and np.isfinite(nodes).all() and np.isfinite(edges).all()),
    }


def _parameter_report() -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for variant in ("plain", "role_info", "role_head", "topology_info", "graph"):
        actor = SharedActor(26, 3, log_std_min=-4.0, log_std_max=0.0,
                            log_std_mode="state_independent_tanh",
                            state_independent_log_std_init=-1.5, actor_variant=variant)
        report[variant] = {
            "environment_observation_dim": actor.obs_dim,
            "effective_actor_input_dim": actor.effective_input_dim,
            "actor_parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
        }
    p3, p4 = report["topology_info"]["actor_parameter_count"], report["graph"]["actor_parameter_count"]
    report["comparison"] = {
        "p3_p4_relative_difference": abs(p3 - p4) / max(p3, p4),
        "within_ten_percent": abs(p3 - p4) / max(p3, p4) <= 0.10,
    }
    return report


def _read_smoke(variant: str, directory: Path) -> dict[str, Any]:
    required_files = (directory / "config.json", directory / "train.csv", directory / "eval.csv",
                      directory / "checkpoints" / "latest.pt", directory / "checkpoints" / "best.pt",
                      directory / "checkpoints" / "actor_final.pt",
                      directory / "eval_checkpoints" / "actor_update_0010.pt")
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    if missing:
        return {"variant": variant, "directory": str(directory.relative_to(ROOT)), "pass": False,
                "missing": missing}
    config_data = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    with (directory / "train.csv").open(newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    with (directory / "eval.csv").open(newline="", encoding="utf-8") as handle:
        eval_rows = list(csv.DictReader(handle))
    final_train = train_rows[-1] if train_rows else {}
    final_eval = eval_rows[-1] if eval_rows else {}
    finite_columns = ("actor_policy_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction",
                      "actor_grad_norm", "critic_grad_norm", "raw_log_std_min", "raw_log_std_mean",
                      "raw_log_std_max", "clamped_log_std_min", "clamped_log_std_mean",
                      "clamped_log_std_max", "action_saturation_ratio")
    finite_logs = bool(train_rows) and all(
        _finite(row.get(column)) and _finite(row.get("finite")) and float(row["finite"]) == 1.0
        for row in train_rows for column in finite_columns
    )
    config_ok = (
        config_data.get("actor_variant") == variant
        and config_data.get("actor_log_std_mode") == "state_independent_tanh"
        and config_data.get("log_std_min") == -4.0 and config_data.get("log_std_max") == 0.0
        and config_data.get("state_independent_log_std_init") == -1.5
        and config_data.get("actor_lr") == 1e-4 and config_data.get("critic_lr") == 3e-4
        and config_data.get("entropy_coef") == 0.0 and config_data.get("local_obs_dim") == 26
        and config_data.get("global_state_dim") == 47
        and config_data.get("topology_node_dim") == 6 and config_data.get("topology_edge_dim") == 7
        and config_data.get("graph_hidden_dim") == 32
    )
    reported_evaluation = json.loads(final_eval["summary_json"]) if final_eval else {}
    actor_final = directory / "checkpoints" / "actor_final.pt"
    reloaded_actor, reloaded_config = _load_actor(actor_final)
    reload_summary, _, _ = evaluate_actor(reloaded_actor, reloaded_config, range(10_000, 10_005))
    reload_matches = bool(reported_evaluation) and all(
        reload_summary[key] == reported_evaluation[key] if isinstance(reload_summary[key], int)
        else math.isclose(float(reload_summary[key]), float(reported_evaluation[key]), rel_tol=1e-10, abs_tol=1e-10)
        for key in ("normal_completion_count", "persistent_outage_count", "collision_count", "boundary_count",
                    "mean_e2e_rate_mbps", "outage_step_ratio", "rate_satisfaction_ratio")
    )
    updates = int(final_train.get("update", 0)) if final_train else 0
    result = {
        "variant": variant,
        "directory": str(directory.relative_to(ROOT)),
        "updates": updates,
        "team_time_samples": int(config_data.get("team_time_samples", 0)),
        "total_samples": updates * int(config_data.get("team_time_samples", 0)),
        "configuration_contract_pass": config_ok,
        "finite_logs": finite_logs,
        "checkpoint_files_present": True,
        "final_train": {key: float(final_train[key]) for key in finite_columns if key in final_train and _finite(final_train[key])},
        "final_evaluation": reported_evaluation,
        "actor_final_reload_evaluation": reload_summary,
        "actor_final_reload_matches_logged_evaluation": reload_matches,
    }
    result["pass"] = (updates == 10 and result["total_samples"] == 10_240 and config_ok and finite_logs
                      and bool(reported_evaluation) and reload_matches)
    return result


def _run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "selfcheck", "-p", "test_mappo*.py"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    output = completed.stdout + completed.stderr
    count = next((int(line.split()[1]) for line in output.splitlines()
                  if line.startswith("Ran ") and " tests" in line), None)
    return {"command": " ".join(command), "return_code": completed.returncode, "tests_run": count,
            "pass": completed.returncode == 0, "output_tail": output.strip().splitlines()[-5:]}


def _markdown(report: dict[str, Any]) -> str:
    topology, parameters, smokes, tests = (report["topology_contract"], report["parameter_report"],
                                           report["smokes"], report["automated_tests"])
    lines = [
        "# Stage 4 P3/P4 Topology / Graph 验证",
        "",
        "本报告仅覆盖实现合同、回归和 10-update smoke；未执行 P3/P4 的 1000-update 正式训练。Environment v1.1、Reward、26 维 local observation、47 维 Critic、PPO/GAE 数学均未改动。",
        "",
        "## Topology 特征合同",
        "",
        f"- node/edge/graph-hidden dimensions: {topology['node_dim']} / {topology['edge_dim']} / {topology['graph_hidden_dim']}。",
        f"- K=3: nodes={topology['shapes']['K3']['nodes']}, edges={topology['shapes']['K3']['edges']}；K=4: nodes={topology['shapes']['K4']['nodes']}, edges={topology['shapes']['K4']['edges']}；K=5: nodes={topology['shapes']['K5']['nodes']}, edges={topology['shapes']['K5']['edges']}。",
        f"- K=4 shape/归一化/有限性: {'PASS' if topology['pass'] else 'FAIL'}。边容量由既有 `relay_env.communication.link_metrics()` 构造；自动测试进一步逐字段验证 edge delta、velocity 和 capacity。",
        "- 仅使用 H/L/Relay 当前 position 与 velocity；不读取 waypoint、cruise speed、step、sim time、outage counter、Role、Agent ID 或 chain index。",
        "",
        "## Actor 参数量",
        "",
        "| Variant | 环境 observation | Actor 实际输入 | Actor parameters |",
        "|---|---:|---:|---:|",
    ]
    for variant in ("plain", "role_info", "role_head", "topology_info", "graph"):
        item = parameters[variant]
        lines.append(f"| {variant} | {item['environment_observation_dim']} | {item['effective_actor_input_dim']} | {item['actor_parameter_count']} |")
    comparison = parameters["comparison"]
    lines += [
        "",
        f"P3/P4 parameter relative difference: {comparison['p3_p4_relative_difference']:.4%} — {'PASS' if comparison['within_ten_percent'] else 'FAIL'} (≤10%)。",
        "",
        "## 自动测试",
        "",
        f"- `{tests['command']}`: {'PASS' if tests['pass'] else 'FAIL'}; {tests['tests_run']} tests executed。",
        "- 覆盖 P3/P4 shapes、归一化、capacity、prohibited-field exclusion、共享 encoder/message/update、K 轮 receptive field、PPO old latent 重算、checkpoint round-trip，以及 P0/P1/P2 checkpoint 兼容。",
        "",
        "## 10-update smoke",
        "",
    ]
    for variant in ("topology_info", "graph"):
        smoke = smokes[variant]
        train, evaluation = smoke.get("final_train", {}), smoke.get("final_evaluation", {})
        lines += [
            f"### {variant}",
            "",
            f"- 目录: `{smoke.get('directory')}`；结果: {'PASS' if smoke.get('pass') else 'FAIL'}；updates={smoke.get('updates')}；samples={smoke.get('total_samples')}。",
            f"- final training: policy loss={train.get('actor_policy_loss')}，critic loss={train.get('critic_loss')}，KL={train.get('approx_kl')}，clip fraction={train.get('clip_fraction')}，actor/critic grad norm={train.get('actor_grad_norm')}/{train.get('critic_grad_norm')}。",
            f"- effective log_std mean={train.get('clamped_log_std_mean')}，action saturation={train.get('action_saturation_ratio')}，finite logs={smoke.get('finite_logs')}。",
            f"- final 5-seed deterministic eval: normal={evaluation.get('normal_completion_count')}，persistent outage={evaluation.get('persistent_outage_count')}，collision={evaluation.get('collision_count')}，boundary={evaluation.get('boundary_count')}，mean e2e={evaluation.get('mean_e2e_rate_mbps')} Mbps。",
            f"- actor_final reload eval matches log: {smoke.get('actor_final_reload_matches_logged_evaluation')}。",
            "",
        ]
    lines += [
        "## Status",
        "",
        "Stage 4 topology/graph implementation = COMPLETE",
        "",
        f"Engineering validation = {'PASS' if report['overall_pass'] else 'FAIL'}",
        "",
        "P3 formal training = NOT STARTED",
        "",
        "P4 formal training = NOT STARTED",
        "",
        "Role+Graph = NOT STARTED",
        "",
        "Smoke 仅验证工程连通性，不构成 P3/P4 性能有效性或优越性的结论。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    report = {
        "stage": "Stage 4 P3/P4 topology/graph implementation validation",
        "topology_contract": _topology_contract(),
        "parameter_report": _parameter_report(),
        "smokes": {variant: _read_smoke(variant, directory) for variant, directory in SMOKES.items()},
    }
    report["automated_tests"] = _run_tests()
    report["overall_pass"] = (report["topology_contract"]["pass"]
                              and report["parameter_report"]["comparison"]["within_ten_percent"]
                              and report["automated_tests"]["pass"]
                              and all(smoke.get("pass", False) for smoke in report["smokes"].values()))
    (ROOT / "stage4_graph_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "stage4_graph_validation.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"overall_pass": report["overall_pass"], "json": "stage4_graph_validation.json",
                      "markdown": "stage4_graph_validation.md"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
