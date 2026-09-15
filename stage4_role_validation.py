"""Reproducible Stage 4 Role-ablation implementation and smoke validation.

This script only loads checkpoints, evaluates the frozen Plain baseline, reads
the two required smoke directories, and runs the MAPPO self-check suite.  It
does not train a formal Role model and never mutates ``RelayEnv``.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from plain_mappo.config import MappoConfig
from plain_mappo.evaluation import evaluate_actor
from plain_mappo.networks import SharedActor
from plain_mappo.roles import role_ids_for_num_relays


ROOT = Path(__file__).resolve().parent
P0_CHECKPOINT = ROOT / "artifacts" / "stage3-stateindependent-full" / "checkpoints" / "actor_final.pt"
SMOKES = {
    "role_info": ROOT / "artifacts" / "stage4-role-info-smoke",
    "role_head": ROOT / "artifacts" / "stage4-role-head-smoke",
}


def _load_actor(checkpoint: Path) -> tuple[SharedActor, MappoConfig]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = MappoConfig.from_dict(payload["config"])
    actor = SharedActor(config.local_obs_dim, config.action_dim, config.actor_hidden_sizes,
                        config.log_std_min, config.log_std_max, config.actor_log_std_mode,
                        config.state_independent_log_std_init, config.actor_variant)
    actor.load_state_dict(payload["actor_state"])
    actor.eval()
    return actor, config


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_smoke(variant: str, directory: Path) -> dict[str, Any]:
    config_path, train_path, eval_path = directory / "config.json", directory / "train.csv", directory / "eval.csv"
    missing = [str(path.relative_to(ROOT)) for path in (config_path, train_path, eval_path) if not path.exists()]
    if missing:
        return {"variant": variant, "directory": str(directory.relative_to(ROOT)), "pass": False, "missing": missing}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with train_path.open(newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    with eval_path.open(newline="", encoding="utf-8") as handle:
        eval_rows = list(csv.DictReader(handle))
    final_train = train_rows[-1] if train_rows else {}
    final_eval = eval_rows[-1] if eval_rows else {}
    finite_columns = ("actor_policy_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction",
                      "actor_grad_norm", "critic_grad_norm", "raw_log_std_mean", "clamped_log_std_mean")
    finite_logs = bool(train_rows) and all(_finite(row.get(column)) and _finite(row.get("finite"))
                                             and float(row["finite"]) == 1.0
                                             for row in train_rows for column in finite_columns)
    checkpoint_paths = {
        name: directory / "checkpoints" / filename
        for name, filename in (("latest", "latest.pt"), ("best", "best.pt"), ("actor_final", "actor_final.pt"))
    }
    periodic = directory / "eval_checkpoints" / "actor_update_0010.pt"
    required_checkpoints = {name: path.exists() for name, path in checkpoint_paths.items()}
    required_checkpoints["periodic_actor_update_0010"] = periodic.exists()
    config_ok = (config.get("actor_variant") == variant and config.get("actor_log_std_mode") == "state_independent_tanh"
                 and config.get("log_std_min") == -4.0 and config.get("log_std_max") == 0.0
                 and config.get("state_independent_log_std_init") == -1.5 and config.get("actor_lr") == 1e-4
                 and config.get("critic_lr") == 3e-4 and config.get("entropy_coef") == 0.0
                 and config.get("local_obs_dim") == 26 and config.get("global_state_dim") == 47)
    updates = int(final_train.get("update", 0)) if final_train else 0
    reported_evaluation = json.loads(final_eval["summary_json"]) if final_eval else {}
    reload_summary: dict[str, Any] = {}
    reload_matches_logged_eval = False
    if checkpoint_paths["actor_final"].exists():
        reloaded_actor, reloaded_config = _load_actor(checkpoint_paths["actor_final"])
        reload_summary, _, _ = evaluate_actor(reloaded_actor, reloaded_config, range(10_000, 10_005))
        reload_matches_logged_eval = bool(reported_evaluation) and all(
            (reload_summary[key] == reported_evaluation[key] if isinstance(reload_summary[key], int)
             else math.isclose(float(reload_summary[key]), float(reported_evaluation[key]), rel_tol=1e-10, abs_tol=1e-10))
            for key in ("normal_completion_count", "persistent_outage_count", "collision_count", "boundary_count",
                        "mean_e2e_rate_mbps", "outage_step_ratio", "rate_satisfaction_ratio")
        )
    result = {
        "variant": variant,
        "directory": str(directory.relative_to(ROOT)),
        "updates": updates,
        "team_time_samples": int(config.get("team_time_samples", 0)),
        "total_samples": updates * int(config.get("team_time_samples", 0)),
        "configuration_contract_pass": config_ok,
        "finite_logs": finite_logs,
        "checkpoints": required_checkpoints,
        "final_train": {column: float(final_train[column]) for column in finite_columns + ("action_saturation_ratio",)
                        if column in final_train and _finite(final_train[column])},
        "final_evaluation": reported_evaluation,
        "actor_final_reload_evaluation": reload_summary,
        "actor_final_reload_matches_logged_evaluation": reload_matches_logged_eval,
    }
    result["pass"] = (updates == 10 and config_ok and finite_logs and all(required_checkpoints.values())
                      and bool(result["final_evaluation"]) and reload_matches_logged_eval)
    return result


def _parameter_report() -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    for variant in ("plain", "role_info", "role_head"):
        actor = SharedActor(26, 3, log_std_min=-4.0, log_std_max=0.0,
                            log_std_mode="state_independent_tanh", state_independent_log_std_init=-1.5,
                            actor_variant=variant)
        report[variant] = {
            "environment_observation_dim": actor.obs_dim,
            "effective_actor_input_dim": actor.effective_input_dim,
            "actor_parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
        }
    return report


def _p0_regression() -> dict[str, Any]:
    if not P0_CHECKPOINT.exists():
        return {"pass": False, "missing": str(P0_CHECKPOINT.relative_to(ROOT))}
    actor, config = _load_actor(P0_CHECKPOINT)
    summary, score, _ = evaluate_actor(actor, config, range(10_000, 10_020))
    passed = (summary["normal_completion_count"] == 0 and summary["persistent_outage_count"] == 19
              and summary["boundary_count"] == 1 and summary["collision_count"] == 0
              and math.isclose(summary["mean_e2e_rate_mbps"], 6.103710197871623, rel_tol=1e-8, abs_tol=1e-8))
    return {"checkpoint": str(P0_CHECKPOINT.relative_to(ROOT)), "config_actor_variant": config.actor_variant,
            "seeds": [10_000, 10_019], "score": list(score), "summary": summary, "pass": passed}


def _run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "selfcheck", "-p", "test_mappo*.py"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    output = completed.stdout + completed.stderr
    count = None
    for line in output.splitlines():
        if line.startswith("Ran ") and " tests" in line:
            count = int(line.split()[1])
            break
    return {"command": " ".join(command), "return_code": completed.returncode, "tests_run": count,
            "pass": completed.returncode == 0, "output_tail": output.strip().splitlines()[-5:]}


def _markdown(report: dict[str, Any]) -> str:
    p0, params, smokes, tests = report["p0_regression"], report["parameter_report"], report["smokes"], report["automated_tests"]
    lines = [
        "# Stage 4 Role 消融验证",
        "",
        "本报告只验证 P0 向后兼容、P1/P2 的实现合同及各 10-update smoke。未运行任何 P1/P2 1000-update 正式训练。",
        "",
        "## P0 第五轮 checkpoint 回归",
        "",
        f"- 结果：{'PASS' if p0.get('pass') else 'FAIL'}",
        f"- checkpoint：`{p0.get('checkpoint', 'missing')}`",
        f"- 固定 seeds：10000--10019；normal completion={p0.get('summary', {}).get('normal_completion_count')}/20，persistent outage={p0.get('summary', {}).get('persistent_outage_count')}，boundary={p0.get('summary', {}).get('boundary_count')}，collision={p0.get('summary', {}).get('collision_count')}，mean e2e={p0.get('summary', {}).get('mean_e2e_rate_mbps')} Mbps。",
        "",
        "## Actor 结构与参数量",
        "",
        "| Variant | 环境 obs | Actor 实际输入 | Actor 参数量 |",
        "|---|---:|---:|---:|",
    ]
    for variant, item in params.items():
        lines.append(f"| {variant} | {item['environment_observation_dim']} | {item['effective_actor_input_dim']} | {item['actor_parameter_count']} |")
    lines += [
        "",
        "P1 仅将静态 3-d one-hot 拼接进 shared backbone；P2 保留 26-d shared backbone，并仅新增两组 128→3 mean head。P1/P2 都只有一组全局共享、平滑有界的 3-d state-independent log_std。Critic 未改变，K=4 仍为 47-d global state。",
        "",
        "## Role mapping",
        "",
        f"- K=3: {report['role_mapping']['K3']}",
        f"- K=4: {report['role_mapping']['K4']}",
        f"- K=5: {report['role_mapping']['K5']}",
        "",
        "## 自动测试",
        "",
        f"- `{tests['command']}`：{'PASS' if tests['pass'] else 'FAIL'}；{tests.get('tests_run')} tests。",
        "",
        "## 10-update smoke",
        "",
    ]
    for variant, smoke in smokes.items():
        train, evaluation = smoke.get("final_train", {}), smoke.get("final_evaluation", {})
        lines += [
            f"### {variant}",
            "",
            f"- 目录：`{smoke.get('directory')}`；结果：{'PASS' if smoke.get('pass') else 'FAIL'}；updates={smoke.get('updates')}；samples={smoke.get('total_samples')}。",
            f"- final: KL={train.get('approx_kl')}，clip fraction={train.get('clip_fraction')}，actor/critic grad norm={train.get('actor_grad_norm')}/{train.get('critic_grad_norm')}，effective log_std mean={train.get('clamped_log_std_mean')}，action saturation={train.get('action_saturation_ratio')}。",
            f"- 5-seed final deterministic eval: normal={evaluation.get('normal_completion_count')}，persistent outage={evaluation.get('persistent_outage_count')}，collision={evaluation.get('collision_count')}，boundary={evaluation.get('boundary_count')}，mean e2e={evaluation.get('mean_e2e_rate_mbps')} Mbps。",
        ]
    lines += [
        "",
        "Smoke 不用于判断 Role 方法性能。Environment v1.1、26-d observation、47-d Critic、Reward 和 PPO 数学流程均未修改；Graph 也未实现。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    report = {
        "stage": "Stage 4 Role ablation implementation validation",
        "role_mapping": {"K3": list(role_ids_for_num_relays(3)), "K4": list(role_ids_for_num_relays(4)),
                         "K5": list(role_ids_for_num_relays(5))},
        "parameter_report": _parameter_report(),
        "p0_regression": _p0_regression(),
        "smokes": {variant: _read_smoke(variant, directory) for variant, directory in SMOKES.items()},
    }
    report["automated_tests"] = _run_tests()
    report["overall_pass"] = (report["p0_regression"].get("pass", False)
                              and report["automated_tests"]["pass"]
                              and all(item.get("pass", False) for item in report["smokes"].values()))
    (ROOT / "stage4_role_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "stage4_role_validation.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"overall_pass": report["overall_pass"], "json": "stage4_role_validation.json",
                      "markdown": "stage4_role_validation.md"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
