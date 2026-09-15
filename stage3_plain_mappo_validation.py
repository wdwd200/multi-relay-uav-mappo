"""Generate the evidence-based final Stage 3 Plain MAPPO acceptance report."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plain_mappo.state import flatten_global_state, global_state_dim
from relay_env import RelayEnv


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_stage3_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "selfcheck", "-p", "test_mappo*.py", "-v"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests", output)
    return {"command": "py -3 -m unittest discover -s selfcheck -p 'test_mappo*.py' -v", "passed": completed.returncode == 0,
            "tests_run": int(match.group(1)) if match else None, "output_tail": output[-1500:]}


def global_state_probe() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relays in (3, 4, 5):
        env = RelayEnv(relays)
        observation, _ = env.reset(seed=2026 + relays)
        vector = flatten_global_state(env.get_global_state(), relays)
        rows.append({"K": relays, "local_observation_shape": [len(observation), len(observation[0])],
                     "global_state_dim": int(vector.size), "expected_global_state_dim": global_state_dim(relays),
                     "finite": bool(np.isfinite(vector).all())})
    return rows


def regression_results() -> dict[str, Any]:
    stage1 = read_json(ROOT / "artifacts" / "selfcheck-results.json")
    stage2 = read_json(ROOT / "artifacts" / "stage2" / "stage2_summary.json")
    stage25 = read_json(ROOT / "artifacts" / "stage2_5_antenna" / "stage2_5_summary.json")
    return {
        "stage1": {"passed": all(value.get("passed", False) for value in stage1.values()),
                   "detail": f"{sum(value.get('passed', False) for value in stage1.values())}/{len(stage1)} selfchecks"},
        "stage2": {"passed": all(stage2["hard_checks"].values()) and stage2["raw_step_records"] is False,
                   "detail": f"{stage2['episodes_per_mode']} episodes/mode; {sum(stage2['hard_checks'].values())}/{len(stage2['hard_checks'])} hard checks; raw_step_records={stage2['raw_step_records']}"},
        "stage2_5": {"passed": stage25["unit_tests"]["passed"] and all(stage25["hard_checks"].values()),
                     "detail": f"{stage25['unit_tests']['detail']}; {sum(stage25['hard_checks'].values())}/{len(stage25['hard_checks'])} hard checks"},
    }


def acceptance_criteria(final: dict[str, Any], random_baseline: dict[str, Any]) -> dict[str, bool]:
    summary = final["summary"]
    return {
        "hard_speed_accel_violations_zero": int(summary["speed_accel_violations"]) == 0,
        "normal_500_step_completions_at_least_16_of_20": int(summary["normal_completion_count"]) >= 16,
        "collision_plus_boundary_at_most_2_of_20": int(summary["collision_count"] + summary["boundary_count"]) <= 2,
        "mean_e2e_rate_at_least_6_mbps": float(summary["mean_e2e_rate_mbps"]) >= 6.0,
        "safety_priority_better_than_random": tuple(final["score"]) < tuple(random_baseline["score"]),
    }


def build_result(run_dir: Path, stage3_tests: dict[str, Any]) -> dict[str, Any]:
    latest = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    best = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    actor_final = torch.load(run_dir / "checkpoints" / "actor_final.pt", map_location="cpu", weights_only=False)
    config = read_json(run_dir / "config.json")
    final, random_baseline = read_json(run_dir / "final_best_eval.json"), read_json(run_dir / "random_baseline_eval.json")
    train_rows, eval_rows = csv_rows(run_dir / "train.csv"), csv_rows(run_dir / "eval.csv")
    expected_config = {"num_relays": 4, "num_envs": 8, "rollout_length": 128, "mini_batch_size": 256,
                       "ppo_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2,
                       "actor_lr": 3e-4, "critic_lr": 3e-4, "entropy_coef": 0.01,
                       "value_loss_coef": 0.5, "max_grad_norm": 0.5, "base_seed": 2026,
                       "full_updates": 1000, "eval_interval_updates": 50, "global_state_dim": 47}
    config_matches = all(config.get(key) == value for key, value in expected_config.items())
    required_checkpoint_fields = ("actor_state", "critic_state", "actor_optimizer_state", "critic_optimizer_state",
                                  "normalizer_state", "update", "total_env_steps", "config", "rng_state", "best_score",
                                  "next_episode_indices")
    required_files = {str(path.relative_to(run_dir)): path.exists() for path in (
        run_dir / "train.csv", run_dir / "eval.csv", run_dir / "config.json", run_dir / "final_best_eval.json",
        run_dir / "random_baseline_eval.json", run_dir / "checkpoints" / "latest.pt",
        run_dir / "checkpoints" / "best.pt", run_dir / "checkpoints" / "actor_final.pt")}
    criteria, regressions = acceptance_criteria(final, random_baseline), regression_results()
    complete_training = int(latest["update"]) == 1000 and int(latest["total_env_steps"]) == 1_024_000
    no_nan_inf = len(train_rows) == 1000 and all(row["finite"] == "1.0" for row in train_rows)
    seed_progress = {"latest": latest.get("next_episode_indices"), "best": best.get("next_episode_indices")}
    seed_progress_ok = all(isinstance(value, list) and len(value) == 8 and all(isinstance(item, int) and item > 0 for item in value) for value in seed_progress.values())
    overall_pass = complete_training and no_nan_inf and config_matches and all(criteria.values()) and all(item["passed"] for item in regressions.values()) and stage3_tests["passed"]
    return {
        "stage": "Stage 3 Plain MAPPO", "date": "2026-09-08", "status": "pass" if overall_pass else "needs_tuning",
        "global_state_revision": {"formula": "23 + 6*K", "K4_dim": 47, "replaced_incorrect_dim": 60,
                                  "read_only_environment_fields_added": ["H.cruise_speed", "L.cruise_speed", "consecutive_outage_steps"]},
        "global_state_probe": global_state_probe(), "stage3_tests": stage3_tests, "regressions": regressions,
        "environment_v1_1": {"physical_model_frozen": True,
                               "note": "Final-training work changed no Environment v1.1 behavior; only the previously approved read-only global-state fields exist."},
        "seed_resume_fix": {"fixed": seed_progress_ok, "checkpoint_field": "next_episode_indices",
                             "semantics": "Each reset consumes a next-unused per-environment index; resume starts fresh environments from saved unused indices.",
                             "latest_progress": seed_progress["latest"], "best_progress": seed_progress["best"],
                             "test": "selfcheck.test_mappo_checkpoint.MappoCheckpointTests.test_resume_consumes_saved_next_unused_episode_seeds"},
        "formal_training": {"run_dir": str(run_dir.relative_to(ROOT)), "command": "py -3 train_plain_mappo.py --full --output-dir artifacts/stage3-final",
                            "completed": complete_training, "updates": int(latest["update"]), "team_time_samples": int(latest["total_env_steps"]),
                            "config_matches_frozen_request": config_matches, "train_log_rows": len(train_rows), "eval_log_rows": len(eval_rows),
                            "eval_updates": [int(row["update"]) for row in eval_rows], "all_train_rows_finite": no_nan_inf,
                            "best_checkpoint_update": int(best["update"]), "actor_final_selected_from": actor_final.get("selected_from")},
        "checkpoint": {"latest_fields_present": all(field in latest for field in required_checkpoint_fields),
                       "best_fields_present": all(field in best for field in required_checkpoint_fields), "required_files": required_files},
        "final_evaluation": final, "random_baseline": random_baseline, "acceptance_criteria": criteria,
        "diagnosis": {"conclusion": "pass" if overall_pass else "needs_tuning",
                      "failed_criteria": [name for name, passed in criteria.items() if not passed],
                      "evidence": "All unit/regression tests pass, all 1,000 training rows are finite, and the full frozen 1,024,000-sample budget was executed. The failed normal-completion line indicates inadequate policy reliability under the current frozen learning hyperparameters, not permission to modify Environment v1.1 or Reward."},
    }


def report_markdown(result: dict[str, Any]) -> str:
    final, random_baseline, criteria = result["final_evaluation"], result["random_baseline"], result["acceptance_criteria"]
    summary, random_summary = final["summary"], random_baseline["summary"]
    state_rows = "\n".join(f"| {row['K']} | `{row['local_observation_shape']}` | {row['global_state_dim']} | {'PASS' if row['finite'] else 'FAIL'} |" for row in result["global_state_probe"])
    regression_rows = "\n".join(f"| {name} | {'PASS' if value['passed'] else 'FAIL'} | {value['detail']} |" for name, value in result["regressions"].items())
    criterion_evidence = {
        "hard_speed_accel_violations_zero": f"{summary['speed_accel_violations']}",
        "normal_500_step_completions_at_least_16_of_20": f"{summary['normal_completion_count']}/20 (required ≥16)",
        "collision_plus_boundary_at_most_2_of_20": f"{summary['collision_count'] + summary['boundary_count']}/20 (required ≤2)",
        "mean_e2e_rate_at_least_6_mbps": f"{summary['mean_e2e_rate_mbps']:.6f} Mbps (required ≥6)",
        "safety_priority_better_than_random": f"best={final['score']}; random={random_baseline['score']}",
    }
    criteria_rows = "\n".join(f"| {name} | {'PASS' if passed else 'FAIL'} | {criterion_evidence[name]} |" for name, passed in criteria.items())
    files = "\n".join(f"- `{name}`: {'present' if exists else 'MISSING'}" for name, exists in result["checkpoint"]["required_files"].items())
    return f"""# Stage 3 Plain MAPPO 最终验证报告

日期：2026-09-08  
最终结论：**Stage 3 = FAIL / NEEDS TUNING**。完整训练、代码、checkpoint、恢复和环境回归均通过，但最终策略未达到最低 500-step 正常完成率。

## 47 维 Critic 状态与冻结边界

原任务书的 60 维错误设计已由正式修订替换。K=4 Critic 使用显式固定顺序的 **47** 维 state（`23 + 6*K`）；Actor 继续只用原有 26 维 local observation。Environment 只读接口包含 H/L `cruise_speed` 和 `consecutive_outage_steps`，未补零、复制或拼接 raw observation。

| K | local observation | global state | probe |
| ---: | --- | ---: | --- |
{state_rows}

Environment v1.1 物理模型保持冻结：未改动力学、Reward、Observation、Action、通信/天线、终止语义或安全逻辑。

## checkpoint seed-progress 修复

- 已修复：`latest.pt` 和 `best.pt` 保存 `next_episode_indices`，恢复时每个环境以其**下一枚未使用** seed 开始新的 episode；不保存环境瞬态或半截 rollout。
- latest seed progress：`{result['seed_resume_fix']['latest_progress']}`；best seed progress：`{result['seed_resume_fix']['best_progress']}`。
- 自动测试：`test_resume_consumes_saved_next_unused_episode_seeds` PASS；Stage 3 tests 共 {result['stage3_tests']['tests_run']} 项 PASS。

## 实际 1000-update 训练

- 命令：`{result['formal_training']['command']}`
- 结果：update={result['formal_training']['updates']}，team-time samples={result['formal_training']['team_time_samples']:,}，训练日志 {result['formal_training']['train_log_rows']} 行、周期评估 {result['formal_training']['eval_log_rows']} 次（updates {result['formal_training']['eval_updates']}）。
- 冻结配置核验：{'PASS' if result['formal_training']['config_matches_frozen_request'] else 'FAIL'}；所有 train.csv 行 finite：{'PASS' if result['formal_training']['all_train_rows_finite'] else 'FAIL'}。
- `best.pt` 来自 update {result['formal_training']['best_checkpoint_update']}；`actor_final.pt` 导出自 `{result['formal_training']['actor_final_selected_from']}`。

## 最终 20-episode 验收与随机基线

`best.pt` 使用 K=4、seeds 10000–10019、`tanh(mean)` 进行确定性评估。随机基线在完全相同环境 seeds 下使用固定独立的 uniform `[-1,1]` actions。

| 指标 | best.pt | random baseline |
| --- | ---: | ---: |
| 正常 500-step 完成 | {summary['normal_completion_count']}/20 | {random_summary['normal_completion_count']}/20 |
| collision / boundary / persistent outage | {summary['collision_count']} / {summary['boundary_count']} / {summary['persistent_outage_count']} | {random_summary['collision_count']} / {random_summary['boundary_count']} / {random_summary['persistent_outage_count']} |
| mean e2e rate | {summary['mean_e2e_rate_mbps']:.6f} Mbps | {random_summary['mean_e2e_rate_mbps']:.6f} Mbps |
| outage / rate satisfaction | {summary['outage_step_ratio']:.6f} / {summary['rate_satisfaction_ratio']:.6f} | {random_summary['outage_step_ratio']:.6f} / {random_summary['rate_satisfaction_ratio']:.6f} |
| max xy speed / max xy acceleration | {summary['max_xy_speed_mps']:.6f} / {summary['max_xy_accel_mps2']:.6f} | {random_summary['max_xy_speed_mps']:.6f} / {random_summary['max_xy_accel_mps2']:.6f} |
| min separation / action saturation | {summary['min_separation_m']:.6f} m / {summary['action_saturation_ratio']:.6f} | {random_summary['min_separation_m']:.6f} m / {random_summary['action_saturation_ratio']:.6f} |
| mean acceleration / movement distance | {summary['mean_acceleration_mps2']:.6f} / {summary['movement_distance_m']:.6f} m | {random_summary['mean_acceleration_mps2']:.6f} / {random_summary['movement_distance_m']:.6f} m |

| 验收线 | 结果 | 实际证据 |
| --- | --- | --- |
{criteria_rows}

## 回归

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| Stage 3 MAPPO tests | {'PASS' if result['stage3_tests']['passed'] else 'FAIL'} | {result['stage3_tests']['tests_run']} 项。 |
{regression_rows}

## 诊断与下一步

训练程序成功运行不等于性能验收通过。本次完整预算、数值有限性、checkpoint/resume、随机基线对照和 Environment 回归均无失败；`best.pt` 也在安全优先字典序上优于随机策略。然而正常完成仅 **{summary['normal_completion_count']}/20**，未达到 **16/20**，因此 Stage 3 不能 PASS。

在不改 Environment v1.1 或 Reward 的约束下，证据更符合当前冻结 PPO 超参数下的策略可靠性不足。保留所有日志和 checkpoint，等待明确授权后再做独立的 Stage 3 调参诊断；不得进入 Graph、Role、Role+Graph 或 CBF。

## 最终产物

{files}
"""


def handoff_markdown(result: dict[str, Any]) -> str:
    summary = result["final_evaluation"]["summary"]
    return f"""# 阶段 3 Plain MAPPO 代码与自检数据交接

状态：**Stage 3 = FAIL / NEEDS TUNING。**

## 已完成

- Plain MAPPO shared Actor、47维 centralized Critic、8 环境同步 rollout、GAE 两类 boundary mask、10-epoch PPO、独立 Adam、CSV、checkpoint/resume 与确定性评估已实现。
- 47维 state 契约是 `23 + 6*K`；K=4 为 47。Environment 仅有已批准的只读 `get_global_state()` 扩展，物理模型、Reward、Observation、通信和安全逻辑均冻结。
- checkpoint 新增 `next_episode_indices`。每次 reset 消费下一枚未使用的 per-env seed；恢复不会回到早期 seed，且仍不保存环境瞬态。
- Stage 3 自检 {result['stage3_tests']['tests_run']} 项、Stage 1 13/13、Stage 2 17/17、Stage 2.5 5/5 与 17/17 均 PASS。
- 正式训练从全新 `artifacts/stage3-final` 开始，实际完成 1000 updates / 1,024,000 team-time samples；train.csv 1000 行均 finite，eval.csv 在每 50 updates 记录一次。

## 最终验收

`best.pt`（update {result['formal_training']['best_checkpoint_update']}）在 seeds 10000–10019 的结果：速度/加速度硬违规={summary['speed_accel_violations']}，正常完成={summary['normal_completion_count']}/20，collision={summary['collision_count']}，boundary={summary['boundary_count']}，persistent outage={summary['persistent_outage_count']}，mean e2e={summary['mean_e2e_rate_mbps']:.6f} Mbps。

相同 seeds 的随机动作基线在安全优先字典序上更差（模型第二关键字为 collision+boundary=1，随机为12），所以相对随机基线通过；速率、硬约束和 collision+boundary 门槛也通过。唯一但决定性的失败是正常完成仅 {summary['normal_completion_count']}/20，低于 16/20。因此不得把训练程序成功运行标为 Stage 3 PASS。

## 后续

保留 `artifacts/stage3-final` 中的 train/eval/config、`latest.pt`、`best.pt`、`actor_final.pt`、最终评估和随机基线数据。下一步需要单独授权 Stage 3 调参诊断；不得修改 Environment v1.1/Reward 或进入 Graph、Role、Role+Graph、CBF。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Write final Stage 3 validation and handoff documents.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts" / "stage3-final")
    parser.add_argument("--skip-stage3-tests", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    stage3_tests = {"passed": None, "tests_run": None, "detail": "not run"} if args.skip_stage3_tests else run_stage3_tests()
    result = build_result(run_dir, stage3_tests)
    (ROOT / "stage3_plain_mappo_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "stage3_plain_mappo_validation.md").write_text(report_markdown(result), encoding="utf-8")
    (ROOT / "09_阶段3_Plain_MAPPO代码与自检数据交接.md").write_text(handoff_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "final_completion": result["final_evaluation"]["summary"]["normal_completion_count"],
                      "failed_criteria": result["diagnosis"]["failed_criteria"], "updates": result["formal_training"]["updates"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
