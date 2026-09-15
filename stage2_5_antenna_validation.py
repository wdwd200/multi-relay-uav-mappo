"""Stage 2.5 elevation-antenna validation; deliberately contains no RL training."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

from relay_env import EnvironmentConfig
from relay_env.communication import link_metrics
from stage2_calibration import CalibrationRun, MODE_NAMES, run_stage1_regression


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "stage2_5_antenna"
TYPICAL_GEOMETRIES = ((300.0, 0.0), (400.0, 0.0), (500.0, 0.0), (400.0, 200.0),
                      (300.0, 200.0), (200.0, 150.0), (200.0, 200.0))


def _metrics(d_xy: float, dz: float) -> dict[str, float]:
    c = EnvironmentConfig().comm
    return link_metrics((0.0, 0.0, 150.0), (d_xy, 0.0, 150.0 + dz), c.reference_gain,
                        c.path_loss_exponent, c.tx_power_w, c.noise_density_w_hz, c.bandwidth_hz)


def typical_rows() -> list[dict[str, Any]]:
    c = EnvironmentConfig().comm
    rows: list[dict[str, Any]] = []
    for d_xy, dz in TYPICAL_GEOMETRIES:
        value = _metrics(d_xy, dz)
        rows.append({"horizontal_distance_m": d_xy, "abs_height_difference_m": dz,
                     "distance_3d_m": value["distance_3d"], "elevation_angle_deg": value["elevation_angle_deg"],
                     "antenna_gain_linear": value["antenna_gain_linear"], "antenna_gain_db": value["antenna_gain_db"],
                     "old_snr_db": value["old_snr_db"], "new_snr_db": value["snr_db"],
                     "new_capacity_mbps": value["capacity_bps"] / 1e6,
                     "below_5db_outage": value["snr_linear"] < c.gamma_min})
    return rows


def write_typical_table(rows: list[dict[str, Any]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    with (OUTPUT / "typical_link_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def merged_hop_summary(run: CalibrationRun, mode: str, metric: str) -> dict[str, float | int | None]:
    stats = [value for (entry_mode, _hop, entry_metric), value in run.hop_metric.items()
             if entry_mode == mode and entry_metric == metric]
    if not stats:
        return {"count": 0, "min": None, "mean": None, "p5": None, "p95": None, "max": None}
    values = sorted(value for stat in stats for value in stat.values)
    count = sum(stat.count for stat in stats)
    total = sum(stat.total for stat in stats)
    return {"count": count, "min": min(stat.minimum for stat in stats), "mean": total / count,
            "p5": values[round((len(values) - 1) * .05)], "p95": values[round((len(values) - 1) * .95)],
            "max": max(stat.maximum for stat in stats)}


def metric_mean(run: CalibrationRun, mode: str, name: str) -> float:
    return float(run.metric[mode][name].summary()["mean"])


def run_unit_tests() -> tuple[bool, str]:
    suite = unittest.defaultTestLoader.loadTestsFromName("selfcheck.test_antenna")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return result.wasSuccessful(), f"ran={result.testsRun}, failures={len(result.failures)}, errors={len(result.errors)}"


def build_report(stage1_ok: bool, unit_ok: bool, unit_text: str, run: CalibrationRun, typical: list[dict[str, Any]]) -> str:
    c = run.config
    rows = "\n".join(
        "| {horizontal_distance_m:.0f} | {abs_height_difference_m:.0f} | {distance_3d_m:.3f} | {elevation_angle_deg:.3f} | "
        "{antenna_gain_linear:.6f} | {antenna_gain_db:.4f} | {old_snr_db:.3f} | {new_snr_db:.3f} | {new_capacity_mbps:.4f} | {outage} |".format(
            **row, outage="yes" if row["below_5db_outage"] else "no") for row in typical)
    mode_rows: list[str] = []
    distribution_rows: list[str] = []
    for mode in MODE_NAMES:
        counts = run.counts[mode]
        samples = counts["hop_samples"]
        below = counts["hops_below_5db"]
        e2e = metric_mean(run, mode, "effective_e2e_mbps")
        mode_rows.append(f"| {mode} | {counts['reset_successes']}/{run.episodes} | {counts['terminated_persistent_outage']} | {counts['terminated_safety_violation']} | {counts['terminated_boundary_violation']} | {counts['truncated']} | {counts['outage_steps']} | {e2e:.4f} | {below}/{samples} ({below / samples:.2%}) |")
        elevation = merged_hop_summary(run, mode, "elevation_angle_deg")
        gain = merged_hop_summary(run, mode, "antenna_gain_linear")
        dz = merged_hop_summary(run, mode, "height_difference_abs_m")
        snr = merged_hop_summary(run, mode, "snr_db")
        distribution_rows.append(f"| {mode} | {dz['mean']:.3f} / {dz['p95']:.3f} | {elevation['mean']:.3f} / {elevation['p95']:.3f} | {gain['mean']:.6f} / {gain['p5']:.6f} | {snr['p5']:.3f} / {snr['p95']:.3f} |")
    max_hl = max(float(run.metric[mode]["hl_distance_m"].maximum) for mode in MODE_NAMES)
    nominal_e2e = metric_mean(run, "nominal", "effective_e2e_mbps")
    all_finite = not run.non_finite_found
    pass_all = stage1_ok and unit_ok and all(run.hard_checks().values()) and all_finite
    return fr"""# Stage 2.5 仰角相关全向偶极子天线验证

日期：2026-09-05  
结论：**{'PASS（Environment v1.1 候选通过本轮局部技术验证，等待人工冻结）' if pass_all else 'FAIL（见下方失败项）'}**。本次未修改 5 dB、6 Mbps 或 2000 m 冻结参数，未执行 RL 训练或 Stage 3 工作。

## 实现范围

- `relay_env/communication.py`：新增 `horizontal_distance`、`elevation_angle_rad`、`antenna_gain_linear`、`antenna_gain_db` 与 `link_metrics`。
- `relay_env/environment.py`：`_communication_info()` 使用新链路增益；`info['link_diagnostics']` 输出水平距离、高度差、仰角、天线增益、SNR 和容量。Observation 仍是 26 维，仅其原有容量值采用新结果。
- `stage2_calibration.py`：在原有三模式框架中增加单跳仰角/增益/高度差统计，以及 relay 高度和上下游高度差诊断。

新链路模型：

\[
d_{{xy}}=\sqrt{{(x_i-x_j)^2+(y_i-y_j)^2}},\quad
\phi=\operatorname{{atan2}}(|z_i-z_j|,d_{{xy}}),\quad
G^{{ant}}=\cos^2\phi,\quad
h_{{new}}=\beta_0G^{{ant}}/d^\alpha.
\]

没有加入 2.15 dBi 或其他绝对增益；`G^ant` 是收发合计的链路级归一化增益。严格垂直链路通过 `atan2` 与有限浮点余弦处理，计算不产生 NaN/Inf。

## 单元与回归测试

- Stage 2.5 五项天线单元测试：{'PASS' if unit_ok else 'FAIL'}（{unit_text}）。覆盖同高度退化、单调性、对称性、0/45 度和垂直链路稳定性。
- Stage 1 原有 13 项自检：{'PASS' if stage1_ok else 'FAIL'}。
- K=3/4/5、固定 seed、D_max preventive guard 与终止语义：{'PASS' if all(run.hard_checks().values()) else 'FAIL'}（由三模式框架 hard checks 复查）。
- 数值有限性：{'PASS' if all_finite else 'FAIL'}。

## 典型几何链路表

| d_xy (m) | |dz| (m) | d_3d (m) | elev (deg) | G_ant | G_ant (dB) | old SNR (dB) | new SNR (dB) | new C (Mbps) | < 5 dB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
{rows}

完整可机读表：`artifacts/stage2_5_antenna/typical_link_table.csv`。

## 20 seeds × 3 modes（每 episode 最多 500 steps）

| mode | reset | persistent outage term. | safety term. | boundary term. | truncated | outage steps | mean effective R_e2e (Mbps) | hops below 5 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(mode_rows)}

| mode | |dz| mean / p95 (m) | elev mean / p95 (deg) | G_ant mean / p5 | SNR p5 / p95 (dB) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(distribution_rows)}

详细结果：`artifacts/stage2_5_antenna/stage2_summary.json`、`stage2_episode_stats.csv`、`stage2_hop_stats.csv`、`stage2_relay_diagnostics.csv`、`stage2_relay_height_trajectories.csv`。后两者分别提供每个 relay 的高度/上下游高度差统计和逐 step 轨迹。

## 冻结参数局部复核（只统计，未调参）

- **gamma_min = 5 dB：暂保留。** 典型表和各模式的 `<5 dB` hop 计数已明确显示门限影响；outage 仍被正确掩蔽为零有效端到端速率。是否改变门限应在人工比较真实链路预算后另行决定。
- **R_ref = 6 Mbps：暂保留，需人工结合下表 mean effective R_e2e 复查。** 本轮没有自动把奖励参考值适配到天线损失后的速率。
- **D_max = 2000 m：保留。** 三模式中的最大实测 H/L 距离为 {max_hl:.6f} m，未越界；其 preventive guard 逻辑未改动。

## 第三维诊断与下一步建议

本轮记录了所有通信跳的 |dz|、仰角和天线增益，以及各 relay 的高度、上游/下游高度差分布。三种执行器都不是训练策略，因此这些数据只能排查数值或环境机制的异常压缩，不能代表未来已训练策略的高度偏好。当前没有新增任何“同高度”奖励或强制逻辑；如果人工审阅分布后认为高度明显塌缩，应在后续独立实验中诊断，而非在本阶段改 Reward。

建议：在人工审核本报告和 CSV 后，若接受 5 dB/6 Mbps/2000 m 的统计表现，再决定是否开展任务书所说的 500×3 大规模重新验收。当前不自动启动该运行。

## 修改文件清单

- `relay_env/communication.py`
- `relay_env/environment.py`
- `stage2_calibration.py`
- `selfcheck/test_antenna.py`
- `stage2_5_antenna_validation.py`
- `stage2_5_antenna_validation.md`（本报告）
"""


def main() -> None:
    unit_ok, unit_text = run_unit_tests()
    stage1_ok, stage1_output = run_stage1_regression(ROOT)
    run = CalibrationRun(ROOT, OUTPUT, episodes=20, steps=500, write_raw=False)
    run.run()
    typical = typical_rows()
    write_typical_table(typical)
    summary = {"unit_tests": {"passed": unit_ok, "detail": unit_text}, "stage1_regression": {"passed": stage1_ok, "output_tail": stage1_output}, "hard_checks": run.hard_checks(), "typical_links": typical}
    (OUTPUT / "stage2_5_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(stage1_ok, unit_ok, unit_text, run, typical)
    (ROOT / "stage2_5_antenna_validation.md").write_text(report, encoding="utf-8")
    print(json.dumps({"validation": "PASS" if "**PASS" in report else "FAIL", "output_dir": str(OUTPUT), "unit_tests": unit_ok, "stage1_regression": stage1_ok, "hard_checks": run.hard_checks()}, ensure_ascii=False))
    if not (unit_ok and stage1_ok and all(run.hard_checks().values())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
