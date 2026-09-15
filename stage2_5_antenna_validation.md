# Stage 2.5 仰角相关全向偶极子天线验证

日期：2026-09-05  
结论：**PASS（Environment v1.1 候选通过本轮局部技术验证，等待人工冻结）**。本次未修改 5 dB、6 Mbps 或 2000 m 冻结参数，未执行 RL 训练或 Stage 3 工作。

## 实现范围

- `relay_env/communication.py`：新增 `horizontal_distance`、`elevation_angle_rad`、`antenna_gain_linear`、`antenna_gain_db` 与 `link_metrics`。
- `relay_env/environment.py`：`_communication_info()` 使用新链路增益；`info['link_diagnostics']` 输出水平距离、高度差、仰角、天线增益、SNR 和容量。Observation 仍是 26 维，仅其原有容量值采用新结果。
- `stage2_calibration.py`：在原有三模式框架中增加单跳仰角/增益/高度差统计，以及 relay 高度和上下游高度差诊断。

新链路模型：

\[
d_{xy}=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2},\quad
\phi=\operatorname{atan2}(|z_i-z_j|,d_{xy}),\quad
G^{ant}=\cos^2\phi,\quad
h_{new}=\beta_0G^{ant}/d^\alpha.
\]

没有加入 2.15 dBi 或其他绝对增益；`G^ant` 是收发合计的链路级归一化增益。严格垂直链路通过 `atan2` 与有限浮点余弦处理，计算不产生 NaN/Inf。

## 单元与回归测试

- Stage 2.5 五项天线单元测试：PASS（ran=5, failures=0, errors=0）。覆盖同高度退化、单调性、对称性、0/45 度和垂直链路稳定性。
- Stage 1 原有 13 项自检：PASS。
- K=3/4/5、固定 seed、D_max preventive guard 与终止语义：PASS（由三模式框架 hard checks 复查）。
- 数值有限性：PASS。

## 典型几何链路表

| d_xy (m) | |dz| (m) | d_3d (m) | elev (deg) | G_ant | G_ant (dB) | old SNR (dB) | new SNR (dB) | new C (Mbps) | < 5 dB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 300 | 0 | 300.000 | 0.000 | 1.000000 | 0.0000 | 9.458 | 9.458 | 32.9659 | no |
| 400 | 0 | 400.000 | 0.000 | 1.000000 | 0.0000 | 6.959 | 6.959 | 25.7641 | no |
| 500 | 0 | 500.000 | 0.000 | 1.000000 | 0.0000 | 5.021 | 5.021 | 20.6258 | no |
| 400 | 200 | 447.214 | 26.565 | 0.800000 | -0.9691 | 5.990 | 5.021 | 20.6258 | no |
| 300 | 200 | 360.555 | 33.690 | 0.692308 | -1.5970 | 7.861 | 6.264 | 23.8685 | no |
| 200 | 150 | 250.000 | 36.870 | 0.640000 | -1.9382 | 11.041 | 9.103 | 31.9123 | no |
| 200 | 200 | 282.843 | 45.000 | 0.500000 | -3.0103 | 9.969 | 6.959 | 25.7641 | no |

完整可机读表：`artifacts/stage2_5_antenna/typical_link_table.csv`。

## 20 seeds × 3 modes（每 episode 最多 500 steps）

| mode | reset | persistent outage term. | safety term. | boundary term. | truncated | outage steps | mean effective R_e2e (Mbps) | hops below 5 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal | 20/20 | 0 | 0 | 0 | 20 | 0 | 7.9384 | 0/50000 (0.00%) |
| zero_action | 20/20 | 18 | 1 | 0 | 1 | 450 | 6.6322 | 450/25060 (1.80%) |
| random_stress | 20/20 | 5 | 0 | 15 | 0 | 180 | 6.8375 | 184/18305 (1.01%) |

| mode | |dz| mean / p95 (m) | elev mean / p95 (deg) | G_ant mean / p5 | SNR p5 / p95 (dB) |
| --- | ---: | ---: | ---: | ---: |
| nominal | 9.187 / 23.598 | 2.361 / 6.504 | 0.997114 / 0.987169 | 8.955 / 15.820 |
| zero_action | 16.013 / 42.087 | 4.089 / 12.094 | 0.986743 / 0.956102 | 6.096 / 15.357 |
| random_stress | 30.320 / 93.482 | 7.118 / 22.457 | 0.968733 / 0.854086 | 6.291 / 14.595 |

详细结果：`artifacts/stage2_5_antenna/stage2_summary.json`、`stage2_episode_stats.csv`、`stage2_hop_stats.csv`、`stage2_relay_diagnostics.csv`、`stage2_relay_height_trajectories.csv`。后两者分别提供每个 relay 的高度/上下游高度差统计和逐 step 轨迹。

## 冻结参数局部复核（只统计，未调参）

- **gamma_min = 5 dB：暂保留。** 典型表和各模式的 `<5 dB` hop 计数已明确显示门限影响；outage 仍被正确掩蔽为零有效端到端速率。是否改变门限应在人工比较真实链路预算后另行决定。
- **R_ref = 6 Mbps：暂保留，需人工结合下表 mean effective R_e2e 复查。** 本轮没有自动把奖励参考值适配到天线损失后的速率。
- **D_max = 2000 m：保留。** 三模式中的最大实测 H/L 距离为 1811.164943 m，未越界；其 preventive guard 逻辑未改动。

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
