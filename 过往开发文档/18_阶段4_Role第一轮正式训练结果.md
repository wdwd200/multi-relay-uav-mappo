# Stage 4 Role 第一轮正式训练结果

## 结论范围

本报告记录两次实际完成的、受控的第一轮正式训练。它们仅改变 Actor variant：

- P1：`role_info`
- P2：`role_head`

没有修改 Environment v1.1、Reward、26 维 local observation、47 维 centralized Critic、PPO/GAE/buffer 数学、Role 定义、网络宽度、学习率、entropy、clip 或 log_std 参数化。没有实现或训练 Graph、Topology-info MLP、Role+Graph、Agent ID、chain index 或新 Role 结构。

## 固定正式配置

| 项目 | P1 Role-info | P2 Role-structured |
|---|---:|---:|
| `actor_variant` | `role_info` | `role_head` |
| K / environments | 4 / 8 | 4 / 8 |
| rollout / mini-batch / PPO epochs | 128 / 256 / 10 | 128 / 256 / 10 |
| gamma / GAE lambda / clip | 0.99 / 0.95 / 0.2 | 0.99 / 0.95 / 0.2 |
| actor / critic Adam LR | 1e-4 / 3e-4 | 1e-4 / 3e-4 |
| entropy / value coefficient / max grad norm | 0 / 0.5 / 0.5 | 0 / 0.5 / 0.5 |
| log_std mode / bounds / effective initialization | `state_independent_tanh` / [-4, 0] / -1.5 | `state_independent_tanh` / [-4, 0] / -1.5 |
| base seed | 2026 | 2026 |
| training updates | 1,000 | 1,000 |
| team-time samples | 1,024,000 | 1,024,000 |

两次实验均从各自全新随机初始化创建。P1 的首次宿主训练进程在 update 740 完整 PPO update 边界后未报告 Python/数值异常即提前退出；随后从其 `latest.pt` 以完全相同 config、optimizer、RNG、normalizer 和 per-env seed progress 恢复额外 260 updates，最终恰为 1,000 updates / 1,024,000 samples。该恢复未重复前 740 条 `train.csv` 记录。P2 从新初始化连续完成 1,000 updates。

训练结束后重新逐字段比对 P0 第五轮、P1 和 P2 的冻结训练字段（K、8 env、rollout、mini-batch、epochs、gamma、GAE、clip、Actor/Critic LR、entropy、value coefficient、gradient norm、log_std mode/bounds/init、base seed、26-d observation、47-d Critic state），全部一致；P1/P2 的 config 差异只有 `actor_variant` 与 `output_dir`。

## 检查点与周期评估

| 项目 | P1 | P2 |
|---|---:|---:|
| `train.csv` rows | 1,000 | 1,000 |
| `eval.csv` periodic evaluations | 20 | 20 |
| periodic Actor checkpoints | 20（50--1000，每 50 updates） | 20（50--1000，每 50 updates） |
| selected `best.pt` update | 550 | 600 |
| `latest.pt` update | 1,000 | 1,000 |
| `actor_final.pt` | saved | saved |

`best.pt` 继续采用既有 completion-first safety-priority ordering；其选择基于每 50 update 的 5-episode deterministic evaluation。最终 20-seed 验收并未参与重选 best checkpoint。

## 训练数值稳定性

全部 2,000 条训练记录的 `finite=1`，所有记录的 Actor loss、Critic loss、entropy、approx KL、clip fraction、gradient norm、raw/effective log_std 和 action saturation 均为 finite；没有 NaN 或 Inf。

训练结束后执行 `python -m unittest discover -s selfcheck -p "test_mappo*.py"`：**38/38 PASS**。本轮训练没有修改任何源码。

| 指标（min / mean / max，跨全部 1,000 updates） | P1 Role-info | P2 Role-structured |
|---|---:|---:|
| Actor policy loss | -0.009174 / -0.004984 / -0.001761 | -0.010468 / -0.006575 / -0.003549 |
| Critic loss | 0.610447 / 7.356183 / 142.826304 | 0.929611 / 7.354252 / 133.247490 |
| approx KL | 0.000031 / 0.005105 / 0.011617 | -0.000855 / 0.005406 / 0.012194 |
| clip fraction | 0.017749 / 0.056673 / 0.106470 | 0.021851 / 0.059109 / 0.118848 |
| Actor grad norm（pre-clip） | 0.472856 / 1.093358 / 2.320516 | 0.454794 / 0.918133 / 1.765629 |
| Critic grad norm（pre-clip） | 8.742287 / 31.312344 / 123.774209 | 12.762234 / 34.609306 / 194.510475 |
| effective log_std 全部轴的范围 | [-2.400935, -1.465093] | [-2.367430, -1.452455] |
| final effective log_std 三轴范围 | [-2.400935, -2.276062] | [-2.367430, -2.235091] |
| action saturation ratio | 0--0.00008138（final 0） | 0（final 0） |

两个 variant 的 effective log_std 都严格位于 smooth bound [-4, 0] 内部，未发生旧 state-dependent raw-log_std clamp 饱和。P2 的偶发负 approx-KL 是基于 mini-batch 差值的估计量，未伴随非有限值或更新失败。

## 最终固定 20-seed deterministic evaluation

评估 checkpoint 为各自 `best.pt`，K=4、seeds=10000--10019、20 个完整 deterministic episodes，动作为 `tanh(mean)`。

| 指标 | P1 Role-info，best update 550 | P2 Role-structured，best update 600 |
|---|---:|---:|
| normal completion | 0/20 | 0/20 |
| persistent outage | 3 | 16 |
| collision | 0 | 0 |
| boundary | 17 | 4 |
| speed hard violations | 0 | 0 |
| acceleration hard violations | 0 | 0 |
| mean e2e rate | 6.739271 Mbps | 5.988173 Mbps |
| outage step ratio | 0.068757 | 0.160407 |
| rate satisfaction ratio | 0.919142 | 0.798355 |
| minimum node distance | 40.623862 m | 23.486659 m |
| max XY speed | 20.000000 m/s | 20.000000 m/s |
| max XY acceleration | 1.300568 m/s² | 1.798244 m/s² |
| action saturation | 0 | 0 |
| mean acceleration | 0.756437 m/s² | 0.636502 m/s² |
| cumulative movement distance | 665.024137 m | 883.536900 m |

`EpisodeMetrics` stores speed and acceleration violations as one aggregate counter; it is exactly zero in both final summaries, therefore both individual hard-violation categories are zero.

## P0 / P1 / P2 实际结果对比

P0 使用已有 Stage 3 第五轮 20-seed regression 结果；P1/P2 使用本次真实 final evaluation。不同 Role 方法只作描述性并排记录，不据此提前宣称方法有效。

| 方法 | normal completion | persistent outage | collision | boundary | mean e2e | outage ratio | rate satisfaction | action saturation | 参考线 normal >=16/20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0 Plain | 0/20 | 19 | 0 | 1 | 6.103710 Mbps | 0.129161 | 0.775567 | 0 | FAIL |
| P1 Role-info | 0/20 | 3 | 0 | 17 | 6.739271 Mbps | 0.068757 | 0.919142 | 0 | FAIL |
| P2 Role-structured | 0/20 | 16 | 0 | 4 | 5.988173 Mbps | 0.160407 | 0.798355 | 0 | FAIL |

因此，两次第一轮 Role 正式训练都没有达到此前 `normal completion >= 16/20` 的任务性能参考线。该事实不等同于 Role 方法无效，也不授权本轮进行第二轮训练或调参。

## 训练产物

### P1 Role-info

- 根目录：`artifacts/stage4-role-info-full/`
- 配置和日志：`config.json`、`train.csv`、`eval.csv`
- 完整 checkpoints：`checkpoints/latest.pt`、`checkpoints/best.pt`、`checkpoints/actor_final.pt`
- 周期 Actor：`eval_checkpoints/actor_update_0050.pt` 至 `actor_update_1000.pt`
- 最终验收：`final_eval_20seeds_best.json`

### P2 Role-structured

- 根目录：`artifacts/stage4-role-head-full/`
- 配置和日志：`config.json`、`train.csv`、`eval.csv`
- 完整 checkpoints：`checkpoints/latest.pt`、`checkpoints/best.pt`、`checkpoints/actor_final.pt`
- 周期 Actor：`eval_checkpoints/actor_update_0050.pt` 至 `actor_update_1000.pt`
- 最终验收：`final_eval_20seeds_best.json`

## 工程事件与停止点

- P1：无 NaN/Inf、无 checkpoint 数据损坏；一次宿主进程提前停止已从 update 740 完整 checkpoint 恢复，并完成到 1,000 updates。
- P2：无 NaN/Inf、无 checkpoint 问题，连续完成。
- 最终 20-seed evaluation：两者 hard violations 均为 0；不存在 checkpoint reload/evaluation 错误。
- 本文档生成后停止。没有启动第二轮 P1/P2 训练，也没有启动任何 Graph 或其他结构实验。
