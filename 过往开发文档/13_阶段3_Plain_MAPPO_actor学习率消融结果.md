# Stage 3 Plain MAPPO：第四轮 actor 学习率受控消融结果

执行日期：2026-09-08  
结论：**Stage 3 = FAIL / NEEDS REDESIGN**

## 实验范围与冻结项

第四轮从全新随机初始化开始，训练目录为 `artifacts/stage3-entropy0-actorlr1e4/`，没有从 A、B、C 任一 checkpoint 恢复。唯一训练动力学变化为：以 B（`entropy_coef=0.0`、`actor_lr=3e-4`）为基线，将 `actor_lr` 降为 `1e-4`。`entropy_coef` 仍为 `0.0`。

其余配置与 B 相同：K=4、8 个同步环境、rollout=128、mini-batch=256 team-time、10 PPO epochs、`critic_lr=3e-4`、`clip_epsilon=0.2`、`max_grad_norm=0.5`、`log_std=[-5,2]`、47 维 centralized critic state、冻结的 Actor/Critic 网络。

本轮没有修改 Environment v1.1 的动力学、物理参数、Observation（仍为 26 维）、Reward、Action、通信/天线、安全或终止逻辑；也没有加入 Graph、Role 或 CBF。没有生成逐 step raw JSONL。

## 实际执行产物

- 真实完成：1,000 / 1,000 PPO updates，`1000 × 8 × 128 = 1,024,000` team-time samples。
- 每 50 updates 进行了固定 5-episode deterministic evaluation，共 20 次。
- 已保存 `train.csv`、`eval.csv`、`config.json`、`checkpoints/latest.pt`、`checkpoints/best.pt`、`checkpoints/actor_final.pt`，以及 20 个轻量 `eval_checkpoints/actor_update_*.pt`。
- 最终 `best.pt` 的固定 20-seed（10000–10019）评估和同种子的随机策略对照均已实际运行。
- 断链复放诊断已输出 `final_checkpoint_outage_diagnosis.json`；其记录了 onset link、step、水平距离、高度差、SNR/margin、R1/R4 动作与速度。

关键路径：

- `artifacts/stage3-entropy0-actorlr1e4/train.csv`
- `artifacts/stage3-entropy0-actorlr1e4/eval.csv`
- `artifacts/stage3-entropy0-actorlr1e4/final_best_eval.json`
- `artifacts/stage3-entropy0-actorlr1e4/random_baseline_eval.json`
- `artifacts/stage3-entropy0-actorlr1e4/final_checkpoint_outage_diagnosis.json`

## Completion-first best 规则已生效

本轮 checkpoint 排序使用已修正的字典序：

```text
(speed_accel_violations,
 -normal_completion_count,
 collision + boundary,
 persistent_outage_count,
 outage_step_ratio,
 -rate_satisfaction_ratio,
 -mean_e2e_rate_mbps)
```

第四轮实际选择 `best.pt = update 150`。在所有 5-episode 周期评估中，正常完成数都为 0，因此第二关键字没有候选可区分。update 150 的 key 为 `(0, 0, 0, 5, 0.12324, -0.80855, -6.25477)`；update 200 虽将 persistent outage 减为 4，但有 1 次 collision，第三关键字更差；后续低 outage 候选也伴随 boundary/collision。因此选择符合规则，并非旧的安全排序残留。

`latest.pt` 是 update 1000；其最后一次周期评估为 0 normal、1 boundary、4 persistent outage、6.372 Mbps，亦因 `collision + boundary=1` 而不应取代 best。该 5-episode 值不是最终 20-seed 验收。

重新执行的 MAPPO 自检为 **24/24 PASS**，其中包括 `test_completion_ranks_before_nonhard_termination_counts`、47 维 critic-state contract 和 checkpoint resume seed-progress 测试。

## actor_lr 对 PPO 优化状态的影响

下表中的梯度范数均为裁剪前值；“>0.5”表示该 update 的 Actor 梯度触发了 `max_grad_norm=0.5` 裁剪。不能仅由梯度范数判断学习率效果，KL/clip fraction 才直接反映实际 policy drift。

| Updates | B: KL / clip / Actor grad / >0.5 | D: KL / clip / Actor grad / >0.5 |
| --- | --- | --- |
| 1–200 | 0.00838 / 0.08189 / 1.188 / 56.0% | 0.00316 / 0.03021 / 0.269 / 0.0% |
| 201–500 | 0.01842 / 0.13474 / 8.104 / 100.0% | 0.00473 / 0.04447 / 3.396 / 88.7% |
| 501–800 | 0.01905 / 0.12362 / 8.965 / 100.0% | 0.02042 / 0.11520 / 25.808 / 100.0% |
| 801–1000 | 0.01348 / 0.08759 / 3.194 / 100.0% | 0.03589 / 0.17445 / 45.302 / 100.0% |

第四轮在前 500 updates 的确显著降低了 KL 和 clip fraction；但 501–800 已回升至 B 的量级，801–1000 更恶化为 KL 0.03589、clip fraction 0.17445。故 `actor_lr=1e-4` 只是延后了策略漂移，不能稳定整个训练过程。

Critic 没有对应的失控证据。B/D 的分段 critic loss 分别为 `23.733/6.541/5.568/9.158` 与 `27.053/7.270/4.817/4.472`；D 后半段反而较低。没有 NaN/Inf。

第四轮分段 entropy 为 `3.351 → 1.326 → -1.593 → -2.785`，rollout action saturation 为 `4.06% → 2.26% → 2.92% → 2.93%`。这避免了 A/C 的高熵、高饱和模式，但也未转化为任务完成。

## log_std 检查

D 的 `best.pt` 在 20-seed deterministic states 上：raw log_std mean = -0.6254、max = 0.0026、`raw > 2` = 0；clamped mean 与 raw 相同。update-1000 latest 的 raw mean = -2.2557、max = 3.5274、`raw > 2` = 0.998%，clamped mean = -2.2259。故不存在 A/C 那样的全局 clamp 饱和。

但 D 后期 rollout 的 raw `log_std > 2` 比例仍从前 200 updates 的 0 上升到最后 200 updates 的 0.656%，且熵持续下降。它说明 state-dependent 方差头仍会产生少量越界状态，同时大部分状态又趋向过小方差；这与随后高 KL/clip 并存，并不构成健康收敛。

## 周期评估轨迹

20 次 5-episode evaluation 中，正常完成数始终为 0。训练中较低 outage 的片段（例如 update 450：2 persistent outage、3 boundary；update 500：1 persistent outage、4 boundary）是以边界失败为代价得到的。后期 800–1000 的 persistent outage 为 3–4/5，boundary 为 1–2/5，mean E2E rate 约为 6.37–6.40 Mbps。

因此 D 没有像 A 的 850–900 区间那样出现完成率峰值；也不能将其理解为“继续训练便会自然通过”。

## 第四轮固定 20-seed 最终验收

评估对象是 `artifacts/stage3-entropy0-actorlr1e4/checkpoints/best.pt`（update 150）。

| 验收项 | 实测 | 结果 |
| --- | ---: | --- |
| 速度/加速度硬违规 | 0 | PASS |
| 正常完成 | 0 / 20（要求 >=16） | **FAIL** |
| collision + boundary | 0 + 1 = 1（要求 <=2） | PASS |
| persistent outage | 19 / 20 | 主要失败模式 |
| outage step ratio | 0.13064 | — |
| rate satisfaction ratio | 0.77467 | — |
| mean E2E rate | 6.08649 Mbps（要求 >=6） | PASS |
| 最大水平/垂直速度 | 20.000 / 4.392 m/s | 无硬违规 |
| 最大水平/垂直加速度 | 1.629 / 1.226 m/s² | 无硬违规 |
| 最小节点间距 | 38.781 m | — |
| deterministic action saturation | 0 | — |
| 平均加速度 / 移动距离 | 0.484 m/s² / 1131.43 m | — |

同一 20 个 seeds 的独立 uniform-random baseline 为：0 normal、12 boundary、8 persistent outage、0 hard violations、outage ratio 0.07343、rate satisfaction 0.90019、6.59652 Mbps。按当前 completion-first 字典序，D 因 `collision + boundary=1 < 12` 而优于该随机基线；但这不足以抵消 0/20 completion 和 19/20 persistent outage，Stage 3 整体仍不合格。

## 四轮并排最终对照

entropy 是最后 200 updates 的 rollout Gaussian entropy；raw log_std 是各轮 best actor 在相同最终 20-seed deterministic states 上的诊断值。

| 轮次 | entropy / actor_lr | best update | normal | persistent outage | collision / boundary | mean E2E Mbps | outage ratio | rate satisfaction | eval action sat. | last-200 entropy | raw log_std mean / max / >2 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 0.01 / 3e-4 | 900 | 6/20 | 13/20 | 1 / 0 | 6.694 | 0.07437 | 0.89972 | 20.30% | 10.254 | 4.986 / 7.379 / 99.988% |
| B | 0 / 3e-4 | 200 | 0/20 | 20/20 | 0 / 0 | 5.909 | 0.16945 | 0.77924 | 0 | 4.200 | -1.292 / 1.790 / 0 |
| C | 0.001 / 3e-4 | 700 | 0/20 | 19/20 | 1 / 0 | 6.135 | 0.13354 | 0.79462 | 0.11% | 10.004 | 4.617 / 8.268 / 96.484% |
| D | 0 / 1e-4 | 150 | 0/20 | 19/20 | 0 / 1 | 6.086 | 0.13064 | 0.77467 | 0 | -2.785 | -0.625 / 0.003 / 0 |

结论：D 解决了 A/C 的大规模 log_std saturation，并在前半程显著降低 B 的 KL/clip；但并未降低最终 persistent outage 到可接受范围，也没有产生任何 normal completion。与 B 相比只少了 1 个 persistent outage，且新增 1 个 boundary；与 A 相比明显退步。

## 端部链路与 outage onset 诊断

D best 的 19 个 persistent-outage terminal events 平均从 step 162.2 开始（范围 128–271），即前期至中期；没有晚期 onset。onset 时链路都接近可用门槛（SNR 约 3.1、margin 略小于 0 dB），但策略未能维持或恢复队形。

| onset 链路 | onset labels | 平均 onset step | 平均水平距离 m | 平均高度差 m | 平均 SNR / margin dB |
| --- | ---: | ---: | ---: | ---: | --- |
| H→R1 | 1 | 137.0 | 501.69 | 5.11 | 3.155 / -0.010 |
| R1→R2 | 6 | — | 502.42 | 5.72 | 3.146 / -0.023 |
| R2→R3 | 4 | — | 502.20 | 18.81 | 3.140 / -0.031 |
| R3→R4 | 4 | — | 503.05 | 15.16 | 3.133 / -0.040 |
| R4→L | 5 | 142.6 (132–160) | 501.30 | 25.49 | 3.136 / -0.037 |

其中 20 个 onset-link labels 中端部 H→R1/R4→L 共 6 个（30%），中段三条链路共 14 个。与 C 的 18/19 端部集中不同，D 的失败已扩散为整条链的队形保持问题，而不是只由端点跟踪造成。

- H→R1 唯一 onset 时，R1 的 deterministic action 为 `(-0.399, -0.260, 0.020)`，水平速度 20.0 m/s、垂直速度 2.77 m/s。
- R4→L 的 5 个 onset 中，R4 平均 action 为 `(-0.052, -0.070, 0.015)`，平均水平/垂直速度为 10.88 / 1.24 m/s（水平速度范围 2.06–20.0 m/s）。
- onset action saturation 均为 0；所以这些断链不是“动作顶在 tanh 边界但速度仍不够”的单一现象。更直接的迹象是 near-threshold 几何下，确定性动作没有形成可靠、持续的链路恢复控制。

## 停止结论与后续边界

本轮不启动第五轮训练，也不继续机械尝试 `critic_lr`、`clip_epsilon`、PPO epochs、网络宽度、Reward 或 Environment。`actor_lr=1e-4` 的受控单变量假设没有得到性能支持：前期漂移改善，后期仍发生更严重的 KL/clip 上升，最终完成率仍为 0。

下一步应先进行动作分布设计审查，而不是开始另一轮 1000-update sweep，重点是：

1. state-dependent `log_std` head 是否导致不同状态间的方差极端漂移；
2. hard clamp `[-5, 2]` 的零梯度边界与少量越界状态；
3. 对 latent Gaussian entropy 的奖励是否与 tanh-squashed action 的实际分布匹配；
4. stochastic rollout action 与 deterministic `tanh(mean)` evaluation 之间的行为差异；
5. endpoint concentration虽未在 D 重现，但连接维持失败仍高度结构化，需评估无 Role/无显式结构信息的 shared Plain MAPPO 能力边界。

上述是下一阶段前的诊断议题，不是本轮授权的实现变更。Environment v1.1、Reward、Observation、网络和通信模型继续保持冻结。
