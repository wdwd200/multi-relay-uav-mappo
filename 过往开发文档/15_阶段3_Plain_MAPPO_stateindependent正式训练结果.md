# Stage 3 Plain MAPPO：state-independent `log_std` 第五轮正式训练结果

日期：2026-09-09  
结论：**Stage 3 = FAIL / NEEDS STRUCTURAL DIAGNOSIS**。

本轮真实完成了从随机初始化开始的 1,000-update 正式训练和固定 20-seed 验收。新的动作分布消除了旧 `state-dependent` 方差 head 的越界/硬裁剪问题，并显著降低了第四轮后期的 PPO KL 与 clip fraction；但没有产生任何正常完成 episode，因而不满足 Stage 3 的策略性能验收线。未启动第六轮训练。

## 1. 本轮受控变量与冻结边界

相对第四轮 D，唯一核心结构变化为：

```text
旧：obs -> shared MLP -> mean_head(3) + state-dependent log_std_head(3) -> hard clamp
新：obs -> shared MLP -> mean_head(3)
    theta = nn.Parameter([theta_x, theta_y, theta_z])
    effective_log_std = -2 + 2 * tanh(theta)   # 严格位于 (-4, 0)
```

`theta` 对所有 relay 和所有 observation 共用；x/y/z 三维仍可各自学习。训练 sampling、缓存同一旧 `latent_z` 的 Gaussian log-prob、PPO ratio、GAE、buffer、Critic 和部署时的 `tanh(mean)` 均未改变。

| 项目 | 第五轮实际值 |
| --- | --- |
| Actor distribution | `state_independent_tanh` |
| effective `log_std` bounds / init | `(-4, 0)` / `-1.5` |
| entropy coefficient / Actor LR | `0.0` / `1e-4` |
| Critic LR | `3e-4` |
| K / environments / rollout | `4 / 8 / 128` |
| mini-batch / PPO epochs | `256 team-time / 10` |
| gamma / GAE lambda / clip | `0.99 / 0.95 / 0.2` |
| local observation / critic state | `26 / 47` |
| base seed | `2026` |

Environment v1.1 保持冻结：未修改动力学、Reward、Observation、通信/天线、outage 与 persistent-outage 规则、安全、terminated/truncated、Actor/Critic 隐藏层、PPO/GAE/rollout 组织，也没有加入 Graph、Role、Agent ID 或 CBF。

## 2. 实际正式训练与产物

实际命令：

```powershell
py -3 train_plain_mappo.py --full --output-dir artifacts\stage3-stateindependent-full `
  --entropy-coef 0.0 --actor-lr 1e-4 `
  --actor-log-std-mode state_independent_tanh `
  --log-std-min -4.0 --log-std-max 0.0 `
  --state-independent-log-std-init -1.5
```

- 已完成：**1,000 updates**，**1,024,000 team-time samples** (`1000 × 8 × 128`)。
- `train.csv` 有 1,000 行，`finite=1.0` 为 1,000/1,000；未发生 NaN/Inf。
- `eval.csv` 有 20 条固定 5-episode deterministic 记录，对应 updates 50 至 1000、每 50 updates 一次。
- completion-first best 规则实际生效。`best.pt` 为 **update 650**：当时 5-episode eval 为 hard violations=0、normal=0、collision+boundary=0、persistent outage=5；与 update 400 同为前四个关键字，但 update 650 的 outage ratio 更低（0.126824 vs. 0.127984），因此按字典序胜出。
- 已保存 `latest.pt`、`best.pt`、`actor_final.pt` 和 20 个轻量 checkpoint (`actor_update_0050.pt` 至 `actor_update_1000.pt`)；未生成逐 step raw JSONL。

产物目录：`artifacts/stage3-stateindependent-full/`。

## 3. 新方差参数与 PPO 数值行为

`best.pt`（update 650）的三维共享参数和有效方差为：

| 动作维度 | theta | effective log_std | std |
| --- | ---: | ---: | ---: |
| x | 0.025193 | -1.949625 | 0.142327 |
| y | -0.011032 | -2.022062 | 0.132382 |
| z | 0.034950 | -1.930128 | 0.145130 |

`latest.pt`（update 1000）仍在 bounds 中部，而非靠近任一边界：

| 动作维度 | theta | effective log_std | std |
| --- | ---: | ---: | ---: |
| x | -0.084672 | -2.168941 | 0.114299 |
| y | -0.164198 | -2.325476 | 0.097737 |
| z | -0.127638 | -2.253898 | 0.104989 |

这不是旧实现的 raw `log_std` head；这里的 `theta` 是平滑映射的内部坐标，不能把它与旧 `raw_log_std > 2` 指标混为一谈。有效 `log_std` 在本轮中从初始化约 -1.50 平滑降至训练末约 -2.25，始终远离 `(-4, 0)` 的边界；action saturation 始终为 0。

第五轮按阶段的实际训练/周期评估均值如下。eval 每项均为该阶段 5-episode eval 的均值，而非 20-seed 最终验收。

| updates | critic loss | entropy | approx KL | clip fraction | effective log_std | rollout saturation | eval normal /5 | eval persistent outage /5 | eval e2e Mbps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–200 | 20.094 | -0.405 | 0.00384 | 0.03731 | -1.554 | 0.000 | 0.00 | 1.75 | 6.937 |
| 201–500 | 6.532 | -0.870 | 0.00424 | 0.04358 | -1.709 | 0.000 | 0.00 | 3.00 | 6.707 |
| 501–800 | 5.076 | -1.627 | 0.00515 | 0.05724 | -1.962 | 0.000 | 0.00 | 3.83 | 6.493 |
| 801–1000 | 5.472 | -2.229 | 0.00603 | 0.06677 | -2.162 | 0.000 | 0.00 | 3.50 | 6.471 |

结论：state-independent 参数化确实修复了 A/C 的全局方差失控；与 D 的后 200 updates KL=0.03589、clip=0.17445 相比，第五轮分别为 0.00603、0.06677，后期 PPO 漂移显著缓和。但周期评估从 update 50 到 1000 的 normal completion 始终为 0，说明数值稳定性改善没有转化为可靠队形保持。

## 4. 固定 20-seed 最终验收

使用 `best.pt`（update 650），K=4，seeds 10000–10019，以 deterministic `tanh(mean)` 运行。结果保存于：

- `artifacts/stage3-stateindependent-full/final_evaluation_20seeds.json`
- `artifacts/stage3-stateindependent-full/final_checkpoint_outage_diagnosis.json`
- `artifacts/stage3-stateindependent-full/random_baseline_eval.json`

| 指标 | 第五轮 best.pt | 同 seeds uniform-random baseline |
| --- | ---: | ---: |
| normal 500-step completion | **0/20** | 0/20 |
| persistent outage | 19 | 8 |
| collision / boundary | 0 / 1 | 0 / 12 |
| speed/acceleration hard violations | 0 | 0 |
| mean e2e rate | **6.103710 Mbps** | 6.596517 Mbps |
| outage step ratio | 0.129161 | 0.073432 |
| rate satisfaction ratio | 0.775567 | 0.900185 |
| minimum node distance | 29.801 m | 77.086 m |
| max xy speed / max abs z speed | 20.000 / 3.506 m/s | 9.343 / 6.000 m/s |
| max xy acceleration / max abs z acceleration | 1.458 / 0.878 m/s² | 2.000 / 2.000 m/s² |
| mean acceleration | 0.567453 m/s² | 1.865689 m/s² |
| action saturation | 0.000000 | 0.049058 |
| cumulative movement distance | 1489.973 m | 390.836 m |

completion-first safety score：best=`[0, 0, 1, 19, 0.129161, -0.775567, -6.103710]`，random=`[0, 0, 12, 8, 0.073432, -0.900185, -6.596517]`。因此模型在该安全优先字典序上优于 random（1 次 collision+boundary 少于 12 次），但这不能抵消 normal completion 的失败。

| Stage 3 验收线 | 结果 | 状态 |
| --- | ---: | --- |
| hard speed/acceleration violations = 0 | 0 | PASS |
| normal completion >= 16/20 | 0/20 | **FAIL** |
| collision + boundary <= 2/20 | 1/20 | PASS |
| mean e2e >= 6 Mbps | 6.103710 | PASS |
| 安全优先排序优于 random | 是 | PASS |

## 5. persistent outage 的实际诊断

20-seed deterministic replay 中 19 个 persistent-outage 终止事件的 onset 平均在 step 166.3（范围 104–248），即主要发生在 episode 的早中期。onset link 覆盖整条链，而非只集中端部：

| onset link | 事件数 |
| --- | ---: |
| H→R1 | 5 |
| R1→R2 | 6 |
| R2→R3 | 1 |
| R3→R4 | 1 |
| R4→L | 6 |

onset 时该动作分布的绝对动作均值为 0.1705、最大值为 0.7032，onset action saturation 为 0；所有 replay action saturation 亦为 0。outage 期间的平均水平速度为 15.94 m/s，部分 relay 已接近 20 m/s 上限，但并非由 `tanh` 动作边界饱和触发。结合整条链的 onset 扩散和 0/20 normal completion，这支持“均值策略尚未学会可靠的中继队形保持/恢复”，而不是“仅剩方差失控”的判断。

## 6. 五轮真实对照

所有最终值均为各轮自己的 `best.pt`、相同 seeds 10000–10019 的 deterministic 20-episode 结果。

| 轮次 | Actor distribution | entropy / Actor LR | best update | normal | persistent outage | collision / boundary | mean e2e Mbps | outage ratio | rate satisfaction | final action saturation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | state-dependent + clamp | 0.01 / 3e-4 | 900 | 6/20 | 13 | 1 / 0 | 6.694106 | 0.074369 | 0.899716 | 0.203004 |
| B | state-dependent + clamp | 0 / 3e-4 | 200 | 0/20 | 20 | 0 / 0 | 5.908689 | 0.169448 | 0.779244 | 0.000000 |
| C | state-dependent + clamp | 0.001 / 3e-4 | 700 | 0/20 | 19 | 1 / 0 | 6.134737 | 0.133537 | 0.794624 | 0.001056 |
| D | state-dependent + clamp | 0 / 1e-4 | 150 | 0/20 | 19 | 0 / 1 | 6.086493 | 0.130640 | 0.774669 | 0.000000 |
| E | **state-independent + smooth bound** | **0 / 1e-4** | **650** | **0/20** | **19** | **0 / 1** | **6.103710** | **0.129161** | **0.775567** | **0.000000** |

下表的 entropy/KL/clip/saturation 为各轮最后 200 updates 均值；log_std 是 best checkpoint 的 deterministic-evaluation states 上的有效均值。A/C 的 `raw > 2` 是旧 raw head 超出 hard-clamp 上界的比例。

| 轮次 | tail entropy | tail KL | tail clip fraction | tail rollout saturation | best effective log_std | best raw > 2 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 10.2540 | 0.00319 | 0.02682 | 0.82344 | 1.99993 | 99.988% | 全局 clamp 饱和 |
| B | 4.2002 | 0.01348 | 0.08759 | 0.29666 | -1.29159 | 0.000% | 无全局饱和但失败 |
| C | 10.0037 | 0.00910 | 0.02609 | 0.78748 | 1.97920 | 96.484% | 再次 clamp 饱和 |
| D | -2.7846 | 0.03589 | 0.17445 | 0.02930 | -0.62545 | 0.000% | 后期 KL/clip 漂移明显 |
| E | -2.2293 | 0.00603 | 0.06677 | 0.00000 | -1.96727 | 不适用（theta） | 方差稳定、PPO 漂移改善，但性能未改善 |

第五轮明确回答了本轮研究问题：**state-independent smooth-bounded `log_std` 解决了旧 Actor 的方差失控和由此导致的 PPO 数值不稳，但未让 Plain MAPPO 获得稳定可靠的通信保持策略。** E 与 D 的最终任务指标几乎相同（均 normal=0/20、persistent outage=19），故不能把后续失败再主要归因于动作方差头。

## 7. 测试、回归与停止条件

本轮训练后实际重跑：

| 验证 | 真实结果 |
| --- | --- |
| Stage 3 MAPPO tests | 28/28 PASS |
| Stage 1 selfcheck | 13/13 PASS |
| Stage 2 hard checks（默认不写 raw JSONL） | 17/17 PASS |
| Stage 2.5 antenna tests | 5/5 PASS |
| Stage 2.5 hard checks | 17/17 PASS |

其中 Stage 3 suite 继续覆盖 state-independent shared parameter、同一旧 latent 的 log-prob 重算、有限梯度、checkpoint round-trip/resume 与 seed-progress；Environment v1.1 未因训练或测试发生修改。

本轮未启动第六次 1,000-update 训练，也不会机械再调 learning rate、entropy、clip、critic LR、网络宽度或 Reward。下一步应先进行 **Plain MAPPO 局部可观测性 / shared Actor 结构能力诊断**：只读检查 26 维 local observation、没有 Role/Agent ID 的共享 Actor，是否能够稳定区分 H–R1、三个中间 hop 与 R4–L 的不同队形责任；在该诊断完成前，不加入 Role 或 Graph。
