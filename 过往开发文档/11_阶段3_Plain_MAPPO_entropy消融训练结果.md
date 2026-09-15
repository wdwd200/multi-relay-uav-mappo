# Stage 3 Plain MAPPO：entropy=0 单变量消融训练结果

日期：2026-09-08  
最终结论：**第二轮训练已实际完成，但 Stage 3 仍为 FAIL / NEEDS TUNING。** `entropy_coef=0` 消除了第一轮 best 的全局 `log_std` clamp 饱和，却没有改善 persistent outage；按既定安全优先 best 选择出的策略反而显著弱于第一轮。

## 1. 实验边界、可复现性与实际执行

- 第一轮 baseline 保留不变：`artifacts/stage3-final/`，`entropy_coef=0.01`。
- 第二轮从随机初始化开始：`artifacts/stage3-entropy0/`。没有传入 `--resume`，没有加载第一轮的 `best.pt`、`latest.pt` 或 `actor_final.pt`。
- 第二轮已实际完成 **1,000 / 1,000 updates**、**1,024,000 team-time samples**；`train.csv` 有 1,000 行（update 1--1000）、所有行 `finite=1`、所有数值字段 finite；`eval.csv` 有 20 行（每 50 update 一次）。
- 已保存：
  - `artifacts/stage3-entropy0/train.csv`
  - `artifacts/stage3-entropy0/eval.csv`
  - `artifacts/stage3-entropy0/config.json`
  - `artifacts/stage3-entropy0/checkpoints/latest.pt`（update 1000）
  - `artifacts/stage3-entropy0/checkpoints/best.pt`（update 200）
  - `artifacts/stage3-entropy0/checkpoints/actor_final.pt`（来自 best.pt）
  - `artifacts/stage3-entropy0/final_best_eval.json`
  - `artifacts/stage3-entropy0/random_baseline_eval.json`
- 两份 `config.json` 的差异严格只有输出目录和 **`entropy_coef: 0.01 -> 0.0`**。K=4、8 环境、rollout=128、mini-batch=256、10 PPO epochs、学习率、网络、`log_std=[-5,2]`、base seed=2026、47 维 critic state 均相同。
- `RelayEnv`、Environment v1.1 物理/通信/安全逻辑、Reward、Observation 和网络结构均未修改。Graph、Role、CBF 和第三轮训练均未启动。

为审计 clamp 现象，仅增加了不参与动作或梯度计算的日志：每个 update 在 rollout local observations 上记录 Actor 原始 `raw_log_std`、clamp 后 `log_std` 与 `raw_log_std > 2` 比例；固定 seed 最终回放也记录同一统计。该诊断不改变 Actor 的 `forward` 数值、PPO 损失或环境 transition。Stage 3 MAPPO 回归测试在开始本轮前为 **23/23 PASS**。

## 2. entropy=0 是否解决 log_std 饱和

### 训练期曲线对照

| 阶段 | 第一轮 entropy | 第二轮 entropy | 第一轮 rollout saturation | 第二轮 rollout saturation | 第二轮 raw log_std mean | 第二轮 raw > 2 ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1--200 | 5.952 | 2.430 | 0.334 | 0.0607 | -0.609 | 0.00026 |
| 201--500 | 10.021 | -0.532 | 0.789 | 0.0121 | -1.595 | 0.00019 |
| 501--800 | 10.252 | 0.082 | 0.810 | 0.0744 | -1.370 | 0.02587 |
| 801--1000 | 10.254 | 4.200 | 0.823 | 0.297 | 0.158 | 0.16865 |

第一轮在 500 update 后 entropy 长期接近三维 Gaussian 的上界 10.2568；第二轮没有保持在此上界。第二轮虽然在后期又出现方差反弹（update 1000：entropy 6.6066、rollout saturation 0.4932、raw `log_std>2` 比例 0.3425），仍显著低于第一轮的 update 1000（entropy 10.2568、saturation 0.8276）。因此 `entropy_coef=0` **显著缓解而非逻辑上永久禁止** raw head 越过 clamp 上界。

### 最终 best.pt 的相同 20-seed 评估状态

固定 seeds 10000--10019 的 Actor 输出统计如下。最终 entropy 由 clamp 后的三维 Gaussian 计算：`3 * (0.5*ln(2*pi*e) + mean(clamped_log_std))`。

| 项目 | 第一轮：entropy=0.01，best update 900 | 第二轮：entropy=0，best update 200 |
|---|---:|---:|
| raw log_std mean | 4.98619 | -1.29159 |
| raw log_std max | 7.37860 | 1.79015 |
| raw log_std > 2 ratio | 0.999884 | 0.000000 |
| clamped log_std mean | 1.99993 | -1.29159 |
| clamped log_std max | 2.00000 | 1.79015 |
| 最终状态的三维 Gaussian entropy | 10.25662 | 0.38204 |
| deterministic action saturation | 0.20300 | 0.00000 |

所以对于最终被安全优先规则选中的第二轮 best，答案是明确的：**是，entropy=0 解决了第一轮那种几乎 100% raw `log_std>2`、clamp 后固定在 2 的饱和。** 但这不是充分条件：训练后期 latest.pt 的 evaluation states 仍有 0.03481 raw `log_std>2`，训练 rollout 的最后一个 update 为 0.34253，说明 PPO surrogate 本身仍可把局部方差推高。

## 3. PPO 优化状态

| 指标（阶段均值） | 第一轮 201--500 | 第二轮 201--500 | 第一轮 501--800 | 第二轮 501--800 | 第一轮 801--1000 | 第二轮 801--1000 |
|---|---:|---:|---:|---:|---:|---:|
| approx KL | 0.00305 | 0.01842 | 0.00280 | 0.01905 | 0.00319 | 0.01348 |
| clip fraction | 0.0203 | 0.1347 | 0.0197 | 0.1236 | 0.0268 | 0.0876 |
| Actor grad norm（裁剪前） | 0.192 | 8.104 | 0.083 | 8.965 | 0.143 | 3.194 |
| Critic grad norm（裁剪前） | 61.84 | 44.66 | 76.68 | 40.59 | 105.85 | 52.00 |
| critic loss | 11.17 | 6.54 | 15.31 | 5.57 | 19.45 | 9.16 |

第二轮没有 NaN/Inf，Critic loss 与预裁剪 Critic norm 反而低于第一轮的后段；但 Actor 的 KL、clip fraction 和预裁剪梯度明显更高，并在个别 update 出现 KL 0.2312、clip fraction 0.2642、Actor grad norm 39.49。故 entropy=0 没有产生数值崩溃，但使 Actor 优化更激进，且后期再次推动一部分 raw log_std 越过上界。

这也解释了“去除 entropy bonus”不能被解读成“策略必然变得更稳定”：方差的正则驱动被移除后，PPO 目标仍可在某些状态把方差上调，且训练状态分布、确定性评估和安全终止之间仍存在未解决的耦合。

## 4. 5-episode 训练曲线：没有复现第一轮 update 900 峰值

第一轮在 850/900/950/1000 的正常完成数为 `2/5, 3/5, 1/5, 0/5`，update 900 曾短暂达到 persistent outage `2/5`、boundary `0/5`。

第二轮在 **20 次**周期评估中正常完成始终为 **0/5**。后段虽有通信/outage 改善，却换成了 boundary：

| update | normal /5 | persistent outage /5 | collision /5 | boundary /5 | outage ratio | mean e2e (Mbps) |
|---:|---:|---:|---:|---:|---:|---:|
| 800 | 0 | 2 | 0 | 3 | 0.07204 | 6.72254 |
| 850 | 0 | 3 | 0 | 2 | 0.09555 | 6.64476 |
| 900 | 0 | 3 | 0 | 2 | 0.09018 | 6.60217 |
| 950 | 0 | 1 | 0 | 4 | 0.05758 | 6.87268 |
| 1000 | 0 | 1 | 0 | 4 | 0.03472 | 7.03056 |

结论：**第一轮 update 900 的短暂完成率峰值没有改善或复现。** 第二轮后期是“更少断链、更多 boundary、零完成”的交换，不能作为安全性能提升。

## 5. best.pt 选择与最终 20-episode 并排比较

既定 safety-first 字典序未修改：

`(speed_accel_violations, collision + boundary, -normal_completion_count, outage_step_ratio, -rate_satisfaction_ratio, -mean_e2e_rate_mbps)`。

第二轮 update 200 的周期评估为 hard violations=0、collision=0、boundary=0、normal=0、persistent outage=5，因此在第一、二优先级上胜过后期的 boundary 策略，被正确保存为 best.pt。它不是误加载或 checkpoint 写错，而是既定安全优先规则对真实曲线的结果。

| 相同 deterministic seeds 10000--10019 | 第一轮 baseline | 第二轮 entropy=0 | 变化 |
|---|---:|---:|---:|
| selected best update | 900 | 200 | 更早 |
| normal completion | 6/20 | 0/20 | -6 |
| persistent outage | 13/20 | 20/20 | +7（更差） |
| collision | 1 | 0 | -1 |
| boundary | 0 | 0 | 0 |
| hard speed/acceleration violations | 0 | 0 | 0 |
| collision + boundary | 1 | 0 | -1 |
| mean e2e rate (Mbps) | 6.69411 | 5.90869 | -0.78542 |
| outage step ratio | 0.07437 | 0.16945 | +0.09508（更差） |
| rate satisfaction ratio | 0.89972 | 0.77924 | -0.12047 |
| deterministic action saturation | 0.20300 | 0.00000 | -0.20300 |

第二轮 `latest.pt`（update 1000）并不是更好的替代：同一 20 seeds 下正常完成 0/20、persistent outage 6、boundary 14、outage ratio 0.07779、mean e2e 6.60822 Mbps。它说明后段确实压低 outage，但以大量边界终止为代价，因此按安全优先规则不能替换 update-200 best。

## 6. 原验收线与随机基线

第二轮 best.pt 的 20-episode 分数是：

`(0, 0, 0, 0.1694480, -0.7792439, -5.9086892)`。

本轮实际运行的同 seeds 随机策略分数是：

`(0, 12, 0, 0.0734318, -0.9001854, -6.5965169)`。

由于第二个字典序关键字 `collision + boundary` 为 0（随机为 12），第二轮 best 仍按规定优于随机策略；这不代表它满足完整任务性能验收。

| 验收项 | 第二轮结果 | 状态 |
|---|---:|---|
| speed/acceleration hard violations = 0 | 0 | PASS |
| normal completion >= 16/20 | 0/20 | **FAIL** |
| collision + boundary <= 2/20 | 0/20 | PASS |
| mean e2e rate >= 6 Mbps | 5.90869 | **FAIL** |
| 安全优先字典序优于随机 | 是 | PASS |

## 7. 结论与下一步边界

- **entropy=0 解决了第一轮 best 的 log_std 全局 clamp 饱和，也显著降低了 action saturation。**
- **它没有降低 persistent outage，也没有提高 normal completion；最终 best 从 6/20 正常完成退化为 0/20，mean e2e 也跌破 6 Mbps。**
- 因此 Stage 3 不能标为 PASS：第二轮具体卡在 `normal completion >= 16/20` 和 `mean_e2e_rate_mbps >= 6` 两项，且 persistent outage 增至 20/20。
- 不应根据这一次失败擅自修改 Environment、Reward、Observation、通信参数、网络或 best 规则，也不应进入 Graph/Role/CBF。
- 按本次单变量实验协议，**不自行启动第三轮**。是否单独测试已诊断出的 `critic_lr=1e-4`，应在审阅本报告后另行决定。

