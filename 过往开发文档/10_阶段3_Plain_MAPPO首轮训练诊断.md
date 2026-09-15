# Stage 3 Plain MAPPO 首轮正式训练诊断

日期：2026-09-08  
结论：**Stage 3 工程实现通过；首轮策略性能仍为 FAIL / NEEDS TUNING。** 本文仅分析正式训练目录 `artifacts/stage3-final/`，未使用 Stage 2 artifact，未把 10-update smoke 数据代替正式数据，也未启动新训练。

## 数据范围与方法

- 正式训练：`train.csv` 共 1,000 updates，全部 `finite=1`；总交互量为 1,024,000 team-time samples。
- 周期评估：`eval.csv` 共 20 次、每 50 update 一次、每次固定 5 个 deterministic episodes。
- 最终验收：`checkpoints/best.pt`（update 900）在 seeds 10000--10019 上的已有 20-episode 结果。
- 为补足 CSV 没有保存的逐步 outage 信息，新增只读脚本 `stage3_training_diagnosis.py`，重新执行相同 20 个 deterministic seeds，输出紧凑诊断数据 `artifacts/stage3-final/final_checkpoint_outage_diagnosis.json`。脚本不训练、不保存 checkpoint、不改变 `RelayEnv`。

## 1. 训练曲线：学习阶段结论

`train.csv` 的 return/length 是每个 rollout 中已结束 episode 的均值，再按 update 做阶段平均；不是将不同长度 episode 错误拼接后的单一 return。`eval.csv` 指标为每次 5 episodes 的均值或计数。

| 训练阶段 | train return | train episode length | eval 正常完成 /5 | eval persistent outage /5 | eval outage ratio | eval rate satisfaction | eval mean e2e (Mbps) | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0--200 | 155.81 | 172.22 | 0.00 | 4.75 | 0.1610 | 0.7941 | 5.9209 | 有早期学习，但尚不能完成 episode；update 200 还出现 1 次 boundary。 |
| 201--500 | 171.97 | 184.78 | 0.00 | 4.83 | 0.1724 | 0.7779 | 5.7956 | 训练 return/长度微升，但评估通信与 outage 没有改善，属于平台/轻微退化。 |
| 501--800 | 237.26 | 244.87 | 0.00 | 4.33 | 0.1223 | 0.8260 | 6.2193 | 通信质量改善，但仍无正常完成；650--750 的评估还合计出现 3 次 collision/boundary。 |
| 801--1000 | 341.02 | 317.73 | 1.50 | 3.50 | 0.0684 | 0.9124 | 6.9151 | 850--900 出现实质突破，随后明显退化，不能视作稳定持续上升。 |

后段的单点序列更能说明问题：update 850 为正常完成 2/5、persistent outage 3/5、outage ratio 0.05659；update 900 改善至 3/5、2/5、0.03479；但 update 950 回落为 1/5、4/5、0.07577，update 1000 又回落为 0/5、5/5、0.10645。对应的 mean e2e rate 是 6.975、**7.394**、6.931、6.361 Mbps。

因此总体最符合 **C：后期发生性能退化**，而不是 A（仍稳定持续进步）、B（全程完全平台）或 D（明显数值不稳定）。训练 return 与完成长度的总体上升说明算法确实学到部分行为；但面对真正的 deterministic 评估，900 是一个短暂峰值，之后没有保持。以当前不变超参数单纯增加 update，不是首选。

## 2. PPO 优化与数值稳定性

| 指标（阶段均值） | 0--200 | 201--500 | 501--800 | 801--1000 | 诊断 |
|---|---:|---:|---:|---:|---|
| actor policy loss | -0.00637 | -0.00283 | -0.00267 | -0.00361 | 不应把它解释为“越低越好”；没有显示目标函数数值发散。 |
| critic loss | 29.17 | 11.17 | 15.31 | 19.45 | 初期快速下降，后期波动/抬升，950 附近有尖峰。 |
| latent Gaussian entropy | 5.952 | 10.021 | 10.252 | 10.254 | 后期贴住上界，是最显著的优化异常。 |
| approx KL | 0.00667 | 0.00305 | 0.00280 | 0.00319 | 长期很低且稳定；仅 281、386、520 有 0.0237、0.0792、0.0184 的孤立点。 |
| clip fraction | 0.0636 | 0.0203 | 0.0197 | 0.0268 | 未长期偏高。最高 0.1204 在 update 160；200 后没有持续的过度裁剪。 |
| actor grad norm（裁剪前） | 0.376 | 0.192 | 0.083 | 0.143 | 后期远低于 0.5，未显示 Actor 更新过猛。 |
| critic grad norm（裁剪前） | 66.17 | 61.84 | 76.68 | 105.85 | 每个 update 的平均值均远高于 0.5，说明 Critic 的大多数梯度已触发 clipping；908 达 263.18，955 达 223.39。 |
| raw advantage mean / std | 2.575 / 8.009 | 0.880 / 6.811 | 1.480 / 7.572 | 1.577 / 8.512 | std 始终非零且有限，之后按实现正确标准化。 |
| rollout action saturation | 0.334 | 0.789 | 0.810 | 0.823 | 后期采样动作大量位于 tanh 边界附近。 |

没有 NaN/Inf、没有持续高 KL、没有持续高 clip fraction，也没有 Actor 梯度爆炸；所以“PPO 更新过猛”或“数值崩溃”不成立。Critic 预裁剪梯度很大且后期 loss 有波动，可能放大了价值目标噪声，但目前只能列为次级风险，不能单独断言为根因。

熵问题有直接证据：3 维高斯在 `log_std=2` 时的理论熵为 10.2568；501--1000 的日志为 10.252--10.254，且 update 900/1000 分别为 10.25682/10.25678。这说明 `log_std` 几乎一直顶在配置上界，而不是适度探索。训练采样动作饱和率从 update 1 的 0.0665 增至 update 250 的 0.7603，后续长期约 0.81--0.84；相对地，best 的 deterministic 动作饱和率仅 0.2030。这是训练状态分布与 `tanh(mean)` 评估状态分布存在明显差异的证据。

## 3. persistent outage 的物理/行为诊断

best.pt 的 20-episode 重放和已有最终验收一致：正常完成 6/20、persistent outage 13、collision 1、boundary 0、硬速度/加速度违规 0、mean e2e 6.6941 Mbps、outage ratio 0.07437。

### 发生时段

- 一共观测到 16 段 outage，13 段最终达到 25 consecutive outage steps 并终止。
- 13 个 persistent event 的起始步骤为 128--455，均值 232；终止步骤为 152--479，均值 256。
- 按起始步骤，persistent event 分布为早期（1--166）3 个、中期（167--333）8 个、后期（334--500）2 个。按 outage step 累计，早/中/后分别为 58/234/56 steps。

所以它不是只在 episode 最后自然失效：主要在中期发生，但早期和后期都可能发生。策略存在通信表现良好、却不能在阈值刚被触发时恢复的缺陷。

### 是端点瓶颈，不是整链拉散

- 348 个 outage steps **每一步均只有一条**链路处于 outage。
- 16 次 outage 起点中，`R4->L` 有 9 次，`H->R1` 有 7 次；没有由中间 `R1->R2`、`R2->R3` 或 `R3->R4` 单独触发的 best.pt outage。
- 起点的线性 SNR 已刚好低于 5 dB 的 3.1623 门限：`H->R1` 平均 3.123，`R4->L` 平均 3.154。对应的平均水平距离为 427.7 m/489.8 m，平均绝对高度差为 48.0 m/67.3 m。由于冻结天线模型同时依赖水平/高度几何，不能把它简化为仅“距离过远”。
- 尽管逻辑 outage 时 effective rate 必为 0，非 outage steps 的 mean e2e rate 仍为 7.3593 Mbps。这解释了为什么总均值能超过 6 Mbps、但仍频繁触发 persistent outage。

因此，失败不是四架 relay 集体漂散或中间 hop 崩溃，而是共享 Actor 对两端移动节点附近 relay 的几何跟踪/恢复不够鲁棒。

### 动作、速度和高度不是直接硬限故障

- outage 开始那一步的 action saturation ratio 为 0.1563，低于正常 steps 的 0.2025；outage 期间为 0.1858。没有证据显示“outage 前单次动作已经明显打满”是直接触发因素。
- best 在全 20 episodes 的最大 relay XY 速度为 12.913 m/s，95% 速度上限（19 m/s）命中比例为 0；outage 时最大 XY 速度为 10.714 m/s。因此并非 UAV 已达到 20 m/s 速度上限仍无法恢复。
- outage 时平均合加速度 2.394 m/s²，略高于全程 2.346 m/s²，说明策略确实在调整，但没有形成足够有效的端点几何修正。硬速度和加速度违规均为 0。
- relay 高度全程为 122.84--271.54 m，outage 时为 125.02--215.17 m；没有越界或单向高度逃逸的异常。垂直速度最大为 5.0 m/s，未触及配置的正向 6.0 m/s 上限。

结论是：动作在训练分布中常因高方差而饱和，但在最终 deterministic 失败起点并非某个立即饱和动作或速度上限造成；主要问题是端点链路临界几何下的持续跟踪/恢复策略不足。

## 4. best.pt 选择是否合理

实现的安全优先字典序为：

`(speed_accel_violations, collision + boundary, -normal_completion_count, outage_step_ratio, -rate_satisfaction_ratio, -mean_e2e_rate_mbps)`

其中越小越好，和任务书规定一致。best.pt 的元数据确认它保存于 update 900 / 921,600 team-time samples，选择分数为：

`(0, 0, -3, 0.0347887, -0.9562640, -7.3937254)`

在前后相同的周期评估中，update 850 为正常完成 2/5、persistent outage 3/5，update 900 为 3/5、2/5；950 为 1/5、4/5；1000 为 0/5、5/5。因此 update 900 在第三优先级“正常完成数”已严格胜出，best 选择正确，并非过度偏向 rate 或 outage ratio。

另外对 `latest.pt`（update 1000）实际重放同一 20 个 seeds：它得到正常完成 0/20、persistent outage 17、collision 0、boundary 3、outage ratio 0.10458、rate satisfaction 0.84594、mean e2e 6.27683 Mbps。相对 best 的 6/20、13、1、0、0.07437、0.89972、6.69411，两者的首个字典序项均为 0，但 latest 在其后的全部关键项上更差。它的最小节点间距较大只是非选择字典序指标，不能推翻上述结论。

需要注意的是，周期选择只使用 5 episodes，故无法充分估计泛化方差：900 的 5-episode 结果为 3/5，而扩大到 20 episodes 为 6/20。这是评估样本量限制，不是 best 规则实现错误；本轮不修改既定 best 规则。

## 5. 根因优先级

1. **最可能：熵奖励使 Gaussian `log_std` 长期顶在上限，导致训练期过度随机、tanh 饱和，并造成训练/确定性评估状态分布错配。** 熵 10.2568 的上界贴合、采样饱和约 0.82、而 deterministic 仅 0.203，是直接且持续的证据。这最能解释已经能在 850--900 学到可行行为，却不能稳定保留的现象。
2. **第二可能：策略没有学到对两端移动节点几何的鲁棒闭环恢复。** 16 次 outage 起点全部由 `H->R1` 或 `R4->L` 触发，且 SNR 仅略低于门限；中间 hops 未成为 best 的主瓶颈。速度没有达到上限、启动瞬间也未特别饱和，说明是跟踪/恢复控制质量不足，而不是不可达的物理约束。
3. **第三可能：Critic 的持续强裁剪和后期波动增加了更新噪声，参与了 900 后的退化。** Critic 预裁剪 norm 远大于 0.5 且 908/955 有尖峰；critic loss 在 900 后波动。不过 KL、clip fraction、Actor 梯度和 finite 标志不支持把它误诊为 PPO 数值崩溃，因此它不是第一调参目标。

## 6. 下一轮最小调参建议

**不建议先用原参数继续加训练步数。** 850--900 的提高证明更多样本有价值，但 950--1000 已经退化；同一配置继续训练更可能反复采样短暂峰值，而不保证提高 20-seed 鲁棒性。

建议先做一个单变量的下一轮对照训练：

1. `entropy_coef: 0.01 -> 0.0`，其余训练配置完全保持不变。理由是熵明确被驱动到 `log_std=2` 上限；先移除持续推高方差的唯一直接源，验证 stochastic rollout 与 deterministic evaluation 的分布错配是否消失。该试验只改一个训练超参数。
2. 若熵对照结果仍显示 900 后显著回落，第二个、单独验证的候选是 `critic_lr: 3e-4 -> 1e-4`。理由是 Critic 在整个训练期都被 0.5 梯度裁剪且后期有尖峰；降低其步长比放宽梯度上限更保守，且不改变环境或目标。不要与第一项同时修改，以保留根因可辨性。

当前明确**不要**修改：Environment v1.1 动力学/安全/终止逻辑、Reward、26 维 Observation、通信或天线参数、47 维 critic state、网络宽度/深度、Graph/Role/CBF；也暂不改 `clip_epsilon=0.2`、`ppo_epochs=10`、Actor learning rate、batch size、GAE/gamma 或 `max_grad_norm`。现有 KL/clip/Actor 梯度没有支持这些方向；网络容量也没有被数据证明是瓶颈。
