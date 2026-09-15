# Stage 3 Plain MAPPO：Actor 动作分布设计审查

日期：2026-09-09  
本轮边界：只审查/实现 Actor 分布，完成单元测试、回归和 10-update smoke；**未启动第五次 1,000-update 正式训练**。

## 结论摘要

应当将下一版 Plain MAPPO Actor 的 `state-dependent log_std_head` 替换为一个所有 relay、所有 observation 共用的三维 `state-independent` learnable vector，并采用平滑有界映射。该修改能直接解决已有的 raw-log_std 全局越界、hard-clamp 死区和按状态方差剧烈漂移问题。

但这不是“方差问题已解释全部任务失败”的结论。D 的 stochastic 与 deterministic 评估均为 0/20 normal completion，20 个周期 Actor checkpoint 重新以相同 20 seeds 评估也全部为 0/20；同时 outage 已扩散到整条链，onset 时动作通常并未饱和。因而 deterministic mean policy 的队形保持能力及局部可观测性仍是独立、未消除的风险。

## 1. 四轮证据与已经排除的假设

| 轮次 | entropy / actor_lr | normal completion | persistent outage | 与分布直接相关的证据 |
| --- | --- | ---: | ---: | --- |
| A | 0.01 / 3e-4 | 6/20 | 13/20 | best 状态 raw log_std mean/max = 4.986/7.379，99.988% > 2；有效 log_std 几乎固定为 2，后 200 updates entropy=10.254、rollout saturation=82.3%。 |
| B | 0 / 3e-4 | 0/20 | 20/20 | best 不再饱和（raw max=1.790），但仍失败；后期 PPO KL/clip 高。 |
| C | 0.001 / 3e-4 | 0/20 | 19/20 | best raw log_std mean/max=4.617/8.268，96.484% > 2；后 200 updates entropy=10.004、saturation=78.7%。 |
| D | 0 / 1e-4 | 0/20 | 19/20 | best 未饱和（raw mean/max=-0.625/0.003），但后 200 updates raw mean=-2.399、max=3.096、少量 >2，KL=0.03589、clip=0.17445。 |

本轮采用的已独立验收事实如下：

- D 的 20 个周期 Actor checkpoint（update 50–1000）均在 seeds 10000–10019 上得到 0/20 normal completion，故不能归因于 `best.pt` 选择或 5-seed 评价噪声。
- D `best.pt` 的 stochastic Gaussian 采样评估仍为 0/20 normal、19/20 persistent outage；故不能将失败主要归于 rollout stochastic action 与 deterministic `tanh(mean)` 的评估失配。
- D 的 outage onset 已不只是 H→R1/R4→L：best 的 19 个 persistent terminal events 有 20 个 onset-link labels，其中 H→R1/R4→L 为 6 个，三条中间链路为 14 个；onset action saturation 为 0。

因此可以排除或显著削弱的解释是：checkpoint 选错、仅 5-seed 的偶然噪声、仅 deterministic evaluation 失配、以及“端部动作顶到 tanh 边界仍无法恢复”是唯一原因。不能排除的是 Actor 均值控制本身未学出可靠队形保持，以及局部信息不足。

## 2. 当前 state-dependent log_std 的代码级审查

当前 Actor 是：

```text
local_obs [..., 26]
  -> shared MLP [..., 128]
  -> mean_head [..., 3]
  -> log_std_head [..., 3]
  -> raw_log_std.clamp(-5, 2)
  -> Normal(mean, exp(log_std))
  -> z -> tanh(z)
```

`mean_head` 与 `log_std_head` 共享同一 MLP backbone。对单个维度，令 `ell=log_std`，则

```text
log pi(z|o) = -0.5 * [((z-mu)/exp(ell))^2 + 2*ell + log(2*pi)]
d log pi / d ell = ((z-mu)^2 / exp(2*ell)) - 1
```

PPO 使用缓存的同一个 old latent `z` 重算 new log-prob，`ratio=exp(logp_new-logp_old)` 不变。只要样本未被 clipped surrogate 截断，ratio 的 policy gradient 就会经由上述导数更新 log_std；三维 latent-Gaussian entropy 对每个 `ell` 的导数为 1，因而正的 entropy bonus 还会持续上推方差。四个 relay 的所有 team-time 样本又把 mean 与 log_std 的梯度同时汇总进同一个 backbone。

这使得“某些高 advantage 的局部状态需要均值修正”与“另一些离群 latent 样本通过加大方差提高似然”耦合在同一套隐藏表示中。它不是必然错误，但对本任务的 shared 多 Agent policy 是额外自由度和额外梯度干扰：任一 relay/状态的方差更新都会改变共享函数在其他 relay/状态上的输出。

真实数据与此一致：

- A 的 entropy reward 将大量状态推入上界；A 后期 entropy 贴近三维 `log_std=2` 的理论上界 10.2568，raw head 虽继续增长，实际分布却固定在上界。
- C 的较小 `entropy_coef=0.001` 只延后而未消除同一模式，best 的 96.484% raw 值仍越过 2。
- B/D 设为 `entropy_coef=0` 后消除了 best 的全局饱和，却没有禁止 PPO ratio 本身推动局部方差。D 后期是“大多数状态低方差、少量状态异常高方差”与高 KL/clip 同时出现，说明已不是单一全局熵问题。

### hard clamp 的具体问题

当前 `ell=clamp(raw,-5,2)` 在 `raw>2` 或 `raw<-5` 时有 `d ell / d raw=0`。因此在越界状态上，PPO log-prob 与 entropy 对 raw head 的直接梯度为零；raw 可继续在上界外漂移，而有效分布不再可由该路径拉回。共享 backbone 仍可能因 mean head 或别的状态间接改变 raw 输出，但这不是一个稳定的方差校正机制。

这正解释 A/C 中“raw 大幅超过 2、有效 log_std 却始终为 2”的现象。D 没有全局进入该死区，但它仍有少量状态进入上界外，且共享 state-dependent head 仍可参与后期 policy drift；不能据此断言 D 的高 KL 全部由方差造成。

## 3. 三种 log_std 参数化比较

| 方案 | 有效 log_std | 上界外梯度 | 当前证据下的评价 |
| --- | --- | --- | --- |
| A. 当前 state-dependent + hard clamp | `clip(f(o), -5, 2)` | clamp 外为 0 | 已在 A/C 发生全局饱和；保留了按状态任意变化的方差。 |
| B. state-independent Parameter + hard clamp | `clip(theta, lmin, lmax)` | clamp 外为 0 | 删除 observation 相关的方差波动，但仍保留死区；优于 A 但非最小风险方案。 |
| C. state-independent Parameter + smooth tanh bound | `mid + half_range*tanh(theta)` | 连续、在边界附近渐小 | 删除按状态方差 head，严格有界且没有突然的 clamp 零梯度；推荐。 |

当前 `[-5,2]` 对 normalized action 过宽：std 范围是约 `[0.0067, 7.39]`。后者使大量 latent 落在 `tanh` 边界，A/C 的约 79–82% rollout saturation 已是实证。反之，过小的下界会让策略近乎确定。推荐的下一版实验区间是 `log_std in (-4, 0)`，即 std 在约 `(0.018, 1.0)`：上界已足以产生有意义探索，但不会系统性制造 `tanh` 饱和；下界允许接近确定性而不使用极端 0.0067。

建议有效初值 `log_std=-1.5`，即 latent std 约 0.223。在均值接近零时，归一化动作的局部标准差约为 0.22，对 2 m/s² 的单维加速度尺度约为 0.45 m/s²；二维水平合加速度的典型量级仍低于最大值，适合作为初始探索而不是接近满幅随机控制。这是下一轮受控 Actor 方案的初值，不是对环境的修改或永久物理参数。

## 4. 推荐的唯一最小 Actor redesign

唯一推荐的结构修改是 **state-independent、平滑有界的三维 log_std vector**。已实现为可显式选择的 `state_independent_tanh`，历史 A–D 则保留 `state_dependent_clamp`，保证已有 checkpoint 仍可加载。

### 参数化

对 `j in {x,y,z}`：

```text
theta = nn.Parameter([theta_x, theta_y, theta_z])             # shape [3]
ell_j = -2 + 2 * tanh(theta_j)                                # ell_j in (-4, 0)
sigma_j = exp(ell_j)
theta_init = atanh((-1.5 - (-2)) / 2) = atanh(0.25) ~= 0.2554
```

网络形状为：`obs [B,K,26] -> shared MLP [B,K,128] -> mean [B,K,3]`；`theta [3]` 广播为 `[B,K,3]`。没有 `log_std_head`，因此所有 relay、所有 observation 共享同一对象和同一组三个可学习数；x/y/z 仍可分别学习。

采样仍是 `epsilon~N(0,I)`、`z=mu+sigma*epsilon`、`a=tanh(z)`。rollout 继续保存 old `z` 与 old summed Gaussian log-prob。更新时以同一 old `z` 计算 old/new `log_prob=sum_j log N(z_j;mu_j,sigma_j)`，所以 PPO ratio、GAE、buffer 和 clipped surrogate 都不改变；tanh Jacobian 在同一个动作的 old/new ratio 中仍相消。部署及 deterministic evaluation 仍只用 `tanh(mean)`，不使用方差。

checkpoint 的 `config` 记录 mode、bounds 和 init；Actor state dict 新增 `log_std_parameter` 而不含 `log_std_head`，Actor Adam state 也自然包含该参数。历史 config 缺少 mode 时默认 `state_dependent_clamp`，实际 A/B/C/D `best.pt` 已逐个重新严格加载成功。

为隔离该结构变化，建议未来获得授权后沿用 D 的训练常量：`entropy_coef=0.0`、`actor_lr=1e-4`，以及既有 PPO/critic/网络/环境常量不变。`entropy_coef=0` 是为了不再次直接奖励最大方差；`actor_lr=1e-4` 是 D 的已验证前期较温和基线。它们不是本轮新增 sweep 参数。

## 5. 方差不是全部根因

修复动作分布是有直接证据支持的必要工程改进，但现有数据不足以证明它足以让任务通过：

- B/D 的 best 本身已经没有全局 log_std 饱和，normal completion 仍为 0/20。
- D 的 stochastic 与 deterministic policy 同样失败，排除了“仅用 mean 才失败”的主要解释。
- D 的全部 20 个周期模型都为 0/20 normal completion，排除了错过短暂好 checkpoint 的解释。
- D best 的 onset 动作不饱和，且 outage 从端部扩散到中间三条链路；这是可靠队形保持/恢复策略没有形成的表现，而非方差单点错误的充分证明。

所以正确的表述是：**动作分布设计存在可验证问题，值得先以单一结构改动修复；但它不能单独解释所有性能失败。** 若该更稳的分布仍无法产生 normal completion，下一步应重新审查 Actor 可观测性与 Plain shared-policy 的能力边界，而不是继续调节同一套分布超参数。

## 6. 26 维 local observation 的可观测性审查

每个 relay 的 26 维字段为：

| 维度 | 内容 |
| --- | --- |
| 0–5 | 自己的绝对 position(3)、velocity(3) |
| 6–12 | 以有序 **upstream** 槽位给出的相对 position(3)、relative velocity(3)、该 hop capacity(1) |
| 13–19 | 以有序 **downstream** 槽位给出的相对 position(3)、relative velocity(3)、该 hop capacity(1) |
| 20–25 | 最近一个非相邻节点的相对 position(3)、relative velocity(3) |

这提供了有价值的信息：Actor 能区分“上游槽位”和“下游槽位”，能据两个即时相对位置/速度及两个 capacity 作局部链距调整，也有自己的绝对坐标可学习边界回避。故不能声称 shared Actor 在局部中继保持上理论必然无解。

但它不是 Actor 所需的全局 Markov state，且没有显式 relay index/Role：

- 未观测 H/L waypoint、cruise speed、step、全局 `consecutive_outage_steps`；相同即时 local vector 可以对应不同下一步移动端机动或不同的 outage 终止紧迫度。
- 除最近非邻居外，其余远端节点不可见。例如固定 R2、自身上下游 R1/R3 和最近的 H 的局部量后，仍可改变远端 R4/L 的位置、速度或 waypoint 且不改变 R2 的 26 维输入；但全链 bottleneck 以及有利的协同行为可以不同。
- 本机绝对 position 使同一时刻的不同 relay 很少取得完全相同向量，但跨 episode 或不同全局状态中，链端和中继的局部几何/速度模式可重合。该映射从全局状态到 26 维输入是多对一，而端点/中段在全链恢复中的责任不同。
- nearest-non-neighbor 仅有几何量，没有节点语义；它可能是 H、L 或另一个 relay。上/下游的有序槽位缓解了部分拓扑歧义，却不能告诉 Actor 自己是 R1/R2/R3/R4，也不能表达所有远端断链。

因此无 Role/ID 的 Plain shared Actor 存在真实的信息不可辨识和部分可观测风险，但本轮不据此加入 Role。A 的 6/20 normal completion 说明环境并非经验上绝对不可解；四轮失败只能表明当前 reactive shared Actor 尚未可靠学会该能力。

## 7. 实现与实际 smoke 结果

已修改的仅是 Plain MAPPO Actor 分布及其调用/诊断/测试：

- `plain_mappo/networks.py`：增加 opt-in `state_independent_tanh`；旧 mode 保持。
- `plain_mappo/config.py`、`trainer.py`、`train_plain_mappo.py`：记录/选择 distribution mode；不改 PPO、GAE、buffer、Critic。
- `evaluate_plain_mappo.py`、`stage3_training_diagnosis.py`：按 checkpoint config 重建 Actor；无 outage 诊断子集安全返回 0 比例，不再产生 NaN JSON。
- 新增 state-independent shape、old-latent log-prob、共享参数/有限梯度、checkpoint round-trip/resume 测试。

实际执行的 smoke：

```text
output: artifacts/stage3-stateindependent-smoke/
updates: 10                 team-time samples: 10,240
mode: state_independent_tanh
effective bounds/init: (-4, 0) / -1.5
entropy_coef / actor_lr: 0.0 / 1e-4
```

update 10 的真实训练记录：`finite=1`、approx KL=0.004676、clip fraction=0.044922、action saturation=0、原始共享坐标 mean=0.25431、有效 log_std mean=-1.50207（范围 -1.51000 至 -1.49712）。2-seed 只读诊断也成功加载 best/latest，并得到完全有限的相同有效 log_std。该 smoke 的 5-episode eval 有 5 次 boundary、0 normal completion；它只验证工程和数值路径，**不得被当作性能评估或第五轮结果**。

验证结果：

- Stage 3 MAPPO tests：28/28 PASS（新增 4 项分布/shape/checkpoint 覆盖）。
- new Actor `actor_final.pt` 已独立加载并执行 2-episode deterministic deployment check。
- state-independent checkpoint round-trip 后实际继续 1 update：PASS。
- A/B/C/D 四个历史 best checkpoint 都按 legacy config 严格加载：PASS。
- Stage 1 regression：13/13 PASS。
- Stage 2 hard checks：17/17 PASS，输出在 `artifacts/stage2-actor-dist-regression/`，raw step JSONL 未生成。
- Stage 2.5 antenna tests：5/5 PASS。
- Environment v1.1、Reward、Observation=26、global state=47、通信、网络隐藏层、Critic 与 PPO GAE/buffer 均未修改。

## 8. 启动第五轮的条件

工程条件已经具备：新 Actor 已有明确参数化、旧 checkpoint 兼容、PPO old-latent log-prob 正确、共享参数/有限梯度/checkpoint/resume/环境回归与 10-update smoke 均通过。

性能条件尚不具备：没有任何证据表明该分布修改已产生 normal completion，更不能据 smoke 预测 20-seed 验收。若用户明确授权下一次正式训练，唯一应变化的是本报告定义的 Actor distribution（state-independent smooth bounded log_std）；其余训练与环境常量保持 D 的基线。未经该授权，不启动第五次 1,000-update 训练。
