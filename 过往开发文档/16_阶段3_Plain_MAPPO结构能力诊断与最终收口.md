# Stage 3 Plain MAPPO：结构能力 / 局部可观测性诊断与最终收口

日期：2026-09-09  
结论：**Stage 3 Plain MAPPO baseline = COMPLETE**。

- Engineering correctness = **PASS**
- PPO numerical stability = **PASS**
- Task-performance acceptance = **FAIL**
- Baseline status = **VALID WEAK BASELINE**

这里的“COMPLETE”表示 Plain MAPPO 基线的工程实现、受控训练和失败模式已经被充分记录，并不表示任务性能验收通过。结论选择本轮规定的 **B**：工程与主要 PPO 数值问题已经基本排除；当前 `26` 维 local observation + parameter-shared Actor + 无 Role / 无 Graph 具有明确的结构信息缺口，足以把 Plain MAPPO 作为性能不足但有效的对照基线结束。该结论不声称 Plain MAPPO 在数学上不可能成功。

本轮未修改 Environment、Reward、Observation、Actor/Critic、PPO 超参数或训练代码，未启动新的 1,000-update 训练，也未实现 Role、Agent ID、Graph 或 CBF。

## 1. 五轮训练事实简表

所有最终结果均为各轮自己 `best.pt`、K=4、同一 deterministic seeds 10000–10019 的 20 complete episodes。所有轮次 speed/acceleration hard violations 均为 0。

| 轮次 | Actor variance | entropy / Actor LR | best update | normal completion | persistent outage | collision / boundary | mean e2e Mbps | outage ratio | rate satisfaction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | state-dependent + clamp | 0.01 / 3e-4 | 900 | 6/20 | 13 | 1 / 0 | 6.694106 | 0.074369 | 0.899716 |
| B | state-dependent + clamp | 0 / 3e-4 | 200 | 0/20 | 20 | 0 / 0 | 5.908689 | 0.169448 | 0.779244 |
| C | state-dependent + clamp | 0.001 / 3e-4 | 700 | 0/20 | 19 | 1 / 0 | 6.134737 | 0.133537 | 0.794624 |
| D | state-dependent + clamp | 0 / 1e-4 | 150 | 0/20 | 19 | 0 / 1 | 6.086493 | 0.130640 | 0.774669 |
| E | state-independent smooth bound | 0 / 1e-4 | 650 | 0/20 | 19 | 0 / 1 | 6.103710 | 0.129161 | 0.775567 |

第五轮 E 从全新随机初始化真实运行 1,000 updates / 1,024,000 team-time samples。其 shared effective `log_std` 从约 -1.50 平滑到训练末约 -2.25，始终位于 `(-4, 0)` 内；rollout action saturation=0，最后 200 updates 的 approx KL=0.00603、clip fraction=0.06677。它显著优于 D 的 0.03589 / 0.17445，且无 NaN/Inf，却仍为 0/20 normal completion。这是后续结构诊断的关键证据。

## 2. 已排除或显著削弱的解释

| 假设 | 证据 | 结论 |
| --- | --- | --- |
| Environment v1.1 / 47 维 critic state 的工程错误 | Stage 1 13/13、Stage 2 17/17、Stage 2.5 tests 5/5 与 hard checks 17/17、Stage 3 28/28 全部 PASS；物理模型未改 | 未发现支持证据 |
| PPO/GAE/buffer、checkpoint 或 resume 错误 | old-latent log-prob、terminated/truncated mask、47 维 flatten、complete checkpoint、seed progress 均有自动测试；E 实际完整运行 | 已基本排除 |
| best.pt 或 5-seed selection 噪声 | D 的 20 个周期 Actor checkpoint 都已用 20 fixed seeds 重评，normal 均为 0/20 | 已排除为主要解释 |
| stochastic rollout 与 deterministic `tanh(mean)` mismatch | D best 的 stochastic 20-seed 重评仍为 normal=0/20、persistent outage=19 | 已排除为主要解释 |
| entropy / state-dependent `log_std` 饱和 | A/C 的严重 clamp 饱和已由 E 的 state-independent smooth-bounded 参数化消除 | 已排除为 E 失败的主要解释 |
| Actor LR 过高及 PPO 后期剧烈漂移 | D 使用 1e-4 仍失败；E 在更低 KL/clip 下仍 0/20 | 已显著削弱 |
| 仅端部追踪失败或动作饱和 | E 首次 outage 覆盖五条 hop，onset action saturation=0 | 已排除为充分解释 |

以上不等于“没有任何训练改进空间”，而是说明再机械扫 Actor LR、critic LR、entropy、clip 或网络宽度，缺少能解释 0/20 的新证据。

## 3. 当前实际 26 维 local observation

实际实现位于 `RelayEnv._get_observations()`。节点数组固定构造为 `[H, R1, R2, R3, R4, L]`；对第 `i` 架 relay（零基）取 `upstream=nodes[i]`、`downstream=nodes[i+2]`。原始字段随后由 `_normalise_observation()` 归一化，Actor 只接收归一化后的数值，不接收节点名称。

| 下标 | 原始字段 | 物理含义 | 归一化 |
| ---: | --- | --- | --- |
| 0–2 | `position.x/y/z` | 自身绝对位置 | x/map_x，y/map_y，`(z-200)/100` |
| 3–5 | `velocity.x/y/z` | 自身速度 | xy/20，z/6 |
| 6–8 | `upstream.position - self.position` | 有序 upstream 槽位的相对位置 | dx/map_x，dy/map_y，dz/200 |
| 9–11 | `upstream.velocity - self.velocity` | upstream 相对速度 | xy/40，z/12 |
| 12 | `up_capacity` | upstream 邻接 hop 容量 | `min(capacity / 60e6, 1)` |
| 13–15 | `downstream.position - self.position` | 有序 downstream 槽位的相对位置 | dx/map_x，dy/map_y，dz/200 |
| 16–18 | `downstream.velocity - self.velocity` | downstream 相对速度 | xy/40，z/12 |
| 19 | `down_capacity` | downstream 邻接 hop 容量 | `min(capacity / 60e6, 1)` |
| 20–22 | `safe.position - self.position` | 最近的**非自身且非当前通信邻居**节点的相对位置 | dx/map_x，dy/map_y，dz/200 |
| 23–25 | `safe.velocity - self.velocity` | 同一 safe 节点的相对速度 | xy/40，z/12 |

`safe` 由候选节点中最小三维距离选择，排除自己、upstream 和 downstream；它不携带节点名称、Role、容量或拓扑关系。固定 seed=2026 的只读探针确认实际槽位为：R1=`H / R2`，R2=`R1 / R3`，R3=`R2 / R4`，R4=`R3 / L`，并确认下标 12/19 分别对应两个相邻 hop 的容量。

## 4. shared Actor 能知道什么、不能知道什么

当前部署 Actor 是同一个函数 `pi(a|o)`：`[B,K,26] -> shared MLP -> mean_head(3)`；没有输入 Agent ID、Role、chain index、global state 或图消息。Actor 所有 relay 的 mean 共享同一组参数。

| 角色问题 | 当前 observation 的隐含信息 | 当前 observation 明确缺失的信息 |
| --- | --- | --- |
| R1 是否知道 upstream 是 H | 环境确实把 H 放入有序 upstream 槽位；H 的局部相对运动在统计上可能看起来与 relay 不同 | 无 `is_H` 标签、无 H waypoint/cruise speed、无“我是第一个 relay”字段；数值槽位本身不说明节点类型 |
| R4 是否知道 downstream 是 L | 环境确实把 L 放入有序 downstream 槽位；可看到它的即时相对位置/速度/容量 | 无 `is_L` 标签、无 L waypoint/cruise speed、无“我是最后 relay”字段 |
| R2/R3 是否知道自己在中部 | 两个有序通信邻居都由 relay 构成；可局部平衡两边相对几何与容量 | 无 ordinal（R2/R3）、无距 H/L 的 hop count、无全链当前瓶颈位置 |
| 所有 relay 的局部控制 | 自身绝对位置/速度、上/下游方向、两条本地容量、最近非邻居几何，足以做即时两跳调整和边界/间距反应 | 非邻接链路容量/SNR、远端 relay 与端点状态、使命 waypoint、global outage history |

因此，“没有 Role”不等于 Actor 完全不知道方向：upstream/downstream 的**有序槽位**明确提供了局部拓扑方向，两个相邻 hop 的容量也提供了本地链路质量。可是它没有可确定地辨认 H/L 身份或自己的链序号；任何这类辨认只能从连续数值的统计模式中推断，不能由接口保证。

## 5. observation aliasing：可证明的部分与尚不能证明的部分

### 已由代码证明：Actor observation 对真实环境状态非单射

`get_global_state()` 含 H/L 的 position、velocity、**waypoint、cruise_speed**，所有 relay position/velocity，以及 **step、sim_time、consecutive_outage_steps**。其中 waypoint、cruise speed、全局 outage counter 和大部分远端节点信息不参与 `_get_observations()`。因此，存在不同全局状态被映射到相同的 26 维 local observation；对 Actor 而言，这不是完整 Markov state。

一个代码级构造如下（数值只为说明，未声称它是第五轮 trace 中的逐帧样本）。令 focal relay 在 `P=(1000,1000,200)`、自身速度为零；其 upstream 在 `P+(-500,0,0)`、downstream 在 `P+(500,0,0)`，对应速度和两个 500 m 的容量完全相同；最近非通信节点也设置为相同的相对位置/速度。

| 状态 | focal 角色与相同 local fields | 未进入 local observation 的全局差别 | 长期控制含义 |
| --- | --- | --- | --- |
| A | focal=R1；upstream 槽位实际是 H，downstream 是 R2；上述 26 个数值固定 | H 的 waypoint / cruise speed、R3/R4/L 的状态和远端链路 | R1 需要为外生 H 的下一步移动及整条下游可用余量留出跟踪空间 |
| B | focal=R2；upstream 实际是 R1，downstream 是 R3；上述 26 个数值逐项相同 | H、R4/L 的位置/速度/waypoint 与远端瓶颈可不同 | R2 需要在两个可控 relay 间平衡，并可能为远端 bottleneck 协同调整 |

两个状态在 Actor 输入上完全相同：节点类型并非字段，链路容量只由相同的局部几何得出，safe 节点也没有 identity。更直接地，对同一个 R1，保持当前 H 的 position/velocity 不变而改变其未观测 waypoint，可使 `_advance_mobile()` 的未来加速度不同；保持当前 local 几何不变而把 `consecutive_outage_steps` 从 0 改到 24，也会改变再发生一个 outage step 的终止紧迫性。这是严格的状态/转移别名证据。

### 尚不能由当前代码或五轮数据严格证明

没有求解该 POMDP 的最优策略，且连续状态在真实采样轨迹中很少精确相等。因此不能严格证明“上表的 A 与 B 在每一个配置下必定要求不同的最优即时动作”，也不能证明 Plain MAPPO 数学上必定失败。严谨结论是：当前接口允许有不同未来动力学、远端奖励后果和终止风险的状态落到同一个 Actor 输入；共享 reactive policy 只能学习这些冲突情形上的折中动作。

## 6. 全局信息、centralized Critic 与参数共享的边界

训练时 Critic 使用显式 47 维 global state，并可见 H/L waypoint/cruise speed、全 relay position/velocity 与 outage counter。它能够给团队 value 提供低方差 baseline 和更好的训练期 credit signal；但部署 Actor 仍严格是 `a_i=f(o_i)`。Critic 不输出消息、不改变 Actor 输入，也无法在执行时告诉 R2 远端 R4–L 已接近 outage。因此 centralized training 不能从信息论上补回 decentralized execution 时缺失的状态。

参数共享本身不是错误：它减少参数、聚合四架 relay 的样本，并为 K=3/5 Actor 推理兼容性提供基础。风险在于组合关系：若同一或近似局部向量在不同链位置对应不同的长期责任，所有梯度都会更新同一 `f`，没有 role-conditioned 分支可分离这些行为。五轮中“数值稳定但整链仍频繁断开”与这一风险一致，但不是参数共享单独有害的证明。

## 7. 第五轮 outage 与结构假设的一致性

第五轮 E 的 19 个 persistent terminal events 首次 onset 分布为：H→R1=5、R1→R2=6、R2→R3=1、R3→R4=1、R4→L=6。平均 onset step=166.3（范围 104–248），onset action saturation=0，所有 deterministic replay action saturation=0。

这说明失败并非单独的端点追踪、tanh 饱和或上界方差问题：所有五条 hop 都可以成为首先断开的 bottleneck，尤其两端加上 R1–R2 覆盖了 17/19 个首次事件。该模式与“每个 Actor 只按两条相邻 hop 作反应、不能直接获知全链最弱点或远端移动趋势”相一致；它不是结构信息不足的单独因果证明，但与第五轮已排除的数值问题共同支持把结构作为下一假设。

## 8. Role 与 Graph 各自可能补充什么（仅理论分析）

| 机制 | 主要补足的信息 | 不会自动解决的内容 |
| --- | --- | --- |
| Role / Agent ID / normalized chain index | 端部或中部的逻辑职责、到 H/L 的链序、让同一局部几何可产生 role-conditioned 行为 | 远端节点的实时状态、全局 bottleneck、端点未来 waypoint |
| Graph / message passing | 多跳拓扑、邻居状态和容量经过消息传递形成的更大感受野、远端 bottleneck 与链形协调线索 | 若节点特征仍无角色语义，不能保证端/中职责可辨；有限 message-passing 层数也未必覆盖全链 |

二者互补而不相同：Role 主要解决“我在链中是谁/应承担什么职责”，Graph 主要解决“远端发生了什么/全链如何协调”。二者都只是待验证实验假设，不是本诊断中已被证明一定有效的修复；本轮没有实现其中任何一个。

## 9. 最终状态与下一阶段建议

在冻结 Environment v1.1、26 维 local observation、parameter-shared Actor、无 Role/Agent ID/Graph 的设定下，经历五轮受控训练、best-selection 复核、stochastic/deterministic 复核和 Actor 分布修正后，Plain MAPPO 未达到稳定任务完成标准。其工程实现和数值训练路径有效，故应保留为 **VALID WEAK BASELINE**，而不是把训练程序成功运行误写为性能 PASS。

建议进入下一阶段的**结构实验设计/诊断**，优先把 Role 与 Graph 视为两个可区分、可消融的假设；在获得明确授权和独立任务书之前，不实现 Role、Graph 或其组合，也不开始第六轮 Plain MAPPO 训练。
