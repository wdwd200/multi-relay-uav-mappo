# Stage 3 Plain MAPPO 最终验证

日期：2026-09-09  
最终状态：**Stage 3 Plain MAPPO baseline = COMPLETE**  
分类：Engineering correctness = **PASS**；PPO numerical stability = **PASS**；Task-performance acceptance = **FAIL**；Baseline status = **VALID WEAK BASELINE**。

## 最终第五轮证据

第五轮使用全新随机初始化的 `state_independent_tanh` Actor，从 `artifacts/stage3-stateindependent-full/` 实际完成：

- 1,000 PPO updates / 1,024,000 team-time samples；`train.csv` 1,000/1,000 行 finite。
- 8 个同步环境、128 rollout、256 team-time mini-batch、10 PPO epochs、`gamma=0.99`、`gae_lambda=0.95`、`clip=0.2`。
- `entropy_coef=0`、Actor LR=`1e-4`、Critic LR=`3e-4`；唯一相对第四轮的结构变化为 shared state-independent smooth-bounded `log_std`。
- 每 50 updates 的 5-episode deterministic evaluation 共 20 次，保存 20 个轻量 Actor checkpoint。
- `best.pt` 为 update 650；`latest.pt`、`best.pt`、`actor_final.pt`、train/eval/config 与 20-seed evaluation 均已保存。

新的三维 shared effective `log_std` 始终在 `(-4,0)` 内，action saturation 为 0，最后 200 updates 的 approx KL=0.00603、clip fraction=0.06677，无 NaN/Inf。旧 state-dependent head 的大规模 raw 越界/hard-clamp 饱和已被消除。

## 固定 20-seed 验收

`best.pt` 在 K=4、seeds 10000–10019、deterministic `tanh(mean)` 下的真实结果：

| 指标 | 结果 | 验收 |
| --- | ---: | --- |
| speed/acceleration hard violations | 0 | PASS |
| normal 500-step completion | 0/20 | **FAIL**（要求 >=16/20） |
| collision + boundary | 0 + 1 = 1/20 | PASS |
| persistent outage | 19/20 | FAIL 的主要模式 |
| mean e2e rate | 6.103710 Mbps | PASS |
| outage step ratio / rate satisfaction | 0.129161 / 0.775567 | 记录 |
| minimum node distance | 29.801 m | 记录 |
| action saturation | 0.000000 | 记录 |

同 seeds 的 uniform-random baseline 的 completion-first score 为 `[0,0,12,8,0.073432,-0.900185,-6.596517]`；第五轮为 `[0,0,1,19,0.129161,-0.775567,-6.103710]`，因此在该安全优先字典序上优于 random，但未达到可靠任务完成线。

## 全局状态与冻结边界

centralized Critic 的实际 K=4 global state 是 **47 维**，公式为 `23 + 6*K`；原先 60 维设计已正式废弃。Actor 仍只接收原始冻结的 26 维 local observation。Environment v1.1 的动力学、Reward、Observation、通信/天线、outage、终止和安全逻辑均未改变。

## 测试与回归

| 验证 | 结果 |
| --- | --- |
| Stage 3 MAPPO tests | 28/28 PASS |
| Stage 1 selfcheck | 13/13 PASS |
| Stage 2 hard checks | 17/17 PASS |
| Stage 2.5 antenna tests | 5/5 PASS |
| Stage 2.5 hard checks | 17/17 PASS |
| checkpoint/resume / next-unused seed progress | PASS |

## 结构能力诊断结论

Actor 的 26 维输入提供有序 upstream/downstream 的局部几何、相对速度与相邻 hop 容量，因此不是完全无拓扑信息；但不含邻居身份、Role、chain index、远端链路状态、H/L waypoint/cruise speed 或 global `consecutive_outage_steps`。Critic 虽在训练期间使用 47 维全局状态，但部署时不会向 Actor 传递该信息。

第五轮的首次 persistent-outage onset 分布覆盖 H→R1=5、R1→R2=6、R2→R3=1、R3→R4=1、R4→L=6，且 onset saturation=0。这与“仅由局部两跳反应的共享 Actor 缺乏全链协调信息”的假设一致。代码可证明 Actor observation 对全局状态非单射；但没有求解 POMDP 最优策略，故不声称 Plain MAPPO 数学上不可能成功。

Stage 3 由此按结论 B 收口：保留 Plain MAPPO 为有效但弱的基线；下一阶段应先以独立授权的 Role/Graph 结构假设设计和消融来验证，而不是继续自动调 Plain MAPPO 超参数或直接实现新算法。

详细训练结果见 `15_阶段3_Plain_MAPPO_stateindependent正式训练结果.md`；字段映射、aliasing 证据和结构诊断见 `16_阶段3_Plain_MAPPO结构能力诊断与最终收口.md`。
