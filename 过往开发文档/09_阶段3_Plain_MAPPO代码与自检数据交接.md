# 阶段 3 Plain MAPPO 代码与自检数据交接

状态：**Stage 3 Plain MAPPO baseline = COMPLETE**  
分类：Engineering correctness = **PASS**；PPO numerical stability = **PASS**；Task-performance acceptance = **FAIL**；Baseline status = **VALID WEAK BASELINE**。

## 已交付的 Plain MAPPO 基线

- shared Plain Actor、47 维 centralized Critic、8 个同步训练环境、128-step rollout、GAE、terminated/truncated mask、10-epoch PPO、独立 Actor/Critic Adam、checkpoint/resume、deterministic evaluation 和周期 Actor checkpoint 已实现。
- Actor 永远只使用冻结的 26 维 local observation；Critic 使用显式顺序 flatten 的 global state，维度公式为 `23 + 6*K`，K=4 为 **47**。原任务书的 60 维假设已废弃。
- checkpoint 保存 `next_episode_indices`；恢复时每个环境使用下一枚未用 seed，而不保存环境瞬时状态或半截 rollout。
- 最新 Actor 分布是可选的 `state_independent_tanh`：所有 relay/observation 共享一组三维 learnable `theta`，`log_std = -2 + 2*tanh(theta)`，范围 `(-4,0)`。历史 A–D `state_dependent_clamp` checkpoint 仍兼容。

## 最终第五轮正式训练

目录：`artifacts/stage3-stateindependent-full/`

- 从全新随机初始化实际完成 **1,000 updates / 1,024,000 team-time samples**。
- 配置：K=4、num_envs=8、rollout=128、mini_batch=256、PPO epochs=10、gamma=0.99、GAE lambda=0.95、clip=0.2、Actor LR=1e-4、Critic LR=3e-4、entropy=0、base_seed=2026。
- 仅相对第四轮改变 Actor 方差参数化；Environment v1.1、Reward、Observation、Critic、PPO 组织、通信、天线与安全规则保持冻结。
- 产物：`train.csv`（1,000 finite rows）、`eval.csv`（20 条周期评估）、`config.json`、`checkpoints/latest.pt`、`best.pt`（update 650）、`actor_final.pt` 与 `eval_checkpoints/actor_update_0050.pt` 至 `actor_update_1000.pt`。
- 无逐 step raw JSONL；第五轮训练无 NaN/Inf。新 shared `log_std` 未饱和，rollout action saturation=0，后 200 updates KL=0.00603、clip fraction=0.06677。

## 最终验收

`best.pt` 在 seeds 10000–10019 的 deterministic 20-episode 结果：

| 指标 | 结果 |
| --- | ---: |
| normal completion | **0/20** |
| persistent outage | 19 |
| collision / boundary | 0 / 1 |
| speed/acceleration hard violations | 0 |
| mean e2e rate | 6.103710 Mbps |
| outage step ratio / rate satisfaction | 0.129161 / 0.775567 |
| min node distance | 29.801 m |
| action saturation | 0.000000 |

因此速度/加速度、collision+boundary、mean-rate 和 completion-first safety ordering versus random 均通过；normal completion 线（至少 16/20）失败。训练程序成功运行不能写成任务性能 PASS。

## 已验证回归

- Stage 3 MAPPO tests：28/28 PASS。
- Stage 1：13/13 PASS。
- Stage 2 hard checks：17/17 PASS，默认未写 raw JSONL。
- Stage 2.5 antenna tests：5/5 PASS；hard checks：17/17 PASS。
- Environment v1.1 物理模型保持冻结。

## 最终结构判断与后续边界

五轮受控实验已经排除或显著削弱了 entropy、state-dependent `log_std` 饱和、Actor LR、best checkpoint 选择、stochastic/deterministic mismatch、单一端点问题和 PPO 数值发散作为主要解释。第五轮 persistent-outage 首次 onset 覆盖五条 hop，且 onset action saturation=0。

当前 26 维 Actor observation 含有序 upstream/downstream 两跳几何、速度和容量，故不是无拓扑信息；但缺少节点 identity、Role、chain index、远端链路/节点状态、H/L waypoint/cruise speed 与 global outage counter。Actor observation 对真实全局状态非单射，centralized Critic 又只在训练期可见全局信息，不能在部署时弥补该缺口。

Plain MAPPO 因此作为 **VALID WEAK BASELINE** 收口。下一阶段只建议在独立授权后进行 Role/Graph 的结构假设与消融设计；当前不自动开始第六轮 Plain MAPPO、Role、Graph、Role+Graph 或 CBF 开发。

详细证据：

- `15_阶段3_Plain_MAPPO_stateindependent正式训练结果.md`
- `16_阶段3_Plain_MAPPO结构能力诊断与最终收口.md`
- `stage3_plain_mappo_validation.md` 与 `stage3_plain_mappo_validation.json`
