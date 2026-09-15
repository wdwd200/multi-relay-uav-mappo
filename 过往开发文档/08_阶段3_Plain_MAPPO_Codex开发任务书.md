---
title: 阶段3 Plain MAPPO Codex开发任务书
project: 多中继无人机通信安全协同位置优化
stage: Stage 3
status: ready_for_implementation
environment: Environment v1.1 frozen
updated: 2026-09-07
---

# 阶段3 Plain MAPPO Codex开发任务书

> 本任务书用于在已经冻结的 Environment v1.1 上实现、测试并初步训练 Plain MAPPO。
>
> 当前阶段只建立可靠的普通 MAPPO 基线，不实现论文后续的 Graph、Role 或 Role+Graph 算法。

---

# 1. Codex 开始前必须读取

开始修改代码前，依次读取：

1. 本任务书；
2. `07_阶段2.5最终验收与阶段3启动交接.md`；
3. 用户提供的最新项目压缩包及实际代码；
4. `00_论文当前总体方案与决策记录_阶段2验收版.md`；
5. `04_阶段1代码与自检数据交接.md`；
6. `03_仿真开发阶段2_学习交接.md`。

若旧文档与本任务书冲突：

- Environment v1.1 的物理参数、接口和冻结范围，以 `07` 和实际已验收代码为准；
- Stage 3 的 MAPPO 训练设计，以本任务书为准；
- 不得用旧候选参数覆盖新冻结值。

修改前先运行现有 Stage 1、Stage 2、Stage 2.5 测试并记录基线结果。不要在未阅读实际目录结构的情况下机械创建重复模块。

---

# 2. 本阶段唯一目标

实现一套可运行、可测试、可保存、可恢复、可独立评估的 Plain MAPPO，回答：

> 在固定的 Environment v1.1 上，普通 MAPPO 能否学习到基本的动态多中继协同位置调整策略？

本阶段主训练固定：

$$
K=4
$$

代码仍需对 `K=3/4/5` 的环境接口保持兼容，但 `K=3/5` 只做 Actor 推理和接口冒烟测试，不做动态 Agent 数量训练，也不参与 `best.pt` 的选择。

---

# 3. 严格禁止扩展的内容

本阶段不得加入：

- Graph、GNN、GAT 或任何消息传递网络；
- Role、Agent ID、位置序号编码或角色 one-hot；
- Role+Graph；
- CBF/HOCBF 或其他 Safety Filter；
- World Model、辅助预测网络；
- 动态增加或删除 Agent；
- 动态路由、中继缓存、功率控制；
- 新 Observation 特征；
- 新 Reward 项或 Reward 权重重调；
- 通信参数、天线公式或 Environment v1.1 动力学重调；
- Stable-Baselines3、RLlib 等黑盒强化学习训练框架。

允许使用 PyTorch、NumPy、标准库及项目已有依赖，MAPPO 核心训练流程必须在项目中明确实现，便于后续论文解释和扩展。

---

# 4. Environment v1.1 冻结边界

必须复用现有 `RelayEnv`，不得复制出一套训练专用环境。

核心接口：

- `reset(seed=...)`：返回形状为 `[K, 26]` 的局部 Observation 和 `info`；
- `step(actions)`：接收形状为 `[K, 3]` 的归一化连续动作；
- `get_global_state()`：返回 centralized critic 使用的未归一化全局状态；
- `get_raw_obs()`：仅允许用于只读诊断和统计，禁止作为 Actor 输入；
- `step()` 返回共享团队奖励、`terminated`、`truncated` 和 `info`。

主训练 `K=4` 时必须验证：

- 单个 Agent 局部 Observation 为 26 维；
- centralized critic 全局状态为 60 维；
- 若实际代码不满足上述维度，立即停止并报告冲突，不得静默删减或填充环境信息。

冻结参数包括但不限于：

| 项目 | 冻结值 |
|---|---:|
| 时间步长 | 0.2 s |
| Episode | 500 steps / 100 s |
| 主训练中继数量 | K=4 |
| Relay 水平速度上限 | 20 m/s |
| Relay 垂直速度范围 | -5～6 m/s |
| Relay 最大水平加速度模长 | 2 m/s² |
| Relay 最大垂直加速度绝对值 | 2 m/s² |
| outage SNR 门限 | 5 dB |
| persistent outage | 25 steps |
| 安全距离 | 20 m |
| 预警距离 | 60 m |
| Reward 参考速率 | 6 Mbps |

Stage 3 不得修改 `relay_env/communication.py` 中已经验收的仰角天线模型，也不得改变固定逻辑链：

$$
H\rightarrow R_1\rightarrow R_2\rightarrow\cdots\rightarrow R_K\rightarrow L
$$

---

# 5. 推荐代码组织

先适配实际项目布局。若不存在同类模块，可采用：

```text
plain_mappo/
  __init__.py
  config.py
  networks.py
  normalization.py
  rollout_buffer.py
  trainer.py
  checkpoint.py
  evaluation.py
  metrics.py

train_plain_mappo.py
evaluate_plain_mappo.py
stage3_plain_mappo_validation.py

selfcheck/
  test_mappo_networks.py
  test_mappo_actions.py
  test_mappo_buffer.py
  test_mappo_gae.py
  test_mappo_update.py
  test_mappo_checkpoint.py
  test_mappo_evaluation.py
```

文件名可以按现有项目风格调整，但职责必须清晰，Actor、Critic、buffer、GAE、PPO 更新、评估和 checkpoint 不得全部堆进一个脚本。

---

# 6. 参数共享 Actor

所有 Relay 共用同一套 Actor 参数：

$$
\pi_\theta(\cdot\mid o_{i,t})
$$

要求：

- 输入仅为第 $i$ 架 Relay 的 26 维局部 Observation；
- 不输入全局状态；
- 不输入 Agent ID、Role 或图结构；
- 对 `[B, K, 26]` 输入可批量输出 `[B, K, 3]` 的分布参数；
- 所有 Agent 的梯度共同更新同一套 Actor。

第一版网络使用普通 MLP，默认配置：

```text
26 → 128 → 128 → mean_head(3)
                 → log_std_head(3)
```

- 隐藏层激活函数使用 `tanh`；
- `mean` 不做动作边界裁剪；
- `log_std` 做数值稳定限幅，默认范围 `[-5, 2]`；
- 网络尺寸和 `log_std` 范围必须写入配置和 checkpoint；
- 不使用循环网络。

---

# 7. 连续动作分布与动作边界

Actor 对每个 Relay 输出三维对角高斯分布：

$$
z_{i,t}\sim\mathcal N(\mu_\theta(o_{i,t}),\sigma_\theta(o_{i,t}))
$$

其中 `latent_z` 的取值范围为整个实数空间。传给环境的归一化动作是：

$$
u_{i,t}=\tanh(z_{i,t})
$$

因此每一维满足：

$$
-1<u_{i,t}^{(j)}<1
$$

实现要求：

- 训练时从高斯分布采样 `latent_z`；
- 评估时不采样，直接使用 `tanh(mean)`；
- Actor 只输出归一化动作，不在网络内再次乘以 `2 m/s²`；
- 物理加速度映射继续由 Environment v1.1 完成；
- 必须测试 `u_x=u_y=1` 等极端输入仍满足环境的水平联合模长约束；
- 不额外加入 Safety Filter 或动作后处理器。

PPO 保存并比较同一个旧 `latent_z` 在旧、新高斯分布下的概率密度。每个 Agent 的三维 log-probability 为各维之和：

$$
\log\pi(z\mid o)=\sum_{j=1}^{3}\log\mathcal N(z_j;\mu_j,\sigma_j)
$$

更新 Actor 时：

1. 使用 buffer 中保存的旧 Observation；
2. 当前 Actor 根据该旧 Observation 输出新的 `mean` 和 `std`；
3. 把 buffer 中保存的同一个旧 `latent_z` 代入新分布；
4. 计算新的 log-probability；
5. 禁止重新采样一个新 `latent_z` 来计算 PPO ratio。

PPO ratio 使用 latent Gaussian 的 log-probability。对于同一个 `latent_z`，`tanh` 变换的 Jacobian 项在新旧比值中抵消。熵奖励使用 latent Gaussian entropy，并在代码注释和报告中说明这是实现约定。

---

# 8. Centralized Critic

Critic 在训练阶段观察整个系统的全局状态：

$$
V_\phi(s_t)\in\mathbb R
$$

要求：

- 输入为 `get_global_state()` 提供的全局状态；
- `K=4` 时准确输入维度为 60；
- 输出一个团队标量价值，不输出每个 Agent 各自的价值；
- Critic 与 Actor 不共享参数；
- Critic 只在训练和价值评估中使用，部署 Actor 时不需要 Critic。

第一版 Critic 使用普通 MLP：

```text
60 → 128 → 128 → value(1)
```

隐藏层激活函数使用 `tanh`。

`get_global_state()` 当前是未归一化状态。应在 MAPPO 模块外实现训练专用 `RunningMeanStd`：

- 只使用训练环境状态更新统计量；
- Critic 实际接收标准化后的60维状态；
- 标准化结果默认裁剪到 `[-10, 10]`；
- buffer 保存采样当时已经标准化的 global state，保证 PPO 多 epoch 更新时输入不变化；
- 评估阶段冻结统计量，不使用评估数据更新；
- normalizer 状态必须写入 checkpoint。

Actor 对环境返回的26维 Observation 直接使用现有归一化结果，不得再用 global-state normalizer 二次归一化。

---

# 9. 同步并行环境

第一版使用一个训练脚本内的8个独立 `RelayEnv` 对象：

```text
num_envs = 8
```

要求：

- 同一进程、同步采样；
- 不是同时启动8次训练脚本；
- 8个环境共享同一个 Actor 和 Critic；
- 每个环境拥有独立初始状态和随机序列；
- 某个环境结束时，只重置该环境，其他环境继续；
- 环境结束发生在 rollout 中间时，后续仍继续收集，直到每个环境均收集满128步。

默认训练种子：

```text
base_seed = 2026
env_seed(env_id, episode_index)
    = base_seed + env_id + episode_index * num_envs
```

同时设置 Python `random`、NumPy、PyTorch CPU 和 CUDA 随机种子。

---

# 10. Rollout Buffer

每次 PPO 更新前，从8个环境各采集128步：

```text
rollout_length = 128
num_envs = 8
team_time_samples = 1024
```

核心字段至少包括：

| 字段 | 推荐形状 | 说明 |
|---|---|---|
| `local_obs` | `[128, 8, 4, 26]` | 动作发生前的局部观测 |
| `global_state` | `[128, 8, 60]` | 采样时已标准化的全局状态 |
| `latent_z` | `[128, 8, 4, 3]` | 高斯分布中的原始动作 |
| `action_u` | `[128, 8, 4, 3]` | 传给环境的 `tanh(z)` |
| `old_log_prob` | `[128, 8, 4]` | 每个Agent三维log-prob之和 |
| `value` | `[128, 8]` | 旧Critic价值 |
| `next_value` | `[128, 8]` | 对真实下一状态计算的价值 |
| `reward` | `[128, 8]` | 共享团队奖励 |
| `terminated` | `[128, 8]` | 真正失败终止 |
| `truncated` | `[128, 8]` | 正常500步时间截断 |
| `bootstrap_mask` | `[128, 8]` | 是否允许使用 `next_value` |
| `trace_mask` | `[128, 8]` | GAE是否能递推到下一采样 |
| `advantage` | `[128, 8]` | 团队 advantage |
| `return_target` | `[128, 8]` | Critic 回归目标 |

使用每个 transition 单独保存 `next_value`，避免环境在 episode 结束后重置，导致把“新 episode 的初始价值”错误当成“旧 episode 的末状态价值”。

buffer 只保存 CPU tensor 或 NumPy array；进入 mini-batch 时再搬到训练设备，避免长期占用 GPU 显存。

---

# 11. terminated、truncated 与 rollout 边界

必须使用两种不同 mask：

## 11.1 Bootstrap mask

只有真正 `terminated=True` 时不 bootstrap：

```text
terminated        → bootstrap_mask = 0
truncated         → bootstrap_mask = 1
普通连续 transition → bootstrap_mask = 1
rollout 最后一项但环境未结束 → bootstrap_mask = 1
```

## 11.2 GAE trace mask

只要当前 transition 后发生 episode reset，就不能把下一个 episode 的 advantage 接进来：

```text
terminated → trace_mask = 0
truncated  → trace_mask = 0
环境继续   → trace_mask = 1
```

因此：

- `terminated`：不 bootstrap，也不跨 episode 递推；
- `truncated`：对旧 episode 的真实末状态 bootstrap，但不跨 reset 递推；
- 单纯 rollout 长度达到128：环境没有结束，既 bootstrap，也允许从 rollout 内的后续 transition 递推。

在调用 `reset()` 前，必须先取得并保存 episode 末状态所需的 `next_value` 和指标。

---

# 12. GAE、Advantage 与 Return

默认参数：

```text
gamma = 0.99
gae_lambda = 0.95
```

TD 残差：

$$
\delta_t=r_t+\gamma b_tV(s_{t+1})-V(s_t)
$$

其中 $b_t$ 是 `bootstrap_mask`。

GAE 从后向前计算：

$$
A_t=\delta_t+\gamma\lambda c_tA_{t+1}
$$

其中 $c_t$ 是 `trace_mask`。

Critic 目标：

$$
R_t^{target}=A_t+V_{old}(s_t)
$$

要求：

- advantage 先按 `[128,8]` 计算；
- 在整个1024个 team-time 样本上做一次均值0、标准差1的标准化；
- 同一个 team-time advantage 广播给当时4个 Relay；
- 不分别为4个 Agent 重新计算4套团队 advantage；
- 必须提供手算小样本测试，分别覆盖 terminated、truncated 和 rollout 边界。

---

# 13. PPO Mini-batch 与 Epoch

每个 rollout 有1024个 team-time 样本。

```text
mini_batch_size = 256 team-time samples
num_mini_batches = 4 per epoch
ppo_epochs = 10
optimizer_updates = 40 per rollout
```

每个 epoch 重新打乱1024个 team-time 索引。

每个包含256个 team-time 的 mini-batch 中：

- Actor 实际处理 `256 × 4 = 1024` 个 agent-time 样本；
- Critic 处理256个全局状态；
- Actor loss 对所有选中 team-time 和 Agent 取平均；
- Critic loss只对256个团队价值取平均；
- 不把4个 Agent 的概率相乘成一个联合 ratio。

---

# 14. PPO Loss

每个 Agent 的概率比：

$$
r_{i,t}(\theta)=
\exp\left(
\log\pi_\theta(z_{i,t}\mid o_{i,t})
-\log\pi_{old}(z_{i,t}\mid o_{i,t})
\right)
$$

裁剪范围：

$$
1-\epsilon\le r_{i,t}\le1+\epsilon,
\qquad \epsilon=0.2
$$

策略目标采用标准 PPO clipped surrogate。代码以最小化 loss 为约定：

$$
L_{actor}
=-operatorname{mean}
\left[
\min\left(
r_{i,t}A_t,
\operatorname{clip}(r_{i,t},0.8,1.2)A_t
\right)
\right]
-c_eH
$$

其中：

```text
entropy_coef = 0.01
```

Critic 使用普通均方误差：

$$
L_{critic}=c_v\operatorname{mean}
\left[
(V_\phi(s_t)-R_t^{target})^2
\right]
$$

```text
value_loss_coef = 0.5
```

Actor、Critic 使用两个独立 Adam：

```text
actor_lr = 3e-4
critic_lr = 3e-4
max_grad_norm = 0.5
```

每个 mini-batch 的更新顺序：

1. 计算 Actor loss；
2. Actor `zero_grad()`；
3. Actor 反向传播；
4. Actor 梯度范数裁剪到0.5；
5. Actor Adam `step()`；
6. 计算 Critic loss；
7. Critic `zero_grad()`；
8. Critic 反向传播；
9. Critic 梯度范数裁剪到0.5；
10. Critic Adam `step()`。

不得把学习率误实现为新旧概率密度比值；学习率只属于 Adam 参数更新。

---

# 15. 首轮冻结训练配置

| 参数 | 值 |
|---|---:|
| `K` | 4 |
| `num_envs` | 8 |
| `rollout_length` | 128 |
| `mini_batch_size` | 256 team-time |
| `ppo_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_epsilon` | 0.2 |
| `actor_lr` | 0.0003 |
| `critic_lr` | 0.0003 |
| `entropy_coef` | 0.01 |
| `value_loss_coef` | 0.5 |
| `max_grad_norm` | 0.5，Actor/Critic分别裁剪 |
| `base_seed` | 2026 |
| `eval_interval_updates` | 50 |
| `periodic_eval_episodes` | 5 |
| `acceptance_eval_episodes` | 20 |
| `smoke_updates` | 10 |
| `full_updates` | 1000 |

所有参数必须集中放入配置对象，并随 checkpoint 和结果报告保存。不得散落成多个脚本中的不可追踪常量。

---

# 16. Checkpoint 与恢复

需要生成：

```text
latest.pt
best.pt
actor_final.pt
```

## 16.1 latest.pt

每完成一次完整 PPO update 后保存并覆盖，用于断点恢复。至少包含：

- Actor 参数；
- Critic 参数；
- Actor Adam 状态；
- Critic Adam 状态；
- global-state normalizer 状态；
- 当前 update 数；
- 总环境交互步数；
- 完整训练配置；
- Python、NumPy、PyTorch CPU/CUDA RNG 状态；
- 当前历史最佳评估记录。

使用临时文件加原子替换，避免中断时损坏 `latest.pt`。

## 16.2 best.pt

仅当固定评估集上的结果优于历史最好结果时覆盖。必须是可恢复的完整 checkpoint，不能只存 Actor。

## 16.3 actor_final.pt

训练结束后导出最终选定 Actor 及部署所需元数据，不含 Critic 和 optimizer。

## 16.4 恢复语义

恢复时加载模型、Adam、normalizer、计数器、配置和 RNG 状态，然后重新创建8个环境继续训练。

Stage 3 不保存环境瞬时内部状态或半截 rollout。因此恢复属于“从相同训练状态继续学习”，不保证与从未中断的运行产生逐位相同的后续轨迹。报告中必须明确这一限制。

---

# 17. 训练日志

至少输出：

```text
train.csv
eval.csv
config.json
```

每个 PPO update 记录：

- `total_env_steps`；
- `update`；
- 完整 episode 的 return 和 length；
- Actor policy loss；
- Critic loss；
- latent Gaussian entropy；
- `approx_kl`；
- `clip_fraction`；
- Actor/Critic 更新前梯度范数；
- advantage 均值与标准差；
- 动作均值、标准差；
- `abs(action_u) > 0.95` 的饱和比例；
- NaN/Inf 检查结果。

日志中不得把 Actor loss 或 Critic loss 简单标成“越小越好”。

---

# 18. 任务物理指标

从现有 `info` 和只读诊断接口统计，不修改 Reward：

- 平均有效端到端速率 `mean_e2e_rate_mbps`；
- 端到端速率达到6 Mbps的时间比例；
- 至少一跳低于5 dB的 outage step 比例；
- 最薄弱链路 margin；
- 正常500步完成率；
- collision、boundary、persistent outage 终止次数；
- 最小空中节点间距；
- 最大水平/垂直速度；
- 最大水平/垂直加速度；
- 速度、加速度硬约束违规次数；
- 平均加速度、动作饱和率和累计移动距离。

若 `info` 的实际键名不同，应在 metrics 适配层读取现有字段；不得为了方便而修改环境动力学或 Reward。

---

# 19. 独立确定性评估

每50个 PPO updates 评估一次：

- 使用独立评估环境；
- 主评估只使用 `K=4`；
- 使用固定且不参与训练的 seeds；
- 周期评估使用 seeds `10000～10004`，共5个完整 episode；
- 评估 Actor 使用 `tanh(mean)`，不进行高斯采样；
- 使用 `torch.no_grad()`；
- 不写入 rollout buffer；
- 不反向传播；
- 不更新 global-state normalizer；
- 不扰动训练 RNG 序列，必要时为评估使用独立 RNG/环境。

最终验收使用 seeds `10000～10019`，共20个 `K=4` episode。

`best.pt` 采用安全优先的字典序。以下元组越小越好：

```text
(
  速度/加速度硬约束违规次数,
  collision + boundary 次数,
  -正常完成次数,
  outage_step_ratio,
  -rate_satisfaction_ratio,
  -mean_e2e_rate_mbps
)
```

不得直接根据训练 reward 或单个 episode 保存 best。

---

# 20. 自动测试要求

至少覆盖以下测试：

## 20.1 网络和共享参数

- Actor 输入输出 shape；
- Critic 输入输出 shape；
- 多个 Agent 确实调用同一 Actor 对象；
- Critic 输出一个团队标量；
- `K=3/4/5` Actor 推理 shape 正确。

## 20.2 动作与概率

- `latent_z` 可取边界外实数；
- `tanh(z)` 始终有限且位于 `(-1,1)`；
- 环境物理加速度满足水平联合约束和垂直约束；
- 三维 log-prob 等于各维 log-prob 之和；
- 同一个旧 `z` 代入新分布得到正确 new log-prob；
- PPO 更新计算中没有重新采样动作；
- 极端 `mean/log_std/z` 不产生 NaN/Inf。

## 20.3 GAE 和边界

使用可手算的小数组分别验证：

- 普通连续 transition；
- `terminated=True` 不 bootstrap；
- `truncated=True` bootstrap，但不跨 reset 递推；
- rollout 在128步处结束但 episode 未结束时正常 bootstrap；
- 一个并行环境提前结束不会污染其他环境；
- 新 episode 的初始价值不会被当成旧 episode 的末状态价值。

## 20.4 PPO 更新

- ratio 为1时的 loss；
- 正 advantage 与负 advantage 的 clipped surrogate；
- `clip_fraction` 数值；
- advantage 广播到4个 Agent；
- team-time mini-batch 与 agent-time 展开关系；
- 一次 update 后 Actor 和 Critic 参数均发生有限变化；
- 梯度裁剪分别作用于 Actor 和 Critic。

## 20.5 Checkpoint 与评估

- checkpoint 保存/恢复后参数一致；
- Adam、normalizer、计数器和 RNG 状态均恢复；
- 相同 Actor 与相同评估 seed 得到相同确定性结果；
- `best.pt` 比较遵循安全优先字典序；
- `latest.pt` 可继续训练至少一个 update。

---

# 21. 运行顺序

Codex 按以下顺序执行：

1. 解压并检查最新代码；
2. 运行并记录现有全部测试；
3. 确认 `RelayEnv` 接口、Observation 26维和 `K=4` global state 60维；
4. 实现配置、Actor、Critic 和 normalizer；
5. 实现同步8环境采样与 rollout buffer；
6. 实现 episode 边界安全的 `next_value`、mask 和 GAE；
7. 实现 PPO mini-batch、10 epochs 和两个 Adam 更新；
8. 实现 CSV 日志、checkpoint 和恢复；
9. 实现独立确定性评估和 `best.pt`；
10. 完成自动测试；
11. 重新运行 Stage 1、Stage 2、Stage 2.5 原回归；
12. 运行10-update 冒烟训练；
13. 若执行预算允许，运行1000-update 完整初训；否则提供准确命令并明确标记“长训练待用户执行”；
14. 生成验证报告和交接文档。

不得把未实际运行的长训练写成 PASS。

---

# 22. 最低验收标准

## 22.1 代码与数值正确性

- 新增测试全部通过；
- Stage 1、Stage 2、Stage 2.5 原测试全部继续通过；
- Environment v1.1 冻结文件没有非必要修改；
- 训练与评估不出现 NaN/Inf；
- `terminated/truncated`、bootstrap 和 GAE 行为通过手算测试；
- `latest.pt`、`best.pt`、`actor_final.pt` 行为正确。

## 22.2 冒烟训练

10个 PPO updates 必须：

- 完整运行；
- Actor/Critic 参数均更新；
- 生成 train/eval 日志；
- 可保存、恢复并再运行至少一个 update；
- 动作、速度和加速度保持有限。

## 22.3 完整初训

1000个 PPO updates 等于：

$$
1000\times8\times128=1{,}024{,}000
$$

条 team-time 环境交互。

使用 `best.pt` 在20个固定 `K=4` episode 上最低满足：

| 指标 | 最低要求 |
|---|---:|
| 速度、加速度硬约束违规 | 0次 |
| 正常完成500 steps | 至少16/20 |
| collision 与 boundary 终止合计 | 不超过2/20 |
| 平均有效端到端速率 | 不低于6 Mbps |
| 相对未训练/随机策略 | 按安全优先排序必须更好 |

若未达到完整初训指标：

- 不得伪造 PASS；
- 保留日志和 checkpoint；
- 判断是实现错误、数值不稳定、训练预算不足还是超参数问题；
- 先提交诊断，再决定是否调参；
- 不得通过修改 Environment v1.1 或 Reward 来掩盖失败。

本阶段不要求 Plain MAPPO 超过 Nominal reference，也不要求证明其优于尚未开发的 Graph/Role 算法。

---

# 23. 必须生成的结果文件

除实际代码和测试外，至少生成：

```text
stage3_plain_mappo_validation.md
stage3_plain_mappo_validation.json
09_阶段3_Plain_MAPPO代码与自检数据交接.md
```

验证报告必须记录：

- 最终代码文件清单；
- 配置；
- 实际执行命令；
- 每组测试的真实 PASS/FAIL 数量；
- 冒烟训练耗时和结果；
- 是否实际完成1000 updates；
- 日志、checkpoint 和评估结果路径；
- 所有未完成项、限制和异常；
- Environment v1.1 是否保持冻结。

交接文档必须让下一个聊天窗口在没有完整训练对话的情况下理解：

- Plain MAPPO 的实际代码结构；
- 完整训练数据流；
- buffer 字段和 tensor shape；
- GAE 与 episode 边界实现；
- 保存/恢复语义；
- 已跑测试和真实训练结果；
- 下一阶段是否可以开始。

Codex 最终回复中必须明确写出上述结果文档的实际文件名，不能只说“报告已生成”。

---

# 24. Codex 最终回复格式

最终回复保持简洁，至少包含：

1. 实现了什么；
2. 修改/新增的主要文件；
3. 实际运行了哪些命令；
4. 测试和训练的真实结果；
5. 是否达到 Stage 3 最低验收标准；
6. 未完成项或风险；
7. 结果报告与交接文档的准确文件名。

不要用“应该可以”“理论通过”替代实际运行结果。

---

# 25. 一句话执行边界

$$
\boxed{
\text{保持 Environment v1.1 不变}
\rightarrow
\text{实现并验证 K=4 Plain MAPPO 基线}
}
$$
