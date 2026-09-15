# Stage 4.1 训练与评估协议修复任务书

日期：2026-09-15  
执行对象：Codex  
任务性质：正式训练前的协议修复、旧检查点重评估与门禁验收

---

# 0. 当前状态

当前项目状态必须按以下口径理解：

| 项目                 | 状态                           | 准确含义                                          |
| -------------------- | ------------------------------ | ------------------------------------------------- |
| Stage 1 / 2 / 2.5    | PASS                           | Environment v1.1、通信与天线模型已冻结            |
| P0 Plain MAPPO       | COMPLETE / VALID WEAK BASELINE | 工程正确，但当前单次训练未通过任务性能参考线      |
| P1 Role-info         | 第一轮训练完成                 | 只能视为单训练种子的探索结果                      |
| P2 Role-head         | 第一轮训练完成                 | 只能视为单训练种子的探索结果                      |
| P3 Topology-info MLP | 工程实现完成                   | 自动测试和 10-update smoke 已完成；正式训练未开始 |
| P4 Graph MAPPO       | 工程实现完成                   | 自动测试和 10-update smoke 已完成；正式训练未开始 |
| P5 / P6 / Role+Graph | NOT STARTED                    | 本任务不得实现或训练                              |

P3/P4 已有工程交接表明：

```text
Stage 4 topology/graph implementation = COMPLETE
Engineering validation = PASS
P3 formal training = NOT STARTED
P4 formal training = NOT STARTED
Role+Graph = NOT STARTED
```

因此，本任务不是继续开发 P3/P4，也不是直接启动正式训练，而是先修复会影响论文结论可信度的实验协议。

---

# 1. 开始前必须读取

Codex 必须先读取并对齐：

1. `20_Stage4_Role收口与Graph启动交接.md`
2. `19_阶段4_Graph消融代码与自检交接.md`
3. `18_阶段4_Role第一轮正式训练结果.md`
4. `00_论文当前总体方案与决策记录_阶段3收口版.md`
5. `16_阶段3_Plain_MAPPO结构能力诊断与最终收口.md`
6. 当前源码中的 `AGENTS.md`
7. `plain_mappo/config.py`
8. `plain_mappo/trainer.py`
9. `plain_mappo/evaluation.py`
10. `plain_mappo/metrics.py`
11. `plain_mappo/checkpoint.py`
12. `train_plain_mappo.py`
13. `evaluate_plain_mappo.py`

若文档与实际源码或训练产物冲突，以可复核的源码和产物为准，并在最终交接中列出冲突，不得静默沿用过时描述。

---

# 2. 本任务的唯一目标

建立一套可用于 P0–P4 公平比较的正式实验协议，并完成以下工作：

1. 修复验证集与最终测试集重叠的问题；
2. 使用全新、独立的验证种子重评 P0/P1/P2 的全部周期检查点；
3. 仅用验证集重新选择检查点，再用从未参与选择的测试集评估；
4. 保存逐 Episode 原始结果，而不只保存汇总值；
5. 为未来多训练种子实验拆分并冻结随机数流；
6. 补强 Critic、梯度裁剪和价值学习诊断；
7. 明确当前 checkpoint resume 的真实语义；
8. 生成 P0–P4 正式训练的命令矩阵和验收门槛，但不执行正式训练。

本任务完成后，才能决定是否授权 P0–P4 的多训练种子正式实验。

---

# 3. 为什么必须先修复协议

## 3.1 单训练种子不足

P0 第五轮、P1 和 P2 当前都只有一次独立训练轨迹。`20` 个 evaluation environment seeds 只能衡量一个已训练策略在不同场景下的表现，不能替代多个独立的训练随机种子。

因此，当前结果只能支持：

> 某一次训练运行产生了这些结果。

不得支持：

> 某种 Actor 结构稳定优于另一种结构。

## 3.2 验证集与测试集发生重叠

当前训练每 50 updates 都使用固定 seeds `10000–10004` 选择 `best.pt`，最终 20-seed 评估使用 `10000–10019`。前 5 个场景同时参与模型选择和最终报告，最终测试不再完全独立。

对现有结果拆分后可见明显敏感性：

| 方法          | 参与选择的 5 seeds              | 其余 15 seeds                    |
| ------------- | ------------------------------- | -------------------------------- |
| P1 update 550 | boundary 3，persistent outage 2 | boundary 14，persistent outage 1 |
| P2 update 600 | boundary 2，persistent outage 3 | boundary 2，persistent outage 13 |

P2 的失败类型在两个子集上差异很大，说明当前 `best.pt` 选择对少量固定场景敏感。

## 3.3 Critic 长期处于强裁剪区

现有训练日志中：

| 项目                             |     P1 |     P2 |
| -------------------------------- | -----: | -----: |
| Critic pre-clip grad norm 均值   | 31.312 | 34.609 |
| Critic pre-clip grad norm 最小值 |  8.742 | 12.762 |
| `max_grad_norm`                  |    0.5 |    0.5 |

现有代码只归一化 Critic 输入，没有记录 explained variance，也未归一化 value target。当前现象不证明 Critic 错误，但必须在正式训练前可观测、可解释。

## 3.4 折扣范围与任务时长可能不匹配

当前 `dt=0.2 s`、每 Episode 最长 `500` steps，即 `100 s`；`gamma=0.99` 的回报半衰期约为 `13.8 s`，第 500 step 的权重约为 `0.00657`。同时 `(gamma × lambda)=0.9405`，GAE 的直接影响衰减更快。

这不是已确认的代码错误，也不授权本任务修改 gamma 或 Reward；但必须形成正式诊断和后续决策门槛。

## 3.5 Resume 不是逐状态精确续训

当前 checkpoint 未保存正在运行的环境内部状态、`current_obs`、当前 Episode return 和 length。恢复时会从尚未使用的新环境 seed 开始新 Episode。

因此当前语义只能写成：

> 在 PPO update 边界恢复模型、优化器、normalizer、部分 RNG 和 seed progress，并以新 Episode 继续。

不得写成：

> 与不中断运行逐步一致的 bitwise-exact continuation。

P1 的已报告最佳检查点位于 update 550，早于 update 740 后发生的恢复，因此该最佳检查点本身不受恢复事件影响；但未来正式实验仍需冻结统一规则。

---

# 4. 全程冻结项

本任务不得修改：

- Environment v1.1 的动力学；
- 通信、天线与 capacity 公式；
- Reward 组成及权重；
- outage / persistent-outage 定义；
- collision、boundary、速度和加速度约束；
- terminated / truncated 语义；
- 26 维 local observation；
- 47 维 centralized Critic 输入定义；
- P0–P4 Actor 已冻结的结构；
- P3/P4 topology node、edge 和 message passing 定义；
- PPO old latent `z` 复用原则；
- shared team reward；
- 当前 safety-priority checkpoint 排序规则。

允许修改的范围仅包括：

- 实验 seed 管理；
- 评估、检查点选择和结果落盘协议；
- 训练诊断日志；
- checkpoint 元数据与 resume 语义标记；
- 自动测试；
- Stage 4.1 专用评估脚本、报告和 manifest。

---

# 5. 子任务 A：冻结三分离数据协议

必须明确区分：

| 数据域                  | 用途                   | 是否可参与 checkpoint 选择 |
| ----------------------- | ---------------------- | -------------------------- |
| Training environments   | PPO rollout 和参数更新 | 否                         |
| Validation environments | 周期检查点排序与选择   | 是                         |
| Final test environments | 最终一次独立报告       | 否                         |

## 5.1 Stage 4.1 旧检查点重评种子

本轮直接冻结：

```text
validation seeds = 50_000_000 ... 50_000_019  （20 episodes）
final test seeds = 60_000_000 ... 60_000_049  （50 episodes）
```

要求：

1. 两组种子严格不重叠；
2. 两组种子不得与现有训练实际使用的环境种子重叠；
3. P0、P1、P2 必须使用完全相同的 validation seeds；
4. P0、P1、P2 必须使用完全相同的 final test seeds；
5. 不得查看 test 结果后重选 checkpoint；
6. seed 列表必须在首次 test 前写入只读 manifest；
7. manifest 必须记录协议版本和生成时间。

## 5.2 未来正式训练的随机数域

为未来 P0–P4 多训练种子实验预留 5 个 paired run：

```text
run_seed = 2026, 2027, 2028, 2029, 2030
```

每个 `run_seed` 必须确定性派生并保存以下独立随机数域：

- Actor initialization seed；
- Critic initialization seed；
- rollout action-noise seed；
- training-environment seed base；
- mini-batch permutation seed。

同一 `run_seed` 下，P0–P4 必须使用相同的 Critic 初始化、action-noise 流、training environment seeds 和 mini-batch permutation 流。不同 Actor 结构不得通过额外消耗全局 RNG，间接改变 Critic 初始化或 rollout 噪声。

建议预留 training-environment seed base：

```text
run 0 → 10_000_000
run 1 → 12_000_000
run 2 → 14_000_000
run 3 → 16_000_000
run 4 → 18_000_000
```

以当前 1000 updates、128 steps/update、8 environments 计算，即使每一步都结束一个 Episode，每个 run 的 seed 范围仍不会进入下一个保留区间。

具体字段名可由实现决定，但以下内容必须序列化到 config 和 checkpoint：

```text
protocol_version
run_seed
actor_init_seed
critic_init_seed
action_noise_seed
minibatch_seed
train_env_seed_base
validation_seed_manifest
final_test_seed_manifest
```

旧 checkpoint 缺少这些字段时必须保持兼容，并明确标记为 `legacy_protocol`；不得伪造它们曾使用新协议。

---

# 6. 子任务 B：实现独立 checkpoint-grid 评估

新增一个单独脚本，建议命名：

```text
reevaluate_checkpoint_grid.py
```

脚本至少支持：

```text
--run-dir
--checkpoint-glob
--seed-manifest
--split validation|test
--output-dir
--device
```

## 6.1 Validation 阶段

对以下三个现有运行分别评估其全部 20 个周期 Actor checkpoint：

```text
P0 Stage 3 第五轮 state-independent run
P1 artifacts/stage4-role-info-full/
P2 artifacts/stage4-role-head-full/
```

检查点范围固定为：

```text
actor_update_0050.pt
actor_update_0100.pt
...
actor_update_1000.pt
```

每个 checkpoint 都使用同一组 20 个 validation seeds，并保存：

- 每个 Episode 的完整原始指标；
- checkpoint update；
- Actor variant；
- checkpoint SHA-256；
- config 摘要；
- seed split 名称和 seed；
- `tanh(mean)` deterministic action 声明；
- safety-priority score tuple；
- 汇总指标。

## 6.2 检查点选择

继续使用当前冻结的 `safety_priority_key`：

```text
1. speed/acceleration hard violations 越少越好
2. normal completion 越多越好
3. collision + boundary 越少越好
4. persistent outage 越少越好
5. outage ratio 越低越好
6. rate satisfaction 越高越好
7. mean e2e rate 越高越好
```

若 score 完全相同，选择更早的 update，避免引入事后偏好。

选定后必须先生成：

```text
selected_checkpoints.json
```

其中记录 P0/P1/P2 的：

- selected update；
- checkpoint path；
- checkpoint hash；
- validation score；
- selection rule；
- validation seed manifest hash；
- 选择时间。

不得覆盖或删除原有 `best.pt`、`actor_final.pt`、`eval.csv` 或周期 checkpoint。

## 6.3 Final test 阶段

只有 `selected_checkpoints.json` 写入完成后，才允许对每种方法唯一选中的 checkpoint 运行 50-episode final test。

禁止：

- 在 test split 上评估全部 20 个 checkpoint；
- 根据 test 结果更换 selected update；
- test 后修改 safety-priority key；
- 丢弃不利 Episode；
- 只保存 summary 而不保存原始 Episode 数据。

---

# 7. 子任务 C：结果文件格式

新增目录：

```text
artifacts/stage4-1-protocol-repair/
```

至少生成：

```text
seed_manifest.json
checkpoint_manifest.json
validation_grid.csv
validation_episodes.jsonl
selected_checkpoints.json
final_test_summary.csv
final_test_episodes.jsonl
protocol_audit.json
protocol_audit.md
```

## 7.1 `validation_grid.csv`

每行对应一个 checkpoint，至少包含：

```text
variant
run_seed
update
checkpoint_path
checkpoint_sha256
validation_seed_set
episodes
score
normal_completion_count
collision_count
boundary_count
persistent_outage_count
speed_accel_violations
outage_step_ratio
rate_satisfaction_ratio
mean_e2e_rate_mbps
mean_return
```

## 7.2 `validation_episodes.jsonl`

每行对应一个真实 Episode，至少包含：

```text
variant
update
checkpoint_sha256
split
seed
return
length
terminated
truncated
termination_reason
mean_e2e_rate_mbps
rate_satisfaction_ratio
outage_step_ratio
min_node_distance_m
max_xy_speed_mps
max_xy_accel_mps2
action_saturation_ratio
movement_distance_m
```

字段名应复用现有 `EpisodeMetrics` 的真实输出；若实际名称不同，以源码为准并在 schema 中说明，不得静默编造缺失字段。

## 7.3 可追溯性

所有 Stage 4.1 输出必须记录：

- Python、PyTorch、NumPy 版本；
- device；
- git commit（若仓库存在）；
- dirty worktree 状态；
- Environment/Actor/Critic config；
- 输入 checkpoint 路径、大小和 SHA-256；
- seed manifest SHA-256；
- 执行命令；
- 开始和结束时间。

---

# 8. 子任务 D：补强 Critic 与梯度诊断

只增加诊断，不改变 PPO 优化目标。

`train.csv` 至少新增：

```text
value_target_mean
value_target_std
value_prediction_mean_pre
value_prediction_std_pre
explained_variance_pre
actor_grad_norm_max
critic_grad_norm_max
actor_grad_clip_fraction
critic_grad_clip_fraction
```

其中：

```text
explained_variance = 1 - Var(return_target - value_prediction) / Var(return_target)
```

若 `Var(return_target)` 接近 0，应输出明确的空值或约定值，并记录原因，不得产生 NaN 后继续训练。

梯度裁剪比例定义为：

```text
本 update 内，pre-clip grad norm > max_grad_norm 的 mini-batch 比例
```

要求：

1. 保留现有 mean grad norm 字段；
2. 新增 max 和 clip fraction；
3. 所有诊断不参与 loss；
4. 所有诊断为 finite，或按 schema 明确标记无定义；
5. 先运行 10-update 诊断 smoke；
6. smoke 只验工程，不评价任务性能。

本任务不得直接启用 value normalization、value clipping 或修改 critic LR。Codex 只需在最终报告中根据现有日志和新增 smoke 提出以下二选一建议：

```text
KEEP：正式 P0–P4 继续使用当前 value loss
或
PRE-EXPERIMENT REQUIRED：在 P0 上先做对称、独立的 value 处理前置实验
```

未经新任务书授权，不执行该前置训练。

---

# 9. 子任务 E：gamma / horizon 诊断

新增纯分析输出，不启动训练。

报告至少列出 `gamma=0.99、0.995、0.997、0.999` 在 `dt=0.2 s` 下的：

- step 半衰期；
- 秒级半衰期；
- 100 s 末端奖励权重；
- 与当前 `lambda=0.95` 组合后的 GAE trace 半衰期。

同时结合现有 P0/P1/P2：

- Episode length 分布；
- persistent outage 首次出现 step；
- boundary termination step；
- terminal penalty 位置；

说明 `gamma=0.99` 是否存在明显的长时域信用分配风险。

本任务不得：

- 修改 gamma；
- 修改 lambda；
- 添加 boundary shaping；
- 修改 terminal penalty；
- 启动任何 gamma sweep。

最终只输出：

```text
KEEP gamma=0.99
或
PRE-EXPERIMENT REQUIRED
```

并给出证据。任何训练性验证必须等待新授权。

---

# 10. 子任务 F：修正 checkpoint / resume 语义

必须在 config、checkpoint metadata 和交接文档中加入明确字段：

```text
resume_mode = fresh_episode_at_update_boundary
exact_environment_resume = false
```

旧 checkpoint 继续可加载。

若不保存完整环境内部状态，不得新增测试声称：

```text
interrupted run == uninterrupted run
```

允许验证：

- Actor/Critic 权重可恢复；
- optimizer 可恢复；
- normalizer 可恢复；
- 保存的 RNG 可恢复；
- never-used training environment seeds 不重复；
- update 和 total steps 单调继续。

未来正式 P0–P4 实验默认要求单次运行不中断完成。若中断，在未实现 exact environment resume 前，应从同一 run seed 重新完整训练，而不是把 fresh-Episode resume 当作完全等价轨迹。

---

# 11. 自动测试要求

新增测试至少覆盖：

1. validation 与 test seed 无交集；
2. 新协议的 training seed 保留区间互不重叠；
3. 同一 run seed 下，不同 Actor variant 的 Critic 初始权重完全一致；
4. 不同 Actor variant 的构造不改变 rollout action-noise 随机流；
5. mini-batch permutation 使用独立且可恢复的 RNG；
6. 新 RNG states 正确保存和加载；
7. legacy checkpoint 可加载并标记为 legacy；
8. deterministic evaluation 重复运行结果逐 Episode 一致；
9. checkpoint grid 按 update 数字顺序加载；
10. validation selection 只读取 validation split；
11. score 相同时选择较早 update；
12. test split 不允许调用 checkpoint-grid selection；
13. 原始 Episode 行数等于 checkpoint 数 × seed 数；
14. checkpoint 和 seed manifest hash 正确写入；
15. explained variance 与人工小样本计算一致；
16. grad clip fraction 与人工小样本计算一致；
17. undefined explained variance 被安全处理；
18. P0/P1/P2/P3/P4 旧 checkpoint 兼容；
19. P3/P4 topology buffer 与 old latent `z` 既有测试继续 PASS；
20. 全部输出无 NaN/Inf，或按 schema 明确标记为 null。

必须真实运行：

```text
python -m unittest discover -s selfcheck -p "test_*.py" -v
```

现有交接记录声称旧环境中 `56/56 PASS`，但 Stage 4.1 必须在具备 PyTorch 的实际执行环境重新验证，不得直接复制旧数字。

---

# 12. 执行顺序

严格按以下顺序执行：

```text
1. 读取文档与源码
2. 盘点 P0/P1/P2 周期 checkpoints，并生成 hash manifest
3. 实现 seed protocol 与独立 RNG streams
4. 实现 checkpoint-grid evaluation 和逐 Episode 落盘
5. 实现 Critic / grad diagnostics
6. 新增自动测试
7. 运行全部自动测试
8. 运行 10-update diagnostics smoke
9. 写入并锁定 validation/test seed manifest
10. 对 P0/P1/P2 全部周期 checkpoints 运行 validation
11. 生成 selected_checkpoints.json
12. 仅对三个 selected checkpoints 运行 final test
13. 生成 gamma/horizon 分析
14. 生成 Stage 4.1 validation 报告与交接文档
15. STOP
```

若任一自动测试失败，停止进入旧检查点评估；不得带着失败测试继续生成正式结论。

---

# 13. 明确禁止

本任务禁止：

- P0/P1/P2 新的 1000-update 训练；
- P3/P4 的 1000-update 正式训练；
- P5、P6、Role+Graph 实现或训练；
- 修改 Environment、Reward 或 observation；
- 修改 P0–P4 Actor 结构；
- 修改 PPO/GAE 数学；
- 修改学习率、entropy、clip、网络宽度或 log_std；
- 启用 value normalization 或 value clipping；
- gamma / lambda sweep；
- 根据 test 结果重新选择 checkpoint；
- 为不同 variant 使用不同 validation/test seeds；
- 覆盖旧训练产物；
- 把 smoke 性能写成方法有效性证据；
- 把单训练种子结果写成稳定方法排名；
- 宣称 Graph 有效、Graph 优于 MLP 或论文创新成立。

---

# 14. Stage 4.1 交付物

代码和测试：

```text
reevaluate_checkpoint_grid.py
plain_mappo/experiment_protocol.py        （或等价模块）
selfcheck/test_mappo_protocol.py          （或等价测试）
必要的最小现有文件修改
```

报告：

```text
stage4_1_protocol_validation.py
stage4_1_protocol_validation.json
stage4_1_protocol_validation.md
22_阶段4.1_训练与评估协议修复交接.md
```

数据：

```text
artifacts/stage4-1-protocol-repair/seed_manifest.json
artifacts/stage4-1-protocol-repair/checkpoint_manifest.json
artifacts/stage4-1-protocol-repair/validation_grid.csv
artifacts/stage4-1-protocol-repair/validation_episodes.jsonl
artifacts/stage4-1-protocol-repair/selected_checkpoints.json
artifacts/stage4-1-protocol-repair/final_test_summary.csv
artifacts/stage4-1-protocol-repair/final_test_episodes.jsonl
artifacts/stage4-1-protocol-repair/protocol_audit.json
artifacts/stage4-1-protocol-repair/protocol_audit.md
```

---

# 15. 最低验收标准

只有同时满足以下条件，Stage 4.1 才可判定 PASS：

| 验收项       | PASS 条件                                                    |
| ------------ | ------------------------------------------------------------ |
| 冻结边界     | Environment、Reward、observation、P0–P4 Actor 和 PPO/GAE 数学未变 |
| Seed 分离    | training / validation / test 严格分离，manifest 完整         |
| 旧检查点评估 | P0/P1/P2 各 20 个周期 checkpoint 均完成 20-seed validation   |
| 模型选择     | 只基于 validation；tie 时选较早 update                       |
| Final test   | 每种方法仅 selected checkpoint 完成 50 个独立 test Episodes  |
| 原始数据     | 逐 Episode 数据完整，行数和汇总可反算                        |
| 可追溯性     | checkpoint、seed manifest 和配置 hash 齐全                   |
| RNG 公平性   | Critic、action noise、mini-batch 和 env seeds 不再被 Actor 构造顺序污染 |
| 诊断         | explained variance、value stats、grad max/clip fraction 可用 |
| Resume 口径  | 明确为 fresh-Episode update-boundary resume，不冒充 exact resume |
| 自动测试     | 旧测试与新测试全部 PASS                                      |
| Smoke        | 10-update 诊断 smoke finite 且 checkpoint reload PASS        |
| 正式训练     | P3/P4 1000-update 均未启动                                   |

任务性能好坏不是 Stage 4.1 的工程 PASS 条件。

---

# 16. 后续正式训练门禁

Stage 4.1 PASS 后，交接报告必须给出但不得执行以下正式矩阵：

| Variant | 定义                                | 独立训练 seeds |
| ------- | ----------------------------------- | -------------: |
| P0      | Plain MAPPO，26-d local observation | 最少 3，目标 5 |
| P1      | Role-info                           | 最少 3，目标 5 |
| P2      | Role-head                           | 最少 3，目标 5 |
| P3      | Topology-info MLP                   | 最少 3，目标 5 |
| P4      | Graph MAPPO                         | 最少 3，目标 5 |

后续正式训练必须满足：

- P0–P4 使用相同 1000-update 预算；
- 使用相同 paired run seeds；
- 使用相同 validation 和 final test seed manifests；
- 每个 run 独立选择 checkpoint；
- 先锁定 checkpoint，再运行 final test；
- 统计单位以独立训练 run 为主，不得把大量 evaluation Episodes 当作大量训练重复；
- 报告每种方法的 run-level mean、standard deviation、95% CI 和 paired differences；
- P3/P4 参数公平性继续满足既有 `≤10%` 合同；
- 任何协议变更必须同时应用于 P0–P4，不得只帮助某个 variant。

正式训练仍需新的明确授权和单独任务书。

---

# 17. Codex 最终回复格式

最终回复必须按以下顺序：

```text
1. 修改文件
2. 协议版本与 seed manifests
3. 自动测试结果
4. 10-update 诊断 smoke 结果
5. P0/P1/P2 validation-grid 结果
6. 新 selected updates
7. 独立 final-test 结果
8. Critic / grad 诊断结论
9. gamma / horizon 诊断结论
10. resume 语义确认
11. 未执行内容
12. 最终状态
```

最终状态只能使用：

```text
Stage 4.1 protocol repair = COMPLETE / INCOMPLETE
Engineering validation = PASS / FAIL
P0/P1/P2 retrospective status = SINGLE-SEED PILOT
P3 formal training = NOT STARTED
P4 formal training = NOT STARTED
Multi-seed P0–P4 training = NOT AUTHORIZED
Role+Graph = NOT STARTED
```

不得提前宣称任何方法有效或论文创新成立。

---

# 18. 一句话执行边界

> 本任务只修复实验协议、重评旧检查点并补强诊断；不修改环境、奖励、Actor 结构或 PPO 数学，不启动任何新的 1000-update 正式训练。
