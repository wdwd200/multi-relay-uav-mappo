# Stage 4 Role 消融代码与自检交接

## 当前状态

```text
Stage 4 Role implementation = COMPLETE
Engineering self-check      = PASS
Formal Role training        = NOT STARTED
Graph implementation        = NOT STARTED
```

本阶段只完成 P1/P2 的最小 Role 消融实现、自动测试与各一次 10-update smoke。Smoke 不用于判断 Role 方法是否优于 Plain MAPPO。

## A. 修改与新增文件

- `plain_mappo/roles.py`（新增）：唯一的静态链位置 Role mapping helper，以及 tensor/one-hot 辅助函数。
- `plain_mappo/config.py`：新增向后兼容的 `actor_variant`（`plain`、`role_info`、`role_head`）；旧 checkpoint/config 缺省为 `plain`；Role variant 显式要求 `state_independent_tanh`。
- `plain_mappo/networks.py`：在不改变 P0 模块命名和参数形状的前提下，增加 P1/P2 Actor 分支。
- `plain_mappo/trainer.py`、`train_plain_mappo.py`、`evaluate_plain_mappo.py`、`stage3_training_diagnosis.py`：从 config 重建正确 Actor variant；训练/评估仍走同一 PPO pipeline。
- `plain_mappo/__init__.py`：导出 Role mapping helper。
- `selfcheck/test_mappo_roles.py`（新增）：10 个 Role 合同、PPO update、checkpoint、旧 config 兼容测试。
- `stage4_role_validation.py`（新增）：复评 P0、读取且重新加载两份 smoke checkpoint、运行 MAPPO 自动测试，生成 JSON/Markdown。
- `stage4_role_validation.json`、`stage4_role_validation.md`（新增）：本次机器可读与可读验证结果。

没有修改 `relay_env/`、Reward、通信/天线、安全、terminated/truncated、26 维 observation、47 维 critic state、Critic、GAE、buffer 或 PPO loss。

## B. P0/P1/P2 Actor 定义

```text
P0 plain:
26 -> shared MLP(128,128) -> shared mean(3)

P1 role_info:
26 + static role one-hot(3) = 29 -> shared MLP(128,128) -> shared mean(3)

P2 role_head:
26 -> shared MLP(128,128) -> selected source/middle/destination mean head(3)
```

P2 不是三个完整 Actor：它只有一个 shared backbone，三个 128→3 线性 mean head。所有三种 variant 都保持一个共享 Actor object；P2 的 R2/R3 共用同一个 middle head。

| Variant | 环境 local obs | backbone 实际输入 | Actor 参数量 |
|---|---:|---:|---:|
| P0 plain | 26 | 26 | 20,358 |
| P1 role_info | 26 | 29 | 20,742 |
| P2 role_head | 26 | 26 | 21,132 |

P2 相对 P0 只增加两个额外 128→3 mean head（每个 387 参数），没有 Role-specific Critic、Role-specific log_std、Role embedding 或独立 backbone。

## C. Role mapping

| K | Relay role IDs |
|---:|---|
| 3 | `[0, 1, 2]` |
| 4 | `[0, 1, 1, 2]` |
| 5 | `[0, 1, 1, 1, 2]` |

其中 `0=source-side`、`1=middle`、`2=destination-side`。映射由 `role_ids_for_num_relays(K)` 单点定义，rollout buffer 仍只保存原始 `[..., K, 26]`，不会重复保存 Role 或 29 维拼接结果。

## D. log_std 与 Critic

P1/P2 固定为一个全局共享的三维 state-independent、smooth-bounded log_std：

```text
mode = state_independent_tanh
log_std bounds = [-4, 0]
initial effective log_std = -1.5
```

它跨 environment、relay 和 role 共用；Role 仅影响 P2 的 mean-head 选择。P0 仍完整保留历史 `state_dependent_clamp` 和 Stage 3 `state_independent_tanh` checkpoint 兼容路径。

Critic 没有 Role 输入，仍是 K=4 的 `47 -> 128 -> 128 -> 1` centralized Critic。

## E. P0 向后兼容回归

实际重新加载：`artifacts/stage3-stateindependent-full/checkpoints/actor_final.pt`。

固定 deterministic seeds 10000--10019 的实际结果：

| 指标 | 结果 |
|---|---:|
| normal completion | 0/20 |
| persistent outage | 19 |
| boundary | 1 |
| collision | 0 |
| hard speed/acceleration violations | 0 |
| mean e2e rate | 6.103710197871623 Mbps |

与第五轮 Stage 3 baseline 记录一致，P0 regression = PASS。

## F. 自动测试

执行命令：

```powershell
python -m unittest discover -s selfcheck -p "test_mappo*.py"
```

结果：**38/38 PASS**（原 Stage 3 MAPPO 28 项 + 新增 Role 10 项）。新增测试覆盖 K=3/4/5 mapping、P0/P1/P2 输入维度、P1 one-hot、P2 三 head/中部共享、受控不同 role 输出、全局共享 log_std、action shape、P1/P2 rollout+GAE+PPO update（不重采样 latent z）、P1/P2 checkpoint round trip、旧 config 缺字段默认 P0。

## G. 10-update smoke

两组 smoke 都从新随机初始化运行；均为 10 updates × 8 envs × 128 steps = 10,240 team-time samples，并生成 `train.csv`、`eval.csv`、`latest.pt`、`best.pt`、`actor_final.pt` 与 `eval_checkpoints/actor_update_0010.pt`。

| Variant | 输出目录 | final KL | final clip fraction | final effective log_std mean | saturation | finite | checkpoint reload |
|---|---|---:|---:|---:|---:|---|---|
| P1 role_info | `artifacts/stage4-role-info-smoke/` | 0.00605386 | 0.05603027 | -1.49536145 | 0.0 | PASS | PASS |
| P2 role_head | `artifacts/stage4-role-head-smoke/` | 0.00421984 | 0.05126953 | -1.49534369 | 0.0 | PASS | PASS |

两份 `actor_final.pt` 均重新加载，并在 fixed seeds 10000--10004 上复评，和相应 `eval.csv` 完全相同。

Smoke 最终 5-episode evaluation 仅作管线检查：

| Variant | normal | persistent outage | collision | boundary | mean e2e |
|---|---:|---:|---:|---:|---:|
| P1 | 0 | 1 | 0 | 4 | 6.907123 Mbps |
| P2 | 0 | 0 | 0 | 5 | 7.162623 Mbps |

这些 10-update 数值不具有正式策略性能结论，不能据此宣称 Role 有效或优于 P0。

## H. 已知事项与边界

- 本工作区中未找到任务书列出的 `00_...` 与 `07_...` 历史文件；实现依照根 `AGENTS.md`、可获得的 Stage 3 交接/收口文档以及当前实际代码完成。没有发现会阻止安全实现的设计冲突。
- Stage 3 Plain MAPPO 是有效但性能弱的 baseline；本阶段没有改变该结论。
- 没有启动 P1/P2 的 1000-update 正式训练；需要独立授权后才可进行。
- 未实现 Graph、GNN、GAT、message passing、Role+Graph、Agent ID、chain index 或 CBF。
