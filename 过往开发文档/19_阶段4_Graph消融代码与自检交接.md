# Stage 4 P3/P4 Topology / Graph 消融代码与自检交接

## A. 范围与冻结合同

本次完成的是 P3 `topology_info` 和 P4 `graph` 的实现、自动测试及各 10-update 工程 smoke。没有执行 P3/P4 的 1000-update 正式训练。

Environment v1.1 保持冻结：未修改动力学、通信/天线、Reward、outage/persistent-outage、安全/边界、terminated/truncated 或 26 维 local observation。Critic 仍为冻结的 47 → 128 → 128 → 1；P3/P4 未向 Critic 加入 topology 或 Graph。

Actor 只通过只读 `env.get_global_state()` 取得 H、L、Relay 的当前 position/velocity。未读取 H/L waypoint、cruise speed、step、sim time、global consecutive outage counter 或未来轨迹。

## B. 修改文件

- `plain_mappo/topology.py`：P3/P4 共用的 topology feature helper。
- `plain_mappo/config.py`：新增 `topology_info` / `graph` variants 与序列化 topology 合同（node=6, edge=7, hidden=32）。
- `plain_mappo/networks.py`：P3 concatenated-topology MLP 和 P4 内部 PyTorch 链式 message passing。
- `plain_mappo/rollout_buffer.py`：仅 topology variants 可选保存 action 前的 nodes/edges。
- `plain_mappo/trainer.py`：统一 rollout/PPO 管线在动作前构造并存储 topology，update 时与 old latent z 一起重算 log-prob。
- `plain_mappo/evaluation.py`：P3/P4 deterministic evaluation 在每一步动作前重算当前 topology。
- `train_plain_mappo.py`：统一 CLI 新增 `topology_info`、`graph`。
- `plain_mappo/__init__.py`：导出 topology helper。
- `selfcheck/test_mappo_topology.py`：新增 13 项 P3/P4 合同、PPO、checkpoint 和 receptive-field 测试。
- `stage4_graph_validation.py`：可复现的 P3/P4 验证与报告生成器。
- `stage4_graph_validation.md` / `stage4_graph_validation.json`：本次真实验证结果。

## C. 精确 topology 定义

所有节点固定排序为 `[H, R1, ..., RK, L]`。

每个 node 为 `[x, y, z, vx, vy, vz]`，归一化为：

- `x / map_x_m`
- `y / map_y_m`
- `(z - 200) / 100`
- `vx / 20`
- `vy / 20`
- `vz / 6`

K=4 时 nodes shape 为 `[6, 6]`；K=3/5 分别为 `[5, 6]` / `[7, 6]`。

固定逻辑 hop 为 `H→R1→...→RK→L`。每条 edge 的七维字段、方向均固定为 downstream minus upstream：

`[dx/2000, dy/2000, dz/200, dvx/40, dvy/40, dvz/12, min(capacity_bps/60_000_000, 1)]`

capacity 始终调用现有 `relay_env.communication.link_metrics()`，没有复制路径损耗、天线、SNR 或 capacity 公式。K=4 时 edges shape 为 `[5, 7]`；K=3/5 分别为 `[4, 7]` / `[6, 7]`。测试逐字段确认 node normalization、edge delta/velocity/capacity 与环境真实值一致，且 features 有限。

## D. P3：Topology-info MLP

每个 Relay 的输入是其原有 26 维 local observation 与同一份全链 topology block：

`26 + (6×6) + (5×7) = 97 → 128 → 128 → mean(3)`

四个 Relay 仍共享同一 Actor；各自 26 维 local observation 保持不同，而 71 维 topology block 完全相同。P3 不使用 Role、Role head、Agent ID、chain index 或 message passing。

## E. P4：Graph MAPPO

P4 使用与 P3 完全相同的 raw node (6-d) 和 edge (7-d) 信息，但不 flatten topology。

- shared node encoder：`6 → 32 → Tanh`
- shared edge encoder：`7 → 32 → Tanh`
- reverse edge：仅反转相对 position/velocity 的前六维，capacity 不变。
- 每个方向的 message：`concat(neighbour embedding, encoded directed edge)`，`64 → 32 → Tanh`。
- 聚合：mean aggregation。
- node update：`concat(old embedding, aggregated message)`，`64 → 32 → Tanh`。
- message/update 参数在所有 node、edge direction 和 K 个 round 之间共享；K=4 使用恰好 4 rounds。
- R1..RK 分别取 graph node index `1..K` 的 own embedding。
- 最终共享 Actor trunk：`local 26 + graph 32 = 58 → 128 → 128 → mean(3)`。

没有 GAT、attention、PyG、Graph critic、Role 或 Role+Graph。

## F. 参数量

| Variant | Actor effective input | Actor parameters |
|---|---:|---:|
| P0 plain | 26 | 20,358 |
| P1 role_info | 29 | 20,742 |
| P2 role_head | 26 | 21,132 |
| P3 topology_info | 97 | 29,446 |
| P4 graph | 58 | 29,094 |

P3/P4 参数量相对差异为 1.1954%，满足 `≤ 10%` 公平性合同。

## G. 自动测试

真实执行：

`python -m unittest discover -s selfcheck -p "test_*.py" -v` → **56/56 PASS**。

其中既有 MAPPO/Role 38 项继续 PASS；新增 P3/P4 13 项覆盖：K=3/4/5 helper shapes、精确 normalization、`link_metrics` capacity、privileged-field exclusion、P3 input/repeated topology、P4 encoder/message/update sharing、R2/R3 own embedding、K-round controlled distant-node receptive field、shared log_std、PPO old latent z reuse、actor/critic finite update、checkpoint deterministic round-trip、P0/P1/P2 checkpoint compatibility、P3/P4 parameter fairness。

最终 validation 还再次执行：

`python -m unittest discover -s selfcheck -p "test_mappo*.py"` → **51/51 PASS**。

## H. 10-update smoke（真实执行）

两组 smoke 均使用冻结配置：K=4、8 env、rollout=128、mini-batch=256、10 PPO epochs、actor lr=`1e-4`、critic lr=`3e-4`、entropy=`0`、state-independent smooth-bounded log_std `[-4, 0]`、effective init `-1.5`、base seed=2026。

| Variant | 目录 | Updates / samples | final KL | clip fraction | effective log_std mean | action saturation | finite / checkpoint reload |
|---|---|---:|---:|---:|---:|---:|---|
| P3 topology_info | `artifacts/stage4-topology-info-smoke/` | 10 / 10,240 | 0.006454 | 0.038135 | -1.497982 | 0.0 | PASS / PASS |
| P4 graph | `artifacts/stage4-graph-smoke/` | 10 / 10,240 | 0.003321 | 0.021582 | -1.504348 | 0.0 | PASS / PASS |

两者的 policy loss、critic loss、entropy、KL、clip fraction、actor/critic grad norm、log_std 和 action saturation 均为有限值，未出现 NaN/Inf。每个 smoke 均保存 `config.json`、`train.csv`、`eval.csv`、`checkpoints/latest.pt`、`best.pt`、`actor_final.pt`，以及 `eval_checkpoints/actor_update_0010.pt`。重载 `actor_final.pt` 后的固定 5-seed deterministic evaluation 与对应日志严格一致。

这些 5-seed smoke 评估仅证明 evaluation 与 checkpoint 管线可运行；不用于判断 P3/P4 的任务性能。

## I. 兼容性

P0、P1、P2 的已有 `actor_final.pt` 均可通过新 Config/Actor 代码加载，且均不要求 topology tensors。普通 P0/P1/P2 buffer 路径不分配 topology fields；仅 P3/P4 保存 `[T,E,K+2,6]` 和 `[T,E,K+1,7]` 的 action-time tensors。

## J. 明确未实现的内容

- Role+Topology：未实现。
- Role+Graph：未实现。
- Agent ID、chain index、任何新的 Role 结构：未实现。
- P3/P4 1000-update 正式训练：未执行。

## K. 最终状态

Stage 4 topology/graph implementation = COMPLETE

Engineering validation = PASS

P3 formal training = NOT STARTED

P4 formal training = NOT STARTED

Role+Graph = NOT STARTED

本阶段不对 Graph 是否有效、是否优于 MLP 或正式性能作任何结论，须等待独立的公平正式训练实验。
