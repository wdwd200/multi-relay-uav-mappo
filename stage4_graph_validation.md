# Stage 4 P3/P4 Topology / Graph 验证

本报告仅覆盖实现合同、回归和 10-update smoke；未执行 P3/P4 的 1000-update 正式训练。Environment v1.1、Reward、26 维 local observation、47 维 Critic、PPO/GAE 数学均未改动。

## Topology 特征合同

- node/edge/graph-hidden dimensions: 6 / 7 / 32。
- K=3: nodes=[5, 6], edges=[4, 7]；K=4: nodes=[6, 6], edges=[5, 7]；K=5: nodes=[7, 6], edges=[6, 7]。
- K=4 shape/归一化/有限性: PASS。边容量由既有 `relay_env.communication.link_metrics()` 构造；自动测试进一步逐字段验证 edge delta、velocity 和 capacity。
- 仅使用 H/L/Relay 当前 position 与 velocity；不读取 waypoint、cruise speed、step、sim time、outage counter、Role、Agent ID 或 chain index。

## Actor 参数量

| Variant | 环境 observation | Actor 实际输入 | Actor parameters |
|---|---:|---:|---:|
| plain | 26 | 26 | 20358 |
| role_info | 26 | 29 | 20742 |
| role_head | 26 | 26 | 21132 |
| topology_info | 26 | 97 | 29446 |
| graph | 26 | 58 | 29094 |

P3/P4 parameter relative difference: 1.1954% — PASS (≤10%)。

## 自动测试

- `C:\Python314\python.exe -m unittest discover -s selfcheck -p test_mappo*.py`: PASS; 51 tests executed。
- 覆盖 P3/P4 shapes、归一化、capacity、prohibited-field exclusion、共享 encoder/message/update、K 轮 receptive field、PPO old latent 重算、checkpoint round-trip，以及 P0/P1/P2 checkpoint 兼容。

## 10-update smoke

### topology_info

- 目录: `artifacts\stage4-topology-info-smoke`；结果: PASS；updates=10；samples=10240。
- final training: policy loss=-0.0040544966701418165，critic loss=76.40289239883423，KL=0.0064541305706370625，clip fraction=0.038134765625，actor/critic grad norm=0.5528275221586227/100.60097236633301。
- effective log_std mean=-1.4979819059371948，action saturation=0.0，finite logs=True。
- final 5-seed deterministic eval: normal=0，persistent outage=2，collision=0，boundary=3，mean e2e=6.8572221208304125 Mbps。
- actor_final reload eval matches log: True。

### graph

- 目录: `artifacts\stage4-graph-smoke`；结果: PASS；updates=10；samples=10240。
- final training: policy loss=-0.0032136889290995895，critic loss=109.66027526855468，KL=0.003320520558918361，clip fraction=0.02158203125，actor/critic grad norm=0.44119224064052104/124.54830474853516。
- effective log_std mean=-1.5043481588363647，action saturation=0.0，finite logs=True。
- final 5-seed deterministic eval: normal=0，persistent outage=0，collision=0，boundary=5，mean e2e=7.460868882818383 Mbps。
- actor_final reload eval matches log: True。

## Status

Stage 4 topology/graph implementation = COMPLETE

Engineering validation = PASS

P3 formal training = NOT STARTED

P4 formal training = NOT STARTED

Role+Graph = NOT STARTED

Smoke 仅验证工程连通性，不构成 P3/P4 性能有效性或优越性的结论。
