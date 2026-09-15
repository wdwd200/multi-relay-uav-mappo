# Stage 4 Role 消融验证

本报告只验证 P0 向后兼容、P1/P2 的实现合同及各 10-update smoke。未运行任何 P1/P2 1000-update 正式训练。

## P0 第五轮 checkpoint 回归

- 结果：PASS
- checkpoint：`artifacts\stage3-stateindependent-full\checkpoints\actor_final.pt`
- 固定 seeds：10000--10019；normal completion=0/20，persistent outage=19，boundary=1，collision=0，mean e2e=6.103710197871623 Mbps。

## Actor 结构与参数量

| Variant | 环境 obs | Actor 实际输入 | Actor 参数量 |
|---|---:|---:|---:|
| plain | 26 | 26 | 20358 |
| role_info | 26 | 29 | 20742 |
| role_head | 26 | 26 | 21132 |

P1 仅将静态 3-d one-hot 拼接进 shared backbone；P2 保留 26-d shared backbone，并仅新增两组 128→3 mean head。P1/P2 都只有一组全局共享、平滑有界的 3-d state-independent log_std。Critic 未改变，K=4 仍为 47-d global state。

## Role mapping

- K=3: [0, 1, 2]
- K=4: [0, 1, 1, 2]
- K=5: [0, 1, 1, 1, 2]

## 自动测试

- `C:\Python314\python.exe -m unittest discover -s selfcheck -p test_mappo*.py`：PASS；38 tests。

## 10-update smoke

### role_info

- 目录：`artifacts\stage4-role-info-smoke`；结果：PASS；updates=10；samples=10240。
- final: KL=0.0060538558478583585，clip fraction=0.0560302734375，actor/critic grad norm=0.6692229211330414/96.64442710876465，effective log_std mean=-1.4953614473342896，action saturation=0.0。
- 5-seed final deterministic eval: normal=0，persistent outage=1，collision=0，boundary=4，mean e2e=6.9071230000827395 Mbps。
### role_head

- 目录：`artifacts\stage4-role-head-smoke`；结果：PASS；updates=10；samples=10240。
- final: KL=0.0042198433613521045，clip fraction=0.05126953125，actor/critic grad norm=0.5700768582522869/95.74258575439453，effective log_std mean=-1.4953436851501465，action saturation=0.0。
- 5-seed final deterministic eval: normal=0，persistent outage=0，collision=0，boundary=5，mean e2e=7.162623363536296 Mbps。

Smoke 不用于判断 Role 方法性能。Environment v1.1、26-d observation、47-d Critic、Reward 和 PPO 数学流程均未修改；Graph 也未实现。
