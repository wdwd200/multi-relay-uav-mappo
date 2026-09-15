# Stage 3 Plain MAPPO：entropy=0.001 第三轮训练结果

日期：2026-09-08  
最终结论：**第三轮已实际完成 1,000 updates，但 Stage 3 仍为 FAIL / NEEDS TUNING。** 新的 completion-first best 选择规则已生效；`entropy_coef=0.001` 没有避免后期 `log_std` 再次全局趋近 clamp 上界，且最终 normal completion 为 0/20。

## 1. 本轮范围、代码变更与实际执行

### 受控实验边界

- 第三轮输出目录：`artifacts/stage3-entropy001/`；目录在训练开始时不存在，未覆盖第一、二轮数据。
- 从全新随机初始化开始，未传入 `--resume`，没有加载任何先前 checkpoint。
- 唯一训练动力学超参数变化为 **`entropy_coef: 0.01/0.0 -> 0.001`**。第三轮 `config.json` 保持 K=4、8 环境、rollout=128、mini-batch=256、10 PPO epochs、Actor/Critic lr=3e-4、`log_std=[-5,2]`、网络、base seed=2026 及全部其他训练配置不变。
- Environment v1.1、26 维 Observation、47 维 critic state、Reward、通信/天线、动力学、terminated/truncated 与安全逻辑均未修改；没有 Graph、Role 或 CBF。

### 新 best 选择规则

旧规则已由用户正式替换为以下“越小越好”的 7 元组：

```text
(
  speed_accel_violations,
  -normal_completion_count,
  collision + boundary,
  persistent_outage_count,
  outage_step_ratio,
  -rate_satisfaction_ratio,
  -mean_e2e_rate_mbps,
)
```

实现位于 `plain_mappo.metrics.safety_priority_key()`；它只影响 deterministic evaluation 的 score 和 best.pt 覆盖判断，不参与 Reward、GAE、advantage、PPO loss 或环境 transition。新增自动测试验证“19 个正常完成、1 次 boundary”的策略严格胜过“零 collision/boundary、20 个 persistent outage、零正常完成”的策略。Stage 3 MAPPO 测试实际结果为 **24/24 PASS**（原 23 项加此新规则测试）。

### 周期 Actor checkpoint

每次 50-update deterministic evaluation 额外保存轻量部署 Actor：

`artifacts/stage3-entropy001/eval_checkpoints/actor_update_0050.pt` … `actor_update_1000.pt`

实际生成 20 个文件，总计 1,756,220 bytes（每个 87,811 bytes）。抽查 update-700 文件仅包含 `actor_state`、`config`、`update`、evaluation seeds、checkpoint type 和 47 维状态契约；不包含 Critic 或 Adam 状态。

### 完成证明

- `train.csv`：1,000 行，update 1--1000，全部 `finite=1` 且数值有限。
- `eval.csv`：20 行，update 50--1000、间隔 50。
- `latest.pt`：update 1000、1,024,000 team-time samples。
- `best.pt`：update 700、716,800 team-time samples。
- `actor_final.pt`：实际从新的 best.pt 导出。

## 2. 第三轮训练曲线与 PPO 指标

| 阶段 | episode return | episode length | entropy | rollout saturation | raw log_std >2 | approx KL | clip fraction | Actor grad norm（裁剪前） | Critic loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1--200 | 165.48 | 178.03 | 4.166 | 0.153 | 0.020 | 0.00735 | 0.0711 | 0.446 | 23.86 |
| 201--500 | 200.38 | 211.46 | 7.224 | 0.498 | 0.273 | 0.00835 | 0.0701 | 0.679 | 12.06 |
| 501--800 | 220.16 | 227.83 | 9.927 | 0.775 | 0.882 | 0.00433 | 0.0235 | 0.265 | 13.86 |
| 801--1000 | 236.87 | 241.57 | 10.004 | 0.787 | 0.917 | 0.00910 | 0.0261 | 0.473 | 11.81 |

return 和长度逐阶段上升，但没有转化为 deterministic 正常完成。第三轮的关键过程是：

- `entropy=0.001` 在 1--200 比 `entropy=0` 保留了更多探索；但 201--500 起 raw `log_std` 和 action saturation 持续抬升。
- 501--800 已有平均 88.2% raw `log_std>2`；801--1000 为 91.7%，而 clamp 后的平均 `log_std` 为 1.916。
- update 1000 单点为 entropy 10.161、rollout saturation 0.803、raw `log_std>2` 0.966。故并未避免第一轮的后期方差饱和，只是发生时间晚于 `entropy=0.01`。
- 训练过程没有 NaN/Inf。Critic loss 不是本轮异常主证据。

### Actor 更新是否仍像 entropy=0 一样过激

第三轮没有出现 entropy=0 的长期高 clip/KL 模式：四段 KL 均值为 0.00735、0.00835、0.00433、0.00910，clip fraction 均值为 0.071、0.070、0.024、0.026。update 853 出现单次 KL 0.7849 尖峰，但相邻 update 回落；不能把这个孤点误报为长期 KL 失控。

Actor 预裁剪 grad norm 大于 0.5 的 update 比例依次为 30.0%、67.0%、7.3%、28.5%。因此 201--500 确有高频 Actor clipping，但后两个阶段没有持续。相对第二轮 entropy=0 的 201--800 两段均为 100% 触发、clip fraction 均值约 0.124--0.135，第三轮明显更温和，但仍留下一个可操作的学习率信号。

## 3. 周期 evaluation 与后期行为

第三轮 20 次 5-episode deterministic evaluation 的 normal completion **全部为 0/5**。不存在第一轮 update 850--900 的短暂完成率峰值，因此不存在“该峰值被稳定保留”的情况。

后段数据如下：

| update | normal /5 | persistent outage /5 | collision /5 | boundary /5 | outage ratio | mean e2e (Mbps) |
|---:|---:|---:|---:|---:|---:|---:|
| 700 | 0 | 5 | 0 | 0 | 0.12376 | 6.14621 |
| 800 | 0 | 3 | 0 | 2 | 0.12133 | 6.08142 |
| 850 | 0 | 4 | 0 | 1 | 0.15818 | 5.89162 |
| 900 | 0 | 3 | 1 | 1 | 0.11990 | 6.20042 |
| 950 | 0 | 4 | 0 | 1 | 0.14536 | 5.97388 |
| 1000 | 0 | 3 | 0 | 2 | 0.10335 | 6.28745 |

新规则在本轮的 score 是正确生效的。由于所有点 normal completion 均为 0，排序接着比较 collision+boundary、persistent outage 和 outage ratio；update 700 的周期 score 为：

`(0, 0, 0, 5, 0.1237583, -0.7933816, -6.1462138)`

它优于后来包含 boundary/collision 的模型，故 best.pt 选择 update 700 合理。规则修复能防止第二轮所示“零安全终止但零完成”压过正常完成策略的结构性错误；本轮没有任何正常完成候选，因而无法凭规则本身生成完成策略。

## 4. 最终 20-episode 验收：新 best.pt（update 700）

已实际执行 seeds 10000--10019 的 deterministic evaluation：

| 指标 | 结果 |
|---|---:|
| normal completion | 0/20 |
| persistent outage | 19/20 |
| collision | 1 |
| boundary | 0 |
| hard speed/acceleration violations | 0 |
| mean e2e rate | 6.13474 Mbps |
| outage step ratio | 0.13354 |
| rate satisfaction ratio | 0.79462 |
| deterministic action saturation | 0.00106 |
| min node separation | 19.968 m |

固定 evaluation states 上，best Actor 的 raw `log_std` mean=4.61668、max=8.26843，**96.484% 大于 2**；clamp 后 mean=1.97920，三维 Gaussian entropy=10.19442。这证明 `entropy=0.001` 没有避免最终 best 的全局 clamp 饱和。它比第二轮 best 保留了更多探索，但这种探索没有转化为 persistent-outage 恢复能力。

同一 seeds 的实际随机基线 score 是：

`(0, 0, 12, 8, 0.0734318, -0.9001854, -6.5965169)`。

第三轮 best score 为：

`(0, 0, 1, 19, 0.1335366, -0.7946240, -6.1347373)`。

二者在 hard violations 和 normal completion 相同；第三关键字 collision+boundary 为 1（随机为 12），所以第三轮 best 按新安全优先规则优于随机策略。

### 验收线

| 验收项 | 第三轮结果 | 状态 |
|---|---:|---|
| hard violations = 0 | 0 | PASS |
| normal completion >=16/20 | 0/20 | **FAIL** |
| collision + boundary <=2/20 | 1/20 | PASS |
| mean e2e >=6 Mbps | 6.13474 | PASS |
| 新安全优先规则优于随机 | 是 | PASS |

唯一但决定性的最终验收失败项是 normal completion；persistent outage 为 19/20 是其直接失败模式。

## 5. 三轮相同 20-seed 结果并排

| 项目 | A：entropy=0.01 | B：entropy=0 | C：entropy=0.001 |
|---|---:|---:|---:|
| selected best update | 900 | 200 | 700 |
| normal completion | 6/20 | 0/20 | 0/20 |
| persistent outage | 13 | 20 | 19 |
| collision / boundary | 1 / 0 | 0 / 0 | 1 / 0 |
| hard violations | 0 | 0 | 0 |
| mean e2e (Mbps) | 6.69411 | 5.90869 | 6.13474 |
| outage step ratio | 0.07437 | 0.16945 | 0.13354 |
| rate satisfaction ratio | 0.89972 | 0.77924 | 0.79462 |
| deterministic action saturation | 0.20300 | 0.00000 | 0.00106 |
| final-state raw log_std >2 | 0.999884 | 0.000000 | 0.964844 |
| final-state Gaussian entropy | 10.25662 | 0.38204 | 10.19442 |

结论：

1. **0.001 不避免全局 log_std 饱和。** 最终 best 的 96.5% raw 值超过 2，且 entropy 接近第一轮上界。
2. **0.001 比 0 保留了更多探索，但没有改善任务完成。** B 的最终 entropy 仅 0.382、C 为 10.194；C 的速率和 persistent outage 略优于 B，但两者均为 0 正常完成。
3. **C 没有超过第一轮的 6/20 normal completion，也没有降低第一轮的 13 个 persistent outage。** 第一轮仍是三者中唯一有 normal completion 的策略。
4. **本轮没有第一轮那种 900 点完成峰值可供“后期退化”判断。** 它从头到尾均未产生周期正常完成；后期是持续的 outage/边界波动，而非峰值后的稳定保留。

## 6. 下一步：actor_lr=1e-4 是否有充分证据

**结论：将 `actor_lr: 3e-4 -> 1e-4` 明确标记为下一项单变量候选，但本轮没有启动它。**

依据是：

- 第三轮 201--500 有 67.0% update 的 Actor 预裁剪 norm 超过 0.5，符合高频 Actor gradient clipping 的触发条件；此阶段也正是 raw `log_std` 开始持续上移的阶段。
- 第三轮未出现长期 KL 或 clip fraction 均值约 0.1 的问题；因此这不是“已经证明 actor_lr 过大”的因果结论，而是一个有针对性的、低风险的单变量实验理由。
- 相比第二轮，C 的 KL/clip 明显缓和，故没有证据优先改 critic_lr、clip epsilon、PPO epochs、网络、Environment 或 Reward。

若获得下一轮授权，唯一应改变的训练参数应是 `actor_lr=1e-4`；保留新 best 规则、周期 Actor checkpoints、`entropy_coef=0.001` 及其余冻结配置，以隔离 Actor 更新步长的影响。当前报告之后没有自动开始任何第四轮训练。

## 7. 结果路径

- 训练日志与配置：`artifacts/stage3-entropy001/train.csv`、`eval.csv`、`config.json`
- 完整/部署 checkpoint：`artifacts/stage3-entropy001/checkpoints/latest.pt`、`best.pt`、`actor_final.pt`
- 周期 Actor checkpoint：`artifacts/stage3-entropy001/eval_checkpoints/`
- 固定 20-seed best 评估：`artifacts/stage3-entropy001/final_best_eval.json`
- 随机基线：`artifacts/stage3-entropy001/random_baseline_eval.json`
- raw/clamped log_std 与 outage 只读回放：`artifacts/stage3-entropy001/final_checkpoint_outage_diagnosis.json`

