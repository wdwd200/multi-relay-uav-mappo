# Stage 2：数值校准、统计测试与 Environment v1 验收数据生成

## 0. 当前状态

Stage 1 Baseline 已完成并验收：

- 13/13 PASS
- 基础环境骨架已经正确
- 不允许重新设计或大规模重构环境
- 本阶段不进行任何强化学习训练

本阶段目标：

1. 在 Stage 1 Baseline 上应用指定的候选参数；
2. 修复 H/L 最终速度轻微超限问题；
3. 编写 Stage 2 数值统计与压力测试；
4. 输出完整原始统计数据和汇总结果；
5. 停止开发，等待人工验收。

禁止自行根据测试结果继续调参。

---

# 1. 严格禁止加入的内容

本阶段禁止实现：

- PPO
- MAPPO
- GNN / GAT
- Graph MAPPO
- Role
- RG-MAPPO
- CBF / HOCBF / QP Safety Filter
- 动态路由
- UAV buffer
- 动态 Agent 数量
- 功率优化
- 能耗优化

Stage 2 只处理环境参数和测试。

---

# 2. 应用以下候选参数

## 2.1 通信

保持：

B = 10 MHz

P = 0.1 W

N0 = -169 dBm/Hz

beta0 = -60 dB @ 1 m

alpha = 2

硬 outage 门限修改为：

gamma_min_db = 5 dB

代码实际使用线性 SNR 时：

gamma_min = 10^(5/10)
          ≈ 3.1622776601683795

注意：

禁止错误地直接写 gamma_min = 5。

---

## 2.2 H/L 任务距离

初始三维欧氏距离继续保持：

1000 m <= d_HL(0) <= 1500 m

运行期最大三维任务距离修改为：

D_max = 2000 m

必须继续保证：

d_HL(t) <= D_max

---

## 2.3 H/L 运动

水平巡航目标速度：

6 ~ 10 m/s

垂直速度：

-3 ~ +3 m/s

候选最大水平加速度：

1 m/s^2

候选最大垂直加速度：

1 m/s^2

waypoint 典型航段：

250 ~ 450 m

继续采用：

Constrained Random Waypoint

不得改成每个 0.2 s Step 随机方向。

---

# 3. 必须修复 Stage 1 已知问题

Stage 1 压力测试曾出现：

水平速度约 10.03 m/s

垂直速度约 3.185 m/s

原因是任务距离等修正以后，最终状态可能重新轻微超过速度上限。

要求：

所有 waypoint / boundary / D_max 等修正完成后，
增加最终严格速度限幅。

最终必须保证：

||v_HL_xy|| <= 10 m/s

|v_HL_z| <= 3 m/s

注意不要破坏运动连续性和同步更新逻辑。

---

# 4. 安全参数

保持硬安全距离：

d_safe = 20 m

修改安全预警距离：

d_warn = 60 m

逻辑继续保持：

d >= 60 m
→ 正常

20 m < d < 60 m
→ warning / soft penalty

d < 20 m
→ safety violation
→ terminated = True

不要加入 Safety Filter。

---

# 5. persistent outage

暂时继续保持：

25 steps

由于：

dt = 0.2 s

所以：

25 steps = 5 s

短时 outage 不立即终止。

恢复连接后：

consecutive_outage_steps = 0

连续达到阈值才：

terminated = True

---

# 6. Reward 候选参数

Reward 结构不得改变：

r_t =
R_e2e_effective / R_ref
- lambda_o * I_outage
- P_warn
- lambda_T * I_failure

候选值：

R_ref = 6 Mbps

lambda_o = 1

lambda_T = 5

注意：

这些仍属于 Stage 2 候选参数。

不要因为测试结果自行再次修改。

当前已有 warning penalty 的函数形式原则上保持，
但必须在统计结果中报告它的实际取值范围。

---

# 7. Observation 归一化候选尺度

保留现有 26 维 Observation 结构，不增加或删除特征。

候选归一化：

水平绝对位置：
2000 m

高度：
使用当前 [100,300] m 范围进行合理中心化/缩放；
推荐：
(z - 200) / 100

自身水平速度：
20 m/s

自身垂直速度：
6 m/s

相对水平位置：
2000 m

相对高度：
200 m

相对水平速度：
40 m/s

相对垂直速度：
12 m/s

单跳容量：
候选 C_norm = 60 Mbps

容量归一化暂可采用：

min(C / 60 Mbps, 1)

但是测试时必须同时记录：

1. normalization 前的原始 Observation；
2. normalization 后的 Observation。

不得只保存归一化结果。

---

# 8. 新增 Stage 2 专用统计程序

不要依赖 RL。

建议新增独立入口，例如：

stage2_calibration.py

或者项目现有结构下含义等价的文件。

不得污染正常 Env API。

Stage 2 测试至少包含三种运行方式：

### A. Nominal / reference 模式

使用简单确定性的参考位置控制方式，
使中继大致追踪 H-L 之间的等距位置。

该控制器只用于 Stage 2 环境数值统计，
不是论文算法，也不能写入 RL 模块。

用途：

观察环境在“合理中继布局”下的正常通信性能范围。

### B. Zero-action 模式

中继输入零动作。

用途：

观察移动 H/L 时环境自然退化、outage 和恢复需求。

### C. Random-action stress 模式

合法动作范围内随机动作。

用途：

压力测试：

- safety
- boundary
- outage
- termination
- NaN / Inf
- 数值稳定性

三种模式的数据必须分开报告，不能混在一起求一个平均值。

---

# 9. 主统计实验

主统计：

K = 4

建议至少：

500 个 Episode

每个：

最多 500 steps

使用连续确定 seed 集合，例如：

seed = 0 ... 499

如果运行时间明显过长，可保留命令行参数允许减少 Episode，
但最终正式统计应尽量完成 500 Episode。

另外进行：

K = 3
K = 5

兼容性 smoke test。

它们不需要和 K=4 做相同规模的大统计。

---

# 10. 必须记录的数据

每个模式至少统计：

## H/L

- reset 成功率
- d_HL(0)
- 每 Step 的 d_HL
- d_HL min / mean / median / P95 / P99 / max
- 达到 1600 / 1800 / 2000 m 附近的频率
- H/L 水平速度 min/max
- H/L 垂直速度 min/max
- H/L 加速度范围
- waypoint 切换次数 / Episode
- waypoint 航段实际长度分布
- D_max 修正触发次数

## 单跳通信

所有 hop 的：

- 三维距离
- SNR linear
- SNR dB
- capacity

至少输出：

min
mean
median
P5
P95
P99
max

并统计：

gamma < 5 dB

的比例。

## 端到端通信

记录：

- raw R_e2e
- effective R_e2e
- outage flag

统计：

min
mean
median
P5
P95
P99
max

单位统一明确为 Mbps。

## outage

统计：

- outage step ratio
- 每 Episode outage 次数
- outage 连续持续长度
- persistent outage termination 数量
- outage 后成功恢复数量
- 最大 consecutive_outage_steps

## safety

统计：

- d < 60 m warning 频率
- d < 20 m violation 次数
- safety termination 次数
- minimum pair distance

## boundary

统计：

- boundary violation 次数
- boundary termination 次数

## Episode

统计：

- 正常 truncated 数量
- safety terminated 数量
- boundary terminated 数量
- persistent outage terminated 数量
- Episode 平均实际长度

## Reward

分开记录：

- throughput reward
- outage penalty
- warning penalty
- failure penalty
- total reward

给出各项实际数值分布。

## Observation

对 26 个维度分别统计原始值：

min / max / mean / std / P1 / P99

并对归一化后的 26 个维度再次统计同样内容。

必须特别标记：

- 是否存在 NaN
- 是否存在 Inf
- 是否大量超出 [-1,1]
- 哪些维度发生 clipping
- clipping 比例

---

# 11. 必须进行的硬性回归检查

Stage 1 原有 13 项测试全部重新运行。

要求：

13/13 PASS

另外增加 Stage 2 硬检查：

1. gamma_min 确认为 5 dB 等价线性值；
2. d_HL(0) 始终位于 1000~1500 m；
3. d_HL(t) 不超过 2000 m；
4. H/L 水平速度最终严格 <= 10 m/s；
5. H/L 垂直速度最终严格 <= 3 m/s；
6. d_warn = 60 m；
7. d_safe = 20 m；
8. persistent outage = 25 steps；
9. outage 时 effective R_e2e = 0；
10. 恢复连接后 consecutive_outage_steps 清零；
11. 无 NaN；
12. 无 Inf；
13. 固定 seed 可复现；
14. K=3/4/5 均能正常 reset + step；
15. terminated / truncated 语义保持正确。

数值比较允许合理浮点 tolerance。

---

# 12. 输出文件

请输出至少：

stage2_summary.json

stage2_episode_stats.csv

stage2_hop_stats.csv

stage2_observation_stats.csv

stage2_validation_report.md

如果方便，可额外生成少量 PNG 图：

- H/L distance distribution
- R_e2e distribution
- hop SNR distribution
- outage duration distribution

图仅用于辅助验收，不要投入大量时间美化。

---

# 13. validation_report.md 必须包含

报告开头明确写：

Stage 2 implementation validation:
PASS / FAIL

注意：

这里的 PASS 只代表：

“代码修改正确，统计数据成功生成。”

不得自行声明：

“Environment v1 已最终冻结。”

Environment v1 是否冻结由人工根据统计结果判断。

报告必须包含：

1. 修改了哪些参数；
2. 修改了哪些文件；
3. 新增了哪些测试；
4. Stage 1 regression 结果；
5. Stage 2 hard checks；
6. 三种运行模式分别的关键统计；
7. 所有异常；
8. 尚未决定的参数；
9. 输出数据文件路径。

---

# 14. 重要开发原则

- 尽量小改动；
- 不重构已验收模块；
- 不改变 Env API，除非确有必要；
- 不删除 Stage 1 测试；
- 不为了让测试 PASS 而弱化断言；
- 不看到数据后自行调参数；
- 不开始 MAPPO；
- 不替用户决定 Environment v1 已经通过。

执行结束后停止。

最终只汇报：

1. 完成情况；
2. 测试结果；
3. 关键统计摘要；
4. 生成文件；
5. 是否存在异常。

等待下一步人工验收。
