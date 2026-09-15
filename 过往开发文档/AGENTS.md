---
title: Stage 2.5 仰角相关全向偶极子天线模型开发任务书
project: 论文
status: ready-for-codex
updated: 2026-09-05
basis:
  - 00_论文当前总体方案与决策记录_阶段2验收版.md
  - 05_阶段2最终验收与阶段3启动交接.md
purpose: 在冻结的 Environment v1 上加入仰角相关天线增益，并进行局部重新验收；不进入 MAPPO、Graph、Role 或世界模型开发
---

# 1. 任务目标

在当前已通过 Stage 1 / Stage 2 验收的 Environment v1 上，加入：

\[
\boxed{\text{仰角相关的竖直全向偶极子天线链路增益}}
\]

得到候选：

\[
\boxed{\text{Environment v1.1}}
\]

本阶段只修改通信模型及其相关数值验收，不改动：

- H/L 运动模型；
- Relay 动力学；
- 安全距离逻辑；
- Episode 时序；
- 26 维 Observation 的结构；
- Reward 的结构；
- MAPPO / PPO；
- Graph；
- Role；
- World Model；
- CBF/HOCBF。

# 2. 当前冻结基线

以下参数以 Stage 2 最终验收版本为准。

## 2.1 时间与场景

\[
\Delta t=0.2\text{ s}
\]

\[
T_{\max}=500\text{ steps}=100\text{ s}
\]

\[
x,y\in[0,2000]\text{ m}
\]

\[
z\in[100,300]\text{ m}
\]

主实验：

\[
K=4
\]

代码继续保持 K=3/4/5 可运行。

## 2.2 任务尺度

初始 H/L 三维距离：

\[
1000\le d_{HL}(0)\le1500\text{ m}
\]

运行期最大三维距离：

\[
D_{\max}=2000\text{ m}
\]

## 2.3 通信基线

\[
B=10\text{ MHz}
\]

\[
P=0.1\text{ W}=20\text{ dBm}
\]

\[
N_0=-169\text{ dBm/Hz}
\]

\[
\beta_0=-60\text{ dB，参考距离 }1\text{ m}
\]

\[
\alpha=2
\]

硬 outage：

\[
\gamma_{\min}=5\text{ dB}
\]

持续 outage：

\[
25\text{ steps}=5\text{ s}
\]

原信道：

\[
h_{ij}^{old}=\frac{\beta_0}{d_{ij}^{\alpha}}
\]

容量：

\[
C_{ij}=B\log_2(1+\gamma_{ij})
\]

DF + TDMA：

\[
R_{\mathrm{e2e}}
=
\frac{1}{\sum_i1/C_i}
\]

任意一跳低于 SNR 门限：

\[
R_{\mathrm{e2e}}^{effective}=0
\]

## 2.4 Reward

\[
R_{\mathrm{ref}}=6\text{ Mbps}
\]

\[
\lambda_o=1
\]

\[
\lambda_T=5
\]

warning penalty 当前保持 0.2。

# 3. 新增天线模型

## 3.1 建模假设

每个空中节点采用：

\[
\boxed{\text{竖直安装的全向偶极子天线}}
\]

仅考虑链路仰角带来的相对方向增益。

明确忽略：

- UAV roll；
- UAV pitch；
- UAV yaw；
- 机体遮挡；
- 天线安装位置扰动；
- 天线主动指向；
- 波束赋形；
- 方位角相关增益。

方位面视为全向。

## 3.2 水平距离

对节点 \(i,j\)：

\[
d_{ij}^{xy}
=
\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}
\]

## 3.3 高度差

\[
\Delta z_{ij}=z_i-z_j
\]

## 3.4 三维距离

继续使用：

\[
d_{ij}
=
\sqrt{(d_{ij}^{xy})^2+(\Delta z_{ij})^2}
\]

## 3.5 仰角

定义相对于水平面的链路仰角：

\[
\boxed{
\phi_{ij}
=
\arctan2(
|\Delta z_{ij}|,
d_{ij}^{xy}
)
}
\]

建议代码使用 `atan2(abs(dz), d_xy)`，不要直接使用 `atan(abs(dz)/d_xy)`，避免 \(d_{xy}\to0\) 时出现除零问题。

仰角范围：

\[
0\le\phi_{ij}\le\frac{\pi}{2}
\]

## 3.6 链路天线增益

采用收发两端竖直全向偶极子合计的归一化链路增益：

\[
\boxed{
G_{ij}^{ant}
=
\cos^2\phi_{ij}
}
\]

这里的 \(G_{ij}^{ant}\) 已经是链路级合计增益，不再额外平方。

等价地：

\[
G_{ij}^{ant}
=
\frac{(d_{ij}^{xy})^2}
{(d_{ij}^{xy})^2+(\Delta z_{ij})^2}
\]

但代码和论文主表达优先保留仰角形式。

## 3.7 新信道

将原模型：

\[
h_{ij}^{old}
=
\frac{\beta_0}{d_{ij}^{\alpha}}
\]

修改为：

\[
\boxed{
h_{ij}^{new}
=
\frac{\beta_0 G_{ij}^{ant}}
{d_{ij}^{\alpha}}
}
\]

其中 \(G_{ij}^{ant}\) 只表示相对于水平最佳方向的归一化方向增益。

不得额外加入 2.15 dBi 等绝对偶极子峰值增益，避免改变原 \(\beta_0\) 基准含义。

# 4. 代码修改要求

## 4.1 不允许大规模重构

优先在现有通信计算模块内局部修改。

不得为了本任务：

- 改写环境主循环；
- 改 Observation 维度；
- 改 Action；
- 改 H/L 运动；
- 改 Relay 动力学；
- 引入新 RL 代码；
- 引入 World Model；
- 引入新的第三方深度学习框架。

## 4.2 新增可检查量

建议通信计算函数至少能够输出/记录：

- `horizontal_distance`
- `height_difference`
- `elevation_angle_rad`
- `elevation_angle_deg`
- `antenna_gain_linear`
- `antenna_gain_db`
- `snr_linear`
- `snr_db`
- `capacity_bps`

这些量不要求全部加入 Agent Observation。

其中 Observation 仍保持原 26 维，只是其中链路容量 \(C^{up},C^{down}\) 自动使用新的通信模型结果。

## 4.3 数值稳定性

必须处理：

### 情况 A：同高度

\[
\Delta z=0
\]

应有：

\[
\phi=0
\]

\[
G^{ant}=1
\]

新模型退化为原模型。

### 情况 B：水平距离趋近 0

使用 `atan2` 保证：

\[
\phi\to90^\circ
\]

\[
G^{ant}\to0
\]

不得出现 NaN / Inf。

若实际代码中需要设置极小 epsilon，只允许用于数值稳定，不得改变正常区间的物理值。

# 5. 单元测试

新增以下测试。

## Test 1：同高度退化测试

构造：

\[
\Delta z=0
\]

检查：

\[
G^{ant}=1
\]

且：

\[
h^{new}=h^{old}
\]

允许浮点误差。

## Test 2：仰角单调性

固定水平距离，例如：

\[
d_{xy}=300\text{ m}
\]

依次设置：

\[
|\Delta z|=0,50,100,150,200\text{ m}
\]

检查：

\[
\phi\uparrow
\]

\[
G^{ant}\downarrow
\]

## Test 3：对称性

交换两节点高度：

\[
\Delta z\rightarrow-\Delta z
\]

必须满足：

\[
G_{ij}^{ant}=G_{ji}^{ant}
\]

链路增益一致。

## Test 4：典型角度校验

检查：

\[
\phi=0^\circ
\Rightarrow G=1
\]

\[
\phi=45^\circ
\Rightarrow G=0.5
\]

对应：

\[
G_{dB}\approx-3.0103\text{ dB}
\]

## Test 5：极端垂直链路稳定性

构造：

\[
d_{xy}\approx0,\quad |\Delta z|>0
\]

检查：

- 不出现 NaN；
- 不出现 Inf；
- \(G^{ant}\) 接近 0；
- SNR / capacity 计算稳定。

# 6. 通信专项数值检查

不训练 RL。

至少输出下列典型组合：

| \(d_{xy}\) | \(|\Delta z|\) |
|---:|---:|
| 300 m | 0 m |
| 400 m | 0 m |
| 500 m | 0 m |
| 400 m | 200 m |
| 300 m | 200 m |
| 200 m | 150 m |
| 200 m | 200 m |

对每一组输出：

1. 三维距离；
2. 仰角；
3. \(G^{ant}\)；
4. 天线损失 dB；
5. 原 SNR；
6. 新 SNR；
7. 新单跳容量；
8. 是否低于 5 dB outage 门限。

目的：

\[
\boxed{\text{确认新模型数值与人工预期一致}}
\]

# 7. Stage 2 局部重新验收

## 7.1 重点重新检查的参数

只重新审查：

\[
\boxed{\gamma_{\min}=5\text{ dB}}
\]

\[
\boxed{R_{\mathrm{ref}}=6\text{ Mbps}}
\]

\[
\boxed{D_{\max}=2000\text{ m}}
\]

注意：

先统计，不允许自动改参数。

任何参数调整都必须先给出数据，再由人工决定。

## 7.2 保持不变的参数

除非测试暴露真实 bug，否则不要修改：

- Step = 0.2 s；
- Episode = 500 steps；
- K=4 主实验；
- H/L 速度与加速度；
- Relay 速度与加速度；
- \(d_{\mathrm{safe}}=20m\)；
- \(d_{\mathrm{warn}}=60m\)；
- persistent outage = 25 steps；
- \(\lambda_o=1\)；
- \(\lambda_T=5\)；
- warning penalty = 0.2；
- Observation 26 维结构。

# 8. 回归统计

使用现有 Stage 2 三模式框架：

- Nominal；
- Zero-action；
- Random-stress。

第一轮先做小规模回归：

\[
20\text{ seeds/mode}
\]

每个 Episode 最多：

\[
500\text{ steps}
\]

统计：

- reset 成功率；
- terminated / truncated；
- outage steps；
- persistent outage termination；
- safety termination；
- boundary termination；
- mean effective \(R_{\mathrm{e2e}}\)；
- 单跳 SNR 分布；
- 仰角分布；
- 天线增益分布；
- H/L 距离范围；
- NaN / Inf。

如果小规模回归正常，再决定是否重新运行 Stage 2 的：

\[
500\text{ Episodes}\times3\text{ modes}
\]

不得默认直接运行大规模统计。

# 9. 额外诊断：检查是否“压死第三维”

新增统计：

- 所有通信跳的 \(|\Delta z|\) 分布；
- 所有通信跳的 \(\phi\) 分布；
- 各 relay 的高度轨迹；
- 各 relay 与上下游的高度差。

目的不是要求它们必须高度分散，而是检查：

> 新天线模型是否导致正常策略被异常强迫到几乎完全同高度。

注意：

这一项当前只做诊断，不人为添加鼓励高度差的 Reward。

# 10. 兼容性要求

修改后必须保证：

- K=3：PASS；
- K=4：PASS；
- K=5：PASS；
- 固定 seed 可复现；
- Stage 1 原有 13 项测试不因本次修改无理由失效；
- D_max preventive guard 逻辑保持不变；
- 无 NaN / Inf。

如原测试中某个“通信固定期望值”因模型变化而合理失效，应更新该测试的期望值，但必须记录为什么更新，不能直接删除测试。

# 11. 输出文件

Codex 完成后至少交付：

1. 修改后的源代码；
2. 新增/更新的测试代码；
3. `stage2_5_antenna_validation.md`
4. 小规模三模式统计结果；
5. 典型几何组合通信表；
6. 修改文件清单。

`stage2_5_antenna_validation.md` 至少包括：

- 改了哪些函数；
- 新公式；
- 单元测试结果；
- 典型 SNR / capacity 表；
- 20 seeds × 3 modes 结果；
- 5 dB 是否仍合理的初步数据；
- \(R_{\mathrm{ref}}=6Mbps\) 是否仍合理的初步数据；
- \(D_{\max}=2000m\) 是否仍合理的初步数据；
- 是否观察到第三维被明显压缩；
- 是否存在 NaN / Inf；
- 当前是否建议进入大规模重新验收。

# 12. 明确禁止

本阶段禁止：

- 实现 PPO / MAPPO；
- 修改 Actor/Critic；
- 加 Graph；
- 加 Role；
- 加 World Model；
- 加辅助预测网络；
- 加仰角到 Observation；
- 加天线增益到 Observation；
- 改 Reward 结构；
- 增加垂直运动奖励；
- 加功率控制；
- 加方位角天线模型；
- 加 UAV 姿态角；
- 加波束赋形；
- 加机体遮挡模型。

# 13. 验收标准

本阶段第一轮完成条件：

1. 仰角公式正确；
2. \(G^{ant}=\cos^2\phi\) 正确；
3. 同高度时严格退化到旧通信模型；
4. 典型几何组合数值正确；
5. K=3/4/5 可运行；
6. seed 可复现；
7. 无 NaN / Inf；
8. Stage 1 非通信相关能力不被破坏；
9. 20 seeds × 3 modes 小规模回归完成；
10. 给出 5 dB、6 Mbps、2000 m 三个冻结参数的“保留/需进一步复查”数据结论；
11. 不擅自修改这些冻结参数；
12. 不引入任何 RL / World Model 新模块。

完成上述内容后停止。

不要直接开始 Stage 3，也不要自行运行新的大规模训练。

等待人工验收后再决定：

\[
\boxed{\text{Environment v1.1 是否冻结}}
\]
