# Stage 4.1.1 协议收口修复与 GitHub 交付任务书

> 执行对象：Codex
>
> 项目仓库：`https://github.com/wdwd200/multi-relay-uav-mappo`
>
> 本地项目目录：以当前 Codex 工作目录为准，预期为 `D:\转移\桌面\work\multi-agent`
>
> 审查基线提交：`d96ef73aa290459849190e384053d1c89430fe71`

## 0. 总目标

完成 **Stage 4.1.1 工程收口修复**，解决 Stage 4.1 GitHub 审查发现的协议可复现性问题，并由 Codex 自行完成 Git 分支、提交、推送和 PR 创建。

本轮只允许修复：

- JSON manifest 跨平台哈希；
- 正式训练冻结配置入口；
- validation manifest 缺失时的错误退回；
- legacy checkpoint 连续恢复语义；
- checkpoint 配置反序列化校验；
- README、依赖、CI 与复现说明；
- 历史评估 checkpoint 的 GitHub Release 打包与上传。

本轮不是新实验，不允许启动任何新的正式训练或重新评估。

## 1. 已确认事实

以下结果已经过外部审查，本轮必须保留：

- validation-grid：60 checkpoints × 20 seeds = 1,200 条 Episode；
- final test：3 checkpoints × 50 seeds = 150 条 Episode；
- selected updates：P0=1000、P1=800、P2=650；
- P0 final：normal=0、persistent outage=44、collision=1、boundary=5、mean e2e=6.338344332983057 Mbps；
- P1 final：normal=0、persistent outage=14、collision=0、boundary=36、mean e2e=6.8470624105779105 Mbps；
- P2 final：normal=0、persistent outage=34、collision=0、boundary=16、mean e2e=6.235161143435673 Mbps；
- 60 个历史 evaluation Actor checkpoint 的文件哈希与 `checkpoint_manifest.json` 对应；
- P0/P1/P2 仍然只是 single-seed retrospective pilot；
- P3/P4 正式训练仍为 NOT STARTED；
- multi-seed P0–P4 training 仍为 NOT AUTHORIZED。

## 2. 绝对禁止项

1. 不运行 P0–P4 的任何 1000-update 正式训练。
2. 不重跑现有 1,200 条 validation 或 150 条 final-test Episode。
3. 不修改 Environment v1.1、通信、天线、Reward、outage、safety 或终止条件。
4. 不修改 P0–P4 Actor 结构、47 维 Critic 输入或 PPO/GAE 数学。
5. 不调整 gamma、lambda、Actor/Critic LR、entropy、value normalization、value clipping 等实验变量。
6. 不改写 Episode 的数值字段、终止类型、selected update 或 checkpoint 文件哈希。
7. 不把 `.pt` 文件或 ZIP 强行提交到普通 Git 历史。
8. 不使用 `git reset --hard`、强制推送或覆盖用户未提交修改。
9. 不合并 PR 到 `main`；等待下一轮 GitHub 审查。

## 3. 开始前检查

先执行并记录：

```text
git status --short
git remote -v
git branch --show-current
git rev-parse HEAD
```

要求：

- 确认远端是 `wdwd200/multi-relay-uav-mappo`；
- 确认基线包含提交 `d96ef73aa290459849190e384053d1c89430fe71`；
- 如存在用户未提交修改，必须保留并判断是否与任务冲突，不得直接清理；
- 从当前基线创建分支：`fix/stage4-1-1-protocol-hardening`。

## 4. 工作包 A：改为跨平台稳定的 JSON 哈希

### A1. 问题

当前 `sha256_file()` 对 JSON 文件原始字节求哈希。Stage 4.1 在 Windows 生成 CRLF 文件后，GitHub 保存为 LF，导致 fresh clone 的哈希不同。

三组旧值是 Windows CRLF 原始文件哈希：

- seed manifest：`d2622a91c144aa7d32ccd5bbf0353694ba281a2044b7985a0290d4c9e7227acb`
- checkpoint manifest：`c7dcbb98493230c890758dbcb76295ce0b34f99a0d651e2b06aa0601f787a435`
- selected checkpoints：`ff21a6534bbe646fa356a9d255695b8c2742faa290c9dad31f6dfaf37cd1df31`

### A2. 实现要求

- 保留 `sha256_file()`，只用于 `.pt`、ZIP 等二进制文件。
- 新增语义明确的 JSON 哈希函数，例如：

```text
sha256_json_payload(payload)
sha256_json_file(path)
```

- JSON 哈希必须基于已有 `canonical_json_bytes(payload)`：UTF-8、键排序、紧凑分隔符，不依赖缩进、CRLF/LF 或操作系统。
- seed manifest、checkpoint manifest、selected checkpoints 及其引用字段统一使用 `canonical-json-v1`。
- 在产物中显式记录 `manifest_hash_scheme = canonical-json-v1`，避免以后再次混用文件字节哈希与 JSON 内容哈希。
- `.pt` checkpoint 的 SHA-256 继续按原始二进制文件计算，禁止更改。
- 增加 `.gitattributes`，至少固定源码、Markdown、JSON、JSONL、CSV 为 LF；但不得把行尾规则当成 JSON 语义哈希的替代品。

### A3. 历史元数据迁移

编写一次性、可重复执行的迁移脚本，不允许手工搜索替换。迁移以下文件中的 manifest 引用：

- `artifacts/stage4-1-protocol-repair/validation_grid.csv`
- `artifacts/stage4-1-protocol-repair/validation_episodes.jsonl`
- `artifacts/stage4-1-protocol-repair/selected_checkpoints.json`
- `artifacts/stage4-1-protocol-repair/final_test_summary.csv`
- `artifacts/stage4-1-protocol-repair/final_test_episodes.jsonl`
- `artifacts/stage4-1-protocol-repair/protocol_audit.json`
- `artifacts/stage4-1-protocol-repair/protocol_audit.md`
- 根目录 `stage4_1_protocol_validation.json`
- 根目录 `stage4_1_protocol_validation.md`
- `22_阶段4.1_训练与评估协议修复交接.md`

`evaluation_invocations.jsonl` 属于历史执行轨迹：不得把旧调用伪装成新调用。可以保留原始字段，并增加明确的旧哈希方案标记或追加 metadata migration 事件。

新增 `artifacts/stage4-1-protocol-repair/hash_migration.json`，至少记录：

- 旧 raw-file/CRLF 哈希；
- 新 canonical JSON 哈希；
- 迁移时间；
- 修改文件清单；
- `episode_numeric_payload_changed=false`；
- checkpoint binary hashes unchanged 的验证结果。

迁移前后必须证明：

- validation 仍为 60 汇总行、1,200 Episode；
- final 仍为 3 汇总行、150 Episode；
- 所有 Episode 数值、seed、checkpoint hash、终止类型逐条不变；
- selected updates 仍为 1000/800/650；
- 从 raw Episode 重算的汇总结果仍完全一致。

## 5. 工作包 B：增加唯一的正式训练冻结配置入口

### B1. 问题

当前交接文档中的 P1–P4 命令会因 `state_dependent_clamp` 默认值而校验失败；P0 虽能启动，却会使用旧默认 `actor_lr=3e-4`、`entropy=0.01`、`log_std=[-5,2]`，不是冻结的 Round-E 公平对照配置。

### B2. 实现要求

- 增加一个原子化 preset，名称统一为 `stage4-formal`。
- 推荐命令形式：

```text
python train_plain_mappo.py --preset stage4-formal --full --actor-variant plain --run-seed 2026 --output-dir ...
```

- `--full` 必须要求显式提供 `--preset stage4-formal`；缺失时立即报错，不得按旧默认值启动 1000 updates。
- preset 必须一次性冻结所有共同实验字段，不能依靠用户逐项输入。
- 至少包含：

```text
actor_log_std_mode = state_independent_tanh
log_std_min = -4.0
log_std_max = 0.0
state_independent_log_std_init = -1.5
actor_lr = 1e-4
critic_lr = 3e-4
entropy_coef = 0.0
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
value_loss_coef = 0.5
max_grad_norm = 0.5
num_envs = 8
rollout_length = 128
mini_batch_size = 256
ppo_epochs = 10
eval_interval_updates = 50
full_updates = 1000
topology_node_dim = 6
topology_edge_dim = 7
graph_hidden_dim = 32
```

- 以实际 Stage 3 state-independent、P1、P2 的历史 config 为依据，逐字段核对共同冻结项；如发现任务书遗漏字段，应纳入 preset 并在交接报告说明。
- P0–P4 在相同 run seed 下，除 `actor_variant`、结构必然产生的参数量和 `output_dir` 外，所有共同训练字段及 Critic/RNG seeds 必须一致。
- CLI 配置构建逻辑应拆成可单测函数，测试配置时不得真正创建 Trainer 或启动训练。
- 更新交接文档中的未来命令矩阵，全部使用该 preset；本轮只验证配置，禁止执行这些 `--full` 命令。

## 6. 工作包 C：正式协议必须 fail closed

修改 `_periodic_validation_seeds()`：

- 对 `stage4.1-v1` 新协议，manifest 缺失、路径错误、内容错误、hash scheme 错误、validation split 数量不是 20、`selection_allowed` 不是 true 时必须抛出清晰异常；
- 禁止退回 `10000..10004`；
- 旧 5-seed 路径只能由明确的 `legacy_protocol` 使用；
- manifest 路径必须相对仓库/配置基准稳定解析，不得依赖启动命令所在的当前工作目录；
- test split 必须继续保持 `selection_allowed=false`。

## 7. 工作包 D：修复 checkpoint 协议一致性

### D1. legacy load → save → load

当前历史 checkpoint 首次加载后，再保存可能形成以下混合状态：

- config 声称 `stage4.1-v1`；
- protocol metadata 声称 `legacy_protocol`；
- `protocol_rng_state=None`。

修复后必须保证：

- legacy checkpoint 加载后仍明确为 legacy；
- 再次保存的 config、metadata、RNG state 三者语义一致；
- 新保存的 legacy checkpoint 可以再次加载；
- 不得把历史 checkpoint 伪装成使用 Stage 4.1 独立 RNG；
- resume 仍明确是 `fresh_episode_at_update_boundary`、`exact_environment_resume=false`。

### D2. `MappoConfig.from_dict()`

- 反序列化新协议 checkpoint 时，必须先检查原始字典中的 protocol seed/path 字段；
- 如果保存值与 `run_seed` 推导值不一致，必须报错；
- 禁止由 `__post_init__()` 静默覆盖错误值后再假装校验通过；
- 历史配置缺少协议字段时，仍显式识别为 `legacy_protocol`。

### D3. CUDA 小修复

若不扩大范围，顺手把 action-noise Generator 的设备从 `self.device.type` 改为完整设备语义，避免 `cuda:1` 等非默认 GPU 与 Generator 设备不一致。增加可静态验证或条件跳过的测试，不要求本轮具备 CUDA。

## 8. 工作包 E：新增协议回归测试

至少新增以下测试：

1. 同一 JSON 内容使用 LF、CRLF、不同缩进和不同键顺序时 canonical hash 相同。
2. JSON 内容真实变化时 canonical hash 变化。
3. `.pt` checkpoint 仍使用二进制文件哈希。
4. P0–P4 的 `stage4-formal` config 均可通过 validate。
5. P0–P4 相同 run seed 的共同字段、Critic seed、action seed、minibatch seed、env seed 完全一致。
6. `--full` 未提供 formal preset 时被拒绝，且没有创建 Trainer/输出目录。
7. 新协议 manifest 缺失时明确失败，不得返回旧 5 seeds。
8. legacy 协议仍可使用明确记录的旧监控 seeds。
9. legacy checkpoint 完成 load → save → load round-trip。
10. stage4.1 checkpoint 的原始 seed 字段被篡改时 `from_dict()` 拒绝加载。
11. migrated artifacts 仍满足 60/1,200、3/150、无重复、无非有限值。
12. raw Episode 重算结果及 selected updates 与本任务书第 1 节完全一致。

运行：

```text
python -m unittest discover -s selfcheck -p "test_*.py" -v
python -m compileall -q .
```

不得只报告“新增测试通过”；必须报告完整测试总数和总结果。

## 9. 工作包 F：README、依赖与 CI

### F1. README

重写过时的 Stage 1 README，至少说明：

- 当前完成到 Stage 4.1.1；
- Environment、Plain/Role/Topology/Graph MAPPO 模块状态；
- P0/P1/P2 只是 single-seed retrospective pilot；
- P3/P4 只有 smoke，正式训练未开始；
- 66/66 是修复前测试记录，本轮应写新的实际测试总数；
- 安装依赖、运行测试、运行只读 artifact verification 的命令；
- 将来正式训练命令仅作为未授权示例，并使用 `--preset stage4-formal`；
- checkpoint Release 的下载与完整性验证方法；
- 不把 evaluation Episodes 当作独立训练重复。

### F2. 依赖

- 增加适合当前项目的 `requirements.txt` 或 `pyproject.toml`；
- 根据实际 import 固定最低必要依赖，至少明确 Python、NumPy、PyTorch 的支持范围；
- 不虚构未实际验证的精确版本兼容性。

### F3. CI

增加 GitHub Actions：

- 在受支持的 Python 版本上安装依赖；
- 执行完整 unittest；
- 执行 compileall；
- 执行只读 Stage 4.1 artifact integrity verification；
- 不运行训练，不写回 artifacts，不依赖 GPU。

## 10. 工作包 G：checkpoint Release

如果本地 60 个历史 periodic Actor checkpoints 都存在：

1. 根据 `checkpoint_manifest.json` 再次验证数量、路径和 SHA-256。
2. 只打包以下内容：
   - P0/P1/P2 共 60 个 `actor_update_*.pt`；
   - `checkpoint_manifest.json`；
   - 简短的 `README_CHECKPOINTS.md`。
3. ZIP 内保留 P0/P1/P2 的清晰目录结构。
4. 计算并报告 ZIP SHA-256。
5. 不把 ZIP 或 `.pt` 加入 Git commit。
6. 使用已经配置好的 GitHub 凭据，由 Codex 自行创建并上传 Release：
   - tag：`stage4.1-retrospective-pilot-checkpoints`
   - title：`Stage 4.1 retrospective pilot checkpoints`
   - Release notes 必须明确写明：single-seed pilot、仅用于复核 60-checkpoint validation-grid、不是正式 multi-seed 结果。

如果 checkpoint 不全或 GitHub 身份验证失败：

- 不得伪造或跳过校验；
- 代码分支仍应正常提交和推送；
- 最终报告准确列出缺失文件或认证错误。

## 11. 最终只读验收

完成修改后必须检查：

```text
git diff --check
git status --short
git diff --stat
```

并验证：

- 没有 Environment/Reward/PPO 数学的意外改动；
- 没有新增正式训练产物；
- 没有 `.pt` 被提交；
- validation/final 数值结果未变化；
- 所有 manifest 引用使用同一 hash scheme；
- README 命令与实际 CLI 一致。

## 12. GitHub 交付授权与要求

本任务明确授权 Codex 自行完成 Git 操作，不要让用户手动复制 Git 命令。

按以下流程执行：

1. 在 `fix/stage4-1-1-protocol-hardening` 分支完成修改。
2. 检查最终 diff 和测试结果。
3. 提交信息：

```text
fix: harden stage4.1 protocol reproducibility
```

4. 将该分支推送到 `origin`，禁止 force push。
5. 如果已安装并登录 GitHub CLI，创建指向 `main` 的 PR：
   - 标题：`fix: harden Stage 4.1 protocol reproducibility`
   - 正文包含问题、修复、测试、未运行内容、artifact 数值未变声明。
6. 不自行合并 PR。
7. 如 GitHub CLI 不可用，至少完成分支推送，并给出可打开的 compare/PR URL。
8. 如上传 checkpoint Release，报告 Release URL 与 ZIP SHA-256。

## 13. 最终汇报格式

完成后只给一份收口报告，必须包含：

```text
Stage 4.1.1 protocol hardening = COMPLETE / BLOCKED
Full automated tests = X/X PASS / FAIL
Artifact integrity = PASS / FAIL
Validation records = 60 summaries / 1,200 Episodes
Final-test records = 3 summaries / 150 Episodes
Selected updates = P0 1000 / P1 800 / P2 650
Episode numeric payload changed = false / true
Formal training executed = false
Evaluation rerun executed = false
P3 formal training = NOT STARTED
P4 formal training = NOT STARTED
Multi-seed P0-P4 training = NOT AUTHORIZED
Git branch = ...
Git commit SHA = ...
PR URL = ...
Checkpoint Release URL = ... / NOT CREATED（说明原因）
Checkpoint ZIP SHA-256 = ... / N/A
Remaining blockers = ...
```

同时列出：

- 修改文件；
- 新增测试；
- 每条测试命令的真实输出摘要；
- 新旧 manifest hash 映射；
- 所有未执行内容。

只有代码、测试、artifact 验证、Git push 全部完成，才能将 Stage 4.1.1 标记为 COMPLETE。不得仅凭“代码已写完”宣称通过。
