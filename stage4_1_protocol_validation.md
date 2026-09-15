# Stage 4.1 训练与评估协议修复验证

本报告记录 Stage 4.1 的真实协议修复、旧 checkpoint 独立重评和最终 test。没有执行任何新的 1000-update 正式训练。

## Seed 与选择协议

- protocol: `stage4.1-v1`；validation/test manifest SHA-256: `d2622a91c144aa7d32ccd5bbf0353694ba281a2044b7985a0290d4c9e7227acb`。
- validation: 20 seeds `50000000..50000019`；final test: 50 seeds `60000000..60000049`；两者严格分离。
- checkpoint 只按 validation 的 frozen safety-priority key 选择；完全同分选择较早 update；test 未参与选择。

## Selected checkpoints

| Variant | update | validation score | SHA-256 |
|---|---:|---|---|
| P0 plain | 1000 | [0.0, -0.0, 0.0, 20.0, 0.12326727291145845, -0.8163723501380936, -6.298344236649277] | `4f42c320fe424e261a049e9a068d7ed44dccb3e055f548b21573209011bf497f` |
| P1 role_info | 800 | [0.0, -0.0, 11.0, 9.0, 0.10212658530594407, -0.8784648995853495, -6.630984683654401] | `fd689fdd214e9cd78aadbd600b5e49dee40c908adbb93bb96fc6cb2d969e51e4` |
| P2 role_head | 650 | [0.0, -0.0, 5.0, 15.0, 0.14521564917170432, -0.8252635142121043, -6.222161383668572] | `31e08acb7501a6f15e3e6eaa4735e7f50cedf2729007ce69a3a821509a0d65b6` |

## Independent 50-episode final test

| Variant | normal completion | persistent outage | collision | boundary | hard violations | mean e2e Mbps | outage ratio | rate satisfaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | 0/50 | 44 | 1 | 5 | 0 | 6.338344 | 0.112996 | 0.817126 |
| role_head | 0/50 | 34 | 0 | 16 | 0 | 6.235161 | 0.133500 | 0.825030 |
| role_info | 0/50 | 14 | 0 | 36 | 0 | 6.847062 | 0.070595 | 0.915737 |

## Critic / gradient diagnostics

| Historic run | critic pre-clip norm min / mean / max | New value diagnostics in historic log |
|---|---|---|
| P0 | 14.335824 / 50.130775 / 159.758854 | False |
| P1 | 8.742287 / 31.312344 / 123.774209 | False |
| P2 | 12.762234 / 34.609306 / 194.510475 | False |

10-update diagnostic smoke final update: value target mean/std=24.978790283203125/12.849007606506348; prediction mean/std=14.383151054382324/6.567276954650879; explained variance=0.46721198071575964; actor clip fraction=0.6; critic clip fraction=1.0; finite=1.0.
- Recommendation: `PRE-EXPERIMENT REQUIRED`. The evidence is diagnostic only; no value normalization, value clipping or critic-LR change was enabled.

## Gamma / horizon diagnosis

| gamma | half-life steps | half-life seconds | terminal weight at 100 s | GAE trace half-life seconds |
|---:|---:|---:|---:|---:|
| 0.990 | 68.968 | 13.794 | 0.00657048 | 2.260 |
| 0.995 | 138.283 | 27.657 | 0.08157186 | 2.462 |
| 0.997 | 230.702 | 46.140 | 0.22262768 | 2.553 |
| 0.999 | 692.801 | 138.560 | 0.60637894 | 2.651 |
- Recommendation: `PRE-EXPERIMENT REQUIRED`. gamma=0.99 gives a 100-s terminal reward weight of 0.00657048 and a 13.79-s half-life; observed terminal/outage timing must be tested under a separately authorized symmetric protocol before changing gamma.

## Resume semantics

`resume_mode = fresh_episode_at_update_boundary`; `exact_environment_resume = false`. A resume restores model/optimizer/normalizer/RNG/unused seed progress but starts fresh episodes; it is not a bitwise-equivalent uninterrupted trajectory.

## Audit caveat

The task-specified `20_...` and `00_...` source documents were absent from this workspace. This audit used current source code plus actual artifacts and records that conflict explicitly.

## Execution trace note

`evaluation_invocations.jsonl` records completed bounded evaluation commands. The initial P0 all-grid command reached the host 30-second limit after writing output; the only duplicated P0 950/1000 records were mechanically deduplicated by checkpoint SHA-256 and seed. The final integrity audit verifies exactly 60×20 unique validation Episode rows and 3×50 final-test rows.

## Status

Stage 4.1 protocol repair = COMPLETE

Engineering validation = PASS

P0/P1/P2 retrospective status = SINGLE-SEED PILOT

P3 formal training = NOT STARTED

P4 formal training = NOT STARTED

Multi-seed P0-P4 training = NOT AUTHORIZED

Role+Graph = NOT STARTED
