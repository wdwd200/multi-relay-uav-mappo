# Multi-relay UAV MAPPO

This repository contains the frozen Environment v1.1 and the Stage 3/4 MAPPO
research pipeline. The current engineering handoff is **Stage 4.1.1 protocol
hardening**. It strengthens reproducibility metadata; it does not introduce a
new policy experiment.

## Current scope and evidence

- Environment v1.1, the 26-dimensional local observation, reward,
  communication model, 47-dimensional centralized Critic state, and PPO/GAE
  mathematics are frozen.
- P0 Plain MAPPO, P1 Role-info, P2 Role-structured, P3 Topology-info, and P4
  Graph Actor implementations are available. P0/P1/P2 have only historical
  **single-seed retrospective pilot** results; they are not multi-seed claims.
- P3 and P4 passed smoke checks only. Their formal training is **NOT
  STARTED**. Multi-seed P0--P4 training is **NOT AUTHORIZED**.
- The historical Stage 4.1 validation grid contains 60 checkpoints × 20
  validation seeds = 1,200 Episodes. The held-out final test contains 3
  selected checkpoints × 50 test seeds = 150 Episodes. Evaluation Episodes
  are not independent training repeats.
- The latest local test run for this handoff was **76/76 PASS**. The obsolete
  `66/66` figure belongs to the pre-Stage-4.1.1 record.

See [`22_阶段4.1_训练与评估协议修复交接.md`](22_阶段4.1_训练与评估协议修复交接.md),
[`stage4_1_protocol_validation.md`](stage4_1_protocol_validation.md), and
`artifacts/stage4-1-protocol-repair/hash_migration.json` for the historical
record and canonical-hash migration audit.

## Install and verify

The source requires Python 3.10+ syntax. The only runtime dependencies are
NumPy and PyTorch; the CI workflow installs the unpinned minimum requirements
on CPython 3.11 and 3.12 rather than asserting an unverified exact version
matrix.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s selfcheck -p "test_*.py" -v
python -m compileall -q .
```

The committed artifact metadata can be checked without writing artifacts or
using a GPU:

```powershell
python verify_stage4_1_artifacts.py --metadata-only
```

With the release checkpoint package extracted, run full binary SHA-256
verification against its P0/P1/P2 directory root instead:

```powershell
python verify_stage4_1_artifacts.py --checkpoint-root <extracted>\stage4.1-retrospective-pilot-checkpoints
```

`canonical-json-v1` hashes JSON semantic content (UTF-8, sorted keys, compact
separators), so CRLF/LF and indentation do not change a manifest identity.
`.pt` and `.zip` values always use raw binary SHA-256.

## Historical checkpoint release

The release tag `stage4.1-retrospective-pilot-checkpoints` packages only the
60 periodic Actor checkpoints used by the retrospective validation grid,
`checkpoint_manifest.json`, and `README_CHECKPOINTS.md`. It is a single-seed
pilot audit asset, not a formal multi-seed result. Download the release ZIP,
verify its published ZIP SHA-256, extract it according to its included README,
then run the full verification command above.

## Future formal command (not authorized here)

The CLI refuses `--full` unless the atomic preset is stated explicitly. The
following is documentation only; do not execute it without separate
authorization:

```powershell
python train_plain_mappo.py --preset stage4-formal --full --actor-variant plain --run-seed 2026 --output-dir artifacts/formal/P0/run_2026
```

The same preset is used for `role_info`, `role_head`, `topology_info`, and
`graph`; only the Actor variant, its structurally required parameters, and the
output directory may differ. It freezes the historical common PPO settings,
state-independent bounded log standard deviation, seed-domain derivation, and
topology dimensions in one testable configuration builder.
