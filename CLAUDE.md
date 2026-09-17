# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **multi-relay UAV MAPPO research pipeline** implementing Stage 3 and Stage 4 policy variants for a frozen Environment v1.1. The current engineering state is **Stage 4.1.1 protocol hardening**, which strengthens reproducibility metadata without introducing new policy experiments.

**Critical constraints:**
- Environment v1.1 (26-dimensional local observation, 47-dimensional centralized Critic state, reward structure, communication model) is **frozen**.
- P0 Plain MAPPO, P1 Role-info, P2 Role-structured have only **single-seed retrospective pilot** results. These are not multi-seed claims.
- P3 Topology-info and P4 Graph Actor have passed smoke checks only. Their formal training is **NOT STARTED**.
- Multi-seed P0–P4 training is **NOT AUTHORIZED** without separate approval.

## Essential Commands

### Installation and verification
```powershell
# Install dependencies (requires Python 3.10+ syntax; tested on CPython 3.11/3.12)
python -m pip install -r requirements.txt

# Run all tests (currently 76 tests)
python -m unittest discover -s selfcheck -p "test_*.py" -v

# Syntax check
python -m compileall -q .

# Verify artifact metadata integrity (read-only, no GPU required)
python verify_stage4_1_artifacts.py --metadata-only
```

### Historical checkpoint verification
With the release checkpoint package extracted:
```powershell
python verify_stage4_1_artifacts.py --checkpoint-root <path>\stage4.1-retrospective-pilot-checkpoints
```

### Training commands (documentation only — NOT AUTHORIZED to execute)
The following demonstrates the frozen `stage4-formal` preset structure for future formal runs. **Do not execute without explicit authorization:**

```powershell
# P0 Plain MAPPO
python train_plain_mappo.py --preset stage4-formal --full --actor-variant plain --run-seed 2026 --output-dir artifacts/formal/P0/run_2026

# P1 Role-info
python train_plain_mappo.py --preset stage4-formal --full --actor-variant role_info --run-seed 2026 --output-dir artifacts/formal/P1/run_2026

# P2 Role-structured
python train_plain_mappo.py --preset stage4-formal --full --actor-variant role_head --run-seed 2026 --output-dir artifacts/formal/P2/run_2026

# P3 Topology-info
python train_plain_mappo.py --preset stage4-formal --full --actor-variant topology_info --run-seed 2026 --output-dir artifacts/formal/P3/run_2026

# P4 Graph Actor
python train_plain_mappo.py --preset stage4-formal --full --actor-variant graph --run-seed 2026 --output-dir artifacts/formal/P4/run_2026
```

The `--preset stage4-formal` is **atomic**: it freezes all common PPO settings, state-independent bounded log standard deviation, seed-domain derivation, and topology dimensions. The `--full` flag requires explicit `--preset stage4-formal` and will refuse to start otherwise.

### Single-update development/debugging
```powershell
# 10-update smoke run (default when --updates and --full are omitted)
python train_plain_mappo.py --actor-variant plain --run-seed 2026 --output-dir artifacts/dev/smoke

# Custom update count for development
python train_plain_mappo.py --actor-variant plain --updates 5 --output-dir artifacts/dev/quick
```

## Architecture

### Module structure
- **`relay_env/`**: Environment v1.1 implementation
  - Deterministic-on-seed K-relay UAV environment
  - 26-dimensional local observation per relay (position, velocity, channel state, safety)
  - Communication model with TDMA allocation, antenna directivity, e2e rate calculation
  - Safety constraints: collision avoidance, boundary limits, persistent outage detection
  - Terminal conditions: normal completion, persistent outage, collision, boundary violation

- **`plain_mappo/`**: MAPPO training pipeline supporting P0–P4 Actor variants
  - **`config.py`**: `MappoConfig` dataclass with Stage 4.1 protocol fields and seed derivation
  - **`networks.py`**: `SharedActor` (supports plain/role_info/role_head/topology_info/graph variants) and `CentralizedCritic`
  - **`trainer.py`**: `MappoTrainer` with independent RNG streams, rollout collection, PPO update, periodic evaluation
  - **`rollout_buffer.py`**: GAE computation and minibatch sampling
  - **`evaluation.py`**: Read-only policy evaluation with frozen seeds
  - **`experiment_protocol.py`**: Stage 4.1 seed derivation, manifest loading, canonical JSON hashing
  - **`presets.py`**: `stage4-formal` atomic configuration preset
  - **`checkpoint.py`**: Atomic `.pt` save/load with protocol metadata
  - **`roles.py`**: Role encoding for P1/P2 (source/middle/destination)
  - **`topology.py`**: Topology feature construction for P3/P4
  - **`state.py`**: 47-dimensional centralized Critic state assembly
  - **`normalization.py`**: Running mean/std for value normalization
  - **`metrics.py`**: Episode-level diagnostics

- **`selfcheck/`**: Automated test suite (10 test files, 76 tests as of Stage 4.1.1)
  - Tests cover environment determinism, PPO mathematics, GAE, checkpoint round-trip, protocol seed derivation, manifest integrity, actor variant configuration

- **`artifacts/`**: Historical training artifacts and evaluation results
  - `stage4-1-protocol-repair/`: Stage 4.1 validation grid (60 checkpoints × 20 seeds = 1,200 Episodes), final test (3 checkpoints × 50 seeds = 150 Episodes), manifests, migration metadata

### Actor variants (P0–P4)
All variants share:
- Same 26-dimensional local observation
- Same 47-dimensional centralized Critic
- Same PPO/GAE mathematics
- Same `stage4-formal` hyperparameters when used for formal comparison

Differences (controlled by `--actor-variant`):
- **`plain`** (P0): Two-hidden-layer MLP, no explicit structure
- **`role_info`** (P1): Concatenates 3-dimensional one-hot role encoding to observation
- **`role_head`** (P2): Separate output heads per role, shared trunk
- **`topology_info`** (P3): Concatenates topology features (node + aggregated edge features)
- **`graph`** (P4): Graph neural network encoder over topology

### Protocol and reproducibility
- **Protocol version**: `stage4.1-v1`
- **Seed domains**: Each `run_seed` (2026–2030 reserved) deterministically derives:
  - `actor_init_seed`: Actor parameter initialization
  - `critic_init_seed`: Critic parameter initialization
  - `action_noise_seed`: Rollout action-noise sampling
  - `minibatch_seed`: PPO minibatch permutation
  - `train_env_seed_base`: Training environment seed reservation (2M-wide non-overlapping ranges)
- **Validation seeds**: `50,000,000..50,000,019` (20 seeds, selection allowed)
- **Final test seeds**: `60,000,000..60,000,049` (50 seeds, selection forbidden)
- **Manifest hashing**: JSON manifests use `canonical-json-v1` (UTF-8, sorted keys, compact separators, CRLF/LF-independent). Binary `.pt` files use raw SHA-256.
- **Resume semantics**: `fresh_episode_at_update_boundary` — model/optimizer/normalizer/RNG state restored, but episodes restart from scratch (not bitwise-equivalent trajectory continuation)

### Historical results (single-seed retrospective pilots only)
- **P0 final (update 1000)**: 0 normal, 44 persistent outage, 1 collision, 5 boundary; mean e2e 6.338 Mbps
- **P1 final (update 800)**: 0 normal, 14 persistent outage, 0 collision, 36 boundary; mean e2e 6.847 Mbps
- **P2 final (update 650)**: 0 normal, 34 persistent outage, 0 collision, 16 boundary; mean e2e 6.235 Mbps

These are **not multi-seed formal results**. They are retrospective pilots on a single run seed.

## Development Guidelines

### When modifying code
1. **Never modify Environment v1.1**: Do not change observation dimensions, reward structure, communication model, antenna directivity, safety constraints, or terminal conditions without explicit authorization.

2. **Never modify PPO/GAE mathematics**: The PPO clipping, GAE computation, value loss, and gradient norms are frozen. Ablations require explicit authorization and must not claim to be part of the Stage 4.1 protocol.

3. **Preserve protocol reproducibility**: Any change that affects seed derivation, manifest loading, checkpoint serialization, or RNG stream isolation must maintain backward compatibility with Stage 4.1 checkpoints and produce identical results given identical seeds.

4. **Actor variant constraints**: P0/P1/P2 are in retrospective pilot state. P3/P4 have only smoke checks. Adding new variants or modifying existing variant structure requires authorization.

### Before running training
- **Smoke runs** (10 updates, default): Safe for development and debugging.
- **Custom `--updates <N>`**: Safe for ablation experiments if not claiming formal results.
- **`--full` (1000 updates)**: Requires `--preset stage4-formal` and explicit authorization. The CLI will refuse `--full` without the preset.

### Testing requirements
- Always run the full test suite after changes: `python -m unittest discover -s selfcheck -p "test_*.py" -v`
- Verify artifact integrity: `python verify_stage4_1_artifacts.py --metadata-only`
- Run `python -m compileall -q .` to catch syntax errors

### Checkpoint and artifact management
- **Never commit `.pt` files or large artifacts to Git**: Use Git LFS or external storage (GitHub Releases).
- **Verify checkpoint binary hashes**: The 60 historical Actor checkpoints have locked SHA-256 values in `artifacts/stage4-1-protocol-repair/checkpoint_manifest.json`.
- **Evaluation Episodes are not training runs**: The validation grid (1,200 Episodes) and final test (150 Episodes) are evaluation records, not independent training repetitions.

### Git workflow
- Work on feature branches, not `main`.
- Do not force-push or use `git reset --hard` on shared branches.
- Commit messages and PR descriptions should end with the attribution specified in `.git/config` or session instructions.

## Stage 4.1.1 Status

**Completed fixes:**
- JSON manifest hashing migrated from raw file bytes to `canonical-json-v1` (CRLF/LF-independent)
- Added atomic `--preset stage4-formal` configuration for future authorized runs
- Enforced fail-closed validation: new protocol refuses to fall back to legacy 5-seed defaults
- Fixed legacy checkpoint load → save → load round-trip consistency
- Added checkpoint config deserialization validation (detects tampered protocol fields)
- Improved README with accurate test counts, installation, verification commands
- Added CI workflow for automated testing on CPython 3.11/3.12
- Documented checkpoint release verification workflow

**Current test status**: 76/76 PASS (was 66/66 before Stage 4.1.1)

**Artifact integrity**: All 1,200 validation Episodes and 150 final-test Episodes preserved with unchanged numeric payloads. Episode immutable-payload hashes verified pre/post migration.

**Remaining scope**: P3/P4 formal training NOT STARTED; multi-seed P0–P4 training NOT AUTHORIZED.
