"""Frozen Stage 4.1 experiment-seed and provenance protocol utilities.

The module deliberately owns the random-domain derivation used by future
formal runs.  It does not change the environment, Actor architecture, or PPO
objective.  Historic checkpoints that lack these fields remain loadable but
are explicitly marked ``legacy_protocol`` rather than being relabelled.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROTOCOL_VERSION = "stage4.1-v1"
LEGACY_PROTOCOL_VERSION = "legacy_protocol"
PREEXPERIMENT_PROTOCOL_VERSION = "stage4.2-preexperiment-v1"
CANONICAL_JSON_HASH_SCHEME = "canonical-json-v1"
VALIDATION_SEEDS = tuple(range(50_000_000, 50_000_020))
FINAL_TEST_SEEDS = tuple(range(60_000_000, 60_000_050))
FORMAL_RUN_SEEDS = (2026, 2027, 2028, 2029, 2030)
PREEXPERIMENT_RUN_SEEDS = (2026, 2027, 2028)
SEED_MANIFEST_PATH = "artifacts/stage4-1-protocol-repair/seed_manifest.json"


def repository_root() -> Path:
    """Return the stable repository/configuration root, never the caller CWD."""
    return Path(__file__).resolve().parents[1]


def resolve_repository_path(path: str | Path) -> Path:
    """Resolve protocol artifact paths relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def _derive_protocol_seed(protocol_version: str, run_seed: int, domain: str) -> int:
    """Derive a stable 63-bit seed without consuming any process RNG state."""
    payload = f"{protocol_version}|{int(run_seed)}|{domain}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def derive_seed(run_seed: int, domain: str) -> int:
    """Derive a Stage-4.1 formal-run seed."""
    return _derive_protocol_seed(PROTOCOL_VERSION, run_seed, domain)


def train_env_seed_base_for(run_seed: int) -> int:
    """Return the disjoint two-million-wide training-seed reservation."""
    try:
        index = FORMAL_RUN_SEEDS.index(int(run_seed))
    except ValueError as exc:
        raise ValueError(f"run_seed must be one of {FORMAL_RUN_SEEDS}") from exc
    return 10_000_000 + 2_000_000 * index


def protocol_fields(run_seed: int) -> dict[str, Any]:
    """Return all serialized random-domain metadata for a new protocol run."""
    run_seed = int(run_seed)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_seed": run_seed,
        "actor_init_seed": derive_seed(run_seed, "actor_initialization"),
        "critic_init_seed": derive_seed(run_seed, "critic_initialization"),
        "action_noise_seed": derive_seed(run_seed, "rollout_action_noise"),
        "minibatch_seed": derive_seed(run_seed, "minibatch_permutation"),
        "train_env_seed_base": train_env_seed_base_for(run_seed),
        "validation_seed_manifest": SEED_MANIFEST_PATH,
        "final_test_seed_manifest": SEED_MANIFEST_PATH,
    }


def preexperiment_train_env_seed_base_for(run_seed: int) -> int:
    """Return the Stage-4.2 pre-experiment-only two-million seed reservation."""
    try:
        index = PREEXPERIMENT_RUN_SEEDS.index(int(run_seed))
    except ValueError as exc:
        raise ValueError(f"pre-experiment run_seed must be one of {PREEXPERIMENT_RUN_SEEDS}") from exc
    return 30_000_000 + 2_000_000 * index


def preexperiment_protocol_fields(run_seed: int) -> dict[str, Any]:
    """Return isolated RNG metadata for the fixed Stage-4.2 pilot matrix."""
    run_seed = int(run_seed)
    return {
        "protocol_version": PREEXPERIMENT_PROTOCOL_VERSION,
        "run_seed": run_seed,
        "actor_init_seed": _derive_protocol_seed(PREEXPERIMENT_PROTOCOL_VERSION, run_seed, "actor_initialization"),
        "critic_init_seed": _derive_protocol_seed(PREEXPERIMENT_PROTOCOL_VERSION, run_seed, "critic_initialization"),
        "action_noise_seed": _derive_protocol_seed(PREEXPERIMENT_PROTOCOL_VERSION, run_seed, "rollout_action_noise"),
        "minibatch_seed": _derive_protocol_seed(PREEXPERIMENT_PROTOCOL_VERSION, run_seed, "minibatch_permutation"),
        "train_env_seed_base": preexperiment_train_env_seed_base_for(run_seed),
        # The Stage-4.1 locked validation split is read-only input.  The
        # final-test path is intentionally absent from this pre-experiment.
        "validation_seed_manifest": SEED_MANIFEST_PATH,
        "final_test_seed_manifest": "",
    }


def is_legacy_protocol(data: dict[str, Any]) -> bool:
    return data.get("protocol_version", LEGACY_PROTOCOL_VERSION) == LEGACY_PROTOCOL_VERSION


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash an opaque binary file such as a checkpoint or release ZIP."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json_payload(payload: Any) -> str:
    """Hash JSON semantics, independent of indentation and line endings."""
    return sha256_bytes(canonical_json_bytes(payload))


def sha256_json_file(path: Path) -> str:
    """Parse then canonically hash a JSON document's semantic payload."""
    return sha256_json_payload(json.loads(path.read_text(encoding="utf-8")))


def seed_manifest_payload() -> dict[str, Any]:
    """Construct the one-time Stage 4.1 validation/final-test manifest."""
    reservations = []
    for run_seed in FORMAL_RUN_SEEDS:
        start = train_env_seed_base_for(run_seed)
        reservations.append({"run_seed": run_seed, "start": start, "stop_exclusive": start + 2_000_000})
    return {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_contract": True,
        "splits": {
            "validation": {"seeds": list(VALIDATION_SEEDS), "episodes": len(VALIDATION_SEEDS), "selection_allowed": True},
            "test": {"seeds": list(FINAL_TEST_SEEDS), "episodes": len(FINAL_TEST_SEEDS), "selection_allowed": False},
        },
        "future_formal_training_seed_reservations": reservations,
    }


def _manifest_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Exclude generation time when deciding whether an existing manifest matches."""
    return {key: value for key, value in payload.items() if key != "generated_at_utc"}


def write_locked_seed_manifest(path: Path) -> tuple[dict[str, Any], str, bool]:
    """Write once, or verify the existing immutable-contract content.

    Returning an existing matching file avoids rewriting its creation time.
    A different manifest is rejected so that test seeds cannot silently change
    after any result has been inspected.
    """
    proposed = seed_manifest_payload()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _manifest_contract(existing) != _manifest_contract(proposed):
            raise ValueError(f"existing seed manifest differs and is locked: {path}")
        return existing, sha256_json_payload(existing), False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proposed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return proposed, sha256_json_payload(proposed), True


def load_seed_manifest(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("seed manifest protocol version is not Stage 4.1")
    if payload.get("manifest_hash_scheme") != CANONICAL_JSON_HASH_SCHEME:
        raise ValueError(f"seed manifest hash scheme must be {CANONICAL_JSON_HASH_SCHEME}")
    if payload.get("read_only_contract") is not True:
        raise ValueError("seed manifest read_only_contract must be true")
    try:
        validation_split, test_split = payload["splits"]["validation"], payload["splits"]["test"]
        validation = validation_split["seeds"]
        test = test_split["seeds"]
    except (KeyError, TypeError) as exc:
        raise ValueError("seed manifest split structure is invalid") from exc

    # These are an immutable evaluation contract, not merely disjoint ranges.
    # Sequence equality deliberately rejects reordering, duplicates, or a
    # same-length replacement with otherwise-unused seeds.
    if (not isinstance(validation, list) or any(type(seed) is not int for seed in validation)
            or tuple(validation) != VALIDATION_SEEDS):
        raise ValueError("validation seeds must exactly match the locked Stage-4.1 sequence")
    if (not isinstance(test, list) or any(type(seed) is not int for seed in test)
            or tuple(test) != FINAL_TEST_SEEDS):
        raise ValueError("test seeds must exactly match the locked Stage-4.1 sequence")
    if validation_split.get("episodes") != 20 or validation_split.get("selection_allowed") is not True:
        raise ValueError("validation split must contain exactly 20 seeds and allow selection")
    if test_split.get("episodes") != 50 or test_split.get("selection_allowed") is not False:
        raise ValueError("test split must contain exactly 50 seeds and forbid selection")
    expected_reservations = [
        {"run_seed": run_seed, "start": train_env_seed_base_for(run_seed),
         "stop_exclusive": train_env_seed_base_for(run_seed) + 2_000_000}
        for run_seed in FORMAL_RUN_SEEDS
    ]
    if payload.get("future_formal_training_seed_reservations") != expected_reservations:
        raise ValueError("future training seed reservations must exactly match the locked Stage-4.1 ranges")
    return payload, sha256_json_payload(payload)


def split_seeds(payload: dict[str, Any], split: str) -> tuple[int, ...]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    return tuple(int(seed) for seed in payload["splits"][split]["seeds"])


def checkpoint_update(path: Path) -> int:
    match = re.fullmatch(r"actor_update_(\d+)\.pt", path.name)
    if not match:
        raise ValueError(f"not a periodic Actor checkpoint name: {path.name}")
    return int(match.group(1))


def provenance_snapshot(device: str) -> dict[str, Any]:
    """Collect non-mutating implementation provenance for audit artifacts."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
    }


def assert_disjoint_ranges(ranges: Iterable[tuple[int, int]]) -> bool:
    normalized = sorted((int(start), int(stop)) for start, stop in ranges)
    return all(left[1] <= right[0] for left, right in zip(normalized, normalized[1:]))
