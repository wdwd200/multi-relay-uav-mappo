"""Frozen Stage-4.2 value-treatment and horizon pre-experiment protocol."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment_protocol import (CANONICAL_JSON_HASH_SCHEME, FINAL_TEST_SEEDS,
                                  PREEXPERIMENT_PROTOCOL_VERSION, PREEXPERIMENT_RUN_SEEDS,
                                  VALIDATION_SEEDS, canonical_json_bytes,
                                  preexperiment_train_env_seed_base_for,
                                  sha256_json_payload)


PREEXPERIMENT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"candidate": "V0H0", "value_normalization": False, "gamma": 0.99, "gae_lambda": 0.95},
    {"candidate": "V0H1", "value_normalization": False, "gamma": 0.995, "gae_lambda": 0.97},
    {"candidate": "V0H2", "value_normalization": False, "gamma": 0.998, "gae_lambda": 0.98},
    {"candidate": "V1H0", "value_normalization": True, "gamma": 0.99, "gae_lambda": 0.95},
    {"candidate": "V1H1", "value_normalization": True, "gamma": 0.995, "gae_lambda": 0.97},
    {"candidate": "V1H2", "value_normalization": True, "gamma": 0.998, "gae_lambda": 0.98},
)

STAGE4_1_CANONICAL_HASHES = {
    "seed_manifest": "99b1cbd618444af64c3ae14eb82f356c44584afa4693c1862a82687d1268ab1f",
    "checkpoint_manifest": "1d81f58bffaad385bb759bc95147ee79401b837581866b29fa907b5eb8532939",
    "selected_checkpoints": "3fa29ec23fc36979d6c6d00ade4ea4b89887dab65341be563f737cbc723ad614",
}


def preexperiment_protocol_payload() -> dict[str, Any]:
    """Build the immutable 18-run Stage-4.2 protocol before execution."""
    reservations = [
        {"run_seed": seed, "start": preexperiment_train_env_seed_base_for(seed),
         "stop_exclusive": preexperiment_train_env_seed_base_for(seed) + 2_000_000}
        for seed in PREEXPERIMENT_RUN_SEEDS
    ]
    return {
        "protocol_version": PREEXPERIMENT_PROTOCOL_VERSION,
        "manifest_hash_scheme": CANONICAL_JSON_HASH_SCHEME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_contract": True,
        "actor_variant": "plain",
        "stage4_1_canonical_hashes": dict(STAGE4_1_CANONICAL_HASHES),
        "run_seeds": list(PREEXPERIMENT_RUN_SEEDS),
        "training_seed_reservations": reservations,
        "validation": {"seeds": list(VALIDATION_SEEDS), "selection_allowed": True,
                       "allowed_updates": [50, 100, 150, 200]},
        "final_test": {"seeds": list(FINAL_TEST_SEEDS), "executed": False,
                       "selection_allowed": False, "permitted": False},
        "updates_per_run": 200,
        "samples_per_update": 8 * 128,
        "candidates": [dict(candidate) for candidate in PREEXPERIMENT_CANDIDATES],
    }


def _contract(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "generated_at_utc"}


def validate_preexperiment_protocol(payload: dict[str, Any]) -> None:
    """Fail closed when any frozen matrix, split, or seed range is altered."""
    expected = _contract(preexperiment_protocol_payload())
    if _contract(payload) != expected:
        raise ValueError("Stage-4.2 pre-experiment protocol differs from its locked contract")
    if payload.get("read_only_contract") is not True:
        raise ValueError("Stage-4.2 pre-experiment protocol must be read-only")


def write_locked_preexperiment_protocol(path: Path) -> tuple[dict[str, Any], str, bool]:
    """Create the protocol once, or validate and reuse its canonical payload."""
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_preexperiment_protocol(payload)
        return payload, sha256_json_payload(payload), False
    payload = preexperiment_protocol_payload()
    validate_preexperiment_protocol(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload, sha256_json_payload(payload), True
