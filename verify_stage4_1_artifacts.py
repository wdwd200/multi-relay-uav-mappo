"""Read-only Stage 4.1 artifact-integrity verification for CI and release."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from migrate_stage4_1_hashes import MIGRATION, _check_historical_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Stage 4.1 artifact integrity verification.")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Validate committed metadata when release checkpoint assets are intentionally absent.")
    parser.add_argument("--checkpoint-root", type=Path,
                        help="Extracted release directory containing P0/, P1/, and P2/ checkpoint folders.")
    args = parser.parse_args()
    result = _check_historical_artifacts(verify_checkpoint_binaries=not args.metadata_only,
                                         checkpoint_root=args.checkpoint_root)
    migration = json.loads(MIGRATION.read_text(encoding="utf-8"))
    if not migration.get("checkpoint_binary_hashes_unchanged") or migration.get("episode_numeric_payload_changed"):
        raise ValueError("hash migration integrity declaration is invalid")
    compact = {key: value for key, value in result.items() if key != "checkpoint_binary_hashes"}
    print(json.dumps({"artifact_integrity": "PASS", **compact}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
