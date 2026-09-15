"""Build the untracked Stage 4.1 retrospective checkpoint release ZIP."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path, PureWindowsPath

from migrate_stage4_1_hashes import (CHECKPOINT_MANIFEST, OUTPUT, _check_historical_artifacts,
                                     _read_json)
from plain_mappo.experiment_protocol import sha256_file


PACKAGE_DIRECTORY = "stage4.1-retrospective-pilot-checkpoints"


def package(output: Path) -> dict[str, object]:
    """Package precisely the audited 60 periodic Actors and their manifest."""
    integrity = _check_historical_artifacts()
    manifest = _read_json(CHECKPOINT_MANIFEST)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_count = 0
    readme = (
        "# Stage 4.1 retrospective pilot checkpoints\n\n"
        "This package contains the 60 periodic Actor checkpoints used solely to audit the "
        "single-seed Stage 4.1 validation grid (P0/P1/P2, 20 checkpoints each). "
        "It is not a formal multi-seed result and must not be interpreted as such.\n\n"
        "The P0/P1/P2 directories map to the historical runs recorded in checkpoint_manifest.json. "
        "Each .pt uses raw binary SHA-256. JSON manifests use canonical-json-v1.\n\n"
        "To verify after extraction from the repository root:\n"
        f"python verify_stage4_1_artifacts.py --checkpoint-root <extracted>\\{PACKAGE_DIRECTORY}\n"
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{PACKAGE_DIRECTORY}/README_CHECKPOINTS.md", readme)
        archive.write(CHECKPOINT_MANIFEST, f"{PACKAGE_DIRECTORY}/checkpoint_manifest.json")
        for label, run in manifest["runs"].items():
            for record in run["checkpoints"]:
                source = Path(__file__).resolve().parent.joinpath(*PureWindowsPath(record["path"]).parts)
                if not source.is_file() or sha256_file(source) != record["sha256"]:
                    raise ValueError(f"checkpoint verification failed before packaging: {record['path']}")
                archive.write(source, f"{PACKAGE_DIRECTORY}/{label}/eval_checkpoints/{source.name}")
                checkpoint_count += 1
    if checkpoint_count != 60:
        raise AssertionError("release package must contain exactly 60 checkpoints")
    return {"zip": str(output), "zip_sha256": sha256_file(output), "checkpoint_count": checkpoint_count,
            "checkpoint_manifest_sha256": integrity["checkpoint_manifest_sha256"],
            "episode_numeric_payload_changed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Package audited Stage 4.1 pilot checkpoints; never commit the ZIP.")
    parser.add_argument("--output", type=Path,
                        default=OUTPUT / "stage4.1-retrospective-pilot-checkpoints.zip")
    args = parser.parse_args()
    print(json.dumps(package(args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
