"""Run the fixed-seed random-action baseline required for Stage 3 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plain_mappo import MappoConfig
from plain_mappo.evaluation import evaluate_random_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a uniform random K=4 action policy on fixed environment seeds.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--output", type=Path, help="Optional JSON path for the baseline result.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    summary, score, _ = evaluate_random_policy(MappoConfig(), range(args.seed_start, args.seed_start + args.episodes))
    result = {"policy": "uniform_random_actions", "action_seed_offset": 50_000, "score": score, "summary": summary}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
