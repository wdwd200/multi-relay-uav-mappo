"""Run independent deterministic evaluation for an exported Plain MAPPO Actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from plain_mappo import MappoConfig
from plain_mappo.evaluation import evaluate_actor
from plain_mappo.networks import SharedActor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tanh(mean) Plain MAPPO actions on fixed seeds.")
    parser.add_argument("checkpoint", type=Path, help="actor_final.pt or complete checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--output", type=Path, help="Optional JSON path for the deterministic evaluation result.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = MappoConfig.from_dict(payload["config"])
    actor = SharedActor(config.local_obs_dim, config.action_dim, config.actor_hidden_sizes,
                        config.log_std_min, config.log_std_max, config.actor_log_std_mode,
                        config.state_independent_log_std_init, config.actor_variant,
                        config.topology_node_dim, config.topology_edge_dim,
                        config.graph_hidden_dim).to(config.device)
    actor.load_state_dict(payload["actor_state"])
    summary, score, _ = evaluate_actor(actor, config, range(args.seed_start, args.seed_start + args.episodes))
    result = {"checkpoint": str(args.checkpoint), "score": score, "summary": summary}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
