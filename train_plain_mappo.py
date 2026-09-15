"""Train or resume the explicit Plain MAPPO baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from plain_mappo import MappoConfig
from plain_mappo.trainer import MappoTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the K=4 Plain MAPPO baseline.")
    parser.add_argument("--updates", type=int, default=None, help="Additional PPO updates (default: 10-update smoke run).")
    parser.add_argument("--full", action="store_true", help="Run the configured 1000-update initial training.")
    parser.add_argument("--resume", type=Path, help="Complete latest.pt/best.pt checkpoint to resume from.")
    parser.add_argument("--output-dir", type=str, default="artifacts/plain_mappo")
    parser.add_argument("--run-seed", type=int,
                        help="Stage-4.1 paired-run seed (2026--2030); derives isolated RNG domains.")
    parser.add_argument("--actor-variant", choices=("plain", "role_info", "role_head", "topology_info", "graph"), default=None,
                        help="Select the shared Actor variant in the single MAPPO pipeline.")
    parser.add_argument("--entropy-coef", type=float, default=None,
                        help="Override only the PPO entropy coefficient for an explicit ablation.")
    parser.add_argument("--actor-lr", type=float, default=None,
                        help="Override only the Actor Adam learning rate for an explicit ablation.")
    parser.add_argument("--actor-log-std-mode", choices=("state_dependent_clamp", "state_independent_tanh"),
                        help="Select the Actor uncertainty parameterization for an explicit controlled experiment.")
    parser.add_argument("--log-std-min", type=float, default=None,
                        help="Override the lower Actor log_std bound for an explicit distribution experiment.")
    parser.add_argument("--log-std-max", type=float, default=None,
                        help="Override the upper Actor log_std bound for an explicit distribution experiment.")
    parser.add_argument("--state-independent-log-std-init", type=float, default=None,
                        help="Initial effective log_std for state_independent_tanh.")
    args = parser.parse_args()
    if args.full and args.updates is not None:
        parser.error("--full and --updates are mutually exclusive")
    overrides = {"output_dir": args.output_dir}
    if args.entropy_coef is not None:
        overrides["entropy_coef"] = args.entropy_coef
    if args.actor_variant is not None:
        overrides["actor_variant"] = args.actor_variant
    if args.actor_lr is not None:
        overrides["actor_lr"] = args.actor_lr
    if args.actor_log_std_mode is not None:
        overrides["actor_log_std_mode"] = args.actor_log_std_mode
    if args.log_std_min is not None:
        overrides["log_std_min"] = args.log_std_min
    if args.log_std_max is not None:
        overrides["log_std_max"] = args.log_std_max
    if args.state_independent_log_std_init is not None:
        overrides["state_independent_log_std_init"] = args.state_independent_log_std_init
    if args.run_seed is not None:
        overrides.update({"run_seed": args.run_seed, "actor_init_seed": None, "critic_init_seed": None,
                          "action_noise_seed": None, "minibatch_seed": None, "train_env_seed_base": None,
                          "validation_seed_manifest": "", "final_test_seed_manifest": ""})
    config = replace(MappoConfig(), **overrides)
    updates = config.full_updates if args.full else (args.updates or config.smoke_updates)
    if updates < 1:
        parser.error("updates must be positive")
    trainer = MappoTrainer(config)
    if args.resume:
        trainer.load(args.resume)
    records = trainer.train(updates)
    print(json.dumps({"updates_completed": updates, "final_update": trainer.update_count,
                      "total_env_steps": trainer.total_env_steps, "latest": str(trainer.checkpoint_dir / "latest.pt"),
                      "best": str(trainer.checkpoint_dir / "best.pt"), "actor_final": str(trainer.checkpoint_dir / "actor_final.pt"),
                      "last_record": records[-1]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
