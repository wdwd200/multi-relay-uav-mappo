"""Stage 4.1 seed, selection, diagnostic, and resume-protocol tests."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from plain_mappo.config import MappoConfig
from plain_mappo.experiment_protocol import (FINAL_TEST_SEEDS, LEGACY_PROTOCOL_VERSION,
                                              VALIDATION_SEEDS, load_seed_manifest, protocol_fields,
                                              sha256_file, train_env_seed_base_for, write_locked_seed_manifest)
from plain_mappo.trainer import MappoTrainer, explained_variance, gradient_clip_fraction
from reevaluate_checkpoint_grid import (assert_episode_completeness, checkpoint_paths, run_grid,
                                        select_checkpoint_from_validation)


def _config(directory: str, variant: str = "plain", run_seed: int = 2026) -> MappoConfig:
    overrides: dict[str, object] = {"num_envs": 2, "rollout_length": 4, "mini_batch_size": 4,
                                    "ppo_epochs": 1, "periodic_eval_episodes": 1, "output_dir": directory,
                                    "actor_variant": variant}
    if variant != "plain":
        overrides.update({"actor_log_std_mode": "state_independent_tanh", "log_std_min": -4.0,
                          "log_std_max": 0.0, "state_independent_log_std_init": -1.5})
    if run_seed != 2026:
        overrides.update({"run_seed": run_seed, "actor_init_seed": None, "critic_init_seed": None,
                          "action_noise_seed": None, "minibatch_seed": None, "train_env_seed_base": None,
                          "validation_seed_manifest": "", "final_test_seed_manifest": ""})
    return replace(MappoConfig(), **overrides)


class MappoProtocolTests(unittest.TestCase):
    def test_validation_test_and_future_training_seed_ranges_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed_manifest.json"
            manifest, digest, created = write_locked_seed_manifest(path)
            self.assertTrue(created); self.assertEqual(digest, sha256_file(path))
            loaded, loaded_digest = load_seed_manifest(path)
            self.assertEqual(manifest, loaded); self.assertEqual(digest, loaded_digest)
            self.assertFalse(set(VALIDATION_SEEDS).intersection(FINAL_TEST_SEEDS))
            ranges = [(train_env_seed_base_for(seed), train_env_seed_base_for(seed) + 2_000_000)
                      for seed in (2026, 2027, 2028, 2029, 2030)]
            self.assertTrue(all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])))
            self.assertTrue(all(not (start <= seed < stop) for start, stop in ranges for seed in VALIDATION_SEEDS + FINAL_TEST_SEEDS))
            _, _, second_created = write_locked_seed_manifest(path)
            self.assertFalse(second_created)

    def test_replacing_run_seed_rederives_every_protocol_domain(self) -> None:
        config = replace(MappoConfig(), run_seed=2027)
        expected = protocol_fields(2027)
        for field, value in expected.items():
            self.assertEqual(getattr(config, field), value)

    def test_same_run_seed_keeps_critic_and_rng_streams_identical_across_actor_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plain = MappoTrainer(_config(str(Path(directory) / "plain"), "plain"))
            graph = MappoTrainer(_config(str(Path(directory) / "graph"), "graph"))
            for left, right in zip(plain.critic.parameters(), graph.critic.parameters()):
                torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
            self.assertTrue(torch.equal(plain.action_noise_generator.get_state(), graph.action_noise_generator.get_state()))  # type: ignore[union-attr]
            self.assertEqual(plain.minibatch_rng.bit_generator.state, graph.minibatch_rng.bit_generator.state)  # type: ignore[union-attr]
            plain.collect_rollout(); graph.collect_rollout()
            self.assertTrue(torch.equal(plain.action_noise_generator.get_state(), graph.action_noise_generator.get_state()))  # type: ignore[union-attr]

    def test_protocol_rng_states_round_trip_through_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            trainer = MappoTrainer(config)
            trainer.collect_rollout()
            checkpoint = trainer.save_latest()
            expected_noise = trainer.action_noise_generator.get_state().clone()  # type: ignore[union-attr]
            expected_minibatch = copy.deepcopy(trainer.minibatch_rng.bit_generator.state)  # type: ignore[union-attr]
            restored = MappoTrainer(config); restored.load(checkpoint)
            self.assertTrue(torch.equal(restored.action_noise_generator.get_state(), expected_noise))  # type: ignore[union-attr]
            self.assertEqual(restored.minibatch_rng.bit_generator.state, expected_minibatch)  # type: ignore[union-attr]

    def test_legacy_checkpoint_loads_and_is_explicitly_labelled_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            trainer = MappoTrainer(config)
            checkpoint = trainer.save_latest()
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            for field in protocol_fields(2026):
                payload["config"].pop(field, None)
            payload.pop("protocol_rng_state", None); payload.pop("protocol_metadata", None)
            torch.save(payload, checkpoint)
            restored = MappoTrainer(config); restored.load(checkpoint)
            self.assertTrue(restored.protocol_is_legacy)
            self.assertEqual(restored.loaded_checkpoint_protocol, LEGACY_PROTOCOL_VERSION)

    def test_grid_is_numeric_and_test_grid_selection_is_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); run_dir = base / "run" / "eval_checkpoints"; run_dir.mkdir(parents=True)
            for update in (1000, 50, 100):
                (run_dir / f"actor_update_{update:04d}.pt").touch()
            paths = checkpoint_paths(base / "run", "eval_checkpoints/actor_update_*.pt")
            self.assertEqual([path.name for path in paths], ["actor_update_0050.pt", "actor_update_0100.pt", "actor_update_1000.pt"])
            manifest_path = base / "seed_manifest.json"; write_locked_seed_manifest(manifest_path)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                run_grid(run_dir=base / "run", checkpoint_glob="eval_checkpoints/actor_update_*.pt",
                         seed_manifest=manifest_path, split="test", output_dir=base / "out", device="cpu")

    def test_validation_selection_uses_frozen_score_and_earlier_update_tie_break(self) -> None:
        base = {"speed_accel_violations": 0, "normal_completion_count": 1, "collision_count": 0,
                "boundary_count": 0, "persistent_outage_count": 0, "outage_step_ratio": 0.1,
                "rate_satisfaction_ratio": 0.9, "mean_e2e_rate_mbps": 7.0}
        score = [0.0, -1.0, 0.0, 0.0, 0.1, -0.9, -7.0]
        winner = select_checkpoint_from_validation([{**base, "update": 100, "score": score, "split": "validation"},
                                                     {**base, "update": 50, "score": score, "split": "validation"}])
        self.assertEqual(winner["update"], 50)
        with self.assertRaisesRegex(ValueError, "test rows"):
            select_checkpoint_from_validation([{**base, "update": 50, "score": score, "split": "test"}])

    def test_explained_variance_and_gradient_clip_fraction_match_hand_calculation(self) -> None:
        target = np.array([1.0, 2.0, 3.0]); prediction = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(explained_variance(target, prediction), 0.0, places=12)
        self.assertIsNone(explained_variance(np.ones(3), np.zeros(3)))
        self.assertAlmostEqual(gradient_clip_fraction([0.1, 0.5, 0.6, 1.0], 0.5), 0.5, places=12)

    def test_raw_episode_rows_must_match_checkpoint_times_seed_without_duplicates(self) -> None:
        hashes, seeds = ("a", "b"), (1, 2)
        rows = [{"checkpoint_sha256": checkpoint_hash, "seed": seed, "return": 1.0, "length": 2,
                 "mean_e2e_rate_mbps": 3.0, "rate_satisfaction_ratio": 0.5, "outage_step_ratio": 0.1,
                 "min_separation_m": 4.0, "max_xy_speed_mps": 5.0, "max_xy_accel_mps2": 0.6,
                 "action_saturation_ratio": 0.0, "movement_distance_m": 7.0}
                for checkpoint_hash in hashes for seed in seeds]
        assert_episode_completeness(rows, hashes, seeds)
        with self.assertRaisesRegex(ValueError, "without duplicates"):
            assert_episode_completeness(rows + [rows[0]], hashes, seeds)

    def test_new_training_log_contains_finite_diagnostics_or_explicit_blank_ev(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = MappoTrainer(_config(directory))
            trainer.train(1)
            with (Path(directory) / "train.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            for field in ("value_target_mean", "value_target_std", "value_prediction_mean_pre",
                          "value_prediction_std_pre", "actor_grad_norm_max", "critic_grad_norm_max",
                          "actor_grad_clip_fraction", "critic_grad_clip_fraction"):
                self.assertTrue(np.isfinite(float(row[field])), field)
            self.assertTrue(row["explained_variance_pre"] == "" or np.isfinite(float(row["explained_variance_pre"])))


if __name__ == "__main__":
    unittest.main()
