"""Stage-4.2 value-treatment and horizon pre-experiment contracts."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from plain_mappo.config import MappoConfig
from plain_mappo.experiment_protocol import (FINAL_TEST_SEEDS, FORMAL_RUN_SEEDS,
                                              PREEXPERIMENT_RUN_SEEDS, VALIDATION_SEEDS,
                                              preexperiment_protocol_fields,
                                              preexperiment_train_env_seed_base_for, protocol_fields,
                                              sha256_json_payload)
from plain_mappo.normalization import RunningScalarMeanStd
from plain_mappo.preexperiment import (PREEXPERIMENT_CANDIDATES,
                                       preexperiment_protocol_payload,
                                       validate_preexperiment_protocol,
                                       write_locked_preexperiment_protocol)
from plain_mappo.presets import stage4_preexperiment_overrides
from plain_mappo.trainer import MappoTrainer, explained_variance


def _preexperiment_config(directory: str, *, candidate: str = "V1H1",
                          value_normalization: bool = True) -> MappoConfig:
    overrides = stage4_preexperiment_overrides(
        candidate=candidate, run_seed=2026, output_dir=directory,
        protocol_sha256="a" * 64, value_normalization=value_normalization,
        gamma=0.995, gae_lambda=0.97,
        evaluation_episode_log_path=str(Path(directory) / "validation_episodes.jsonl"),
    )
    return replace(MappoConfig(**overrides), num_envs=2, rollout_length=4,
                   mini_batch_size=4, ppo_epochs=1)


class Stage42PreexperimentTests(unittest.TestCase):
    def test_preexperiment_seed_namespace_is_disjoint_from_formal_and_evaluation_ranges(self) -> None:
        ranges = [(preexperiment_train_env_seed_base_for(seed), preexperiment_train_env_seed_base_for(seed) + 2_000_000)
                  for seed in PREEXPERIMENT_RUN_SEEDS]
        formal_ranges = [(protocol_fields(seed)["train_env_seed_base"], protocol_fields(seed)["train_env_seed_base"] + 2_000_000)
                         for seed in FORMAL_RUN_SEEDS]
        self.assertEqual(ranges, [(30_000_000, 32_000_000), (32_000_000, 34_000_000), (34_000_000, 36_000_000)])
        self.assertTrue(all(not (left[0] < right[1] and right[0] < left[1]) for left in ranges for right in formal_ranges))
        self.assertTrue(all(not (start <= seed < stop) for start, stop in ranges for seed in VALIDATION_SEEDS + FINAL_TEST_SEEDS))

    def test_candidates_share_all_rng_initial_conditions_for_each_paired_seed(self) -> None:
        fields = ("actor_init_seed", "critic_init_seed", "action_noise_seed", "minibatch_seed", "train_env_seed_base")
        for run_seed in PREEXPERIMENT_RUN_SEEDS:
            expected = preexperiment_protocol_fields(run_seed)
            configs = [MappoConfig(**stage4_preexperiment_overrides(
                candidate=candidate["candidate"], run_seed=run_seed,
                output_dir=f"artifacts/test-{candidate['candidate']}-{run_seed}",
                protocol_sha256="a" * 64, value_normalization=candidate["value_normalization"],
                gamma=candidate["gamma"], gae_lambda=candidate["gae_lambda"],
                evaluation_episode_log_path=f"artifacts/test-{candidate['candidate']}-{run_seed}/episodes.jsonl"))
                       for candidate in PREEXPERIMENT_CANDIDATES]
            for field in fields:
                self.assertEqual({getattr(config, field) for config in configs}, {expected[field]})

    def test_value_normalizer_round_trip_and_inverse_are_exact(self) -> None:
        normalizer = RunningScalarMeanStd()
        values = np.asarray([-3.0, 2.0, 9.0, 12.0], dtype=np.float32)
        normalizer.update(values)
        normalized = normalizer.normalize(values)
        np.testing.assert_allclose(normalizer.denormalize(normalized), values, rtol=0.0, atol=1e-6)
        restored = RunningScalarMeanStd(); restored.load_state_dict(normalizer.state_dict())
        np.testing.assert_allclose(restored.normalize(values), normalized, rtol=0.0, atol=0.0)

    def test_v1_checkpoint_round_trip_restores_value_normalizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _preexperiment_config(directory)
            trainer = MappoTrainer(config); trainer.train(1, final_evaluation=False)
            self.assertIsNotNone(trainer.value_normalizer)
            restored = MappoTrainer(config); restored.load(Path(directory) / "checkpoints" / "latest.pt")
            self.assertIsNotNone(restored.value_normalizer)
            self.assertEqual(restored.value_normalizer.state_dict(), trainer.value_normalizer.state_dict())  # type: ignore[union-attr]

    def test_v0_keeps_raw_value_targets_and_legacy_checkpoint_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            v0 = MappoTrainer(_preexperiment_config(directory, value_normalization=False))
            values = torch.tensor([1.25, -2.0])
            self.assertIs(v0._value_predictions_to_raw(values), values)
            legacy = replace(MappoConfig(protocol_version="legacy_protocol", output_dir=str(Path(directory) / "legacy")),
                             num_envs=2, rollout_length=4, mini_batch_size=4, ppo_epochs=1)
            legacy_trainer = MappoTrainer(legacy); checkpoint = legacy_trainer.save_latest()
            restored = MappoTrainer(legacy); restored.load(checkpoint)
            self.assertTrue(restored.protocol_is_legacy)
            self.assertIsNone(restored.value_normalizer)

    def test_explained_variance_uses_denormalized_value_scale(self) -> None:
        normalizer = RunningScalarMeanStd(); target = np.asarray([10.0, 12.0, 14.0, 16.0], dtype=np.float32)
        normalizer.update(target)
        normalized_prediction = np.asarray([-1.0, -0.2, 0.4, 1.1], dtype=np.float32)
        raw_prediction = normalizer.denormalize(normalized_prediction)
        self.assertEqual(explained_variance(target, raw_prediction), explained_variance(target, normalizer.denormalize(normalized_prediction)))
        self.assertNotEqual(explained_variance(target, raw_prediction), explained_variance(target, normalized_prediction))

    def test_preexperiment_forbids_final_test_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = MappoTrainer(_preexperiment_config(directory))
            with self.assertRaisesRegex(ValueError, "final test is forbidden"):
                trainer.run_evaluation(list(FINAL_TEST_SEEDS))

    def test_protocol_matrix_tampering_fails_and_locked_protocol_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            payload, digest, created = write_locked_preexperiment_protocol(path)
            self.assertTrue(created)
            self.assertEqual(digest, sha256_json_payload(payload))
            changed = copy.deepcopy(payload); changed["candidates"].pop()
            with self.assertRaisesRegex(ValueError, "locked contract"):
                validate_preexperiment_protocol(changed)
            loaded, loaded_digest, created_again = write_locked_preexperiment_protocol(path)
            self.assertEqual(loaded, payload); self.assertEqual(loaded_digest, digest); self.assertFalse(created_again)


if __name__ == "__main__":
    unittest.main()
