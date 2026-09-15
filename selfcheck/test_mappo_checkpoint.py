"""Complete checkpoint, RNG, normalizer, and one-update resume tests."""

from __future__ import annotations

import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from plain_mappo.config import MappoConfig
from plain_mappo.trainer import MappoTrainer


class MappoCheckpointTests(unittest.TestCase):
    def test_checkpoint_restores_complete_training_state_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(MappoConfig(), num_envs=2, rollout_length=4, mini_batch_size=4, ppo_epochs=1,
                             periodic_eval_episodes=1, output_dir=directory)
            trainer = MappoTrainer(config)
            trainer.train(1)
            latest = Path(directory) / "checkpoints" / "latest.pt"
            best, final = Path(directory) / "checkpoints" / "best.pt", Path(directory) / "checkpoints" / "actor_final.pt"
            self.assertTrue(latest.exists() and best.exists() and final.exists())
            expected_random, expected_numpy, expected_torch = random.random(), float(np.random.random()), float(torch.rand(()))
            restored = MappoTrainer(config)
            restored.load(latest)
            self.assertEqual(restored.update_count, trainer.update_count)
            self.assertEqual(restored.total_env_steps, trainer.total_env_steps)
            self.assertEqual(restored.best_score, trainer.best_score)
            np.testing.assert_allclose(restored.normalizer.mean, trainer.normalizer.mean)
            np.testing.assert_allclose(restored.normalizer.var, trainer.normalizer.var)
            for original, loaded in zip(trainer.actor.parameters(), restored.actor.parameters()):
                torch.testing.assert_close(original, loaded)
            for original, loaded in zip(trainer.critic.parameters(), restored.critic.parameters()):
                torch.testing.assert_close(original, loaded)
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(float(np.random.random()), expected_numpy)
            self.assertEqual(float(torch.rand(())), expected_torch)
            restored.train(1)
            self.assertEqual(restored.update_count, trainer.update_count + 1)
            self.assertFalse((Path(directory) / "checkpoints" / "latest.pt.tmp").exists())

    def test_resume_consumes_saved_next_unused_episode_seeds(self) -> None:
        """A resume must not silently restart its environments at early seeds."""
        with tempfile.TemporaryDirectory() as directory:
            config = replace(MappoConfig(), num_envs=2, rollout_length=4, mini_batch_size=4, ppo_epochs=1,
                             periodic_eval_episodes=1, output_dir=directory)
            trainer = MappoTrainer(config)
            # Treat these as already-consumed episodes; the checkpoint stores
            # the next unused values, not the previous/current episode IDs.
            trainer.next_episode_indices = [3, 7]
            trainer.save_latest()
            restored = MappoTrainer(config)
            restored.load(Path(directory) / "checkpoints" / "latest.pt")
            # Loading starts a fresh replacement episode with seed indices 3/7
            # and advances the progress to the next never-used 4/8.
            self.assertEqual(restored.next_episode_indices, [4, 8])
            expected = MappoTrainer(config)
            expected.next_episode_indices = [3, 7]
            expected._reset_training_envs()
            np.testing.assert_array_equal(restored.current_obs[0], expected.current_obs[0])
            restored._reset_one_env(0)
            expected._reset_one_env(0)
            self.assertEqual(restored.next_episode_indices[0], 5)
            np.testing.assert_array_equal(restored.current_obs[0], expected.current_obs[0])

    def test_state_independent_distribution_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(MappoConfig(), num_envs=2, rollout_length=4, mini_batch_size=4, ppo_epochs=1,
                             periodic_eval_episodes=1, output_dir=directory,
                             actor_log_std_mode="state_independent_tanh", log_std_min=-4.0, log_std_max=0.0,
                             state_independent_log_std_init=-1.5)
            trainer = MappoTrainer(config)
            trainer.train(1)
            restored = MappoTrainer(config)
            restored.load(Path(directory) / "checkpoints" / "latest.pt")
            self.assertEqual(tuple(restored.actor.log_std_parameter.shape), (3,))
            torch.testing.assert_close(trainer.actor.log_std_parameter, restored.actor.log_std_parameter)
            restored.train(1)
            self.assertEqual(restored.update_count, 2)


if __name__ == "__main__":
    unittest.main()
