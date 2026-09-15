"""PPO clipping, team-to-agent broadcasting, and separate optimizer tests."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from plain_mappo.config import MappoConfig
from plain_mappo.trainer import MappoTrainer


class MappoUpdateTests(unittest.TestCase):
    def _config(self, directory: str) -> MappoConfig:
        return replace(MappoConfig(), num_envs=2, rollout_length=4, mini_batch_size=4, ppo_epochs=1,
                       periodic_eval_episodes=1, output_dir=directory)

    def test_clipped_surrogate_signs_and_clip_fraction(self) -> None:
        ratio = torch.tensor([[1.5], [1.5], [1.0]])
        advantage = torch.tensor([[1.0], [-1.0], [1.0]])
        unclipped, clipped = ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
        loss = -torch.minimum(unclipped, clipped).mean()
        self.assertAlmostEqual(float(loss), -((1.2 - 1.5 + 1.0) / 3.0), places=7)
        self.assertAlmostEqual(float((torch.abs(ratio - 1.0) > 0.2).float().mean()), 2.0 / 3.0, places=7)

    def test_team_advantage_broadcasts_to_four_agents(self) -> None:
        advantage = torch.tensor([2.5, -1.0]).unsqueeze(-1)
        broadcast = advantage.expand(-1, 4)
        self.assertEqual(tuple(broadcast.shape), (2, 4))
        torch.testing.assert_close(broadcast[0], torch.full((4,), 2.5))
        torch.testing.assert_close(broadcast[1], torch.full((4,), -1.0))

    def test_update_changes_both_models_without_resampling_and_clips_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = MappoTrainer(self._config(directory))
            buffer, _ = trainer.collect_rollout()
            actor_before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
            critic_before = [parameter.detach().clone() for parameter in trainer.critic.parameters()]
            trainer.actor.sample_actions = lambda *_: (_ for _ in ()).throw(AssertionError("PPO update must not resample latent z"))  # type: ignore[method-assign]
            actual_clip = torch.nn.utils.clip_grad_norm_
            parameter_sets: list[set[int]] = []
            def record_clip(parameters, max_norm, *args, **kwargs):
                parameter_list = list(parameters)
                parameter_sets.append({id(parameter) for parameter in parameter_list})
                return actual_clip(parameter_list, max_norm, *args, **kwargs)
            with patch("torch.nn.utils.clip_grad_norm_", side_effect=record_clip):
                result = trainer.update(buffer)
            self.assertTrue(result["finite"])
            self.assertTrue(any(not torch.equal(before, after) for before, after in zip(actor_before, trainer.actor.parameters())))
            self.assertTrue(any(not torch.equal(before, after) for before, after in zip(critic_before, trainer.critic.parameters())))
            actor_ids = {id(parameter) for parameter in trainer.actor.parameters()}
            critic_ids = {id(parameter) for parameter in trainer.critic.parameters()}
            self.assertEqual(len(parameter_sets), 4)  # two mini-batches, Actor then Critic each.
            self.assertEqual(parameter_sets[0], actor_ids); self.assertEqual(parameter_sets[1], critic_ids)
            self.assertEqual(parameter_sets[2], actor_ids); self.assertEqual(parameter_sets[3], critic_ids)


if __name__ == "__main__":
    unittest.main()
