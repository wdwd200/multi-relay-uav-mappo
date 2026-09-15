"""Stage 4 Role-ablation contracts without changing the RelayEnv interface."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from plain_mappo.config import MappoConfig
from plain_mappo.networks import SharedActor
from plain_mappo.roles import role_ids_for_num_relays
from plain_mappo.trainer import MappoTrainer


ROLE_VARIANTS = ("role_info", "role_head")


def _role_config(directory: str, variant: str) -> MappoConfig:
    return replace(MappoConfig(), actor_variant=variant, actor_log_std_mode="state_independent_tanh",
                   log_std_min=-4.0, log_std_max=0.0, state_independent_log_std_init=-1.5,
                   entropy_coef=0.0, actor_lr=1e-4, critic_lr=3e-4,
                   num_envs=2, rollout_length=4, mini_batch_size=4, ppo_epochs=1,
                   periodic_eval_episodes=1, output_dir=directory)


class MappoRoleTests(unittest.TestCase):
    def _actor(self, variant: str) -> SharedActor:
        return SharedActor(26, 3, log_std_min=-4.0, log_std_max=0.0,
                           log_std_mode="state_independent_tanh", state_independent_log_std_init=-1.5,
                           actor_variant=variant)

    def test_static_role_mapping_for_k3_k4_k5(self) -> None:
        self.assertEqual(role_ids_for_num_relays(3), (0, 1, 2))
        self.assertEqual(role_ids_for_num_relays(4), (0, 1, 1, 2))
        self.assertEqual(role_ids_for_num_relays(5), (0, 1, 1, 1, 2))

    def test_input_dimensions_keep_environment_observation_at_26(self) -> None:
        observation = torch.randn(2, 4, 26)
        plain = self._actor("plain")
        role_info = self._actor("role_info")
        role_head = self._actor("role_head")
        self.assertEqual(plain.obs_dim, 26); self.assertEqual(plain.effective_input_dim, 26)
        self.assertEqual(role_info.obs_dim, 26); self.assertEqual(role_info.effective_input_dim, 29)
        self.assertEqual(role_head.obs_dim, 26); self.assertEqual(role_head.effective_input_dim, 26)
        self.assertEqual(tuple(plain.effective_actor_input(observation).shape), (2, 4, 26))
        self.assertEqual(tuple(role_info.effective_actor_input(observation).shape), (2, 4, 29))
        self.assertEqual(tuple(role_head.effective_actor_input(observation).shape), (2, 4, 26))

    def test_role_info_appends_correct_distinct_one_hot_features(self) -> None:
        actor = self._actor("role_info")
        observation = torch.zeros(1, 4, 26)
        effective = actor.effective_actor_input(observation)
        expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                 [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        torch.testing.assert_close(effective[0, :, :26], observation[0])
        torch.testing.assert_close(effective[0, :, 26:], expected)
        self.assertFalse(torch.equal(effective[0, 0, 26:], effective[0, 1, 26:]))

    def test_role_head_has_three_independent_heads_and_middle_is_shared(self) -> None:
        actor = self._actor("role_head")
        self.assertIsNot(actor.source_mean_head, actor.middle_mean_head)
        self.assertIsNot(actor.middle_mean_head, actor.destination_mean_head)
        self.assertEqual(sum(parameter.numel() for parameter in actor.source_mean_head.parameters()), 387)
        self.assertEqual(sum(parameter.numel() for parameter in actor.middle_mean_head.parameters()), 387)
        self.assertEqual(sum(parameter.numel() for parameter in actor.destination_mean_head.parameters()), 387)
        with torch.no_grad():
            for parameter in actor.backbone.parameters():
                parameter.zero_()
            for bias, head in ((1.0, actor.source_mean_head), (2.0, actor.middle_mean_head),
                               (3.0, actor.destination_mean_head)):
                head.weight.zero_()
                head.bias.fill_(bias)
        same_observation = torch.zeros(1, 4, 26)
        mean, _ = actor(same_observation)
        torch.testing.assert_close(mean[0, 0], torch.full((3,), 1.0))
        torch.testing.assert_close(mean[0, 1], torch.full((3,), 2.0))
        torch.testing.assert_close(mean[0, 2], torch.full((3,), 2.0))
        torch.testing.assert_close(mean[0, 3], torch.full((3,), 3.0))

    def test_all_actor_variants_have_k3_k4_k5_action_shapes(self) -> None:
        for variant in ("plain", *ROLE_VARIANTS):
            actor = self._actor(variant)
            for relays in (3, 4, 5):
                observation = torch.randn(2, relays, 26)
                mean, log_std = actor(observation)
                self.assertEqual(tuple(mean.shape), (2, relays, 3))
                self.assertEqual(tuple(log_std.shape), (2, relays, 3))

    def test_role_variants_use_one_globally_shared_log_std_parameter(self) -> None:
        for variant in ROLE_VARIANTS:
            actor = self._actor(variant)
            first, second = torch.randn(2, 4, 26), torch.randn(3, 4, 26)
            _, raw_first, effective_first = actor.distribution_parameters(first)
            _, raw_second, effective_second = actor.distribution_parameters(second)
            self.assertEqual(tuple(actor.log_std_parameter.shape), (3,))
            self.assertEqual(len([name for name, _ in actor.named_parameters() if "log_std" in name]), 1)
            torch.testing.assert_close(raw_first, raw_second[:2])
            torch.testing.assert_close(effective_first, effective_first[:, :1].expand_as(effective_first))
            torch.testing.assert_close(effective_second, effective_second[:, :1].expand_as(effective_second))
            latent = torch.randn(2, 4, 3)
            (-actor.log_prob_entropy(first, latent)[0].mean()).backward()
            self.assertIsNotNone(actor.log_std_parameter.grad)
            self.assertTrue(torch.isfinite(actor.log_std_parameter.grad).all())

    def test_role_variants_reject_legacy_state_dependent_log_std(self) -> None:
        for variant in ROLE_VARIANTS:
            with self.assertRaisesRegex(ValueError, "require state_independent_tanh"):
                SharedActor(26, 3, actor_variant=variant)

    def test_role_variants_complete_finite_ppo_update_without_resampling_latent(self) -> None:
        for variant in ROLE_VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                trainer = MappoTrainer(_role_config(directory, variant))
                buffer, _ = trainer.collect_rollout()
                actor_before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
                critic_before = [parameter.detach().clone() for parameter in trainer.critic.parameters()]
                trainer.actor.sample_actions = lambda *_: (_ for _ in ()).throw(AssertionError("PPO update resampled latent z"))  # type: ignore[method-assign]
                result = trainer.update(buffer)
                self.assertTrue(result["finite"])
                self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in result.values()))
                self.assertTrue(any(not torch.equal(before, after) for before, after in zip(actor_before, trainer.actor.parameters())))
                self.assertTrue(any(not torch.equal(before, after) for before, after in zip(critic_before, trainer.critic.parameters())))

    def test_role_checkpoint_round_trip_preserves_deterministic_actions(self) -> None:
        for variant in ROLE_VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                config = _role_config(directory, variant)
                trainer = MappoTrainer(config)
                observation = torch.randn(2, 4, 26)
                expected = trainer.actor.deterministic_actions(observation).detach().clone()
                checkpoint = trainer.save_latest()
                restored = MappoTrainer(config)
                restored.load(checkpoint)
                self.assertEqual(restored.config.actor_variant, variant)
                torch.testing.assert_close(restored.actor.deterministic_actions(observation), expected)
                torch.testing.assert_close(restored.actor.log_std_parameter, trainer.actor.log_std_parameter)

    def test_old_config_without_actor_variant_defaults_to_plain(self) -> None:
        old_config = MappoConfig().to_dict()
        old_config.pop("actor_variant")
        restored = MappoConfig.from_dict(old_config)
        self.assertEqual(restored.actor_variant, "plain")


if __name__ == "__main__":
    unittest.main()
