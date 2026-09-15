"""Tanh-Gaussian action and probability tests."""

from __future__ import annotations

import math
import unittest

import torch

from plain_mappo.networks import SharedActor
from relay_env import RelayEnv


class MappoActionTests(unittest.TestCase):
    def test_latent_actions_are_unbounded_but_tanh_actions_are_finite(self) -> None:
        # These values are already outside the normalized action range while
        # remaining distinguishable from +/-1 in float32 tanh arithmetic.
        latent = torch.tensor([[[-2.0, 0.0, 2.0]]])
        action = torch.tanh(latent)
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue(torch.all(action > -1.0))
        self.assertTrue(torch.all(action < 1.0))

    def test_log_probability_is_the_sum_of_three_dimensions(self) -> None:
        actor = SharedActor(26, 3)
        observation = torch.randn(2, 4, 26)
        latent = torch.randn(2, 4, 3) * 3.0
        distribution = actor.distribution(observation)
        summed, _ = actor.log_prob_entropy(observation, latent)
        torch.testing.assert_close(summed, distribution.log_prob(latent).sum(dim=-1))

    def test_same_old_latent_is_evaluated_under_new_distribution(self) -> None:
        actor = SharedActor(26, 3)
        observation, latent = torch.randn(1, 4, 26), torch.tensor([[[5.0, -4.0, 2.0]] * 4])
        old_log_prob, _ = actor.log_prob_entropy(observation, latent)
        with torch.no_grad():
            actor.mean_head.bias.add_(0.5)
        new_log_prob, _ = actor.log_prob_entropy(observation, latent)
        self.assertFalse(torch.equal(old_log_prob, new_log_prob))
        self.assertTrue(torch.isfinite(new_log_prob).all())

    def test_state_independent_log_std_is_shared_and_observation_independent(self) -> None:
        actor = SharedActor(26, 3, log_std_min=-4.0, log_std_max=0.0,
                            log_std_mode="state_independent_tanh", state_independent_log_std_init=-1.5)
        first, second = torch.randn(2, 4, 26), torch.randn(2, 4, 26)
        _, raw_first, effective_first = actor.distribution_parameters(first)
        _, raw_second, effective_second = actor.distribution_parameters(second)
        self.assertEqual(tuple(actor.log_std_parameter.shape), (3,))
        self.assertFalse(hasattr(actor, "log_std_head"))
        torch.testing.assert_close(raw_first, raw_second)
        torch.testing.assert_close(effective_first, effective_second)
        self.assertTrue(torch.all(effective_first > -4.0) and torch.all(effective_first < 0.0))
        latent = torch.randn(2, 4, 3)
        log_prob, _ = actor.log_prob_entropy(first, latent)
        (-log_prob.mean()).backward()
        self.assertIsNotNone(actor.log_std_parameter.grad)
        self.assertTrue(torch.isfinite(actor.log_std_parameter.grad).all())

    def test_state_independent_same_old_latent_recomputes_log_prob(self) -> None:
        actor = SharedActor(26, 3, log_std_min=-4.0, log_std_max=0.0,
                            log_std_mode="state_independent_tanh", state_independent_log_std_init=-1.5)
        observation, latent = torch.randn(1, 4, 26), torch.randn(1, 4, 3)
        old_log_prob, _ = actor.log_prob_entropy(observation, latent)
        with torch.no_grad():
            actor.log_std_parameter.add_(0.1)
        new_log_prob, _ = actor.log_prob_entropy(observation, latent)
        self.assertFalse(torch.equal(old_log_prob, new_log_prob))
        self.assertTrue(torch.isfinite(new_log_prob).all())

    def test_extreme_network_values_stay_finite(self) -> None:
        actor = SharedActor(26, 3)
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.fill_(1e3)
        observation, latent = torch.full((2, 4, 26), 1e3), torch.full((2, 4, 3), -1e3)
        mean, log_std = actor(observation)
        log_prob, entropy = actor.log_prob_entropy(observation, latent)
        self.assertTrue(torch.isfinite(mean).all() and torch.isfinite(log_std).all())
        self.assertTrue(torch.isfinite(log_prob).all() and torch.isfinite(entropy).all())

    def test_environment_extreme_normalized_actions_obey_physical_acceleration_limits(self) -> None:
        env = RelayEnv(4, debug=True)
        env.reset(seed=88)
        _, _, _, _, info = env.step([(1.0, 1.0, 1.0)] * 4)
        for acceleration in info["action_debug"]["requested_accelerations"]:
            self.assertLessEqual(math.hypot(acceleration[0], acceleration[1]), env.config.relay_max_xy_accel_mps2 + 1e-12)
            self.assertLessEqual(abs(acceleration[2]), env.config.relay_max_z_accel_mps2 + 1e-12)


if __name__ == "__main__":
    unittest.main()
