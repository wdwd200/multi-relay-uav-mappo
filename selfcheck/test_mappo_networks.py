"""Network and revised 47-dimensional global-state contract tests."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from plain_mappo.networks import CentralizedCritic, SharedActor
from plain_mappo.state import flatten_global_state, global_state_dim, global_state_feature_names
from relay_env import RelayEnv


class MappoNetworkAndStateTests(unittest.TestCase):
    def test_shared_actor_shapes_for_k3_k4_k5(self) -> None:
        actor = SharedActor(26, 3)
        for relays in (3, 4, 5):
            observation = torch.randn(2, relays, 26)
            mean, log_std = actor(observation)
            self.assertEqual(tuple(mean.shape), (2, relays, 3))
            self.assertEqual(tuple(log_std.shape), (2, relays, 3))
            independently_applied = torch.stack([actor(observation[:, index])[0] for index in range(relays)], dim=1)
            torch.testing.assert_close(mean, independently_applied)  # the same Actor object handles every relay.

    def test_state_independent_actor_shapes_for_k3_k4_k5(self) -> None:
        actor = SharedActor(26, 3, log_std_min=-4.0, log_std_max=0.0,
                            log_std_mode="state_independent_tanh", state_independent_log_std_init=-1.5)
        for relays in (3, 4, 5):
            observation = torch.randn(2, relays, 26)
            mean, log_std = actor(observation)
            self.assertEqual(tuple(mean.shape), (2, relays, 3))
            self.assertEqual(tuple(log_std.shape), (2, relays, 3))
            torch.testing.assert_close(log_std, log_std[:, :1].expand_as(log_std))

    def test_critic_outputs_one_team_value(self) -> None:
        critic = CentralizedCritic(47)
        self.assertEqual(tuple(critic(torch.zeros(7, 47)).shape), (7,))

    def test_explicit_global_state_contract_is_47_finite_and_ordered(self) -> None:
        env = RelayEnv(4)
        env.reset(seed=123)
        state = env.get_global_state()
        vector = flatten_global_state(state, 4)
        names = global_state_feature_names(4)
        self.assertEqual(vector.shape, (47,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(global_state_dim(3), 41)
        self.assertEqual(global_state_dim(4), 47)
        self.assertEqual(global_state_dim(5), 53)
        self.assertEqual(names[:10], ("H.position.x", "H.position.y", "H.position.z", "H.velocity.x", "H.velocity.y", "H.velocity.z", "H.waypoint.x", "H.waypoint.y", "H.waypoint.z", "H.cruise_speed"))
        self.assertEqual(names[-3:], ("step", "sim_time", "consecutive_outage_steps"))
        self.assertEqual(vector[9], state["H"]["cruise_speed"])
        self.assertEqual(vector[19], state["L"]["cruise_speed"])
        self.assertEqual(vector[-1], state["consecutive_outage_steps"])
        env.consecutive_outage_steps = 7  # read-only interface must expose the live diagnostic counter.
        self.assertEqual(flatten_global_state(env.get_global_state(), 4)[-1], 7.0)
        # Same semantic mapping in a different insertion order must flatten identically.
        reordered = {"consecutive_outage_steps": state["consecutive_outage_steps"], "relays": state["relays"], "L": state["L"], "sim_time": state["sim_time"], "H": state["H"], "step": state["step"]}
        np.testing.assert_array_equal(vector, flatten_global_state(reordered, 4))

    def test_local_observation_remains_26_after_global_interface_extension(self) -> None:
        for relays in (3, 4, 5):
            env = RelayEnv(relays)
            observation, _ = env.reset(seed=200 + relays)
            self.assertEqual((len(observation), len(observation[0])), (relays, 26))
            self.assertEqual(flatten_global_state(env.get_global_state(), relays).shape, (global_state_dim(relays),))


if __name__ == "__main__":
    unittest.main()
