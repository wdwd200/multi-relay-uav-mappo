"""Hand-calculated GAE and rollout-boundary regression tests."""

from __future__ import annotations

import unittest

import numpy as np

from plain_mappo.rollout_buffer import RolloutBuffer


def buffer_for_gae(time: int, envs: int = 1) -> RolloutBuffer:
    buffer = RolloutBuffer(time, envs, 4, 26, 47, 3)
    buffer.position = time
    return buffer


class MappoBufferTests(unittest.TestCase):
    def test_required_shapes_and_cpu_storage(self) -> None:
        buffer = RolloutBuffer(128, 8, 4, 26, 47, 3)
        self.assertEqual(buffer.local_obs.shape, (128, 8, 4, 26))
        self.assertEqual(buffer.global_state.shape, (128, 8, 47))
        self.assertEqual(buffer.latent_z.shape, (128, 8, 4, 3))
        self.assertEqual(buffer.old_log_prob.shape, (128, 8, 4))
        self.assertEqual(buffer.value.shape, (128, 8))
        self.assertEqual(buffer.local_obs.dtype, np.float32)

    def test_continuing_transition_gae_matches_hand_calculation(self) -> None:
        buffer = buffer_for_gae(2)
        buffer.reward[:, 0] = [1.0, 1.0]
        buffer.value[:, 0] = [0.5, 0.5]
        buffer.next_value[:, 0] = [0.5, 0.5]
        buffer.bootstrap_mask[:, 0] = 1.0
        buffer.trace_mask[:, 0] = 1.0
        advantage = buffer.compute_gae(0.99, 0.95)[:, 0]
        delta = 1.0 + 0.99 * 0.5 - 0.5
        np.testing.assert_allclose(advantage, [delta + 0.99 * 0.95 * delta, delta], rtol=1e-6)

    def test_terminated_does_not_bootstrap(self) -> None:
        buffer = buffer_for_gae(1)
        buffer.reward[0, 0], buffer.value[0, 0], buffer.next_value[0, 0] = 1.0, 0.5, 999.0
        buffer.bootstrap_mask[0, 0], buffer.trace_mask[0, 0] = 0.0, 0.0
        self.assertAlmostEqual(float(buffer.compute_gae(0.99, 0.95)[0, 0]), 0.5)

    def test_truncated_bootstraps_but_does_not_cross_reset(self) -> None:
        buffer = buffer_for_gae(2)
        buffer.reward[:, 0] = [1.0, 100.0]
        buffer.value[:, 0] = [0.5, 0.0]
        buffer.next_value[:, 0] = [0.5, 0.0]
        buffer.bootstrap_mask[:, 0] = 1.0
        buffer.trace_mask[:, 0] = [0.0, 1.0]
        advantage = buffer.compute_gae(0.99, 0.95)[:, 0]
        self.assertAlmostEqual(float(advantage[0]), 0.995, places=6)
        self.assertAlmostEqual(float(advantage[1]), 100.0, places=6)

    def test_rollout_boundary_bootstraps_when_episode_continues(self) -> None:
        buffer = buffer_for_gae(1)
        buffer.reward[0, 0], buffer.value[0, 0], buffer.next_value[0, 0] = 1.0, 0.5, 0.5
        buffer.bootstrap_mask[0, 0], buffer.trace_mask[0, 0] = 1.0, 1.0
        self.assertAlmostEqual(float(buffer.compute_gae(0.99, 0.95)[0, 0]), 0.995, places=6)

    def test_parallel_trace_masks_do_not_cross_contaminate(self) -> None:
        buffer = buffer_for_gae(2, 2)
        buffer.reward[0] = [1.0, 1.0]
        buffer.reward[1] = [100.0, 2.0]
        buffer.bootstrap_mask[:] = 1.0
        buffer.trace_mask[0] = [0.0, 1.0]
        buffer.trace_mask[1] = 1.0
        advantage = buffer.compute_gae(1.0, 1.0)
        self.assertAlmostEqual(float(advantage[0, 0]), 1.0)
        self.assertAlmostEqual(float(advantage[0, 1]), 3.0)


if __name__ == "__main__":
    unittest.main()
