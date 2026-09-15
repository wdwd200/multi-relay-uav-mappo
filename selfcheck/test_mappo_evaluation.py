"""Deterministic independent evaluation and safety-first ranking tests."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

from plain_mappo.config import MappoConfig
from plain_mappo.evaluation import evaluate_actor, evaluate_random_policy
from plain_mappo.metrics import safety_priority_key
from plain_mappo.trainer import MappoTrainer


class MappoEvaluationTests(unittest.TestCase):
    def test_deterministic_actor_evaluation_is_repeatable_and_no_grad(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(MappoConfig(), periodic_eval_episodes=2, output_dir=directory)
            trainer = MappoTrainer(config)
            first_summary, first_score, first_episodes = evaluate_actor(trainer.actor, config, [10_000, 10_001])
            second_summary, second_score, second_episodes = evaluate_actor(trainer.actor, config, [10_000, 10_001])
            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first_score, second_score)
            self.assertEqual(first_episodes, second_episodes)
            self.assertTrue(all(parameter.grad is None for parameter in trainer.actor.parameters()))

    def test_safety_priority_lexicographic_order(self) -> None:
        safe = {"speed_accel_violations": 0, "collision_count": 0, "boundary_count": 0, "normal_completion_count": 1,
                "persistent_outage_count": 0, "outage_step_ratio": 0.9, "rate_satisfaction_ratio": 0.0, "mean_e2e_rate_mbps": 0.0}
        unsafe_but_fast = {**safe, "collision_count": 1, "mean_e2e_rate_mbps": 100.0}
        self.assertLess(safety_priority_key(safe), safety_priority_key(unsafe_but_fast))

    def test_completion_ranks_before_nonhard_termination_counts(self) -> None:
        all_outage_but_no_collision = {"speed_accel_violations": 0, "collision_count": 0, "boundary_count": 0,
                                        "normal_completion_count": 0, "persistent_outage_count": 20,
                                        "outage_step_ratio": 0.2, "rate_satisfaction_ratio": 0.8,
                                        "mean_e2e_rate_mbps": 6.0}
        mostly_complete_with_one_boundary = {**all_outage_but_no_collision, "normal_completion_count": 19,
                                              "persistent_outage_count": 0, "boundary_count": 1,
                                              "outage_step_ratio": 0.01, "rate_satisfaction_ratio": 0.99,
                                              "mean_e2e_rate_mbps": 8.0}
        self.assertLess(safety_priority_key(mostly_complete_with_one_boundary),
                        safety_priority_key(all_outage_but_no_collision))

    def test_fixed_seed_random_baseline_is_repeatable(self) -> None:
        config = MappoConfig()
        first_summary, first_score, _ = evaluate_random_policy(config, [10_000, 10_001])
        second_summary, second_score, _ = evaluate_random_policy(config, [10_000, 10_001])
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_score, second_score)


if __name__ == "__main__":
    unittest.main()
