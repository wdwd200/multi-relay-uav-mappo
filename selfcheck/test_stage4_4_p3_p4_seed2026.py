"""Stage-4.4 paired P3/P4 seed-2026 protocol contracts."""

from __future__ import annotations

import unittest

from plain_mappo.experiment_protocol import FINAL_TEST_SEEDS, VALIDATION_SEEDS
from plain_mappo.presets import STAGE4_FORMAL_FIELDS
from stage4_4_p3_p4_seed2026 import (EVAL_UPDATES, RUN_SEED, SAMPLES_PER_UPDATE, UPDATES,
                                     _config_for, _protocol_contract, _protocol_payload,
                                     _select_validation_summary)


class Stage44P3P4Seed2026Tests(unittest.TestCase):
    def test_p3_p4_configs_are_exact_stage4_formal_v0h0_pair(self) -> None:
        p3, p4 = _config_for("P3", "topology_info"), _config_for("P4", "graph")
        self.assertEqual((p3.run_seed, p4.run_seed), (RUN_SEED, RUN_SEED))
        self.assertEqual((p3.value_normalization, p3.gamma, p3.gae_lambda), (False, 0.99, 0.95))
        self.assertEqual((p3.full_updates, p3.team_time_samples), (UPDATES, SAMPLES_PER_UPDATE))
        self.assertEqual(tuple(EVAL_UPDATES), tuple(range(50, 1001, 50)))
        for field, value in STAGE4_FORMAL_FIELDS.items():
            self.assertEqual(getattr(p3, field), value, field)
            self.assertEqual(getattr(p4, field), value, field)
        left, right = p3.to_dict(), p4.to_dict()
        for field in ("actor_variant", "output_dir"):
            left.pop(field); right.pop(field)
        self.assertEqual(left, right)

    def test_protocol_declares_locked_validation_and_forbids_final_test(self) -> None:
        payload = _protocol_payload()
        self.assertEqual(tuple(payload["validation"]["seeds"]), VALIDATION_SEEDS)
        self.assertEqual(tuple(payload["final_test"]["seeds"]), FINAL_TEST_SEEDS)
        self.assertFalse(payload["final_test"]["executed"])
        self.assertFalse(payload["final_test"]["permitted"])
        self.assertEqual(_protocol_contract({**payload, "generated_at_utc": "changed"}), _protocol_contract(payload))

    def test_selection_obeys_safety_key_then_earlier_exact_tie(self) -> None:
        winner = _select_validation_summary([
            {"update": 100, "score": [0.0, -1.0, 0.0, 0.0, 0.1, -0.9, -7.0]},
            {"update": 50, "score": [0.0, -1.0, 0.0, 0.0, 0.1, -0.9, -7.0]},
            {"update": 150, "score": [0.0, -1.0, 1.0, 0.0, 0.0, -1.0, -8.0]},
        ])
        self.assertEqual(winner["update"], 50)
        safety_winner = _select_validation_summary([
            {"update": 50, "score": [1.0, -20.0, 0.0, 0.0, 0.0, -1.0, -10.0]},
            {"update": 100, "score": [0.0, 0.0, 10.0, 10.0, 1.0, 0.0, 0.0]},
        ])
        self.assertEqual(safety_winner["update"], 100)


if __name__ == "__main__":
    unittest.main()
