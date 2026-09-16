"""Regression tests for the Stage-4.5 diagnostic/training gate."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from plain_mappo.networks import SharedActor
from plain_mappo.topology import build_topology_features
from relay_env import RelayEnv
import stage4_5_p3_p4_diagnostics_multiseed as stage45


class Stage45DiagnosticTests(unittest.TestCase):
    def test_paired_2027_config_keeps_every_frozen_common_field(self) -> None:
        p3 = stage45._config_for("P3", "topology_info", 2027).to_dict()
        p4 = stage45._config_for("P4", "graph", 2027).to_dict()
        for field in ("actor_variant", "output_dir"):
            p3.pop(field)
            p4.pop(field)
        self.assertEqual(p3, p4)
        self.assertEqual((p3["value_normalization"], p3["gamma"], p3["gae_lambda"]), (False, 0.99, 0.95))
        self.assertEqual(p3["full_updates"], 1000)

    def test_diagnostic_protocol_uses_only_locked_validation_and_forbids_final_test(self) -> None:
        payload = stage45._diagnostic_payload()
        self.assertFalse(payload["final_test"]["executed"])
        self.assertFalse(payload["final_test"]["permitted"])
        self.assertEqual(tuple(payload["locked_validation"]["seeds"]), stage45.VALIDATION_SEEDS)
        self.assertTrue(set(payload["locked_validation"]["seeds"]).isdisjoint(stage45.FINAL_TEST_SEEDS))
        self.assertIn("edge_perturbation", payload["graph_probe"])

    def test_graph_trace_matches_actor_and_message_block_has_finite_shape(self) -> None:
        config = stage45._config_for("P4", "graph", 2027)
        actor = SharedActor(config.local_obs_dim, config.action_dim, config.actor_hidden_sizes,
                            config.log_std_min, config.log_std_max, config.actor_log_std_mode,
                            config.state_independent_log_std_init, config.actor_variant,
                            config.topology_node_dim, config.topology_edge_dim, config.graph_hidden_dim)
        env = RelayEnv(config.num_relays)
        observation, _ = env.reset(seed=stage45.VALIDATION_SEEDS[0])
        nodes, edges = build_topology_features(env, env.get_global_state())
        obs_tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
        nodes_tensor = torch.as_tensor(nodes).unsqueeze(0)
        edges_tensor = torch.as_tensor(edges).unsqueeze(0)
        traced, rounds = stage45._graph_trace(actor, obs_tensor, nodes_tensor, edges_tensor)
        direct = actor.graph_embeddings(obs_tensor, nodes_tensor, edges_tensor)
        blocked = stage45._graph_embeddings_without_messages(actor, obs_tensor, nodes_tensor, edges_tensor)
        actions = stage45._actions_from_embeddings(actor, obs_tensor, blocked)
        self.assertEqual(len(rounds), config.num_relays)
        self.assertTrue(torch.allclose(traced, direct, atol=1e-7, rtol=0.0))
        self.assertEqual(tuple(blocked.shape), tuple(direct.shape))
        self.assertTrue(torch.isfinite(actions).all())

    def test_summary_recompute_comparison_allows_only_round_trip_float_noise(self) -> None:
        self.assertTrue(stage45._same_summary({"a": 1.0, "b": 2}, {"a": 1.0 + 5e-13, "b": 2}))
        self.assertFalse(stage45._same_summary({"a": 1.0}, {"a": 1.0 + 2e-12}))


if __name__ == "__main__":
    unittest.main()
