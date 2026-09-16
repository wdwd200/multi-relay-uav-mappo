"""Stage-4.3 P3/P4 engineering-gate regression contracts."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from plain_mappo.engineering_gate import (ENGINEERING_GATE_VARIANTS, SAMPLES_PER_UPDATE,
                                          SMOKE_RUN_SEED, SMOKE_UPDATES, actor_parameter_counts,
                                          implementation_audit, stage43_smoke_config)
from plain_mappo.networks import SharedActor
from plain_mappo.topology import build_topology_features
from plain_mappo.trainer import MappoTrainer
from relay_env import RelayEnv


class Stage43EngineeringGateTests(unittest.TestCase):
    def test_p3_p4_k3_k4_k5_actions_and_chain_topology_are_finite(self) -> None:
        for relays in (3, 4, 5):
            env = RelayEnv(relays)
            observation, _ = env.reset(seed=60_000 + relays)
            nodes, edges = build_topology_features(env, env.get_global_state())
            self.assertEqual(nodes.shape, (relays + 2, 6))
            self.assertEqual(edges.shape, (relays + 1, 7))
            self.assertTrue(np.isfinite(nodes).all() and np.isfinite(edges).all())
            local = torch.as_tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
            node_tensor, edge_tensor = torch.as_tensor(nodes).unsqueeze(0), torch.as_tensor(edges).unsqueeze(0)
            for variant in ENGINEERING_GATE_VARIANTS:
                with self.subTest(relays=relays, variant=variant):
                    actor = SharedActor(26, 3, log_std_min=-4.0, log_std_max=0.0,
                                        log_std_mode="state_independent_tanh",
                                        state_independent_log_std_init=-1.5, actor_variant=variant)
                    action = actor.deterministic_actions(local, node_tensor, edge_tensor)
                    _, raw_log_std, log_std = actor.distribution_parameters(local, node_tensor, edge_tensor)
                    self.assertEqual(tuple(action.shape), (1, relays, 3))
                    if variant == "topology_info":
                        self.assertEqual(tuple(actor.effective_actor_input(local, node_tensor, edge_tensor).shape),
                                         (1, relays, 97))
                    self.assertTrue(torch.isfinite(action).all())
                    self.assertTrue(torch.isfinite(raw_log_std).all() and torch.isfinite(log_std).all())
                    self.assertEqual(tuple(actor.log_std_parameter.shape), (3,))

    def test_stage43_smokes_keep_formal_v0h0_and_paired_critic_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configs = [stage43_smoke_config(actor_variant=variant, output_dir=str(Path(directory) / variant))
                       for variant in ENGINEERING_GATE_VARIANTS]
            for config in configs:
                self.assertEqual(config.run_seed, SMOKE_RUN_SEED)
                self.assertEqual((config.value_normalization, config.gamma, config.gae_lambda), (False, 0.99, 0.95))
                self.assertEqual((config.num_envs, config.rollout_length, config.smoke_updates), (8, 128, SMOKE_UPDATES))
                self.assertEqual(config.team_time_samples, SAMPLES_PER_UPDATE)
            p3, p4 = (MappoTrainer(config) for config in configs)
            for left, right in zip(p3.critic.parameters(), p4.critic.parameters()):
                torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
            self.assertTrue(torch.equal(p3.action_noise_generator.get_state(), p4.action_noise_generator.get_state()))  # type: ignore[union-attr]
            self.assertEqual(p3.minibatch_rng.bit_generator.state, p4.minibatch_rng.bit_generator.state)  # type: ignore[union-attr]

    def test_parameter_fairness_and_static_audit_are_frozen(self) -> None:
        counts = actor_parameter_counts()
        self.assertEqual(counts, {"topology_info": 29_446, "graph": 29_094})
        audit = implementation_audit()
        self.assertTrue(audit["parameter_fairness"]["passes"])
        self.assertLessEqual(audit["parameter_fairness"]["relative_difference_of_larger"], 0.10)
        self.assertEqual(audit["P3"]["actor_input_dim_k4"], 97)
        self.assertEqual(audit["P4"]["graph_hidden_dim"], 32)
        self.assertFalse(audit["frozen_scientific_contract_changed"])

    def test_smoke_can_skip_selection_export_but_keeps_reloadable_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            formal = stage43_smoke_config(actor_variant="graph", output_dir=directory)
            # Smaller dimensions only make this a unit test; the actual smoke
            # uses the unmodified formal 8 x 128 collection contract.
            config = replace(formal, num_envs=2, rollout_length=4, mini_batch_size=4, ppo_epochs=1)
            trainer = MappoTrainer(config)
            trainer.train(1, final_evaluation=False, export_final_actor=False)
            latest = Path(directory) / "checkpoints" / "latest.pt"
            self.assertTrue(latest.is_file())
            self.assertFalse((Path(directory) / "checkpoints" / "actor_final.pt").exists())
            restored = MappoTrainer(config); restored.load(latest)
            self.assertEqual((restored.update_count, restored.total_env_steps), (1, 8))
            self.assertEqual(restored.config.resume_mode, "fresh_episode_at_update_boundary")
            self.assertFalse(restored.config.exact_environment_resume)


if __name__ == "__main__":
    unittest.main()
