"""Stage 4 P3/P4 topology and Graph MAPPO contracts."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from plain_mappo.checkpoint import load_checkpoint
from plain_mappo.config import MappoConfig
from plain_mappo.networks import SharedActor
from plain_mappo.topology import build_topology_features, topology_feature_shapes
from plain_mappo.trainer import MappoTrainer
from relay_env import RelayEnv
from relay_env.communication import link_metrics


TOPOLOGY_VARIANTS = ("topology_info", "graph")


def _actor(variant: str) -> SharedActor:
    return SharedActor(26, 3, log_std_min=-4.0, log_std_max=0.0,
                       log_std_mode="state_independent_tanh", state_independent_log_std_init=-1.5,
                       actor_variant=variant)


def _trainer_config(directory: str, variant: str) -> MappoConfig:
    return replace(MappoConfig(), actor_variant=variant, actor_log_std_mode="state_independent_tanh",
                   log_std_min=-4.0, log_std_max=0.0, state_independent_log_std_init=-1.5,
                   entropy_coef=0.0, actor_lr=1e-4, critic_lr=3e-4,
                   num_envs=2, rollout_length=4, mini_batch_size=4, ppo_epochs=1,
                   periodic_eval_episodes=1, output_dir=directory)


class MappoTopologyTests(unittest.TestCase):
    def _environment_features(self, relays: int = 4) -> tuple[RelayEnv, np.ndarray, np.ndarray, np.ndarray, dict]:
        env = RelayEnv(relays)
        observation, _ = env.reset(seed=400 + relays)
        state = env.get_global_state()
        nodes, edges = build_topology_features(env, state)
        return env, np.asarray(observation, dtype=np.float32), nodes, edges, state

    def test_topology_shapes_for_k3_k4_k5(self) -> None:
        for relays in (3, 4, 5):
            _, _, nodes, edges, _ = self._environment_features(relays)
            self.assertEqual(nodes.shape, (relays + 2, 6))
            self.assertEqual(edges.shape, (relays + 1, 7))
            self.assertEqual(topology_feature_shapes(relays), ((relays + 2, 6), (relays + 1, 7)))

    def test_node_normalization_is_exact_and_topology_excludes_privileged_state(self) -> None:
        env, _, nodes, _, state = self._environment_features(4)
        raw_nodes = (state["H"],) + tuple(state["relays"]) + (state["L"],)
        for index, node in enumerate(raw_nodes):
            position, velocity = node["position"], node["velocity"]
            expected = (position[0] / env.config.map_x_m, position[1] / env.config.map_y_m,
                        (position[2] - 200.0) / 100.0, velocity[0] / 20.0,
                        velocity[1] / 20.0, velocity[2] / 6.0)
            np.testing.assert_allclose(nodes[index], expected, rtol=0.0, atol=1e-7)
        altered = copy.deepcopy(state)
        altered["H"]["waypoint"] = (999.0, 999.0, 999.0); altered["L"]["waypoint"] = (1.0, 1.0, 1.0)
        altered["H"]["cruise_speed"] = 123.0; altered["L"]["cruise_speed"] = -123.0
        altered["step"] = 999; altered["sim_time"] = -999.0; altered["consecutive_outage_steps"] = 999
        altered_nodes, altered_edges = build_topology_features(env, altered)
        original_nodes, original_edges = build_topology_features(env, state)
        np.testing.assert_array_equal(altered_nodes, original_nodes)
        np.testing.assert_array_equal(altered_edges, original_edges)
        self.assertEqual(nodes.shape[-1], 6)  # no ID, Role, chain-index, or flags can be present.

    def test_edge_deltas_and_capacity_match_existing_communication_function(self) -> None:
        env, _, _, edges, state = self._environment_features(4)
        raw_nodes = (state["H"],) + tuple(state["relays"]) + (state["L"],)
        comm = env.config.comm
        for index, (upstream, downstream) in enumerate(zip(raw_nodes[:-1], raw_nodes[1:])):
            up_p, down_p = upstream["position"], downstream["position"]
            up_v, down_v = upstream["velocity"], downstream["velocity"]
            metrics = link_metrics(up_p, down_p, comm.reference_gain, comm.path_loss_exponent,
                                   comm.tx_power_w, comm.noise_density_w_hz, comm.bandwidth_hz)
            expected = ((down_p[0] - up_p[0]) / 2000.0, (down_p[1] - up_p[1]) / 2000.0,
                        (down_p[2] - up_p[2]) / 200.0, (down_v[0] - up_v[0]) / 40.0,
                        (down_v[1] - up_v[1]) / 40.0, (down_v[2] - up_v[2]) / 12.0,
                        min(metrics["capacity_bps"] / 60_000_000.0, 1.0))
            np.testing.assert_allclose(edges[index], expected, rtol=0.0, atol=1e-7)

    def test_topology_info_has_97_input_and_repeats_only_the_chain_block(self) -> None:
        _, observation, nodes, edges, _ = self._environment_features(4)
        actor = _actor("topology_info")
        local = torch.as_tensor(observation).unsqueeze(0)
        effective = actor.effective_actor_input(local, torch.as_tensor(nodes).unsqueeze(0), torch.as_tensor(edges).unsqueeze(0))
        self.assertEqual(actor.effective_input_dim, 97)
        self.assertEqual(tuple(effective.shape), (1, 4, 97))
        torch.testing.assert_close(effective[..., :26], local)
        torch.testing.assert_close(effective[:, 0, 26:], effective[:, 1, 26:])
        torch.testing.assert_close(effective[:, 1, 26:], effective[:, 2, 26:])
        torch.testing.assert_close(effective[:, 2, 26:], effective[:, 3, 26:])

    def test_topology_variants_require_complete_topology(self) -> None:
        local = torch.zeros(1, 4, 26)
        for variant in TOPOLOGY_VARIANTS:
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "requires topology_nodes and topology_edges"):
                _actor(variant)(local)

    def test_graph_uses_single_shared_encoders_and_shared_message_update_across_rounds(self) -> None:
        _, observation, nodes, edges, _ = self._environment_features(4)
        actor = _actor("graph")
        self.assertEqual(actor.effective_input_dim, 58)
        self.assertFalse(any("round" in name for name, _ in actor.named_modules()))
        message_calls: list[int] = []; update_calls: list[int] = []
        message_hook = actor.message_mlp.register_forward_hook(lambda *_: message_calls.append(1))
        update_hook = actor.update_mlp.register_forward_hook(lambda *_: update_calls.append(1))
        try:
            embedding = actor.graph_embeddings(torch.as_tensor(observation).unsqueeze(0),
                                               torch.as_tensor(nodes).unsqueeze(0), torch.as_tensor(edges).unsqueeze(0))
        finally:
            message_hook.remove(); update_hook.remove()
        self.assertEqual(tuple(embedding.shape), (1, 6, 32))
        self.assertEqual(len(message_calls), 8)  # two directions times K=4 shared rounds.
        self.assertEqual(len(update_calls), 4)

    def test_graph_actor_uses_each_relay_own_embedding(self) -> None:
        actor = _actor("graph")
        local = torch.zeros(1, 4, 26)
        nodes, edges = torch.zeros(1, 6, 6), torch.zeros(1, 5, 7)
        controlled = torch.arange(6 * 32, dtype=torch.float32).reshape(1, 6, 32)
        with patch.object(actor, "graph_embeddings", return_value=controlled):
            effective = actor.effective_actor_input(local, nodes, edges)
        torch.testing.assert_close(effective[..., :26], local)
        torch.testing.assert_close(effective[0, 0, 26:], controlled[0, 1])
        torch.testing.assert_close(effective[0, 1, 26:], controlled[0, 2])
        torch.testing.assert_close(effective[0, 2, 26:], controlled[0, 3])
        torch.testing.assert_close(effective[0, 3, 26:], controlled[0, 4])

    def test_graph_remote_endpoint_influences_relay_after_k_shared_rounds(self) -> None:
        actor = _actor("graph")
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
            actor.node_encoder[0].weight[0, 0] = 1.0
            actor.message_mlp[0].weight[0, 0] = 1.0  # neighbour embedding coordinate zero
            actor.update_mlp[0].weight[0, 32] = 1.0  # aggregated-message coordinate zero
        local, edges = torch.zeros(1, 4, 26), torch.zeros(1, 5, 7)
        without_remote = torch.zeros(1, 6, 6)
        with_remote = without_remote.clone(); with_remote[0, 0, 0] = 0.5  # H is four hops from R4.
        embedding_without = actor.graph_embeddings(local, without_remote, edges)
        embedding_with = actor.graph_embeddings(local, with_remote, edges)
        self.assertNotEqual(float(embedding_without[0, 4, 0].detach()),
                            float(embedding_with[0, 4, 0].detach()))

    def test_p3_p4_action_shapes_and_single_shared_log_std(self) -> None:
        _, observation, nodes, edges, _ = self._environment_features(4)
        local = torch.as_tensor(observation).unsqueeze(0).repeat(2, 1, 1)
        node_tensor, edge_tensor = torch.as_tensor(nodes).unsqueeze(0).repeat(2, 1, 1), torch.as_tensor(edges).unsqueeze(0).repeat(2, 1, 1)
        for variant in TOPOLOGY_VARIANTS:
            actor = _actor(variant)
            mean, log_std = actor(local, node_tensor, edge_tensor)
            self.assertEqual(tuple(mean.shape), (2, 4, 3)); self.assertEqual(tuple(log_std.shape), (2, 4, 3))
            torch.testing.assert_close(log_std, log_std[:, :1].expand_as(log_std))
            self.assertEqual(tuple(actor.log_std_parameter.shape), (3,))
            self.assertEqual(len([name for name, _ in actor.named_parameters() if "log_std" in name]), 1)

    def test_topology_rollout_gae_and_ppo_update_reuses_old_latent(self) -> None:
        for variant in TOPOLOGY_VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                trainer = MappoTrainer(_trainer_config(directory, variant))
                buffer, _ = trainer.collect_rollout()
                self.assertIsNotNone(buffer.topology_nodes); self.assertIsNotNone(buffer.topology_edges)
                self.assertEqual(buffer.topology_nodes.shape, (4, 2, 6, 6))  # type: ignore[union-attr]
                self.assertEqual(buffer.topology_edges.shape, (4, 2, 5, 7))  # type: ignore[union-attr]
                actor_before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
                critic_before = [parameter.detach().clone() for parameter in trainer.critic.parameters()]
                trainer.actor.sample_actions = lambda *_: (_ for _ in ()).throw(AssertionError("PPO update resampled latent z"))  # type: ignore[method-assign]
                result = trainer.update(buffer)
                self.assertTrue(result["finite"])
                self.assertTrue(any(not torch.equal(before, after) for before, after in zip(actor_before, trainer.actor.parameters())))
                self.assertTrue(any(not torch.equal(before, after) for before, after in zip(critic_before, trainer.critic.parameters())))

    def test_topology_checkpoint_round_trip_is_deterministic(self) -> None:
        for variant in TOPOLOGY_VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                config = _trainer_config(directory, variant)
                trainer = MappoTrainer(config)
                env, observation, nodes, edges, _ = self._environment_features(4)
                del env
                local, node_tensor, edge_tensor = (torch.as_tensor(observation).unsqueeze(0),
                                                   torch.as_tensor(nodes).unsqueeze(0), torch.as_tensor(edges).unsqueeze(0))
                expected = trainer.actor.deterministic_actions(local, node_tensor, edge_tensor).detach().clone()
                checkpoint = trainer.save_latest()
                restored = MappoTrainer(config); restored.load(checkpoint)
                self.assertEqual(restored.config.actor_variant, variant)
                torch.testing.assert_close(restored.actor.deterministic_actions(local, node_tensor, edge_tensor), expected)

    def test_p0_p1_p2_checkpoint_loading_remains_compatible(self) -> None:
        checkpoints = (
            Path("artifacts/stage3-stateindependent-full/checkpoints/actor_final.pt"),
            Path("artifacts/stage4-role-info-full/checkpoints/actor_final.pt"),
            Path("artifacts/stage4-role-head-full/checkpoints/actor_final.pt"),
        )
        for checkpoint in checkpoints:
            self.assertTrue(checkpoint.exists(), str(checkpoint))
            payload = load_checkpoint(checkpoint, "cpu")
            config = MappoConfig.from_dict(payload["config"])
            actor = SharedActor(config.local_obs_dim, config.action_dim, config.actor_hidden_sizes,
                                config.log_std_min, config.log_std_max, config.actor_log_std_mode,
                                config.state_independent_log_std_init, config.actor_variant,
                                config.topology_node_dim, config.topology_edge_dim, config.graph_hidden_dim)
            actor.load_state_dict(payload["actor_state"])
            self.assertFalse(actor.uses_topology)

    def test_p3_p4_parameter_counts_are_within_ten_percent(self) -> None:
        p3, p4 = _actor("topology_info"), _actor("graph")
        p3_count, p4_count = sum(parameter.numel() for parameter in p3.parameters()), sum(parameter.numel() for parameter in p4.parameters())
        self.assertEqual((p3_count, p4_count), (29_446, 29_094))
        self.assertLessEqual(abs(p3_count - p4_count) / max(p3_count, p4_count), 0.10)


if __name__ == "__main__":
    unittest.main()
