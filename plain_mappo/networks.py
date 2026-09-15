"""Shared local Actor and separate centralized Critic for Plain MAPPO."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from .roles import DESTINATION_ROLE, MIDDLE_ROLE, NUM_ROLES, SOURCE_ROLE, role_ids_tensor, role_one_hot
from .topology import GRAPH_HIDDEN_DIM, TOPOLOGY_EDGE_DIM, TOPOLOGY_NODE_DIM


def _tanh_mlp(input_dim: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_sizes:
        layers.extend((nn.Linear(previous, hidden), nn.Tanh()))
        previous = hidden
    return nn.Sequential(*layers)


class SharedActor(nn.Module):
    """One parameter-shared local-observation Gaussian policy for all relays.

    ``state_dependent_clamp`` retains the historical Stage-3 Actor so old
    A--D checkpoints remain deployable.  ``state_independent_tanh`` has one
    three-dimensional learned uncertainty vector shared by every observation
    and relay; a smooth map keeps its effective log standard deviation inside
    the configured finite interval.  P3/P4 receive optional action-time
    topology tensors, never Critic state or modified environment observations.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (128, 128),
                 log_std_min: float = -5.0, log_std_max: float = 2.0,
                 log_std_mode: str = "state_dependent_clamp", state_independent_log_std_init: float = -1.5,
                 actor_variant: str = "plain", topology_node_dim: int = TOPOLOGY_NODE_DIM,
                 topology_edge_dim: int = TOPOLOGY_EDGE_DIM, graph_hidden_dim: int = GRAPH_HIDDEN_DIM) -> None:
        super().__init__()
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
        self.log_std_mode = str(log_std_mode)
        self.actor_variant = str(actor_variant)
        self.topology_node_dim = int(topology_node_dim)
        self.topology_edge_dim = int(topology_edge_dim)
        self.graph_hidden_dim = int(graph_hidden_dim)
        if self.log_std_mode not in {"state_dependent_clamp", "state_independent_tanh"}:
            raise ValueError("unsupported log_std_mode")
        if self.actor_variant not in {"plain", "role_info", "role_head", "topology_info", "graph"}:
            raise ValueError("actor_variant must be plain, role_info, role_head, topology_info, or graph")
        if self.actor_variant != "plain" and self.log_std_mode != "state_independent_tanh":
            raise ValueError("non-plain Actor variants require state_independent_tanh log_std")
        if (self.topology_node_dim, self.topology_edge_dim, self.graph_hidden_dim) != (6, 7, 32):
            raise ValueError("topology/graph dimensions are frozen to node=6, edge=7, hidden=32")
        # The plain branch deliberately retains its historical module names
        # and layer shapes so existing Stage-3 checkpoints load unchanged.
        if self.actor_variant == "role_info":
            self.effective_input_dim = self.obs_dim + NUM_ROLES
        elif self.actor_variant == "topology_info":
            self.effective_input_dim = self.obs_dim + 6 * self.topology_node_dim + 5 * self.topology_edge_dim
        elif self.actor_variant == "graph":
            self.effective_input_dim = self.obs_dim + self.graph_hidden_dim
        else:
            self.effective_input_dim = self.obs_dim
        if self.actor_variant == "graph":
            # One shared encoder/edge encoder/message/update set, reused for
            # every node, hop direction, and the K message-passing rounds.
            self.node_encoder = _tanh_mlp(self.topology_node_dim, (self.graph_hidden_dim,))
            self.edge_encoder = _tanh_mlp(self.topology_edge_dim, (self.graph_hidden_dim,))
            self.message_mlp = _tanh_mlp(2 * self.graph_hidden_dim, (self.graph_hidden_dim,))
            self.update_mlp = _tanh_mlp(2 * self.graph_hidden_dim, (self.graph_hidden_dim,))
        self.backbone = _tanh_mlp(self.effective_input_dim, hidden_sizes)
        final_dim = hidden_sizes[-1] if hidden_sizes else self.effective_input_dim
        if self.actor_variant == "role_head":
            # P2 has one shared backbone and exactly three small mean heads;
            # uncertainty remains the single globally shared vector below.
            self.source_mean_head = nn.Linear(final_dim, self.action_dim)
            self.middle_mean_head = nn.Linear(final_dim, self.action_dim)
            self.destination_mean_head = nn.Linear(final_dim, self.action_dim)
        else:
            self.mean_head = nn.Linear(final_dim, self.action_dim)
        if self.log_std_mode == "state_dependent_clamp":
            self.log_std_head = nn.Linear(final_dim, self.action_dim)
        else:
            if not self.log_std_min < state_independent_log_std_init < self.log_std_max:
                raise ValueError("state_independent_log_std_init must lie strictly inside log_std bounds")
            midpoint = 0.5 * (self.log_std_min + self.log_std_max)
            half_range = 0.5 * (self.log_std_max - self.log_std_min)
            unconstrained_init = math.atanh((float(state_independent_log_std_init) - midpoint) / half_range)
            self.log_std_parameter = nn.Parameter(torch.full((self.action_dim,), unconstrained_init))

    @property
    def uses_topology(self) -> bool:
        """Whether this Actor requires action-time P3/P4 topology tensors."""
        return self.actor_variant in {"topology_info", "graph"}

    def _roles_for(self, local_obs: Tensor) -> Tensor:
        """Recover static relay roles from the explicit K axis, never from data."""
        if local_obs.ndim < 3:
            raise ValueError("Role Actor variants require local_obs shaped [..., K, obs_dim]")
        return role_ids_tensor(int(local_obs.shape[-2]), device=local_obs.device)

    def _topology_tensors(self, local_obs: Tensor, topology_nodes: Tensor | None,
                          topology_edges: Tensor | None) -> tuple[Tensor, Tensor]:
        if topology_nodes is None or topology_edges is None:
            raise ValueError(f"{self.actor_variant} Actor requires topology_nodes and topology_edges")
        relays = int(local_obs.shape[-2])
        expected_nodes = tuple(local_obs.shape[:-2]) + (relays + 2, self.topology_node_dim)
        expected_edges = tuple(local_obs.shape[:-2]) + (relays + 1, self.topology_edge_dim)
        if tuple(topology_nodes.shape) != expected_nodes:
            raise ValueError(f"expected topology_nodes shape {expected_nodes}, got {tuple(topology_nodes.shape)}")
        if tuple(topology_edges.shape) != expected_edges:
            raise ValueError(f"expected topology_edges shape {expected_edges}, got {tuple(topology_edges.shape)}")
        return topology_nodes.to(device=local_obs.device, dtype=local_obs.dtype), topology_edges.to(device=local_obs.device, dtype=local_obs.dtype)

    def graph_embeddings(self, local_obs: Tensor, topology_nodes: Tensor | None = None,
                         topology_edges: Tensor | None = None) -> Tensor:
        """Run exactly K shared mean-aggregation message-passing rounds."""
        if self.actor_variant != "graph":
            raise ValueError("graph_embeddings is only available for actor_variant=graph")
        nodes, edges = self._topology_tensors(local_obs, topology_nodes, topology_edges)
        node_embedding = self.node_encoder(nodes)
        # The reverse graph direction changes only relative position/velocity
        # signs.  Capacity is an undirected physical property of this hop.
        reverse_edges = torch.cat((-edges[..., :6], edges[..., 6:]), dim=-1)
        forward_edge_embedding = self.edge_encoder(edges)
        reverse_edge_embedding = self.edge_encoder(reverse_edges)
        relays = int(local_obs.shape[-2])
        for _ in range(relays):
            forward_message = self.message_mlp(torch.cat((node_embedding[..., :-1, :], forward_edge_embedding), dim=-1))
            reverse_message = self.message_mlp(torch.cat((node_embedding[..., 1:, :], reverse_edge_embedding), dim=-1))
            zero = torch.zeros_like(forward_message[..., :1, :])
            # Forward messages are received by downstream nodes; reverse
            # messages by upstream nodes.  Chain endpoints have degree one.
            aggregate = torch.cat((zero, forward_message), dim=-2) + torch.cat((reverse_message, zero), dim=-2)
            degree = torch.ones((relays + 2,), device=node_embedding.device, dtype=node_embedding.dtype)
            degree[1:-1] = 2.0
            view_shape = (1,) * (node_embedding.ndim - 2) + (relays + 2, 1)
            node_embedding = self.update_mlp(torch.cat((node_embedding, aggregate / degree.view(view_shape)), dim=-1))
        return node_embedding

    def effective_actor_input(self, local_obs: Tensor, topology_nodes: Tensor | None = None,
                              topology_edges: Tensor | None = None) -> Tensor:
        """Return the tensor entering the Actor backbone.

        This exposes P1's 26+3 construction for testing while leaving the
        environment-facing observation and rollout buffer unchanged at 26.
        """
        if local_obs.shape[-1] != self.obs_dim:
            raise ValueError(f"expected local observation dim {self.obs_dim}, got {local_obs.shape[-1]}")
        if self.actor_variant == "role_info":
            one_hot = role_one_hot(int(local_obs.shape[-2]), device=local_obs.device, dtype=local_obs.dtype)
            view_shape = (1,) * (local_obs.ndim - 2) + one_hot.shape
            return torch.cat((local_obs, one_hot.view(view_shape).expand(*local_obs.shape[:-2], *one_hot.shape)), dim=-1)
        if self.actor_variant == "topology_info":
            nodes, edges = self._topology_tensors(local_obs, topology_nodes, topology_edges)
            topology_block = torch.cat((nodes.reshape(*nodes.shape[:-2], -1), edges.reshape(*edges.shape[:-2], -1)), dim=-1)
            repeated_block = topology_block.unsqueeze(-2).expand(*local_obs.shape[:-1], topology_block.shape[-1])
            return torch.cat((local_obs, repeated_block), dim=-1)
        if self.actor_variant == "graph":
            embedding = self.graph_embeddings(local_obs, topology_nodes, topology_edges)
            return torch.cat((local_obs, embedding[..., 1:-1, :]), dim=-1)
        if topology_nodes is not None or topology_edges is not None:
            raise ValueError(f"{self.actor_variant} Actor does not consume topology tensors")
        if self.actor_variant != "role_info":
            return local_obs

    def _mean_from_hidden(self, hidden: Tensor, local_obs: Tensor) -> Tensor:
        if self.actor_variant != "role_head":
            return self.mean_head(hidden)
        roles = self._roles_for(local_obs)
        source = self.source_mean_head(hidden)
        middle = self.middle_mean_head(hidden)
        destination = self.destination_mean_head(hidden)
        role_shape = (1,) * (hidden.ndim - 2) + (roles.shape[0], 1)
        role_ids = roles.view(role_shape)
        return torch.where(role_ids == SOURCE_ROLE, source,
                           torch.where(role_ids == MIDDLE_ROLE, middle, destination))

    def raw_parameters(self, local_obs: Tensor, topology_nodes: Tensor | None = None,
                       topology_edges: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return mean and the pre-bound log-std coordinate for diagnostics."""
        if local_obs.shape[-1] != self.obs_dim:
            raise ValueError(f"expected local observation dim {self.obs_dim}, got {local_obs.shape[-1]}")
        effective_input = self.effective_actor_input(local_obs, topology_nodes, topology_edges)
        hidden = self.backbone(effective_input)
        mean = self._mean_from_hidden(hidden, local_obs)
        if self.log_std_mode == "state_dependent_clamp":
            return mean, self.log_std_head(hidden)
        view_shape = (1,) * (mean.ndim - 1) + (self.action_dim,)
        return mean, self.log_std_parameter.view(view_shape).expand_as(mean)

    def effective_log_std(self, raw_log_std: Tensor) -> Tensor:
        """Map a pre-bound coordinate to the log standard deviation used by PPO."""
        if self.log_std_mode == "state_dependent_clamp":
            return raw_log_std.clamp(self.log_std_min, self.log_std_max)
        midpoint = 0.5 * (self.log_std_min + self.log_std_max)
        half_range = 0.5 * (self.log_std_max - self.log_std_min)
        return midpoint + half_range * torch.tanh(raw_log_std)

    def distribution_parameters(self, local_obs: Tensor, topology_nodes: Tensor | None = None,
                                topology_edges: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Return mean, pre-bound coordinate, and bounded effective log_std."""
        mean, raw_log_std = self.raw_parameters(local_obs, topology_nodes, topology_edges)
        return mean, raw_log_std, self.effective_log_std(raw_log_std)

    def forward(self, local_obs: Tensor, topology_nodes: Tensor | None = None,
                topology_edges: Tensor | None = None) -> tuple[Tensor, Tensor]:
        mean, _, log_std = self.distribution_parameters(local_obs, topology_nodes, topology_edges)
        return mean, log_std

    def distribution(self, local_obs: Tensor, topology_nodes: Tensor | None = None,
                     topology_edges: Tensor | None = None) -> Normal:
        mean, log_std = self(local_obs, topology_nodes, topology_edges)
        return Normal(mean, log_std.exp())

    def sample_actions(self, local_obs: Tensor, topology_nodes: Tensor | None = None,
                       topology_edges: Tensor | None = None, *, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return latent z, tanh(z), summed log-probability, and entropy."""
        distribution = self.distribution(local_obs, topology_nodes, topology_edges)
        # A caller may supply the Stage-4.1 dedicated action-noise generator.
        # The formula is exactly a Normal sample and therefore leaves PPO's
        # latent-z/log-probability contract unchanged.
        if generator is None:
            latent_z = distribution.sample()
        else:
            noise = torch.randn(distribution.loc.shape, dtype=distribution.loc.dtype,
                                device=distribution.loc.device, generator=generator)
            latent_z = distribution.loc + distribution.scale * noise
        action_u = torch.tanh(latent_z)
        return latent_z, action_u, distribution.log_prob(latent_z).sum(dim=-1), distribution.entropy().sum(dim=-1)

    def deterministic_actions(self, local_obs: Tensor, topology_nodes: Tensor | None = None,
                              topology_edges: Tensor | None = None) -> Tensor:
        mean, _ = self(local_obs, topology_nodes, topology_edges)
        return torch.tanh(mean)

    def log_prob_entropy(self, local_obs: Tensor, latent_z: Tensor, topology_nodes: Tensor | None = None,
                         topology_edges: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # PPO compares this same latent Gaussian z under old/new policies.  The
        # tanh Jacobian therefore cancels in the ratio; entropy is deliberately
        # the latent-Gaussian entropy stipulated by the Stage 3 contract.
        distribution = self.distribution(local_obs, topology_nodes, topology_edges)
        return distribution.log_prob(latent_z).sum(dim=-1), distribution.entropy().sum(dim=-1)


class CentralizedCritic(nn.Module):
    """Team-scalar Critic fed only the explicit global-state representation."""

    def __init__(self, state_dim: int, hidden_sizes: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.backbone = _tanh_mlp(self.state_dim, hidden_sizes)
        final_dim = hidden_sizes[-1] if hidden_sizes else self.state_dim
        self.value_head = nn.Linear(final_dim, 1)

    def forward(self, global_state: Tensor) -> Tensor:
        if global_state.shape[-1] != self.state_dim:
            raise ValueError(f"expected global-state dim {self.state_dim}, got {global_state.shape[-1]}")
        return self.value_head(self.backbone(global_state)).squeeze(dim=-1)
