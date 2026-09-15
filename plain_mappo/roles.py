"""Static relay-role helpers for the Stage 4 Role ablations.

Roles are properties of a relay's position in the ordered communication
chain, not environment observations.  Keeping this mapping in one module
avoids silently duplicating a K-specific convention across rollout, training,
and evaluation code.
"""

from __future__ import annotations

import torch
from torch import Tensor


SOURCE_ROLE = 0
MIDDLE_ROLE = 1
DESTINATION_ROLE = 2
NUM_ROLES = 3


def role_ids_for_num_relays(num_relays: int) -> tuple[int, ...]:
    """Return source/middle/destination role IDs for an ordered relay chain.

    The Stage 4 experiments use K=3--5 for Actor shape tests and K=4 for
    training.  A chain needs at least a source-side and destination-side relay
    for this three-role convention to be meaningful.
    """
    if num_relays < 2:
        raise ValueError("role mapping requires at least two relays")
    return (SOURCE_ROLE,) + (MIDDLE_ROLE,) * (num_relays - 2) + (DESTINATION_ROLE,)


def role_ids_tensor(num_relays: int, *, device: torch.device | None = None) -> Tensor:
    """Return the static role mapping as a non-trainable long tensor."""
    return torch.tensor(role_ids_for_num_relays(num_relays), dtype=torch.long, device=device)


def role_one_hot(num_relays: int, *, device: torch.device | None = None,
                 dtype: torch.dtype = torch.float32) -> Tensor:
    """Return K-by-3 source/middle/destination one-hot role features."""
    return torch.nn.functional.one_hot(role_ids_tensor(num_relays, device=device), NUM_ROLES).to(dtype=dtype)
