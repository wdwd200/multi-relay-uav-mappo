"""Explicit centralized-critic state contract for RelayEnv."""

from __future__ import annotations

from typing import Any

import numpy as np


def global_state_dim(num_relays: int) -> int:
    """Return 23 + 6*K for the revised read-only global-state contract."""
    if num_relays < 1:
        raise ValueError("num_relays must be positive")
    return 23 + 6 * int(num_relays)


def global_state_feature_names(num_relays: int) -> tuple[str, ...]:
    """Names in the exact stable flatten order; never depend on dict ordering."""
    if num_relays < 1:
        raise ValueError("num_relays must be positive")
    names: list[str] = []
    for mobile in ("H", "L"):
        names += [f"{mobile}.position.{axis}" for axis in "xyz"]
        names += [f"{mobile}.velocity.{axis}" for axis in "xyz"]
        names += [f"{mobile}.waypoint.{axis}" for axis in "xyz"]
        names.append(f"{mobile}.cruise_speed")
    for index in range(1, num_relays + 1):
        names += [f"R{index}.position.{axis}" for axis in "xyz"]
        names += [f"R{index}.velocity.{axis}" for axis in "xyz"]
    names += ["step", "sim_time", "consecutive_outage_steps"]
    return tuple(names)


def flatten_global_state(state: dict[str, Any], num_relays: int) -> np.ndarray:
    """Flatten a `RelayEnv.get_global_state()` mapping in its documented order.

    The Actor never calls this function.  It is exclusively the centralized
    Critic's training-side representation and includes no synthetic features.
    """
    relays = state.get("relays")
    if not isinstance(relays, (tuple, list)) or len(relays) != num_relays:
        raise ValueError(f"expected exactly {num_relays} relay states")
    values: list[float] = []
    try:
        for mobile_name in ("H", "L"):
            mobile = state[mobile_name]
            for field in ("position", "velocity", "waypoint"):
                vector = mobile[field]
                if len(vector) != 3:
                    raise ValueError(f"{mobile_name}.{field} must have exactly three entries")
                values.extend(float(value) for value in vector)
            values.append(float(mobile["cruise_speed"]))
        for relay in relays:
            for field in ("position", "velocity"):
                vector = relay[field]
                if len(vector) != 3:
                    raise ValueError(f"relay.{field} must have exactly three entries")
                values.extend(float(value) for value in vector)
        values.extend((float(state["step"]), float(state["sim_time"]), float(state["consecutive_outage_steps"])))
    except KeyError as error:
        raise ValueError(f"missing revised global-state field: {error.args[0]}") from error
    result = np.asarray(values, dtype=np.float32)
    expected = global_state_dim(num_relays)
    if result.shape != (expected,):
        raise AssertionError(f"expected {expected} global-state dimensions, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("global state contains NaN or Inf")
    return result
