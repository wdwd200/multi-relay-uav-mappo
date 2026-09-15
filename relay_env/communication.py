"""Pure communication functions, intentionally independent of ``RelayEnv``.

The antenna model is deliberately kept here so the environment, unit tests and
calibration scripts share one numerical implementation.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean 3-D distance in metres."""
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def horizontal_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Horizontal (x-y plane) distance in metres."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def elevation_angle_rad(a: Sequence[float], b: Sequence[float]) -> float:
    """Unsigned link elevation relative to the horizontal plane in radians."""
    return math.atan2(abs(float(a[2]) - float(b[2])), horizontal_distance(a, b))


def antenna_gain_linear(elevation_rad: float) -> float:
    """Normalized two-terminal vertical-dipole link gain, ``cos(phi)^2``.

    ``cos(pi / 2)`` is a tiny positive finite floating-point number, which is
    useful at an exactly vertical link: it represents the physical null while
    retaining finite dB, SNR and capacity diagnostic values.
    """
    return math.cos(float(elevation_rad)) ** 2


def antenna_gain_db(gain_linear: float) -> float:
    """Finite diagnostic dB form of normalized antenna gain.

    The floor is only for reporting an exact mathematical null; it does not
    enter the channel-gain calculation.
    """
    return 10.0 * math.log10(max(0.0, float(gain_linear), 1.0e-300))


def channel_gain(distance_m: float, reference_gain: float, path_loss_exponent: float) -> float:
    """Distance-only path gain; zero distance is evaluated at one metre."""
    return reference_gain * max(float(distance_m), 1.0) ** (-path_loss_exponent)


def link_metrics(
    a: Sequence[float],
    b: Sequence[float],
    reference_gain: float,
    path_loss_exponent: float,
    tx_power_w: float,
    noise_density_w_hz: float,
    bandwidth_hz: float,
) -> dict[str, float]:
    """Return the elevation-aware link budget and its inspectable components.

    ``channel_gain`` remains the frozen distance-only model.  The new channel
    is exactly ``old_channel_gain * cos(phi)**2`` with ``phi`` evaluated using
    ``atan2(abs(dz), d_xy)``.
    """
    d_xy = horizontal_distance(a, b)
    dz = float(a[2]) - float(b[2])
    d_3d = distance(a, b)
    phi = elevation_angle_rad(a, b)
    ant_gain = antenna_gain_linear(phi)
    old_gain = channel_gain(d_3d, reference_gain, path_loss_exponent)
    new_gain = old_gain * ant_gain
    old_snr = snr(old_gain, tx_power_w, noise_density_w_hz, bandwidth_hz)
    new_snr = snr(new_gain, tx_power_w, noise_density_w_hz, bandwidth_hz)
    return {
        "horizontal_distance": d_xy,
        "height_difference": dz,
        "distance_3d": d_3d,
        "elevation_angle_rad": phi,
        "elevation_angle_deg": math.degrees(phi),
        "antenna_gain_linear": ant_gain,
        "antenna_gain_db": antenna_gain_db(ant_gain),
        "old_channel_gain": old_gain,
        "channel_gain": new_gain,
        "old_snr_linear": old_snr,
        "old_snr_db": antenna_gain_db(old_snr),
        "snr_linear": new_snr,
        "snr_db": antenna_gain_db(new_snr),
        "capacity_bps": capacity(bandwidth_hz, new_snr),
    }


def snr(gain: float, tx_power_w: float, noise_density_w_hz: float, bandwidth_hz: float) -> float:
    """Linear SNR."""
    noise_w = noise_density_w_hz * bandwidth_hz
    if noise_w <= 0.0:
        raise ValueError("noise power must be positive")
    return max(0.0, tx_power_w * gain / noise_w)


def capacity(bandwidth_hz: float, linear_snr: float) -> float:
    """Shannon capacity in bit/s."""
    return bandwidth_hz * math.log2(1.0 + max(0.0, linear_snr))


def tdma_allocation(capacities_bps: Iterable[float]) -> tuple[float, ...]:
    """Max-min serial-chain TDMA allocation: tau_i is proportional to 1/C_i."""
    values = tuple(float(x) for x in capacities_bps)
    if not values or any(x <= 0.0 or not math.isfinite(x) for x in values):
        raise ValueError("TDMA requires one or more positive finite capacities")
    inverse_sum = sum(1.0 / x for x in values)
    return tuple((1.0 / x) / inverse_sum for x in values)


def e2e_rate(capacities_bps: Iterable[float]) -> float:
    """Effective serial DF + TDMA rate (harmonic aggregate)."""
    values = tuple(float(x) for x in capacities_bps)
    if not values or any(x <= 0.0 or not math.isfinite(x) for x in values):
        return 0.0
    return 1.0 / sum(1.0 / x for x in values)
