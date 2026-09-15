"""Read-only physical metrics and safety-first model selection for evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from relay_env import EnvironmentConfig


class EpisodeMetrics:
    """Accumulate only diagnostics already exposed by RelayEnv state/info."""

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        self.steps = 0
        self.return_total = 0.0
        self.rate_total_mbps = 0.0
        self.rate_satisfied_steps = 0
        self.outage_steps = 0
        self.first_outage_step: int | None = None
        self.min_link_margin_db = math.inf
        self.min_separation_m = math.inf
        self.max_xy_speed_mps = 0.0
        self.max_abs_z_speed_mps = 0.0
        self.max_xy_accel_mps2 = 0.0
        self.max_abs_z_accel_mps2 = 0.0
        self.speed_accel_violations = 0
        self.accel_sum = 0.0
        self.accel_count = 0
        self.action_saturation_count = 0
        self.action_count = 0
        self.movement_distance_m = 0.0

    def add_step(self, *, reward: float, info: dict[str, Any], previous_state: dict[str, Any],
                 state: dict[str, Any], action_u: np.ndarray) -> None:
        c = self.config
        self.steps += 1
        self.return_total += float(reward)
        rate_mbps = float(info["effective_e2e_rate"]) / 1e6
        self.rate_total_mbps += rate_mbps
        self.rate_satisfied_steps += int(rate_mbps >= c.reward.reference_rate_bps / 1e6)
        is_outage = bool(info["outage"])
        self.outage_steps += int(is_outage)
        if is_outage and self.first_outage_step is None:
            self.first_outage_step = self.steps
        self.min_separation_m = min(self.min_separation_m, float(info["min_separation"]))
        snr_margins = [10.0 * math.log10(max(float(value), 1e-300)) - 5.0 for value in info["link_snrs"].values()]
        self.min_link_margin_db = min(self.min_link_margin_db, *snr_margins)
        previous_relays, relays = previous_state["relays"], state["relays"]
        for before, after in zip(previous_relays, relays):
            velocity = after["velocity"]
            xy_speed, abs_z_speed = math.hypot(velocity[0], velocity[1]), abs(velocity[2])
            self.max_xy_speed_mps, self.max_abs_z_speed_mps = max(self.max_xy_speed_mps, xy_speed), max(self.max_abs_z_speed_mps, abs_z_speed)
            acceleration = tuple((after["velocity"][axis] - before["velocity"][axis]) / c.dt_s for axis in range(3))
            xy_accel, abs_z_accel = math.hypot(acceleration[0], acceleration[1]), abs(acceleration[2])
            self.max_xy_accel_mps2, self.max_abs_z_accel_mps2 = max(self.max_xy_accel_mps2, xy_accel), max(self.max_abs_z_accel_mps2, abs_z_accel)
            self.speed_accel_violations += int(xy_speed > c.relay_max_xy_speed_mps + 1e-7 or not (c.relay_min_z_speed_mps - 1e-7 <= velocity[2] <= c.relay_max_z_speed_mps + 1e-7))
            self.speed_accel_violations += int(xy_accel > c.relay_max_xy_accel_mps2 + 1e-7 or abs_z_accel > c.relay_max_z_accel_mps2 + 1e-7)
            self.accel_sum += math.sqrt(sum(value * value for value in acceleration))
            self.accel_count += 1
            self.movement_distance_m += math.sqrt(sum((after["position"][axis] - before["position"][axis]) ** 2 for axis in range(3)))
        self.action_saturation_count += int(np.count_nonzero(np.abs(action_u) > 0.95))
        self.action_count += int(action_u.size)

    def as_dict(self, *, terminated: bool, truncated: bool, reason: str | None) -> dict[str, Any]:
        return {"return": self.return_total, "length": self.steps, "terminated": bool(terminated), "truncated": bool(truncated),
                "termination_reason": reason or "", "mean_e2e_rate_mbps": self.rate_total_mbps / max(1, self.steps),
                "rate_satisfaction_ratio": self.rate_satisfied_steps / max(1, self.steps),
                "outage_step_ratio": self.outage_steps / max(1, self.steps), "min_link_margin_db": self.min_link_margin_db,
                "min_separation_m": self.min_separation_m, "max_xy_speed_mps": self.max_xy_speed_mps,
                "max_abs_z_speed_mps": self.max_abs_z_speed_mps, "max_xy_accel_mps2": self.max_xy_accel_mps2,
                "max_abs_z_accel_mps2": self.max_abs_z_accel_mps2, "speed_accel_violations": self.speed_accel_violations,
                "mean_acceleration_mps2": self.accel_sum / max(1, self.accel_count),
                "action_saturation_ratio": self.action_saturation_count / max(1, self.action_count),
                "movement_distance_m": self.movement_distance_m,
                # These are read-only diagnostics used for the Stage-4.1
                # horizon analysis; no termination/reward logic changes.
                "first_outage_step": self.first_outage_step,
                "terminal_step": self.steps}


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate required evaluation metrics without changing the environment."""
    if not episodes:
        raise ValueError("at least one episode is required")
    total_steps = sum(int(episode["length"]) for episode in episodes)
    collision_count = sum(episode["termination_reason"] == "safety_violation" for episode in episodes)
    boundary_count = sum(episode["termination_reason"] == "boundary_violation" for episode in episodes)
    persistent_outage_count = sum(episode["termination_reason"] == "persistent_outage" for episode in episodes)
    numeric_means = ("return", "mean_e2e_rate_mbps", "rate_satisfaction_ratio", "outage_step_ratio", "mean_acceleration_mps2", "action_saturation_ratio", "movement_distance_m")
    summary: dict[str, Any] = {f"mean_{name}" if name == "return" else name: float(np.mean([float(episode[name]) for episode in episodes])) for name in numeric_means}
    summary.update({"episodes": len(episodes), "total_steps": total_steps,
                    "normal_completion_count": sum(bool(episode["truncated"]) for episode in episodes),
                    "collision_count": int(collision_count), "boundary_count": int(boundary_count),
                    "persistent_outage_count": int(persistent_outage_count),
                    "speed_accel_violations": int(sum(int(episode["speed_accel_violations"]) for episode in episodes)),
                    "min_link_margin_db": float(min(float(episode["min_link_margin_db"]) for episode in episodes)),
                    "min_separation_m": float(min(float(episode["min_separation_m"]) for episode in episodes)),
                    "max_xy_speed_mps": float(max(float(episode["max_xy_speed_mps"]) for episode in episodes)),
                    "max_abs_z_speed_mps": float(max(float(episode["max_abs_z_speed_mps"]) for episode in episodes)),
                    "max_xy_accel_mps2": float(max(float(episode["max_xy_accel_mps2"]) for episode in episodes)),
                    "max_abs_z_accel_mps2": float(max(float(episode["max_abs_z_accel_mps2"]) for episode in episodes))})
    return summary


def safety_priority_key(summary: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    """Lower is better under the Stage 3 completion-first safety ordering.

    Hard speed/acceleration violations always dominate.  Among physically
    compliant policies, completing an episode is more important than avoiding
    a comparatively rare collision/boundary termination; persistent outage is
    then ranked before communication-quality tie breakers.
    """
    return (float(summary["speed_accel_violations"]), -float(summary["normal_completion_count"]),
            float(summary["collision_count"] + summary["boundary_count"]), float(summary["persistent_outage_count"]),
            float(summary["outage_step_ratio"]), -float(summary["rate_satisfaction_ratio"]),
            -float(summary["mean_e2e_rate_mbps"]))
