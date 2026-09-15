"""K-configurable, deterministic-on-seed stage-1 UAV relay environment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .communication import distance, e2e_rate, link_metrics, tdma_allocation
from .config import EnvironmentConfig

Vec3 = tuple[float, float, float]


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _norm(a: Vec3) -> float:
    return math.sqrt(sum(x * x for x in a))


def _unit(a: Vec3) -> Vec3:
    n = _norm(a)
    return (0.0, 0.0, 0.0) if n == 0.0 else _scale(a, 1.0 / n)


def _finite_vec(a: Sequence[float]) -> Vec3:
    if len(a) != 3:
        raise ValueError("each action must contain exactly three values")
    result = tuple(float(x) for x in a)
    if not all(math.isfinite(x) for x in result):
        raise ValueError("actions must be finite")
    return result  # type: ignore[return-value]


@dataclass
class _Mobile:
    position: Vec3
    velocity: Vec3
    waypoint: Vec3
    cruise_speed: float


class RelayEnv:
    """A cooperative K-agent relay environment with a Gymnasium-like interface.

    Observations are a tuple of K 26-element tuples.  Rewards are a tuple of K
    identical cooperative rewards.  The environment does not silently clip an
    out-of-bound position; such a state terminates the episode.
    """

    observation_size = 26

    def __init__(self, num_relays: int = 4, config: EnvironmentConfig | None = None, debug: bool = False):
        if num_relays < 1:
            raise ValueError("num_relays must be at least one")
        self.num_relays = int(num_relays)
        self.config = config or EnvironmentConfig()
        self.debug = debug
        self._rng = random.Random()
        self._ready = False
        self.step_count = 0
        self.consecutive_outage_steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[tuple[tuple[float, ...], ...], dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)
        reasons: list[str] = []
        for _attempt in range(1, self.config.reset_max_attempts + 1):
            self._sample_initial_state()
            safety = self._safety_info()
            comm = self._communication_info()
            bounds = self._boundary_agents()
            hl_distance = distance(self.h.position, self.l.position)
            if bounds:
                reasons.append("sampled node outside boundary")
            elif hl_distance > self.config.mobile_max_hl_distance_m:
                reasons.append("initial H/L distance exceeds mobile_max_hl_distance_m")
            elif safety["violation_pairs"]:
                reasons.append("initial safety violation")
            elif comm["outage"]:
                reasons.append("initial logical chain outage")
            else:
                self._ready = True
                self.step_count = 0
                self.consecutive_outage_steps = 0
                info = self._make_info(comm, safety, [], None)
                info["hl_distance_m"] = hl_distance
                info["hl_distance_guard_applied"] = False
                return self.get_obs(), info
        detail = reasons[-1] if reasons else "unknown sampling failure"
        raise RuntimeError(f"reset failed after {self.config.reset_max_attempts} attempts: {detail}")

    def step(self, actions: Iterable[Sequence[float]]) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], bool, bool, dict[str, Any]]:
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        requested_actions = tuple(_finite_vec(a) for a in actions)
        if len(requested_actions) != self.num_relays:
            raise ValueError(f"expected {self.num_relays} actions, got {len(requested_actions)}")

        # All proposed next-state values below are calculated from the same s_t.
        accelerations = tuple(self._map_action(a) for a in requested_actions)
        candidate_velocities = tuple(_add(v, _scale(a, self.config.dt_s)) for v, a in zip(self.relay_velocities, accelerations))
        applied_velocities = tuple(self._limit_relay_velocity(v) for v in candidate_velocities)
        next_relays = tuple(
            _add(p, _scale(_add(v_old, v_new), 0.5 * self.config.dt_s))
            for p, v_old, v_new in zip(self.relay_positions, self.relay_velocities, applied_velocities)
        )
        # Each mobile node receives only the other node's old position.  Neither
        # calculation can observe an already-updated t+1 H/L state or relays.
        preventive_guard = self._requires_hl_distance_guard(self.h, self.l)
        h_guard_accel, l_guard_accel = self._hl_guard_accelerations(self.h, self.l) if preventive_guard else (None, None)
        next_h = self._advance_mobile(self.h, self.l, h_guard_accel)
        next_l = self._advance_mobile(self.l, self.h, l_guard_accel)
        # This invariant must be achieved by the bounded acceleration commands
        # above.  Deliberately do not project positions or overwrite velocities.
        if distance(next_h.position, next_l.position) > self.config.mobile_max_hl_distance_m + 1e-9:
            raise RuntimeError("preventive D_max guard failed to keep the H/L trajectory feasible")

        # Form s_(t+1) atomically before checks and communication.
        self.relay_positions, self.relay_velocities = next_relays, applied_velocities
        self.h, self.l = next_h, next_l
        self.step_count += 1

        safety = self._safety_info()
        boundary_agents = self._boundary_agents()
        comm = self._communication_info()
        if comm["outage"]:
            self.consecutive_outage_steps += 1
        else:
            self.consecutive_outage_steps = 0

        termination_reason: str | None = None
        if safety["violation_pairs"]:
            termination_reason = "safety_violation"
        elif boundary_agents:
            termination_reason = "boundary_violation"
        elif self.consecutive_outage_steps >= self.config.persistent_outage_steps:
            termination_reason = "persistent_outage"
        terminated = termination_reason is not None
        truncated = not terminated and self.step_count >= self.config.max_episode_steps
        warning_penalty = self.config.reward.warning_penalty if safety["warning_pairs"] else 0.0
        failure_penalty = self.config.reward.lambda_failure if terminated else 0.0
        reward = (comm["effective_e2e_rate"] / self.config.reward.reference_rate_bps
                  - self.config.reward.lambda_outage * float(comm["outage"])
                  - warning_penalty - failure_penalty)
        info = self._make_info(comm, safety, boundary_agents, termination_reason)
        info["hl_distance_m"] = distance(self.h.position, self.l.position)
        info["hl_distance_guard_applied"] = preventive_guard
        if self.debug:
            info["action_debug"] = {
                "requested_actions": requested_actions,
                "requested_accelerations": accelerations,
                "candidate_velocities": candidate_velocities,
                "applied_velocities": applied_velocities,
            }
        return self.get_obs(), tuple(reward for _ in range(self.num_relays)), terminated, truncated, info

    def get_obs(self) -> tuple[tuple[float, ...], ...]:
        """Return the 26-dimensional, network-ready normalized observations."""
        return self._get_observations(normalised=True)

    def get_raw_obs(self) -> tuple[tuple[float, ...], ...]:
        """Return the same 26 features in physical units for calibration logs."""
        return self._get_observations(normalised=False)

    def _get_observations(self, normalised: bool) -> tuple[tuple[float, ...], ...]:
        if not self._ready:
            raise RuntimeError("call reset() before get_obs()")
        nodes = [("H", self.h.position, self.h.velocity)] + [(f"R{i + 1}", p, v) for i, (p, v) in enumerate(zip(self.relay_positions, self.relay_velocities))] + [("L", self.l.position, self.l.velocity)]
        comm = self._communication_info()
        observations: list[tuple[float, ...]] = []
        for i, (position, velocity) in enumerate(zip(self.relay_positions, self.relay_velocities)):
            upstream = nodes[i]
            downstream = nodes[i + 2]
            excluded = {f"R{i + 1}", upstream[0], downstream[0]}
            candidates = [node for node in nodes if node[0] not in excluded]
            safe = min(candidates, key=lambda node: distance(position, node[1]))
            up_capacity = comm["capacities"][i]
            down_capacity = comm["capacities"][i + 1]
            raw = (position + velocity + _sub(upstream[1], position) + _sub(upstream[2], velocity) + (up_capacity,)
                   + _sub(downstream[1], position) + _sub(downstream[2], velocity) + (down_capacity,)
                   + _sub(safe[1], position) + _sub(safe[2], velocity))
            assert len(raw) == self.observation_size
            observations.append(tuple(self._normalise_observation(raw) if normalised else raw))
        return tuple(observations)

    def get_global_state(self) -> dict[str, Any]:
        """Unnormalised, read-only state for debugging and centralized critics.

        This method intentionally exposes state only; it does not alter the
        environment transition, reward, observation, or termination logic.
        Stage 3 consumes it through an explicit flatten helper rather than
        relying on this mapping's insertion order.
        """
        if not self._ready:
            raise RuntimeError("call reset() before get_global_state()")
        return {"H": {"position": self.h.position, "velocity": self.h.velocity,
                      "waypoint": self.h.waypoint, "cruise_speed": self.h.cruise_speed},
                "L": {"position": self.l.position, "velocity": self.l.velocity,
                      "waypoint": self.l.waypoint, "cruise_speed": self.l.cruise_speed},
                "relays": tuple({"position": p, "velocity": v} for p, v in zip(self.relay_positions, self.relay_velocities)),
                "step": self.step_count, "sim_time": self.step_count * self.config.dt_s,
                "consecutive_outage_steps": self.consecutive_outage_steps}

    def _sample_initial_state(self) -> None:
        c, r = self.config, self._rng
        # Centre-and-angle construction guarantees both H and L are in bounds.
        centre = (r.uniform(c.initial_hl_distance_max_m / 2, c.map_x_m - c.initial_hl_distance_max_m / 2),
                  r.uniform(c.initial_hl_distance_max_m / 2, c.map_y_m - c.initial_hl_distance_max_m / 2),
                  r.uniform(c.min_altitude_m + 60, c.max_altitude_m - 60))
        d = r.uniform(c.initial_hl_distance_min_m, c.initial_hl_distance_max_m)
        angle = r.uniform(0.0, 2.0 * math.pi)
        vertical = r.uniform(-120.0, 120.0)
        horizontal = math.sqrt(max(1.0, d * d - vertical * vertical))
        offset = (0.5 * horizontal * math.cos(angle), 0.5 * horizontal * math.sin(angle), 0.5 * vertical)
        hp, lp = _add(centre, offset), _sub(centre, offset)
        h_speed = r.uniform(c.mobile_min_speed_mps, c.mobile_max_speed_mps)
        l_speed = r.uniform(c.mobile_min_speed_mps, c.mobile_max_speed_mps)
        h_waypoint = self._random_waypoint(reference_position=lp, origin_position=hp)
        l_waypoint = self._random_waypoint(reference_position=hp, origin_position=lp)
        self.h = _Mobile(hp, self._limit_mobile_velocity(self._desired_mobile_velocity(hp, h_waypoint, h_speed)), h_waypoint, h_speed)
        self.l = _Mobile(lp, self._limit_mobile_velocity(self._desired_mobile_velocity(lp, l_waypoint, l_speed)), l_waypoint, l_speed)
        relays: list[Vec3] = []
        for i in range(self.num_relays):
            f = (i + 1) / (self.num_relays + 1)
            baseline = _add(hp, _scale(_sub(lp, hp), f))
            jitter = (r.uniform(-c.relay_initial_jitter_m, c.relay_initial_jitter_m), r.uniform(-c.relay_initial_jitter_m, c.relay_initial_jitter_m), r.uniform(-c.relay_initial_jitter_m, c.relay_initial_jitter_m))
            relays.append(_add(baseline, jitter))
        self.relay_positions = tuple(relays)
        self.relay_velocities = tuple((0.0, 0.0, 0.0) for _ in relays)

    def _random_waypoint(self, reference_position: Vec3 | None = None, origin_position: Vec3 | None = None) -> Vec3:
        """Sample a legal waypoint, optionally bounded by the old peer position.

        The peer constraint enforces the configurable H-L mission scale.  It
        never reads relay positions and therefore cannot make the task easier
        in response to an agent action.
        """
        if origin_position is None:
            raise ValueError("origin_position is required for constrained random waypoint sampling")
        c, r = self.config, self._rng
        m = c.mobile_waypoint_margin_m
        for _ in range(c.mobile_waypoint_max_attempts):
            segment_length = r.uniform(c.mobile_waypoint_min_segment_m, c.mobile_waypoint_max_segment_m)
            azimuth = r.uniform(0.0, 2.0 * math.pi)
            vertical_fraction = r.uniform(-0.35, 0.35)
            horizontal_fraction = math.sqrt(1.0 - vertical_fraction * vertical_fraction)
            displacement = (segment_length * horizontal_fraction * math.cos(azimuth),
                            segment_length * horizontal_fraction * math.sin(azimuth),
                            segment_length * vertical_fraction)
            waypoint = _add(origin_position, displacement)
            in_bounds = (m <= waypoint[0] <= c.map_x_m - m and m <= waypoint[1] <= c.map_y_m - m
                         and c.min_altitude_m + m / 3 <= waypoint[2] <= c.max_altitude_m - m / 3)
            if in_bounds and (reference_position is None or distance(waypoint, reference_position) <= c.mobile_max_hl_distance_m):
                return waypoint
        raise RuntimeError("could not sample a legal H/L waypoint with the configured segment and D_max constraints")

    def _desired_mobile_velocity(self, position: Vec3, waypoint: Vec3, cruise_xy_speed: float) -> Vec3:
        """A desired velocity with frozen-spec horizontal and configured z limits."""
        c = self.config
        delta = _sub(waypoint, position)
        horizontal = (delta[0], delta[1], 0.0)
        desired_xy = _scale(_unit(horizontal), cruise_xy_speed)
        desired_z = max(c.mobile_min_z_speed_mps, min(c.mobile_max_z_speed_mps, delta[2]))
        return (desired_xy[0], desired_xy[1], desired_z)

    def _advance_mobile(self, mobile: _Mobile, peer_old: _Mobile, guard_acceleration: Vec3 | None = None) -> _Mobile:
        c = self.config
        target = _sub(mobile.waypoint, mobile.position)
        if _norm(target) <= c.mobile_waypoint_reach_m:
            # This sampling depends only on its own arriving state and RNG, never relays.
            mobile = _Mobile(mobile.position, mobile.velocity, self._random_waypoint(reference_position=peer_old.position, origin_position=mobile.position), mobile.cruise_speed)
            target = _sub(mobile.waypoint, mobile.position)
        if guard_acceleration is None:
            desired_velocity = self._desired_mobile_velocity(mobile.position, mobile.waypoint, mobile.cruise_speed)
            delta = _sub(desired_velocity, mobile.velocity)
            acceleration = _scale(_unit(delta), min(_norm(delta) / c.dt_s, c.mobile_max_accel_mps2))
        else:
            acceleration = guard_acceleration
        new_velocity = self._limit_mobile_velocity(_add(mobile.velocity, _scale(acceleration, c.dt_s)))
        new_position = _add(mobile.position, _scale(_add(mobile.velocity, new_velocity), 0.5 * c.dt_s))
        return _Mobile(new_position, new_velocity, mobile.waypoint, mobile.cruise_speed)

    def _requires_hl_distance_guard(self, h_old: _Mobile, l_old: _Mobile) -> bool:
        """Return whether bounded inward braking must begin from this s_t."""
        c = self.config
        relative_position = _sub(h_old.position, l_old.position)
        separation = _norm(relative_position)
        if separation == 0.0:
            return False
        direction = _unit(relative_position)
        outward_relative_speed = max(0.0, sum(a * b for a, b in zip(_sub(h_old.velocity, l_old.velocity), direction)))
        relative_braking_accel = 2.0 * c.mobile_max_accel_mps2
        braking_distance = outward_relative_speed ** 2 / (2.0 * relative_braking_accel)
        # The fixed margin absorbs discrete trapezoidal integration and
        # tangential motion; the dynamic term covers current radial momentum.
        guard_distance = max(c.mobile_hl_distance_guard_m, braking_distance + outward_relative_speed * c.dt_s + relative_braking_accel * c.dt_s ** 2)
        return separation >= c.mobile_max_hl_distance_m - guard_distance

    def _hl_guard_accelerations(self, h_old: _Mobile, l_old: _Mobile) -> tuple[Vec3, Vec3]:
        """Pairwise maximum inward acceleration, bounded before integration."""
        direction = _unit(_sub(h_old.position, l_old.position))
        magnitude = self.config.mobile_max_accel_mps2
        return _scale(direction, -magnitude), _scale(direction, magnitude)

    def _limit_mobile_velocity(self, velocity: Vec3) -> Vec3:
        """Final strict H/L speed cap applied after every waypoint/D_max correction."""
        c = self.config
        xy = (velocity[0], velocity[1], 0.0)
        limited_xy = _scale(_unit(xy), min(_norm(xy), c.mobile_max_speed_mps))
        return (limited_xy[0], limited_xy[1], max(c.mobile_min_z_speed_mps, min(c.mobile_max_z_speed_mps, velocity[2])))

    def _map_action(self, action: Vec3) -> Vec3:
        c = self.config
        clipped = tuple(max(-1.0, min(1.0, x)) for x in action)
        xy = (clipped[0], clipped[1], 0.0)
        xy_accel = _scale(_unit(xy), min(_norm(xy), 1.0) * c.relay_max_xy_accel_mps2)
        return (xy_accel[0], xy_accel[1], clipped[2] * c.relay_max_z_accel_mps2)

    def _limit_relay_velocity(self, velocity: Vec3) -> Vec3:
        c = self.config
        xy = (velocity[0], velocity[1], 0.0)
        limited_xy = _scale(_unit(xy), min(_norm(xy), c.relay_max_xy_speed_mps))
        return (limited_xy[0], limited_xy[1], max(c.relay_min_z_speed_mps, min(c.relay_max_z_speed_mps, velocity[2])))

    def _communication_info(self) -> dict[str, Any]:
        nodes = (self.h.position,) + self.relay_positions + (self.l.position,)
        labels = tuple(["H->R1"] + [f"R{i}->R{i + 1}" for i in range(1, self.num_relays)] + [f"R{self.num_relays}->L"])
        diagnostics = tuple(
            link_metrics(a, b, self.config.comm.reference_gain, self.config.comm.path_loss_exponent,
                         self.config.comm.tx_power_w, self.config.comm.noise_density_w_hz,
                         self.config.comm.bandwidth_hz)
            for a, b in zip(nodes[:-1], nodes[1:])
        )
        gains = tuple(item["channel_gain"] for item in diagnostics)
        snrs = tuple(item["snr_linear"] for item in diagnostics)
        capacities = tuple(item["capacity_bps"] for item in diagnostics)
        # The frozen specification defines gamma_min on linear SNR, not gain.
        outage_indices = tuple(i for i, value_snr in enumerate(snrs) if value_snr < self.config.comm.gamma_min)
        outage = bool(outage_indices)
        allocation = tuple(0.0 for _ in capacities) if outage else tdma_allocation(capacities)
        raw_rate = e2e_rate(capacities)
        return {"labels": labels, "gains": gains, "snrs": snrs, "capacities": capacities, "diagnostics": diagnostics, "tdma": allocation,
                "e2e_rate": raw_rate, "effective_e2e_rate": 0.0 if outage else raw_rate,
                "outage": outage, "outage_links": tuple(labels[i] for i in outage_indices)}

    def _safety_info(self) -> dict[str, Any]:
        nodes = [("H", self.h.position), ("L", self.l.position)] + [(f"R{i + 1}", p) for i, p in enumerate(self.relay_positions)]
        pairs: list[tuple[str, str, float]] = []
        for i, (a_name, a_pos) in enumerate(nodes):
            for b_name, b_pos in nodes[i + 1:]:
                if a_name.startswith("R") or b_name.startswith("R"):
                    pairs.append((a_name, b_name, distance(a_pos, b_pos)))
        min_sep = min((x[2] for x in pairs), default=math.inf)
        c = self.config
        warning = tuple((a, b) for a, b, d in pairs if c.hard_safety_distance_m < d < c.warning_distance_m)
        violation = tuple((a, b) for a, b, d in pairs if d < c.hard_safety_distance_m)
        # H/L are retained in the pair for diagnosis; only controllable relay
        # UAVs belong in the RL-agent attribution field.
        involved = tuple(sorted({name for pair in violation for name in pair if name.startswith("R")}))
        return {"min_separation": min_sep, "warning_pairs": warning, "violation_pairs": violation, "involved_agents": involved}

    def _boundary_agents(self) -> tuple[str, ...]:
        c = self.config
        nodes = [("H", self.h.position), ("L", self.l.position)] + [(f"R{i + 1}", p) for i, p in enumerate(self.relay_positions)]
        return tuple(name for name, (x, y, z) in nodes if not (0.0 <= x <= c.map_x_m and 0.0 <= y <= c.map_y_m and c.min_altitude_m <= z <= c.max_altitude_m))

    def _make_info(self, comm: dict[str, Any], safety: dict[str, Any], boundary_agents: list[str] | tuple[str, ...], reason: str | None) -> dict[str, Any]:
        return {"step": self.step_count, "sim_time": self.step_count * self.config.dt_s,
                "e2e_rate": comm["e2e_rate"], "effective_e2e_rate": comm["effective_e2e_rate"],
                "link_capacities": dict(zip(comm["labels"], comm["capacities"])), "link_snrs": dict(zip(comm["labels"], comm["snrs"])),
                "link_diagnostics": dict(zip(comm["labels"], comm["diagnostics"])),
                "tdma_allocation": dict(zip(comm["labels"], comm["tdma"])), "outage": comm["outage"], "outage_links": comm["outage_links"],
                "consecutive_outage_steps": self.consecutive_outage_steps, "min_separation": safety["min_separation"],
                "warning_pairs": safety["warning_pairs"], "violation_pairs": safety["violation_pairs"], "involved_agents": safety["involved_agents"],
                "boundary_violation_agents": tuple(boundary_agents), "termination_reason": reason}

    def _normalise_observation(self, raw: Sequence[float]) -> list[float]:
        c = self.config
        result = list(raw)
        # Absolute positions: horizontal coordinates use map scale; altitude is
        # centred on the [100, 300] m operational band.
        result[0] /= c.map_x_m; result[1] /= c.map_y_m; result[2] = (result[2] - 200.0) / 100.0
        result[3] /= c.relay_max_xy_speed_mps; result[4] /= c.relay_max_xy_speed_mps; result[5] /= c.relay_max_z_speed_mps
        for start in (6, 13, 20):
            result[start] /= c.map_x_m; result[start + 1] /= c.map_y_m; result[start + 2] /= 200.0
        for start in (9, 16, 23):
            result[start] /= 40.0; result[start + 1] /= 40.0; result[start + 2] /= 12.0
        capacity_scale = 60_000_000.0
        result[12] = min(result[12] / capacity_scale, 1.0)
        result[19] = min(result[19] / capacity_scale, 1.0)
        return result
