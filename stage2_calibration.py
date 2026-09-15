"""Stage-2 numerical calibration, regression and stress-test entry point.

This module deliberately has no learning code.  It runs deterministic reference,
zero-action, and random-action controllers against the existing environment and
writes streaming statistics plus optional raw step records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from relay_env import EnvironmentConfig, RelayEnv
from relay_env.communication import distance


MODE_NAMES = ("nominal", "zero_action", "random_stress")
OBS_NAMES = (
    "self_x", "self_y", "self_z", "self_vx", "self_vy", "self_vz",
    "up_dx", "up_dy", "up_dz", "up_dvx", "up_dvy", "up_dvz", "up_capacity",
    "down_dx", "down_dy", "down_dz", "down_dvx", "down_dvy", "down_dvz", "down_capacity",
    "safe_dx", "safe_dy", "safe_dz", "safe_dvx", "safe_dvy", "safe_dvz",
)


class RunningStats:
    """Streaming moments plus a deterministic reservoir for percentile estimates."""

    def __init__(self, reservoir_size: int = 4096, seed: int = 0) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.reservoir_size = reservoir_size
        self.values: list[float] = []
        self.rng = random.Random(seed)

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("non-finite statistic value")
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        if len(self.values) < self.reservoir_size:
            self.values.append(value)
        else:
            index = self.rng.randrange(self.count)
            if index < self.reservoir_size:
                self.values[index] = value

    def summary(self) -> dict[str, float | int | None]:
        if not self.count:
            return {"count": 0, "min": None, "mean": None, "median": None, "p1": None, "p5": None, "p95": None, "p99": None, "max": None, "std": None}
        ordered = sorted(self.values)
        def quantile(q: float) -> float:
            return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))]
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {"count": self.count, "min": self.minimum, "mean": mean, "median": quantile(0.5), "p1": quantile(0.01), "p5": quantile(0.05), "p95": quantile(0.95), "p99": quantile(0.99), "max": self.maximum, "std": math.sqrt(variance)}


def finite_values(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def snr_db(value: float) -> float:
    return 10.0 * math.log10(max(value, 1e-300))


def nominal_actions(env: RelayEnv) -> list[tuple[float, float, float]]:
    """Non-learning reference controller that keeps relays near equal spacing."""
    state = env.get_global_state()
    h, l = state["H"], state["L"]
    delta = tuple(l["position"][axis] - h["position"][axis] for axis in range(3))
    actions = []
    for index, relay in enumerate(state["relays"], start=1):
        fraction = index / (env.num_relays + 1)
        target = tuple(h["position"][axis] + fraction * delta[axis] for axis in range(3))
        error = tuple(target[axis] - relay["position"][axis] for axis in range(3))
        # Action is normalized acceleration demand; it is not an RL actor.
        actions.append(tuple(max(-1.0, min(1.0, 0.025 * error[axis] - 0.12 * relay["velocity"][axis])) for axis in range(3)))
    return actions


def choose_actions(mode: str, env: RelayEnv, rng: random.Random) -> list[tuple[float, float, float]]:
    if mode == "nominal":
        return nominal_actions(env)
    if mode == "zero_action":
        return [(0.0, 0.0, 0.0)] * env.num_relays
    return [tuple(rng.uniform(-1.0, 1.0) for _ in range(3)) for _ in range(env.num_relays)]


def run_stage1_regression(project_root: Path) -> tuple[bool, str]:
    completed = subprocess.run([sys.executable, "-m", "selfcheck.run_selfcheck"], cwd=project_root, text=True, capture_output=True)
    return completed.returncode == 0, completed.stdout.strip()[-2000:] + completed.stderr.strip()[-1000:]


class CalibrationRun:
    def __init__(self, project_root: Path, output_dir: Path, episodes: int, steps: int, write_raw: bool) -> None:
        self.project_root, self.output_dir = project_root, output_dir
        self.episodes, self.steps, self.write_raw = episodes, steps, write_raw
        self.config = EnvironmentConfig()
        self.metric: dict[str, dict[str, RunningStats]] = {mode: defaultdict(RunningStats) for mode in MODE_NAMES}
        self.hop_metric: dict[tuple[str, str, str], RunningStats] = defaultdict(RunningStats)
        self.relay_metric: dict[tuple[str, str, str], RunningStats] = defaultdict(RunningStats)
        self.relay_trajectory_rows: list[dict[str, Any]] = []
        self.obs_metric: dict[tuple[str, str, int], RunningStats] = defaultdict(RunningStats)
        self.counts: dict[str, Counter[str]] = {mode: Counter() for mode in MODE_NAMES}
        self.episode_rows: list[dict[str, Any]] = []
        self.non_finite_found = False
        self.outage_effective_error = False
        self.recovery_error = False
        self._dmax_stress_result: dict[str, Any] | None = None

    def add(self, mode: str, name: str, value: float) -> None:
        try:
            self.metric[mode][name].add(value)
        except ValueError:
            self.non_finite_found = True

    def add_observations(self, mode: str, raw_obs: tuple[tuple[float, ...], ...], norm_obs: tuple[tuple[float, ...], ...]) -> None:
        for representation, observations in (("raw", raw_obs), ("normalized", norm_obs)):
            for observation in observations:
                if not finite_values(observation):
                    self.non_finite_found = True
                for dimension, value in enumerate(observation):
                    self.obs_metric[(mode, representation, dimension)].add(float(value))

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        episode_path = self.output_dir / "stage2_episode_stats.csv"
        raw_path = self.output_dir / "stage2_step_records.jsonl"
        with episode_path.open("w", newline="", encoding="utf-8") as episode_file, (raw_path.open("w", encoding="utf-8") if self.write_raw else _NullWriter()) as raw_file:
            episode_writer = csv.DictWriter(episode_file, fieldnames=("mode", "episode", "seed", "reset_success", "initial_hl_distance_m", "steps", "terminated", "truncated", "termination_reason", "outage_steps", "warning_steps", "min_separation_m", "max_hl_distance_m", "max_hl_xy_speed_mps", "max_hl_abs_z_speed_mps", "waypoint_switches", "dmax_guard_steps", "total_reward"))
            episode_writer.writeheader()
            for mode_index, mode in enumerate(MODE_NAMES):
                for episode in range(self.episodes):
                    seed = mode_index * 1_000_000 + episode
                    row = self.run_episode(mode, episode, seed, raw_file)
                    self.episode_rows.append(row)
                    episode_writer.writerow(row)
        self.write_aggregate_files()

    def run_episode(self, mode: str, episode: int, seed: int, raw_file: Any) -> dict[str, Any]:
        env = RelayEnv(4, self.config, debug=False)
        action_rng = random.Random(seed + 42)
        try:
            normalized_obs, reset_info = env.reset(seed=seed)
        except Exception:
            self.counts[mode]["reset_failures"] += 1
            return {"mode": mode, "episode": episode, "seed": seed, "reset_success": False, "initial_hl_distance_m": "", "steps": 0, "terminated": False, "truncated": False, "termination_reason": "reset_failure", "outage_steps": 0, "warning_steps": 0, "min_separation_m": "", "max_hl_distance_m": "", "max_hl_xy_speed_mps": "", "max_hl_abs_z_speed_mps": "", "waypoint_switches": 0, "dmax_guard_steps": 0, "total_reward": 0.0}
        raw_obs = env.get_raw_obs()
        state = env.get_global_state()
        initial_distance = reset_info["hl_distance_m"]
        self.counts[mode]["reset_successes"] += 1
        self.add(mode, "initial_hl_distance_m", initial_distance)
        for name in ("H", "L"):
            self.add(mode, "waypoint_segment_length_m", distance(state[name]["position"], state[name]["waypoint"]))
        max_hl_distance, min_separation = initial_distance, math.inf
        max_xy_speed = max(abs(math.hypot(state[name]["velocity"][0], state[name]["velocity"][1])) for name in ("H", "L"))
        max_abs_z_speed = max(abs(state[name]["velocity"][2]) for name in ("H", "L"))
        previous_h_velocity, previous_l_velocity = state["H"]["velocity"], state["L"]["velocity"]
        previous_outage, outage_run, waypoint_switches, dmax_guard_steps = False, 0, 0, 0
        outage_steps = warning_steps = 0
        total_reward, terminated, truncated, reason = 0.0, False, False, None
        for step in range(self.steps):
            self.add_observations(mode, raw_obs, normalized_obs)
            before_state = state
            before_waypoints = (before_state["H"]["waypoint"], before_state["L"]["waypoint"])
            actions = choose_actions(mode, env, action_rng)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            raw_next = env.get_raw_obs()
            state = env.get_global_state()
            self.record_step(mode, episode, seed, step, state, previous_h_velocity, previous_l_velocity, info, rewards)
            current_distance = info["hl_distance_m"]
            max_hl_distance = max(max_hl_distance, current_distance)
            for threshold in (1600.0, 1800.0, 2000.0):
                self.counts[mode][f"hl_distance_at_or_above_{int(threshold)}m"] += int(current_distance >= threshold)
            min_separation = min(min_separation, info["min_separation"])
            max_xy_speed = max(max_xy_speed, *(math.hypot(state[name]["velocity"][0], state[name]["velocity"][1]) for name in ("H", "L")))
            max_abs_z_speed = max(max_abs_z_speed, *(abs(state[name]["velocity"][2]) for name in ("H", "L")))
            for name, previous_waypoint in zip(("H", "L"), before_waypoints):
                if state[name]["waypoint"] != previous_waypoint:
                    waypoint_switches += 1
                    self.add(mode, "waypoint_segment_length_m", distance(state[name]["position"], state[name]["waypoint"]))
            dmax_guard_steps += int(info["hl_distance_guard_applied"])
            is_outage = bool(info["outage"])
            outage_steps += int(is_outage); warning_steps += int(bool(info["warning_pairs"]))
            self.counts[mode]["warning_pair_count"] += len(info["warning_pairs"])
            self.counts[mode]["safety_violation_pair_count"] += len(info["violation_pairs"])
            self.counts[mode]["boundary_violation_agent_count"] += len(info["boundary_violation_agents"])
            if is_outage:
                outage_run += 1
                if info["effective_e2e_rate"] != 0.0: self.outage_effective_error = True
            elif previous_outage:
                self.add(mode, "outage_duration_steps", outage_run)
                self.counts[mode]["outage_recoveries"] += 1
                if info["consecutive_outage_steps"] != 0: self.recovery_error = True
                outage_run = 0
            previous_outage = is_outage
            total_reward += rewards[0]
            if self.write_raw:
                nodes = [state["H"]["position"]] + [relay["position"] for relay in state["relays"]] + [state["L"]["position"]]
                link_distances = {label: distance(nodes[index], nodes[index + 1]) for index, label in enumerate(info["link_capacities"])}
                raw_file.write(json.dumps({"mode": mode, "episode": episode, "seed": seed, "step": step, "raw_observation": raw_obs, "normalized_observation": normalized_obs, "actions": actions, "reward": rewards[0], "state": state, "link_distances_m": link_distances, "info": info}, ensure_ascii=False) + "\n")
            raw_obs, normalized_obs = raw_next, next_obs
            previous_h_velocity, previous_l_velocity = state["H"]["velocity"], state["L"]["velocity"]
            if terminated or truncated:
                reason = info["termination_reason"]
                break
        if outage_run: self.add(mode, "outage_duration_steps", outage_run)
        self.counts[mode]["outage_steps"] += outage_steps
        self.counts[mode]["warning_steps"] += warning_steps
        self.counts[mode]["dmax_guard_steps"] += dmax_guard_steps
        self.counts[mode]["terminated_" + str(reason)] += int(terminated)
        self.counts[mode]["truncated"] += int(truncated)
        self.add(mode, "episode_length_steps", step + 1)
        return {"mode": mode, "episode": episode, "seed": seed, "reset_success": True, "initial_hl_distance_m": initial_distance, "steps": step + 1, "terminated": terminated, "truncated": truncated, "termination_reason": reason or "", "outage_steps": outage_steps, "warning_steps": warning_steps, "min_separation_m": min_separation, "max_hl_distance_m": max_hl_distance, "max_hl_xy_speed_mps": max_xy_speed, "max_hl_abs_z_speed_mps": max_abs_z_speed, "waypoint_switches": waypoint_switches, "dmax_guard_steps": dmax_guard_steps, "total_reward": total_reward}

    def record_step(self, mode: str, episode: int, seed: int, step: int, state: dict[str, Any], old_h_velocity: tuple[float, ...], old_l_velocity: tuple[float, ...], info: dict[str, Any], rewards: tuple[float, ...]) -> None:
        dt = self.config.dt_s
        self.add(mode, "hl_distance_m", info["hl_distance_m"])
        self.add(mode, "min_separation_m", info["min_separation"])
        self.add(mode, "raw_e2e_mbps", info["e2e_rate"] / 1e6)
        self.add(mode, "effective_e2e_mbps", info["effective_e2e_rate"] / 1e6)
        self.add(mode, "total_reward", rewards[0])
        components = {"throughput_reward": info["effective_e2e_rate"] / self.config.reward.reference_rate_bps, "outage_penalty": -self.config.reward.lambda_outage * float(info["outage"]), "warning_penalty": -self.config.reward.warning_penalty * float(bool(info["warning_pairs"])), "failure_penalty": -self.config.reward.lambda_failure * float(info["termination_reason"] is not None)}
        for key, value in components.items(): self.add(mode, key, value)
        for name, old_velocity in (("H", old_h_velocity), ("L", old_l_velocity)):
            velocity = state[name]["velocity"]
            self.add(mode, f"{name.lower()}_xy_speed_mps", math.hypot(velocity[0], velocity[1]))
            self.add(mode, f"{name.lower()}_z_speed_mps", velocity[2])
            delta_velocity = tuple((velocity[i] - old_velocity[i]) / dt for i in range(3))
            self.add(mode, f"{name.lower()}_xy_accel_mps2", math.hypot(delta_velocity[0], delta_velocity[1]))
            self.add(mode, f"{name.lower()}_z_accel_mps2", abs(delta_velocity[2]))
        nodes = [state["H"]["position"]] + [relay["position"] for relay in state["relays"]] + [state["L"]["position"]]
        for label, diagnostic in info["link_diagnostics"].items():
            snr_value = diagnostic["snr_linear"]
            for metric, value in (("distance_m", diagnostic["distance_3d"]),
                                  ("horizontal_distance_m", diagnostic["horizontal_distance"]),
                                  ("height_difference_abs_m", abs(diagnostic["height_difference"])),
                                  ("elevation_angle_deg", diagnostic["elevation_angle_deg"]),
                                  ("antenna_gain_linear", diagnostic["antenna_gain_linear"]),
                                  ("antenna_gain_db", diagnostic["antenna_gain_db"]),
                                  ("snr_linear", snr_value), ("snr_db", diagnostic["snr_db"]),
                                  ("capacity_mbps", diagnostic["capacity_bps"] / 1e6)):
                self.hop_metric[(mode, label, metric)].add(value)
            self.counts[mode]["hops_below_5db"] += int(snr_value < self.config.comm.gamma_min)
            self.counts[mode]["hop_samples"] += 1
        for index, relay in enumerate(state["relays"], start=1):
            position = relay["position"]
            upstream, downstream = nodes[index - 1], nodes[index + 1]
            for metric, value in (("altitude_m", position[2]),
                                  ("upstream_height_difference_m", upstream[2] - position[2]),
                                  ("downstream_height_difference_m", position[2] - downstream[2])):
                self.relay_metric[(mode, f"R{index}", metric)].add(value)
            self.relay_trajectory_rows.append({"mode": mode, "episode": episode, "seed": seed, "step": step,
                                               "relay": f"R{index}", "altitude_m": position[2],
                                               "upstream_height_difference_m": upstream[2] - position[2],
                                               "downstream_height_difference_m": position[2] - downstream[2]})

    def write_aggregate_files(self) -> None:
        with (self.output_dir / "stage2_hop_stats.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ("mode", "hop", "metric", "count", "min", "mean", "median", "p1", "p5", "p95", "p99", "max", "std")
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for (mode, hop, metric), stats in sorted(self.hop_metric.items()): writer.writerow({"mode": mode, "hop": hop, "metric": metric, **stats.summary()})
        with (self.output_dir / "stage2_relay_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ("mode", "relay", "metric", "count", "min", "mean", "median", "p1", "p5", "p95", "p99", "max", "std")
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for (mode, relay, metric), stats in sorted(self.relay_metric.items()): writer.writerow({"mode": mode, "relay": relay, "metric": metric, **stats.summary()})
        with (self.output_dir / "stage2_relay_height_trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ("mode", "episode", "seed", "step", "relay", "altitude_m", "upstream_height_difference_m", "downstream_height_difference_m")
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(self.relay_trajectory_rows)
        with (self.output_dir / "stage2_observation_stats.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ("mode", "representation", "dimension", "name", "count", "min", "mean", "median", "p1", "p5", "p95", "p99", "max", "std", "outside_unit_ratio", "clipping_ratio")
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for (mode, representation, dimension), stats in sorted(self.obs_metric.items()):
                summary = stats.summary(); values = stats.values
                outside = sum(abs(x) > 1.0 for x in values) / len(values) if values else 0.0
                clipping = sum(abs(x) >= 1.0 for x in values) / len(values) if values else 0.0
                writer.writerow({"mode": mode, "representation": representation, "dimension": dimension, "name": OBS_NAMES[dimension], **summary, "outside_unit_ratio": outside, "clipping_ratio": clipping})
        mode_summaries = {mode: {name: stats.summary() for name, stats in metrics.items()} for mode, metrics in self.metric.items()}
        summary = {"configuration": asdict(self.config), "episodes_per_mode": self.episodes, "max_steps": self.steps, "raw_step_records": self.write_raw, "modes": mode_summaries, "counts": {mode: dict(counter) for mode, counter in self.counts.items()}, "dmax_preventive_stress": self.dmax_preventive_stress(), "hard_checks": self.hard_checks(), "percentile_method": "deterministic reservoir estimate (4096 samples per metric)"}
        (self.output_dir / "stage2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def hard_checks(self) -> dict[str, bool]:
        c = self.config
        all_rows = [row for row in self.episode_rows if row["reset_success"]]
        def bounded(metric: str, maximum: float) -> bool:
            return all(not metrics.get(metric) or metrics[metric].maximum <= maximum + 1e-9 for metrics in self.metric.values())
        return {"gamma_min_is_5db_linear": math.isclose(c.comm.gamma_min, 10.0 ** 0.5, rel_tol=1e-14), "initial_hl_distance_in_range": all(1000.0 <= float(row["initial_hl_distance_m"]) <= 1500.0 for row in all_rows), "actual_hl_distance_within_dmax": all(float(row["max_hl_distance_m"]) <= c.mobile_max_hl_distance_m + 1e-9 for row in all_rows), "hl_xy_speed_capped": all(float(row["max_hl_xy_speed_mps"]) <= c.mobile_max_speed_mps + 1e-9 for row in all_rows), "hl_z_speed_capped": all(float(row["max_hl_abs_z_speed_mps"]) <= c.mobile_max_z_speed_mps + 1e-9 for row in all_rows), "hl_xy_acceleration_capped": bounded("h_xy_accel_mps2", c.mobile_max_accel_mps2) and bounded("l_xy_accel_mps2", c.mobile_max_accel_mps2), "hl_z_acceleration_capped": bounded("h_z_accel_mps2", c.mobile_max_accel_mps2) and bounded("l_z_accel_mps2", c.mobile_max_accel_mps2), "d_warn_is_60m": c.warning_distance_m == 60.0, "d_safe_is_20m": c.hard_safety_distance_m == 20.0, "persistent_outage_is_25": c.persistent_outage_steps == 25, "outage_masks_effective_rate": not self.outage_effective_error, "recovery_resets_counter": not self.recovery_error, "no_nan_or_inf": not self.non_finite_found, "seed_reproducibility": self.seed_reproducibility(), "k_smoke": self.k_smoke(), "termination_truncation_semantics": all(not (bool(row["terminated"]) and bool(row["truncated"])) for row in all_rows), "dmax_preventive_acceleration_stress": self.dmax_preventive_stress()["passed"]}

    def dmax_preventive_stress(self) -> dict[str, Any]:
        """Exercise a reachable high-separation state without any position projection.

        H/L begin 150 m inside D_max with legal, maximum outward horizontal
        speeds.  The pairwise guard must brake them through a 500-step episode
        while preserving every final speed and acceleration bound.
        """
        if self._dmax_stress_result is not None:
            return self._dmax_stress_result
        env = RelayEnv(4, self.config)
        env.reset(seed=424242)
        env.h.position, env.h.velocity, env.h.waypoint = (75.0, 1_000.0, 200.0), (-10.0, 0.0, 0.0), (300.0, 1_000.0, 200.0)
        env.l.position, env.l.velocity, env.l.waypoint = (1_925.0, 1_000.0, 200.0), (10.0, 0.0, 0.0), (1_700.0, 1_000.0, 200.0)
        env.relay_positions = tuple((75.0 + index * 370.0, 1_000.0, 200.0) for index in range(1, 5))
        env.relay_velocities = tuple((0.0, 0.0, 0.0) for _ in range(4))
        maximum_distance, maximum_xy_accel, maximum_z_accel = 1_850.0, 0.0, 0.0
        maximum_xy_speed, maximum_abs_z_speed, guard_steps = 10.0, 0.0, 0
        terminated = truncated = False
        for _ in range(self.config.max_episode_steps):
            before_h, before_l = env.h.velocity, env.l.velocity
            _, _, terminated, truncated, info = env.step([(0.0, 0.0, 0.0)] * 4)
            for before, after in ((before_h, env.h.velocity), (before_l, env.l.velocity)):
                maximum_xy_accel = max(maximum_xy_accel, math.hypot(after[0] - before[0], after[1] - before[1]) / self.config.dt_s)
                maximum_z_accel = max(maximum_z_accel, abs(after[2] - before[2]) / self.config.dt_s)
                maximum_xy_speed = max(maximum_xy_speed, math.hypot(after[0], after[1]))
                maximum_abs_z_speed = max(maximum_abs_z_speed, abs(after[2]))
            maximum_distance = max(maximum_distance, info["hl_distance_m"])
            guard_steps += int(info["hl_distance_guard_applied"])
            if terminated or truncated:
                break
        c = self.config
        passed = (not terminated and truncated and guard_steps > 0 and maximum_distance <= c.mobile_max_hl_distance_m + 1e-9
                  and maximum_xy_accel <= c.mobile_max_accel_mps2 + 1e-9 and maximum_z_accel <= c.mobile_max_accel_mps2 + 1e-9
                  and maximum_xy_speed <= c.mobile_max_speed_mps + 1e-9 and maximum_abs_z_speed <= c.mobile_max_z_speed_mps + 1e-9)
        self._dmax_stress_result = {"passed": passed, "steps": self.config.max_episode_steps, "terminated": terminated, "truncated": truncated, "guard_steps": guard_steps, "max_hl_distance_m": maximum_distance, "max_hl_xy_accel_mps2": maximum_xy_accel, "max_hl_z_accel_mps2": maximum_z_accel, "max_hl_xy_speed_mps": maximum_xy_speed, "max_hl_abs_z_speed_mps": maximum_abs_z_speed}
        return self._dmax_stress_result

    def seed_reproducibility(self) -> bool:
        first, second = RelayEnv(4, self.config), RelayEnv(4, self.config)
        obs_a, _ = first.reset(seed=9182); obs_b, _ = second.reset(seed=9182)
        action = [(0.2, -0.3, 0.1)] * 4
        return obs_a == obs_b and first.step(action) == second.step(action)

    def k_smoke(self) -> bool:
        try:
            for k in (3, 4, 5):
                env = RelayEnv(k, self.config); obs, _ = env.reset(seed=7000 + k); next_obs, *_ = env.step([(0.0, 0.0, 0.0)] * k)
                if len(obs) != k or len(next_obs) != k or any(len(value) != 26 for value in obs + next_obs): return False
            return True
        except Exception:
            return False


class _NullWriter:
    def write(self, _: str) -> int: return 0
    def __enter__(self) -> "_NullWriter": return self
    def __exit__(self, *_: Any) -> None: return None


def write_validation_report(output_dir: Path, stage1_ok: bool, stage1_output: str, run: CalibrationRun) -> bool:
    hard_checks = run.hard_checks()
    passed = stage1_ok and all(hard_checks.values())
    stress = run.dmax_preventive_stress()
    def mean(mode: str, metric: str) -> str:
        value = run.metric[mode].get(metric)
        return "n/a" if value is None or not value.count else f"{value.summary()['mean']:.4g}"
    mode_lines = "\n".join(f"- {mode}: reset={run.counts[mode]['reset_successes']}, truncated={run.counts[mode]['truncated']}, outage steps={run.counts[mode]['outage_steps']}, warnings={run.counts[mode]['warning_steps']}, mean effective R_e2e={mean(mode, 'effective_e2e_mbps')} Mbps, mean H-L distance={mean(mode, 'hl_distance_m')} m" for mode in MODE_NAMES)
    hard_check_lines = "\n".join(f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in hard_checks.items())
    report = f"""# Stage 2 implementation validation: {'PASS' if passed else 'FAIL'}

This result means only that the requested code modifications and statistics generation completed correctly. It does **not** declare Environment v1 frozen; that decision remains for human review.

## Candidate parameters applied

- gamma_min = 5 dB = {run.config.comm.gamma_min:.15g} linear; D_max = {run.config.mobile_max_hl_distance_m:.0f} m.
- H/L cruise target = [{run.config.mobile_min_speed_mps:.0f}, {run.config.mobile_max_speed_mps:.0f}] m/s; z speed = [{run.config.mobile_min_z_speed_mps:.0f}, {run.config.mobile_max_z_speed_mps:.0f}] m/s; acceleration = {run.config.mobile_max_accel_mps2:.0f} m/s².
- waypoint segment = [{run.config.mobile_waypoint_min_segment_m:.0f}, {run.config.mobile_waypoint_max_segment_m:.0f}] m; d_warn = {run.config.warning_distance_m:.0f} m; R_ref = {run.config.reward.reference_rate_bps / 1e6:.0f} Mbps.

## Modified and added files

- `relay_env/config.py`, `relay_env/environment.py`: candidate parameters, waypoint span, final H/L speed caps, and raw observation access.
- `relay_env/environment.py`: replaces post-step D_max position/velocity projection with synchronous, pairwise preventive bounded braking.
- `stage2_calibration.py`: independent three-mode calibration, hard checks, raw records, summaries, and this report.

## Stage 1 regression

- {'PASS' if stage1_ok else 'FAIL'} — stage1 selfcheck process.

## Stage 2 hard checks

{hard_check_lines}

## D_max / acceleration stress test

- Three-mode regression scale: {run.episodes} seeds per mode, up to {run.steps} steps each.
- Active-guard scenario: one forced near-D_max, maximum-outward-speed trajectory for {stress['steps']} steps.
- Preventive guard interventions: {stress['guard_steps']} steps; maximum H-L distance: {stress['max_hl_distance_m']:.12g} m.
- Maximum actual horizontal acceleration: {stress['max_hl_xy_accel_mps2']:.12g} m/s²; vertical acceleration: {stress['max_hl_z_accel_mps2']:.12g} m/s².
- Maximum actual horizontal speed: {stress['max_hl_xy_speed_mps']:.12g} m/s; absolute vertical speed: {stress['max_hl_abs_z_speed_mps']:.12g} m/s.
- The stress episode terminated={stress['terminated']} and truncated={stress['truncated']}; no position projection or velocity overwrite is used.

## Mode summary

{mode_lines}

## Exceptions and still-undecided parameters

- No automatic re-tuning was performed. gamma_min, D_max, vertical motion range, waypoint behaviour, warning range, reward scales, observation scales, and communication realism still require human review of the generated statistics.
- Stage1 subprocess tail: `{stage1_output[-500:].replace('`', "'")}`

## Output data

- `stage2_summary.json`
- `stage2_episode_stats.csv`
- `stage2_hop_stats.csv`
- `stage2_observation_stats.csv`
- `stage2_step_records.jsonl` {'(written via explicit --write-raw)' if run.write_raw else '(disabled by default; pass --write-raw to enable)'}
"""
    (output_dir / "stage2_validation_report.md").write_text(report, encoding="utf-8")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 environment calibration and stress tests; no RL training.")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per mode; taskbook formal run uses 500.")
    parser.add_argument("--steps", type=int, default=500, help="Maximum steps per episode.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage2"))
    parser.add_argument("--write-raw", action=argparse.BooleanOptionalAction, default=False,
                        help="Write complete step-level raw/normalized observation records (disabled by default; use explicitly for large diagnostics).")
    parser.add_argument("--skip-stage1-regression", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1: parser.error("episodes and steps must be positive")
    root = Path(__file__).resolve().parent
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    stage1_ok, stage1_output = (True, "skipped") if args.skip_stage1_regression else run_stage1_regression(root)
    run = CalibrationRun(root, output_dir, args.episodes, args.steps, args.write_raw)
    run.run()
    passed = write_validation_report(output_dir, stage1_ok, stage1_output, run)
    print(json.dumps({"validation": "PASS" if passed else "FAIL", "output_dir": str(output_dir), "hard_checks": run.hard_checks()}, ensure_ascii=False))
    if not passed: raise SystemExit(1)


if __name__ == "__main__":
    main()
