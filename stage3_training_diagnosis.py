"""Read-only diagnostics for a completed Stage 3 Plain MAPPO run.

This script never trains, writes checkpoints, or changes RelayEnv.  It loads a
saved actor and replays deterministic evaluation episodes, recording compact
per-outage diagnostics that are not retained in train.csv/eval.csv.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from relay_env import RelayEnv
from plain_mappo.checkpoint import load_checkpoint
from plain_mappo.config import MappoConfig
from plain_mappo.metrics import EpisodeMetrics, summarize_episodes
from plain_mappo.networks import SharedActor


def _finite_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _summarise(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(values), "mean": float(np.mean(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def _ratio_above(values: list[float], threshold: float) -> float:
    """Return a finite ratio for possibly empty diagnostic subsets."""
    return float(np.mean(np.asarray(values) > threshold)) if values else 0.0


def _time_bin(step: int) -> str:
    if step <= 166:
        return "early_1_166"
    if step <= 333:
        return "middle_167_333"
    return "late_334_500"


def _actor_from_checkpoint(path: Path) -> tuple[SharedActor, MappoConfig, dict[str, Any]]:
    payload = load_checkpoint(path, "cpu")
    config = MappoConfig.from_dict(payload["config"])
    actor = SharedActor(config.local_obs_dim, config.action_dim, config.actor_hidden_sizes,
                        config.log_std_min, config.log_std_max, config.actor_log_std_mode,
                        config.state_independent_log_std_init, config.actor_variant,
                        config.topology_node_dim, config.topology_edge_dim,
                        config.graph_hidden_dim)
    actor.load_state_dict(payload["actor_state"])
    actor.eval()
    return actor, config, payload


def diagnose_checkpoint(path: Path, seeds: list[int]) -> dict[str, Any]:
    actor, config, payload = _actor_from_checkpoint(path)
    episodes: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    outage_link_steps: Counter[str] = Counter()
    onset_link_events: Counter[str] = Counter()
    active_outage_link_count: Counter[int] = Counter()
    outage_time_bins: Counter[str] = Counter()
    all_action_abs: list[float] = []
    outage_action_abs: list[float] = []
    nonoutage_action_abs: list[float] = []
    onset_action_abs: list[float] = []
    all_xy_speeds: list[float] = []
    outage_xy_speeds: list[float] = []
    all_abs_z_speeds: list[float] = []
    outage_abs_z_speeds: list[float] = []
    all_accel_norms: list[float] = []
    outage_accel_norms: list[float] = []
    relay_altitudes: list[float] = []
    outage_relay_altitudes: list[float] = []
    rates_outage: list[float] = []
    rates_nonoutage: list[float] = []
    onset_link_observations: dict[str, dict[str, list[float]]] = {}
    raw_log_std_values: list[float] = []
    clamped_log_std_values: list[float] = []

    with torch.no_grad():
        for seed in seeds:
            env = RelayEnv(config.num_relays)
            observation, _ = env.reset(seed=seed)
            metrics = EpisodeMetrics(env.config)
            terminated = truncated = False
            reason: str | None = None
            was_outage = False
            current_event: dict[str, Any] | None = None
            episode_events: list[dict[str, Any]] = []

            while not (terminated or truncated):
                before = env.get_global_state()
                tensor_obs = torch.as_tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
                mean, raw_log_std = actor.raw_parameters(tensor_obs)
                clamped_log_std = actor.effective_log_std(raw_log_std)
                raw_log_std_values.extend(float(value) for value in raw_log_std.cpu().numpy().reshape(-1))
                clamped_log_std_values.extend(float(value) for value in clamped_log_std.cpu().numpy().reshape(-1))
                action = torch.tanh(mean).squeeze(0).cpu().numpy()
                observation, rewards, terminated, truncated, info = env.step(action.tolist())
                after = env.get_global_state()
                metrics.add_step(reward=float(rewards[0]), info=info, previous_state=before,
                                 state=after, action_u=action)
                reason = info["termination_reason"]

                step = int(info["step"])
                outage = bool(info["outage"])
                action_abs = np.abs(action).reshape(-1)
                all_action_abs.extend(float(x) for x in action_abs)
                (outage_action_abs if outage else nonoutage_action_abs).extend(float(x) for x in action_abs)
                if outage and not was_outage:
                    onset_action_abs.extend(float(x) for x in action_abs)

                rate_mbps = float(info["effective_e2e_rate"]) / 1e6
                (rates_outage if outage else rates_nonoutage).append(rate_mbps)
                for relay_before, relay_after in zip(before["relays"], after["relays"]):
                    velocity = relay_after["velocity"]
                    xy_speed = math.hypot(velocity[0], velocity[1])
                    z_speed = abs(velocity[2])
                    acceleration = tuple((relay_after["velocity"][axis] - relay_before["velocity"][axis]) /
                                         env.config.dt_s for axis in range(3))
                    accel_norm = math.sqrt(sum(x * x for x in acceleration))
                    all_xy_speeds.append(xy_speed)
                    all_abs_z_speeds.append(z_speed)
                    all_accel_norms.append(accel_norm)
                    relay_altitudes.append(float(relay_after["position"][2]))
                    if outage:
                        outage_xy_speeds.append(xy_speed)
                        outage_abs_z_speeds.append(z_speed)
                        outage_accel_norms.append(accel_norm)
                        outage_relay_altitudes.append(float(relay_after["position"][2]))

                if outage:
                    labels = [str(label) for label in info["outage_links"]]
                    outage_link_steps.update(labels)
                    active_outage_link_count[len(labels)] += 1
                    outage_time_bins[_time_bin(step)] += 1
                    if not was_outage:
                        onset_link_events.update(labels)
                        onset = {"seed": seed, "start_step": step, "end_step": step,
                                 "length": 1, "links_at_onset": labels,
                                 "link_step_counts": Counter(labels),
                                 "trigger_action_abs_mean": float(np.mean(action_abs)),
                                 "trigger_action_saturation_ratio": float(np.mean(action_abs > 0.95)),
                                 "link_physics_at_onset": {},
                                 "relay_controls_at_onset": {}}
                        for label in labels:
                            diagnostic = info["link_diagnostics"][label]
                            snr_linear = float(info["link_snrs"][label])
                            onset["link_physics_at_onset"][label] = {
                                "horizontal_distance_m": float(diagnostic["horizontal_distance"]),
                                "height_difference_abs_m": abs(float(diagnostic["height_difference"])),
                                "snr_linear": snr_linear,
                                "snr_margin_db": 10.0 * math.log10(max(snr_linear, 1e-300)) - 5.0,
                                "capacity_mbps": float(info["link_capacities"][label]) / 1e6}
                            values = onset_link_observations.setdefault(label, {
                                "snr_linear": [], "horizontal_distance_m": [],
                                "height_difference_abs_m": [], "snr_margin_db": [], "capacity_mbps": []})
                            values["snr_linear"].append(snr_linear)
                            values["horizontal_distance_m"].append(float(diagnostic["horizontal_distance"]))
                            values["height_difference_abs_m"].append(abs(float(diagnostic["height_difference"])))
                            values["snr_margin_db"].append(10.0 * math.log10(max(snr_linear, 1e-300)) - 5.0)
                            values["capacity_mbps"].append(float(info["link_capacities"][label]) / 1e6)
                        # These are read-only post-transition values from the
                        # same deterministic replay.  They diagnose endpoint
                        # tracking only; they are never fed back to the Actor.
                        for relay_name, relay_index in (("R1", 0), ("R4", 3)):
                            if relay_index >= config.num_relays:
                                continue
                            velocity = after["relays"][relay_index]["velocity"]
                            onset["relay_controls_at_onset"][relay_name] = {
                                "deterministic_action": [float(value) for value in action[relay_index]],
                                "velocity_mps": [float(value) for value in velocity],
                                "xy_speed_mps": math.hypot(float(velocity[0]), float(velocity[1])),
                                "abs_z_speed_mps": abs(float(velocity[2]))}
                        current_event = onset
                    elif current_event is not None:
                        current_event["end_step"] = step
                        current_event["length"] += 1
                        current_event["link_step_counts"].update(labels)
                elif current_event is not None:
                    current_event["link_step_counts"] = dict(current_event["link_step_counts"])
                    current_event["ended_by"] = "recovery"
                    episode_events.append(current_event)
                    event_records.append(current_event)
                    current_event = None

                was_outage = outage

            if current_event is not None:
                current_event["link_step_counts"] = dict(current_event["link_step_counts"])
                current_event["ended_by"] = reason or "episode_end"
                episode_events.append(current_event)
                event_records.append(current_event)
            record = metrics.as_dict(terminated=terminated, truncated=truncated, reason=reason)
            record.update({"seed": seed, "outage_events": episode_events})
            episodes.append(record)

    summary = summarize_episodes(episodes)
    raw_exceeds_hard_max_ratio = (float(np.mean(np.asarray(raw_log_std_values) > config.log_std_max))
                                   if actor.log_std_mode == "state_dependent_clamp" else 0.0)
    onset_link_summary = {
        label: {metric: _summarise(values) for metric, values in metrics.items()}
        for label, metrics in onset_link_observations.items()
    }
    terminal_events = [event for event in event_records if event["ended_by"] == "persistent_outage"]
    max_xy = max(all_xy_speeds, default=0.0)
    max_z = max(all_abs_z_speeds, default=0.0)
    result = {
        "purpose": "Read-only deterministic replay for Stage 3 first-run diagnosis",
        "checkpoint": str(path).replace("\\", "/"),
        "checkpoint_update": int(payload.get("update", -1)),
        "checkpoint_total_env_steps": int(payload.get("total_env_steps", -1)),
        "checkpoint_best_score": payload.get("best_score"),
        "seeds": seeds,
        "summary": summary,
        "outage_timing": {"outage_step_time_bins": dict(outage_time_bins),
                           "outage_event_count": len(event_records),
                           "persistent_terminal_event_count": len(terminal_events),
                           "outage_event_length": _summarise([float(event["length"]) for event in event_records]),
                           "persistent_event_start_step": _summarise([float(event["start_step"]) for event in terminal_events]),
                           "persistent_event_terminal_step": _summarise([float(event["end_step"]) for event in terminal_events])},
        "outage_bottlenecks": {"outage_link_step_counts": dict(outage_link_steps),
                                "outage_onset_link_event_counts": dict(onset_link_events),
                                "simultaneously_outaged_link_count_steps": {str(k): v for k, v in active_outage_link_count.items()},
                                "onset_link_physics": onset_link_summary},
        "controls": {"all_action_abs": _summarise(all_action_abs),
                     "outage_action_abs": _summarise(outage_action_abs),
                     "nonoutage_action_abs": _summarise(nonoutage_action_abs),
                     "outage_onset_action_abs": _summarise(onset_action_abs),
                     "all_action_saturation_ratio": _ratio_above(all_action_abs, 0.95),
                     "outage_action_saturation_ratio": _ratio_above(outage_action_abs, 0.95),
                     "nonoutage_action_saturation_ratio": _ratio_above(nonoutage_action_abs, 0.95),
                     "outage_onset_action_saturation_ratio": _ratio_above(onset_action_abs, 0.95)},
        "actor_log_std_on_deterministic_eval_states": {
            "mode": actor.log_std_mode,
            "raw": _summarise(raw_log_std_values),
            "clamped": _summarise(clamped_log_std_values),
            "effective": _summarise(clamped_log_std_values),
            "raw_gt_log_std_max_ratio": raw_exceeds_hard_max_ratio,
            "raw_lt_log_std_min_ratio": (float(np.mean(np.asarray(raw_log_std_values) < config.log_std_min))
                                           if actor.log_std_mode == "state_dependent_clamp" else 0.0),
            "log_std_min": config.log_std_min, "log_std_max": config.log_std_max},
        "physical_state": {"xy_speed_mps": _summarise(all_xy_speeds),
                           "outage_xy_speed_mps": _summarise(outage_xy_speeds),
                           "xy_speed_cap_mps": float(env.config.relay_max_xy_speed_mps),
                           "xy_speed_at_or_above_95pct_cap_ratio": float(np.mean(np.asarray(all_xy_speeds) >= 0.95 * env.config.relay_max_xy_speed_mps)),
                           "z_speed_abs_mps": _summarise(all_abs_z_speeds),
                           "outage_z_speed_abs_mps": _summarise(outage_abs_z_speeds),
                           "z_speed_cap_mps": float(env.config.relay_max_z_speed_mps),
                           "acceleration_norm_mps2": _summarise(all_accel_norms),
                           "outage_acceleration_norm_mps2": _summarise(outage_accel_norms),
                           "relay_altitude_m": _summarise(relay_altitudes),
                           "outage_relay_altitude_m": _summarise(outage_relay_altitudes)},
        "rates_mbps": {"outage": _summarise(rates_outage), "nonoutage": _summarise(rates_nonoutage)},
        "episodes": episodes,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/stage3-final"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10000, 10020)))
    args = parser.parse_args()
    run_dir = args.run_dir
    output = {
        "best": diagnose_checkpoint(run_dir / "checkpoints" / "best.pt", args.seeds),
        "latest": diagnose_checkpoint(run_dir / "checkpoints" / "latest.pt", args.seeds),
    }
    destination = run_dir / "final_checkpoint_outage_diagnosis.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
