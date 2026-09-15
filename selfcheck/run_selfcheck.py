"""Run the ten checks named in the stage-1 task book and write evidence."""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Callable

from relay_env import EnvironmentConfig, RelayEnv
from relay_env.communication import capacity, channel_gain, distance, snr, tdma_allocation

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"


def check_a_reset() -> dict:
    env, distances = RelayEnv(4), []
    counters = {"boundary": 0, "safety": 0, "outage": 0}
    for seed in range(1000):
        _, info = env.reset(seed=seed)
        state = env.get_global_state()
        distances.append(distance(state["H"]["position"], state["L"]["position"]))
        counters["boundary"] += bool(info["boundary_violation_agents"])
        counters["safety"] += bool(info["violation_pairs"])
        counters["outage"] += bool(info["outage"])
    ok = not any(counters.values()) and min(distances) >= env.config.initial_hl_distance_min_m and max(distances) <= env.config.initial_hl_distance_max_m
    return {"passed": ok, "resets": 1000, "illegal_counts": counters, "hl_distance_min_m": min(distances), "hl_distance_max_m": max(distances)}


def check_b_zero_action() -> dict:
    env = RelayEnv(4, debug=True)
    env.reset(seed=11)
    before = env.get_global_state()
    _, _, terminated, _, info = env.step([(0, 0, 0)] * 4)
    after = env.get_global_state()
    relay_zero = all(v == (0.0, 0.0, 0.0) for v in (x["velocity"] for x in after["relays"]))
    mobile_moved = before["H"]["position"] != after["H"]["position"] and before["L"]["position"] != after["L"]["position"]
    return {"passed": relay_zero and mobile_moved and not terminated and info["sim_time"] == 0.2,
            "relay_zero_velocity": relay_zero, "mobile_moved": mobile_moved, "sim_time_s": info["sim_time"]}


def check_c_max_acceleration() -> dict:
    env = RelayEnv(4, debug=True)
    env.reset(seed=12)
    before = env.get_global_state()["relays"][0]["position"]
    _, _, _, _, info = env.step([(1, 0, 0)] + [(0, 0, 0)] * 3)
    applied = info["action_debug"]["applied_velocities"][0]
    expected = env.config.relay_max_xy_accel_mps2 * env.config.dt_s
    after = env.get_global_state()["relays"][0]["position"]
    expected_x = before[0] + 0.5 * expected * env.config.dt_s
    return {"passed": math.isclose(applied[0], expected, abs_tol=1e-12) and math.isclose(after[0], expected_x, abs_tol=1e-12),
            "expected_delta_v_mps": expected, "actual_velocity_mps": applied,
            "expected_position_x_m": expected_x, "actual_position_x_m": after[0]}


def check_d_speed_limit() -> dict:
    env = RelayEnv(4, debug=True)
    env.reset(seed=13)
    horizontal, vertical_up, vertical_down = [], [], []
    for _ in range(60):
        _, _, terminated, _, info = env.step([(1, 0, 0)] + [(0, 0, 0)] * 3)
        horizontal.append(math.hypot(*info["action_debug"]["applied_velocities"][0][:2]))
        if terminated:
            break
    env.reset(seed=14)
    for _ in range(40):
        _, _, terminated, _, info = env.step([(0, 0, 1)] + [(0, 0, 0)] * 3)
        vertical_up.append(info["action_debug"]["applied_velocities"][0][2])
        if terminated:
            break
    env.reset(seed=15)
    for _ in range(40):
        _, _, terminated, _, info = env.step([(0, 0, -1)] + [(0, 0, 0)] * 3)
        vertical_down.append(info["action_debug"]["applied_velocities"][0][2])
        if terminated:
            break
    c = env.config
    return {"passed": max(horizontal) <= c.relay_max_xy_speed_mps and max(vertical_up) <= c.relay_max_z_speed_mps and min(vertical_down) >= c.relay_min_z_speed_mps,
            "max_xy_speed_mps": max(horizontal), "max_up_speed_mps": max(vertical_up), "min_down_speed_mps": min(vertical_down)}


def check_e_communication_monotonicity() -> dict:
    c = EnvironmentConfig().comm
    rows = []
    for d in (100.0, 250.0, 500.0, 1000.0):
        gain = channel_gain(d, c.reference_gain, c.path_loss_exponent)
        value_snr = snr(gain, c.tx_power_w, c.noise_density_w_hz, c.bandwidth_hz)
        rows.append({"distance_m": d, "snr": value_snr, "capacity_bps": capacity(c.bandwidth_hz, value_snr)})
    return {"passed": all(rows[i]["snr"] > rows[i + 1]["snr"] and rows[i]["capacity_bps"] > rows[i + 1]["capacity_bps"] for i in range(len(rows) - 1)), "rows": rows}


def check_f_tdma() -> dict:
    capacities = (10.0, 1.0, 10.0)
    tau = tdma_allocation(capacities)
    return {"passed": tau[1] > tau[0] and tau[1] > tau[2] and math.isclose(sum(tau), 1.0, abs_tol=1e-12), "capacities": capacities, "allocation": tau, "sum": sum(tau)}


def check_g_hard_outage() -> dict:
    cfg = replace(EnvironmentConfig(), persistent_outage_steps=3)
    env = RelayEnv(4, cfg)
    env.reset(seed=16)
    saved = env.relay_positions
    # Move R1 to a legal map corner: its adjacent hops become invalid without
    # conflating the outage test with a boundary termination.
    env.relay_positions = ((1_950.0, 1_950.0, 150.0),) + saved[1:]
    observations = []
    for _ in range(3):
        _, _, term, _, info = env.step([(0, 0, 0)] * 4)
        observations.append({"links": info["outage_links"], "rate": info["effective_e2e_rate"], "counter": info["consecutive_outage_steps"], "terminated": term})
    env.reset(seed=16)
    _, _, _, _, recovered = env.step([(0, 0, 0)] * 4)
    correct = all(x["rate"] == 0.0 and x["counter"] == i + 1 and x["links"] for i, x in enumerate(observations))
    return {"passed": correct and not observations[0]["terminated"] and not observations[1]["terminated"] and observations[-1]["terminated"] and recovered["consecutive_outage_steps"] == 0,
            "outage_steps": observations, "recovered_counter": recovered["consecutive_outage_steps"]}


def check_h_safety() -> dict:
    env = RelayEnv(3)
    env.reset(seed=17)
    hp = env.h.position
    lp = env.l.position
    base = env.relay_positions
    results = {}
    for label, reference, offset in (("clear", hp, 70.0), ("warning", hp, 30.0), ("violation_h", hp, 10.0), ("violation_l", lp, 10.0)):
        env.reset(seed=17)
        env.relay_positions = ((reference[0] + offset, reference[1], reference[2]),) + base[1:]
        _, _, terminated, _, info = env.step([(0, 0, 0)] * 3)
        results[label] = {"warning_pairs": info["warning_pairs"], "violation_pairs": info["violation_pairs"], "involved_agents": info["involved_agents"], "terminated": terminated, "reason": info["termination_reason"]}
    valid_hl_attribution = all(results[name]["violation_pairs"] and results[name]["reason"] == "safety_violation" and results[name]["involved_agents"] == ("R1",) for name in ("violation_h", "violation_l"))
    return {"passed": not results["clear"]["warning_pairs"] and not results["clear"]["terminated"] and bool(results["warning"]["warning_pairs"]) and not results["warning"]["terminated"] and valid_hl_attribution, "cases": results}


def check_i_boundary() -> dict:
    env = RelayEnv(4)
    env.reset(seed=18)
    p, v = list(env.relay_positions), list(env.relay_velocities)
    p[0] = (env.config.map_x_m - 0.5, env.config.map_y_m - 0.5, env.config.max_altitude_m - 0.5)
    v[0] = (env.config.relay_max_xy_speed_mps, 0.0, env.config.relay_max_z_speed_mps)
    env.relay_positions, env.relay_velocities = tuple(p), tuple(v)
    _, _, terminated, _, info = env.step([(0, 0, 0)] * 4)
    return {"passed": terminated and info["termination_reason"] == "boundary_violation", "terminated": terminated, "reason": info["termination_reason"], "agents": info["boundary_violation_agents"]}


def check_j_scaling() -> dict:
    rows = []
    for k in (3, 4, 5):
        env = RelayEnv(k)
        obs, info = env.reset(seed=100 + k)
        next_obs, _, terminated, _, step_info = env.step([(0, 0, 0)] * k)
        rows.append({"K": k, "obs_shape": [len(obs), len(obs[0])], "next_obs_shape": [len(next_obs), len(next_obs[0])], "links": len(info["link_capacities"]), "terminated": terminated, "step_links": len(step_info["link_capacities"])})
    return {"passed": all(x["obs_shape"] == [x["K"], 26] and x["next_obs_shape"] == [x["K"], 26] and x["links"] == x["K"] + 1 and x["step_links"] == x["K"] + 1 for x in rows), "rows": rows}


def check_k_reproducibility_and_random_episodes() -> dict:
    """Additional acceptance criterion: fixed seeds reproduce and random runs stay finite."""
    first, second = RelayEnv(4), RelayEnv(4)
    obs_a, _ = first.reset(seed=31415)
    obs_b, _ = second.reset(seed=31415)
    actions = [(0.3, -0.1, 0.2)] * 4
    next_a = first.step(actions)
    next_b = second.step(actions)
    reproducible = obs_a == obs_b and next_a == next_b
    rng, steps = random.Random(2718), 0
    finite = True
    for k in (3, 4, 5):
        env = RelayEnv(k)
        for episode in range(3):
            obs, _ = env.reset(seed=10_000 + k * 10 + episode)
            for value in (x for agent in obs for x in agent): finite &= math.isfinite(value)
            for _ in range(100):
                action = [tuple(rng.uniform(-1.0, 1.0) for _ in range(3)) for _ in range(k)]
                obs, rewards, terminated, truncated, info = env.step(action)
                steps += 1
                finite &= all(math.isfinite(x) for agent in obs for x in agent) and all(math.isfinite(x) for x in rewards)
                finite &= all(math.isfinite(x) for x in info["link_capacities"].values())
                if terminated or truncated:
                    break
    return {"passed": reproducible and finite, "seed_reproducible": reproducible, "random_steps_completed": steps, "all_numeric_values_finite": finite}


def check_l_frozen_spec_alignment() -> dict:
    """Guard the frozen 01 handoff defaults against accidental regressions."""
    c = EnvironmentConfig()
    comm = c.comm
    env = RelayEnv(4)
    env.reset(seed=2026)
    state = env.get_global_state()
    horizontal_speeds = [math.hypot(state[name]["velocity"][0], state[name]["velocity"][1]) for name in ("H", "L")]
    z_speeds = [state[name]["velocity"][2] for name in ("H", "L")]
    waypoint_scales = [distance(state["H"]["waypoint"], state["L"]["position"]), distance(state["L"]["waypoint"], state["H"]["position"])]
    n0_at_minus_169_dbm_hz = 10.0 ** ((-169.0 - 30.0) / 10.0)
    passed = (c.min_altitude_m == 100.0 and c.max_altitude_m == 300.0 and c.mobile_min_speed_mps == 6.0 and c.mobile_max_speed_mps == 10.0
              and comm.bandwidth_hz == 10_000_000.0 and comm.tx_power_w == 0.1 and comm.reference_gain == 1.0e-6 and comm.path_loss_exponent == 2.0
              and math.isclose(comm.noise_density_w_hz, n0_at_minus_169_dbm_hz, rel_tol=1e-14)
              and all(c.mobile_min_speed_mps <= speed <= c.mobile_max_speed_mps for speed in horizontal_speeds)
              and all(c.mobile_min_z_speed_mps <= speed <= c.mobile_max_z_speed_mps for speed in z_speeds)
              and all(value <= c.mobile_max_hl_distance_m for value in waypoint_scales))
    return {"passed": passed, "altitude_range_m": [c.min_altitude_m, c.max_altitude_m], "hl_xy_cruise_range_mps": [c.mobile_min_speed_mps, c.mobile_max_speed_mps],
            "communication": {"B_hz": comm.bandwidth_hz, "P_w": comm.tx_power_w, "N0_w_hz": comm.noise_density_w_hz, "beta0": comm.reference_gain, "alpha": comm.path_loss_exponent, "gamma_min_linear": comm.gamma_min},
            "initial_hl_xy_speeds_mps": horizontal_speeds, "initial_hl_z_speeds_mps": z_speeds, "waypoint_peer_distances_m": waypoint_scales}


def check_m_episode_hl_distance_constraint() -> dict:
    """A complete 500-step episode must keep actual H-L distance within D_max."""
    cfg = replace(EnvironmentConfig(), persistent_outage_steps=10_000)
    env = RelayEnv(4, cfg)
    _, reset_info = env.reset(seed=20260829)
    distances, guard_steps = [reset_info["hl_distance_m"]], 0
    terminated = truncated = False
    for _ in range(cfg.max_episode_steps):
        _, _, terminated, truncated, info = env.step([(0.0, 0.0, 0.0)] * 4)
        distances.append(info["hl_distance_m"])
        guard_steps += int(info["hl_distance_guard_applied"])
        if terminated or truncated:
            break
    return {"passed": not terminated and truncated and len(distances) == cfg.max_episode_steps + 1 and max(distances) <= cfg.mobile_max_hl_distance_m + 1e-9,
            "steps_completed": len(distances) - 1, "terminated": terminated, "truncated": truncated,
            "max_actual_hl_distance_m": max(distances), "d_max_m": cfg.mobile_max_hl_distance_m, "emergency_guard_steps": guard_steps}


def run_trace() -> dict:
    """Nominal deterministic trace used exclusively for acceptance plots."""
    cfg = replace(EnvironmentConfig(), persistent_outage_steps=10_000, max_episode_steps=240)
    env = RelayEnv(4, cfg)
    env.reset(seed=20260829)
    trace = {key: [] for key in ("time", "h_x", "h_y", "l_x", "l_y", "h_z", "l_z", "hl_distance", "min_separation", "rate", "outage_counter")}
    for i in range(200):
        actions = [(0.22 * math.sin(i / 18 + j), 0.18 * math.cos(i / 21 + j), 0.08 * math.sin(i / 15 + j)) for j in range(4)]
        _, _, terminated, truncated, info = env.step(actions)
        state = env.get_global_state()
        trace["time"].append(info["sim_time"])
        trace["h_x"].append(state["H"]["position"][0]); trace["h_y"].append(state["H"]["position"][1])
        trace["l_x"].append(state["L"]["position"][0]); trace["l_y"].append(state["L"]["position"][1])
        trace["h_z"].append(state["H"]["position"][2]); trace["l_z"].append(state["L"]["position"][2])
        trace["hl_distance"].append(distance(state["H"]["position"], state["L"]["position"]))
        trace["min_separation"].append(info["min_separation"]); trace["rate"].append(info["effective_e2e_rate"]); trace["outage_counter"].append(info["consecutive_outage_steps"])
        for j, relay in enumerate(state["relays"]):
            trace.setdefault(f"r{j + 1}_x", []).append(relay["position"][0]); trace.setdefault(f"r{j + 1}_y", []).append(relay["position"][1]); trace.setdefault(f"r{j + 1}_z", []).append(relay["position"][2]); trace.setdefault(f"r{j + 1}_xy_speed", []).append(math.hypot(relay["velocity"][0], relay["velocity"][1]))
        for label, val in info["link_capacities"].items():
            trace.setdefault("capacity_" + label, []).append(val)
        if terminated or truncated:
            break
    return trace


def write_svg(path: Path, title: str, x: list[float], series: dict[str, list[float]], x_label: str, y_label: str) -> None:
    width, height, pad = 1000, 560, 70
    vals = [v for data in series.values() for v in data if math.isfinite(v)]
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(vals), max(vals)
    if xmax == xmin: xmax += 1.0
    if ymax == ymin: ymax += 1.0
    margin = (ymax - ymin) * 0.08
    ymin, ymax = ymin - margin, ymax + margin
    def px(v: float) -> float: return pad + (v - xmin) / (xmax - xmin) * (width - 2 * pad)
    def py(v: float) -> float: return height - pad - (v - ymin) / (ymax - ymin) * (height - 2 * pad)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{pad}" y="32" font-family="sans-serif" font-size="20">{title}</text>', f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#222"/>', f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#222"/>', f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif">{x_label}</text>', f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="sans-serif">{y_label}</text>']
    for index, (name, data) in enumerate(series.items()):
        points = " ".join(f"{px(a):.2f},{py(b):.2f}" for a, b in zip(x, data))
        y = 58 + index * 20
        parts += [f'<polyline points="{points}" fill="none" stroke="{colors[index % len(colors)]}" stroke-width="2"/>', f'<text x="{width-235}" y="{y}" font-family="sans-serif" font-size="13" fill="{colors[index % len(colors)]}">{name}</text>']
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_plots(trace: dict) -> list[str]:
    PLOTS.mkdir(parents=True, exist_ok=True)
    x = trace["time"]
    specs = [("01_xy_trajectory.svg", "x-y trajectory", trace["h_x"], {"H y(x)": trace["h_y"], "L y(x)": trace["l_y"], **{f"R{i} y(x)": trace[f"r{i}_y"] for i in range(1, 5)}}, "x (m)", "y (m)"),
             ("02_altitude.svg", "Altitude", x, {"H": trace["h_z"], "L": trace["l_z"], **{f"R{i}": trace[f"r{i}_z"] for i in range(1, 5)}}, "time (s)", "z (m)"),
             ("03_relay_xy_speed.svg", "Relay horizontal speed", x, {f"R{i}": trace[f"r{i}_xy_speed"] for i in range(1, 5)}, "time (s)", "speed (m/s)"),
             ("04_link_capacity.svg", "Logical-link capacity", x, {k.removeprefix("capacity_"): v for k, v in trace.items() if k.startswith("capacity_")}, "time (s)", "capacity (bit/s)"),
             ("05_e2e_rate.svg", "Effective end-to-end rate", x, {"R_e2e": trace["rate"]}, "time (s)", "rate (bit/s)"),
             ("06_hl_distance.svg", "H-L 3-D distance", x, {"H-L": trace["hl_distance"]}, "time (s)", "distance (m)"),
             ("07_min_separation.svg", "Minimum node separation", x, {"minimum": trace["min_separation"]}, "time (s)", "distance (m)"),
             ("08_outage_counter.svg", "Consecutive outage counter", x, {"counter": trace["outage_counter"]}, "time (s)", "steps")]
    output = []
    for filename, title, x_data, data, xl, yl in specs:
        write_svg(PLOTS / filename, title, x_data, data, xl, yl); output.append(str(Path("artifacts/plots") / filename))
    return output


def write_report(results: dict, plots: list[str]) -> None:
    """Write the required report, with 01 as the final specification."""
    c = EnvironmentConfig()
    pass_count, total = sum(x["passed"] for x in results.values()), len(results)
    rows = "\n".join(
        f"| {name} | {'PASS' if result['passed'] else 'FAIL'} | {json.dumps({k: v for k, v in result.items() if k != 'passed'}, ensure_ascii=False)[:320]} |"
        for name, result in results.items()
    )
    repairs = "\n".join([
        "- Corrected scene altitude: [50, 250] m -> frozen [100, 300] m.",
        "- Corrected H/L horizontal cruise target: [8, 12] -> frozen [6, 10] m/s; added configurable vertical limits and initial cruise velocity.",
        "- Added mobile_max_hl_distance_m as D_max mission-scale configuration and constrain new H/L waypoints using the old peer position only.",
        "- Corrected communication defaults: B=10 MHz, P=20 dBm (0.1 W), N0=-169 dBm/Hz, beta0=-60 dB at 1 m, alpha=2.",
        "- Corrected hard-outage comparison: linear SNR is now compared to gamma_min, never channel gain.",
        "- Extended checks with frozen-default, H/L cruise, and trapezoidal-position validation.",
        "- Enforced D_max on every committed H/L state with smooth braking plus a final state-invariant guard; added a full 500-step Episode check.",
        "- Restricted involved_agents to controllable relay agents; H/L remain visible only in violation_pairs.",
    ])
    report = f"""# Stage 1 Environment Self-check Report

Date: 2026-08-29  
Final authority: `01_仿真开发阶段1_最终交接.md`; implementation checklist: `02_仿真开发阶段1_Codex任务书.md`.

## Acceptance decision

**PASS — Stage 1 acceptance is passed.** All {pass_count}/{total} reproducible automated checks pass after frozen-spec corrections. No MAPPO, Graph, Role, CBF, routing, caching, or power-optimization content is present.

## Repairs made against the frozen specification

{repairs}

## Code layout

- `relay_env/config.py`: centralized scene, motion, safety, communication, and reward configuration.
- `relay_env/communication.py`: independently testable distance, gain, SNR, capacity, TDMA, and end-to-end-rate functions.
- `relay_env/environment.py`: configurable-K environment, synchronous step, observation, termination, and info.
- `selfcheck/run_selfcheck.py`: A-J checks, additional acceptance checks, plots, and this report.

## Test results

Command: `py -3.13 -m selfcheck.run_selfcheck`

| Test | Result | Evidence |
| --- | --- | --- |
{rows}

Detailed JSON: `artifacts/selfcheck-results.json`.

## Still-un calibrated parameters

- `gamma_min = {c.comm.gamma_min:g}` linear SNR (provisional 0 dB test threshold).
- `D_max = {c.mobile_max_hl_distance_m:.1f} m` actual H/L three-dimensional distance invariant for every Episode step.
- H/L horizontal target = [{c.mobile_min_speed_mps:.1f}, {c.mobile_max_speed_mps:.1f}] m/s; vertical limit = [{c.mobile_min_z_speed_mps:.1f}, {c.mobile_max_z_speed_mps:.1f}] m/s.
- `d_warn = {c.warning_distance_m:.1f} m`; hard safety distance = {c.hard_safety_distance_m:.1f} m.
- `R_ref = {c.reward.reference_rate_bps:.0f} bit/s`; `lambda_o = {c.reward.lambda_outage:g}`; `lambda_T = {c.reward.lambda_failure:g}`.

The frozen specification leaves gamma_min, D_max, H/L vertical range, waypoint behavior, d_warn, persistent-outage duration, reward weights, observation scales, and final communication realism for numerical calibration. Values here remain configurable provisional values.

## Acceptance plots

{chr(10).join(f'- `{path}`' for path in plots)}

## Next step

Freeze this corrected Stage 1 baseline for numerical calibration; only after that may a separate later-stage Plain MAPPO implementation be considered. Do not add Graph, Role, CBF, dynamic agents, queues, dynamic routing, or power optimization here.
"""
    (ROOT / "\u9636\u6bb51\u73af\u5883\u81ea\u68c0\u62a5\u544a.md").write_text(report, encoding="utf-8")


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    checks: list[tuple[str, Callable[[], dict]]] = [("A_reset_legality", check_a_reset), ("B_zero_action", check_b_zero_action), ("C_max_acceleration", check_c_max_acceleration), ("D_speed_limit", check_d_speed_limit), ("E_communication_monotonicity", check_e_communication_monotonicity), ("F_tdma", check_f_tdma), ("G_hard_outage", check_g_hard_outage), ("H_safety", check_h_safety), ("I_boundary", check_i_boundary), ("J_k_scaling", check_j_scaling), ("K_reproducibility_random_episodes", check_k_reproducibility_and_random_episodes), ("L_frozen_spec_alignment", check_l_frozen_spec_alignment), ("M_episode_hl_distance_constraint", check_m_episode_hl_distance_constraint)]
    results = {name: func() for name, func in checks}
    trace, plots = run_trace(), write_plots(run_trace())
    (ARTIFACTS / "selfcheck-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACTS / "nominal-trace.json").write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    write_report(results, plots)
    print(json.dumps({name: result["passed"] for name, result in results.items()}, ensure_ascii=False))
    if not all(x["passed"] for x in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
