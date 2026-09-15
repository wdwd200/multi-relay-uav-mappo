"""All physical, motion, safety and reward constants for stage 1."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommunicationConfig:
    """Frozen stage-1 path-loss communication model in SI units."""

    bandwidth_hz: float = 10_000_000.0
    tx_power_w: float = 0.1  # 20 dBm
    noise_density_w_hz: float = 1.2589254117941661e-20  # -169 dBm/Hz
    reference_gain: float = 1.0e-6  # -60 dB at 1 m
    path_loss_exponent: float = 2.0
    gamma_min: float = 10.0 ** (5.0 / 10.0)  # 5 dB, stored as linear SNR


@dataclass(frozen=True)
class RewardConfig:
    reference_rate_bps: float = 6_000_000.0
    lambda_outage: float = 1.0
    lambda_failure: float = 5.0
    warning_penalty: float = 0.2


@dataclass(frozen=True)
class EnvironmentConfig:
    """One source of truth for stage-1 environment parameters."""

    map_x_m: float = 2_000.0
    map_y_m: float = 2_000.0
    min_altitude_m: float = 100.0
    max_altitude_m: float = 300.0
    dt_s: float = 0.2
    max_episode_steps: int = 500
    initial_hl_distance_min_m: float = 1_000.0
    initial_hl_distance_max_m: float = 1_500.0
    relay_initial_jitter_m: float = 4.0
    reset_max_attempts: int = 200
    relay_max_xy_accel_mps2: float = 2.0
    relay_max_z_accel_mps2: float = 2.0
    relay_max_xy_speed_mps: float = 20.0
    relay_min_z_speed_mps: float = -5.0
    relay_max_z_speed_mps: float = 6.0
    mobile_min_speed_mps: float = 6.0
    mobile_max_speed_mps: float = 10.0
    mobile_min_z_speed_mps: float = -3.0
    mobile_max_z_speed_mps: float = 3.0
    mobile_max_accel_mps2: float = 1.0
    mobile_waypoint_reach_m: float = 4.0
    mobile_waypoint_margin_m: float = 80.0
    mobile_max_hl_distance_m: float = 2_000.0
    # Internal preventive margin (not D_max): leaves enough distance for two
    # 1 m/s² mobile nodes to cancel their worst permitted relative separation.
    mobile_hl_distance_guard_m: float = 250.0
    mobile_waypoint_max_attempts: int = 100
    mobile_waypoint_min_segment_m: float = 250.0
    mobile_waypoint_max_segment_m: float = 450.0
    hard_safety_distance_m: float = 20.0
    warning_distance_m: float = 60.0
    persistent_outage_steps: int = 25
    comm: CommunicationConfig = field(default_factory=CommunicationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
