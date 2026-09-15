# Stage 2 implementation validation: PASS

This result means only that the requested code modifications and statistics generation completed correctly. It does **not** declare Environment v1 frozen; that decision remains for human review.

## Candidate parameters applied

- gamma_min = 5 dB = 3.16227766016838 linear; D_max = 2000 m.
- H/L cruise target = [6, 10] m/s; z speed = [-3, 3] m/s; acceleration = 1 m/s².
- waypoint segment = [250, 450] m; d_warn = 60 m; R_ref = 6 Mbps.

## Modified and added files

- `relay_env/config.py`, `relay_env/environment.py`: candidate parameters, waypoint span, final H/L speed caps, and raw observation access.
- `relay_env/environment.py`: replaces post-step D_max position/velocity projection with synchronous, pairwise preventive bounded braking.
- `stage2_calibration.py`: independent three-mode calibration, hard checks, raw records, summaries, and this report.

## Stage 1 regression

- PASS — stage1 selfcheck process.

## Stage 2 hard checks

- gamma_min_is_5db_linear: PASS
- initial_hl_distance_in_range: PASS
- actual_hl_distance_within_dmax: PASS
- hl_xy_speed_capped: PASS
- hl_z_speed_capped: PASS
- hl_xy_acceleration_capped: PASS
- hl_z_acceleration_capped: PASS
- d_warn_is_60m: PASS
- d_safe_is_20m: PASS
- persistent_outage_is_25: PASS
- outage_masks_effective_rate: PASS
- recovery_resets_counter: PASS
- no_nan_or_inf: PASS
- seed_reproducibility: PASS
- k_smoke: PASS
- termination_truncation_semantics: PASS
- dmax_preventive_acceleration_stress: PASS

## D_max / acceleration stress test

- Three-mode regression scale: 20 seeds per mode, up to 500 steps each.
- Active-guard scenario: one forced near-D_max, maximum-outward-speed trajectory for 500 steps.
- Preventive guard interventions: 126 steps; maximum H-L distance: 1950 m.
- Maximum actual horizontal acceleration: 1 m/s²; vertical acceleration: 1 m/s².
- Maximum actual horizontal speed: 10 m/s; absolute vertical speed: 3 m/s.
- The stress episode terminated=False and truncated=True; no position projection or velocity overwrite is used.

## Mode summary

- nominal: reset=20, truncated=20, outage steps=0, warnings=0, mean effective R_e2e=7.938 Mbps, mean H-L distance=1190 m
- zero_action: reset=20, truncated=1, outage steps=450, warnings=258, mean effective R_e2e=6.632 Mbps, mean H-L distance=1177 m
- random_stress: reset=20, truncated=0, outage steps=180, warnings=39, mean effective R_e2e=6.837 Mbps, mean H-L distance=1232 m

## Exceptions and still-undecided parameters

- No automatic re-tuning was performed. gamma_min, D_max, vertical motion range, waypoint behaviour, warning range, reward scales, observation scales, and communication realism still require human review of the generated statistics.
- Stage1 subprocess tail: `skipped`

## Output data

- `stage2_summary.json`
- `stage2_episode_stats.csv`
- `stage2_hop_stats.csv`
- `stage2_observation_stats.csv`
- `stage2_step_records.jsonl` (disabled by default; pass --write-raw to enable)
