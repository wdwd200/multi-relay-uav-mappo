"""Focused Stage 2.5 tests for the elevation-dependent dipole link model."""

from __future__ import annotations

import math
import unittest

from relay_env import EnvironmentConfig
from relay_env.communication import link_metrics


def metrics(a: tuple[float, float, float], b: tuple[float, float, float]) -> dict[str, float]:
    c = EnvironmentConfig().comm
    return link_metrics(a, b, c.reference_gain, c.path_loss_exponent,
                        c.tx_power_w, c.noise_density_w_hz, c.bandwidth_hz)


class ElevationDipoleTests(unittest.TestCase):
    def test_same_height_degenerates_to_distance_only_channel(self) -> None:
        result = metrics((0.0, 0.0, 150.0), (300.0, 400.0, 150.0))
        self.assertEqual(result["elevation_angle_rad"], 0.0)
        self.assertEqual(result["antenna_gain_linear"], 1.0)
        self.assertAlmostEqual(result["channel_gain"], result["old_channel_gain"], places=18)
        self.assertAlmostEqual(result["snr_linear"], result["old_snr_linear"], places=12)

    def test_elevation_is_monotonic_and_gain_decreases(self) -> None:
        rows = [metrics((0.0, 0.0, 100.0), (300.0, 0.0, 100.0 + dz)) for dz in (0, 50, 100, 150, 200)]
        self.assertTrue(all(rows[i]["elevation_angle_rad"] < rows[i + 1]["elevation_angle_rad"] for i in range(len(rows) - 1)))
        self.assertTrue(all(rows[i]["antenna_gain_linear"] > rows[i + 1]["antenna_gain_linear"] for i in range(len(rows) - 1)))

    def test_height_exchange_is_symmetric(self) -> None:
        forward = metrics((100.0, 10.0, 120.0), (400.0, 10.0, 260.0))
        reverse = metrics((100.0, 10.0, 260.0), (400.0, 10.0, 120.0))
        self.assertAlmostEqual(forward["antenna_gain_linear"], reverse["antenna_gain_linear"], places=15)
        self.assertAlmostEqual(forward["channel_gain"], reverse["channel_gain"], places=24)

    def test_known_angles(self) -> None:
        horizontal = metrics((0.0, 0.0, 100.0), (100.0, 0.0, 100.0))
        forty_five = metrics((0.0, 0.0, 100.0), (100.0, 0.0, 200.0))
        self.assertAlmostEqual(horizontal["antenna_gain_linear"], 1.0, places=15)
        self.assertAlmostEqual(forty_five["elevation_angle_deg"], 45.0, places=12)
        self.assertAlmostEqual(forty_five["antenna_gain_linear"], 0.5, places=12)
        self.assertAlmostEqual(forty_five["antenna_gain_db"], -3.01029995664, places=9)

    def test_vertical_link_remains_finite(self) -> None:
        result = metrics((200.0, 200.0, 100.0), (200.0, 200.0, 250.0))
        self.assertAlmostEqual(result["elevation_angle_deg"], 90.0, places=12)
        self.assertLess(result["antenna_gain_linear"], 1e-30)
        self.assertTrue(all(math.isfinite(value) for value in result.values()))
        self.assertGreaterEqual(result["capacity_bps"], 0.0)


if __name__ == "__main__":
    unittest.main()
