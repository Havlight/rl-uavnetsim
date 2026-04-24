from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.rl_interface.mdp import _safe_norm


@dataclass(frozen=True)
class ScenarioGeometry:
    map_length_m: float = config.MAP_LENGTH
    map_width_m: float = config.MAP_WIDTH

    def __post_init__(self) -> None:
        if self.map_length_m <= 0.0 or self.map_width_m <= 0.0:
            raise ValueError("Scenario map dimensions must be positive.")

    @property
    def x_bounds_m(self) -> tuple[float, float]:
        return (0.0, float(self.map_length_m))

    @property
    def y_bounds_m(self) -> tuple[float, float]:
        return (0.0, float(self.map_width_m))

    @property
    def center_xy(self) -> np.ndarray:
        return np.asarray([self.map_length_m / 2.0, self.map_width_m / 2.0], dtype=float)

    @property
    def satellite_position(self) -> np.ndarray:
        return np.asarray([self.center_xy[0], self.center_xy[1], config.SAT_ALTITUDE], dtype=float)

    @property
    def ground_base_station_position(self) -> np.ndarray:
        return np.asarray([0.05 * self.map_length_m, 0.05 * self.map_width_m, 0.0], dtype=float)

    def normalize_uav_position(self, position: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                _safe_norm(float(position[0]), self.map_length_m),
                _safe_norm(float(position[1]), self.map_width_m),
                _safe_norm(float(position[2]), config.UAV_HEIGHT),
            ],
            dtype=float,
        )

    def normalize_user_position(self, position: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                _safe_norm(float(position[0]), self.map_length_m),
                _safe_norm(float(position[1]), self.map_width_m),
            ],
            dtype=float,
        )
