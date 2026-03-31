from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.mobility.base_mobility import BaseMobilityModel, MobilityState
from rl_uavnetsim.utils.helpers import clamp, ensure_2d_velocity, ensure_3d_position


@dataclass
class RandomWalkMobility(BaseMobilityModel):
    x_bounds_m: tuple[float, float] = (0.0, config.MAP_LENGTH)
    y_bounds_m: tuple[float, float] = (0.0, config.MAP_WIDTH)
    speed_mean_mps: float = config.USER_SPEED_MEAN
    speed_max_mps: float = config.USER_SPEED_MAX
    direction_sigma_rad: float = config.USER_DIR_SIGMA
    fixed_altitude_m: float = 0.0

    def step(
        self,
        *,
        position: np.ndarray,
        velocity: np.ndarray,
        speed: float,
        delta_t_s: float,
        rng: Optional[np.random.Generator] = None,
    ) -> MobilityState:
        rng = rng or np.random.default_rng()
        position = ensure_3d_position(position, default_z=self.fixed_altitude_m)
        velocity = ensure_2d_velocity(velocity)
        delta_t_s = float(delta_t_s)

        if np.linalg.norm(velocity) > 0.0:
            direction_rad = math.atan2(float(velocity[1]), float(velocity[0]))
        else:
            direction_rad = float(rng.uniform(-math.pi, math.pi))

        direction_rad += float(rng.normal(0.0, self.direction_sigma_rad))
        speed_mps = clamp(float(speed) if speed > 0.0 else self.speed_mean_mps, 0.0, self.speed_max_mps)
        next_velocity = np.array(
            [speed_mps * math.cos(direction_rad), speed_mps * math.sin(direction_rad)],
            dtype=float,
        )
        next_position = position.copy()
        next_position[:2] = next_position[:2] + next_velocity * delta_t_s

        next_position[0], next_velocity[0] = self._reflect(next_position[0], next_velocity[0], self.x_bounds_m)
        next_position[1], next_velocity[1] = self._reflect(next_position[1], next_velocity[1], self.y_bounds_m)
        next_position[2] = self.fixed_altitude_m

        return MobilityState(
            position=next_position,
            velocity=next_velocity,
            speed=float(np.linalg.norm(next_velocity)),
        )

    @staticmethod
    def _reflect(
        coordinate_m: float,
        velocity_component_mps: float,
        bounds_m: tuple[float, float],
    ) -> tuple[float, float]:
        lower_bound_m, upper_bound_m = bounds_m
        while coordinate_m < lower_bound_m or coordinate_m > upper_bound_m:
            if coordinate_m < lower_bound_m:
                coordinate_m = lower_bound_m + (lower_bound_m - coordinate_m)
                velocity_component_mps *= -1.0
            elif coordinate_m > upper_bound_m:
                coordinate_m = upper_bound_m - (coordinate_m - upper_bound_m)
                velocity_component_mps *= -1.0
        return coordinate_m, velocity_component_mps
