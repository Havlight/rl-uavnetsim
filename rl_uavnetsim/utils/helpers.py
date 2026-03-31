from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from rl_uavnetsim import config


def clamp(value: float, lower_bound: float, upper_bound: float) -> float:
    return max(float(lower_bound), min(float(value), float(upper_bound)))


def ensure_3d_position(position: Iterable[float], default_z: float = 0.0) -> np.ndarray:
    position_array = np.asarray(position, dtype=float)
    if position_array.shape == (2,):
        position_array = np.array([position_array[0], position_array[1], float(default_z)], dtype=float)
    if position_array.shape != (3,):
        raise ValueError("Position must be a length-2 or length-3 vector.")
    return position_array.astype(float, copy=True)


def ensure_2d_velocity(velocity: Iterable[float]) -> np.ndarray:
    velocity_array = np.asarray(velocity, dtype=float)
    if velocity_array.shape == (3,):
        velocity_array = velocity_array[:2]
    if velocity_array.shape != (2,):
        raise ValueError("Velocity must be a length-2 or length-3 vector.")
    return velocity_array.astype(float, copy=True)


def euclidean_distance_2d(point_a: Iterable[float], point_b: Iterable[float]) -> float:
    point_a_array = ensure_3d_position(point_a)
    point_b_array = ensure_3d_position(point_b)
    return float(np.linalg.norm(point_a_array[:2] - point_b_array[:2]))


def euclidean_distance_3d(point_a: Iterable[float], point_b: Iterable[float]) -> float:
    point_a_array = ensure_3d_position(point_a)
    point_b_array = ensure_3d_position(point_b)
    return float(np.linalg.norm(point_a_array - point_b_array))


def db_to_linear_power(value_db: float) -> float:
    return 10.0 ** (float(value_db) / 10.0)


def linear_to_db(value_linear: float) -> float:
    return 10.0 * math.log10(max(float(value_linear), config.EPSILON))


def noise_power_watts(bandwidth_hz: float, noise_figure_db: float, n0_w_per_hz: float = config.N0) -> float:
    return float(n0_w_per_hz) * float(bandwidth_hz) * db_to_linear_power(noise_figure_db)


def shannon_capacity_bps(bandwidth_hz: float, sinr_linear: float) -> float:
    return float(bandwidth_hz) * math.log2(1.0 + max(float(sinr_linear), 0.0))
