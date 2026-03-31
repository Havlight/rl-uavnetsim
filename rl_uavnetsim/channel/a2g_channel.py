from __future__ import annotations

import math

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.utils.helpers import (
    db_to_linear_power,
    ensure_3d_position,
    euclidean_distance_3d,
    noise_power_watts,
    shannon_capacity_bps,
)


def elevation_angle_deg(uav_position: np.ndarray, user_position: np.ndarray) -> float:
    uav_position = ensure_3d_position(uav_position, default_z=config.UAV_HEIGHT)
    user_position = ensure_3d_position(user_position, default_z=0.0)
    distance_m = euclidean_distance_3d(uav_position, user_position)
    if distance_m <= config.EPSILON:
        return 90.0
    vertical_distance_m = max(0.0, float(uav_position[2] - user_position[2]))
    vertical_ratio = min(1.0, max(0.0, vertical_distance_m / distance_m))
    return math.degrees(math.asin(vertical_ratio))


def los_probability(
    elevation_angle_deg_value: float,
    a_env: float = config.A_ENV,
    b_env: float = config.B_ENV,
) -> float:
    return 1.0 / (1.0 + float(a_env) * math.exp(-float(b_env) * (float(elevation_angle_deg_value) - float(a_env))))


def los_path_loss_db(
    distance_m: float,
    shadow_fading_db: float = config.SHADOW_STD,
    alpha_los_db: float = config.ALPHA_LOS,
    beta_los: float = config.BETA_LOS,
) -> float:
    distance_m = max(float(distance_m), 1.0)
    return float(alpha_los_db) + 10.0 * float(beta_los) * math.log10(distance_m) + float(shadow_fading_db)


def nlos_path_loss_db(
    distance_m: float,
    shadow_fading_db: float = config.SHADOW_STD,
    alpha_nlos_db: float = config.ALPHA_NLOS,
    beta_nlos: float = config.BETA_NLOS,
) -> float:
    distance_m = max(float(distance_m), 1.0)
    return float(alpha_nlos_db) + 10.0 * float(beta_nlos) * math.log10(distance_m) + float(shadow_fading_db)


def average_path_loss_db(
    uav_position: np.ndarray,
    user_position: np.ndarray,
    shadow_fading_db: float = config.SHADOW_STD,
) -> float:
    distance_m = euclidean_distance_3d(
        ensure_3d_position(uav_position, default_z=config.UAV_HEIGHT),
        ensure_3d_position(user_position, default_z=0.0),
    )
    elevation_deg = elevation_angle_deg(uav_position, user_position)
    los_prob = los_probability(elevation_deg)
    los_loss_db = los_path_loss_db(distance_m, shadow_fading_db=shadow_fading_db)
    nlos_loss_db = nlos_path_loss_db(distance_m, shadow_fading_db=shadow_fading_db)
    return los_prob * los_loss_db + (1.0 - los_prob) * nlos_loss_db


def channel_gain_linear(
    uav_position: np.ndarray,
    user_position: np.ndarray,
    shadow_fading_db: float = config.SHADOW_STD,
) -> float:
    return db_to_linear_power(-average_path_loss_db(uav_position, user_position, shadow_fading_db=shadow_fading_db))


def a2g_noise_power_w(
    bandwidth_hz: float = config.SUBCHANNEL_BW,
    noise_figure_db: float = config.NF_ACCESS,
) -> float:
    return noise_power_watts(bandwidth_hz=bandwidth_hz, noise_figure_db=noise_figure_db)


def a2g_sinr_linear(
    uav_position: np.ndarray,
    user_position: np.ndarray,
    interference_power_w: float = 0.0,
    transmit_power_w: float = config.P_TX_RF,
    bandwidth_hz: float = config.SUBCHANNEL_BW,
) -> float:
    signal_power_w = float(transmit_power_w) * channel_gain_linear(uav_position, user_position)
    noise_and_interference_w = a2g_noise_power_w(bandwidth_hz=bandwidth_hz) + max(0.0, float(interference_power_w))
    return signal_power_w / noise_and_interference_w


def a2g_subchannel_rate_bps(
    uav_position: np.ndarray,
    user_position: np.ndarray,
    interference_power_w: float = 0.0,
    bandwidth_hz: float = config.SUBCHANNEL_BW,
    transmit_power_w: float = config.P_TX_RF,
) -> float:
    sinr_linear_value = a2g_sinr_linear(
        uav_position=uav_position,
        user_position=user_position,
        interference_power_w=interference_power_w,
        transmit_power_w=transmit_power_w,
        bandwidth_hz=bandwidth_hz,
    )
    return shannon_capacity_bps(bandwidth_hz=bandwidth_hz, sinr_linear=sinr_linear_value)


def a2g_upper_bound_rate_bps(
    uav_position: np.ndarray,
    user_position: np.ndarray,
    num_subchannels: int = config.NUM_SUBCHANNELS,
) -> float:
    per_subchannel_rate_bps = a2g_subchannel_rate_bps(
        uav_position=uav_position,
        user_position=user_position,
        interference_power_w=0.0,
        bandwidth_hz=config.SUBCHANNEL_BW,
    )
    return int(num_subchannels) * per_subchannel_rate_bps
