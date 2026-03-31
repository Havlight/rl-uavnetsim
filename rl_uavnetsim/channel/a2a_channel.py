from __future__ import annotations

import math

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.utils.helpers import ensure_3d_position, euclidean_distance_3d, linear_to_db, noise_power_watts, shannon_capacity_bps


def a2a_channel_gain_linear(
    position_a: np.ndarray,
    position_b: np.ndarray,
    rho_0: float = config.RHO_0,
) -> float:
    distance_m = euclidean_distance_3d(
        ensure_3d_position(position_a, default_z=config.UAV_HEIGHT),
        ensure_3d_position(position_b, default_z=config.UAV_HEIGHT),
    )
    distance_m = max(distance_m, 1.0)
    return float(rho_0) / (distance_m ** 2)


def a2a_noise_power_w(
    bandwidth_hz: float = config.A2A_BW,
    noise_figure_db: float = config.NF_A2A,
) -> float:
    return noise_power_watts(bandwidth_hz=bandwidth_hz, noise_figure_db=noise_figure_db)


def a2a_snr_linear(
    position_a: np.ndarray,
    position_b: np.ndarray,
    transmit_power_w: float = config.P_TX_UAV,
) -> float:
    signal_power_w = float(transmit_power_w) * a2a_channel_gain_linear(position_a, position_b)
    return signal_power_w / a2a_noise_power_w()


def a2a_snr_db(
    position_a: np.ndarray,
    position_b: np.ndarray,
    transmit_power_w: float = config.P_TX_UAV,
) -> float:
    return linear_to_db(a2a_snr_linear(position_a, position_b, transmit_power_w=transmit_power_w))


def a2a_capacity_bps(
    position_a: np.ndarray,
    position_b: np.ndarray,
    bandwidth_hz: float = config.A2A_BW,
    transmit_power_w: float = config.P_TX_UAV,
) -> float:
    snr_linear_value = a2a_snr_linear(position_a, position_b, transmit_power_w=transmit_power_w)
    return shannon_capacity_bps(bandwidth_hz=bandwidth_hz, sinr_linear=snr_linear_value)


def a2a_link_is_active(
    position_a: np.ndarray,
    position_b: np.ndarray,
    threshold_db: float = config.GAMMA_TH_DB,
) -> bool:
    return a2a_snr_db(position_a, position_b) >= float(threshold_db)
