from __future__ import annotations

import math
from typing import Any

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.utils.helpers import (
    db_to_linear_power,
    ensure_3d_position,
    euclidean_distance_3d,
    noise_power_watts,
    shannon_capacity_bps,
)


def free_space_path_loss_db(distance_m: float, carrier_freq_hz: float = config.CARRIER_FREQ) -> float:
    distance_m = max(float(distance_m), 1.0)
    return 20.0 * math.log10((4.0 * math.pi * distance_m * float(carrier_freq_hz)) / config.LIGHT_SPEED)


def received_power_dbw(
    tx_power_w: float,
    path_loss_db: float,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = 0.0,
    additional_loss_db: float = 0.0,
) -> float:
    tx_power_dbw = 10.0 * math.log10(max(float(tx_power_w), config.EPSILON))
    return tx_power_dbw + float(tx_gain_db) + float(rx_gain_db) - float(path_loss_db) - float(additional_loss_db)


def satellite_backhaul_capacity_bps(
    anchor_position: np.ndarray,
    satellite_position: np.ndarray = np.asarray(config.SAT_POSITION, dtype=float),
    tx_power_w: float = config.P_TX_UAV,
    bandwidth_hz: float = config.B_SAT,
    tx_gain_db: float = config.G_TX_DB,
    rx_gain_db: float = config.G_RX_SAT_DB,
    atmospheric_loss_db: float = config.L_ATM_DB,
    noise_figure_db: float = config.NF_SAT,
) -> float:
    distance_m = euclidean_distance_3d(
        ensure_3d_position(anchor_position, default_z=config.UAV_HEIGHT),
        ensure_3d_position(satellite_position, default_z=config.SAT_ALTITUDE),
    )
    path_loss_db = free_space_path_loss_db(distance_m)
    rx_power_dbw = received_power_dbw(
        tx_power_w=tx_power_w,
        path_loss_db=path_loss_db,
        tx_gain_db=tx_gain_db,
        rx_gain_db=rx_gain_db,
        additional_loss_db=atmospheric_loss_db,
    )
    rx_power_w = db_to_linear_power(rx_power_dbw)
    sinr_linear_value = rx_power_w / noise_power_watts(bandwidth_hz=bandwidth_hz, noise_figure_db=noise_figure_db)
    return shannon_capacity_bps(bandwidth_hz=bandwidth_hz, sinr_linear=sinr_linear_value)


def gbs_backhaul_capacity_bps(
    anchor_position: np.ndarray,
    gbs_position: np.ndarray,
    tx_power_w: float = config.P_TX_UAV,
    bandwidth_hz: float = config.GBS_BW,
    tx_gain_db: float = 0.0,
    rx_gain_db: float = config.G_RX_GBS_DB,
    noise_figure_db: float = config.NF_GBS,
) -> float:
    distance_m = euclidean_distance_3d(
        ensure_3d_position(anchor_position, default_z=config.UAV_HEIGHT),
        ensure_3d_position(gbs_position, default_z=0.0),
    )
    path_loss_db = free_space_path_loss_db(distance_m)
    rx_power_dbw = received_power_dbw(
        tx_power_w=tx_power_w,
        path_loss_db=path_loss_db,
        tx_gain_db=tx_gain_db,
        rx_gain_db=rx_gain_db,
    )
    rx_power_w = db_to_linear_power(rx_power_dbw)
    sinr_linear_value = rx_power_w / noise_power_watts(bandwidth_hz=bandwidth_hz, noise_figure_db=noise_figure_db)
    return shannon_capacity_bps(bandwidth_hz=bandwidth_hz, sinr_linear=sinr_linear_value)


def backhaul_capacity_bps(
    anchor_position: np.ndarray,
    backhaul_node: Any,
    backhaul_type: str | None = None,
) -> float:
    node_type = (backhaul_type or getattr(backhaul_node, "backhaul_type", "")).lower()
    position = getattr(backhaul_node, "position", backhaul_node)

    if node_type == "satellite" or backhaul_node.__class__.__name__.lower() == "satellite":
        return satellite_backhaul_capacity_bps(
            anchor_position=anchor_position,
            satellite_position=position,
            bandwidth_hz=getattr(backhaul_node, "bandwidth_hz", config.B_SAT),
            rx_gain_db=getattr(backhaul_node, "rx_gain_db", config.G_RX_SAT_DB),
            atmospheric_loss_db=getattr(backhaul_node, "atmospheric_loss_db", config.L_ATM_DB),
            noise_figure_db=getattr(backhaul_node, "noise_figure_db", config.NF_SAT),
        )

    if node_type == "gbs" or backhaul_node.__class__.__name__.lower() == "groundbasestation":
        return gbs_backhaul_capacity_bps(
            anchor_position=anchor_position,
            gbs_position=position,
            bandwidth_hz=getattr(backhaul_node, "bandwidth_hz", config.GBS_BW),
            rx_gain_db=getattr(backhaul_node, "rx_gain_db", config.G_RX_GBS_DB),
            noise_figure_db=getattr(backhaul_node, "noise_figure_db", config.NF_GBS),
        )

    raise ValueError("Unsupported backhaul node type. Expected Satellite or GroundBaseStation.")
