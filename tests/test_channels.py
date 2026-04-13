from __future__ import annotations

import math

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.channel.a2a_channel import a2a_capacity_bps, a2a_link_is_active, a2a_snr_db
from rl_uavnetsim.channel.a2g_channel import (
    a2g_sinr_linear,
    a2g_subchannel_rate_bps,
    average_path_loss_db,
    elevation_angle_deg,
    los_probability,
)
from rl_uavnetsim.channel.backhaul_channel import backhaul_capacity_bps, gbs_backhaul_capacity_bps, satellite_backhaul_capacity_bps
from rl_uavnetsim.energy import SimplifiedEnergyModel, Zeng2019EnergyModel
from rl_uavnetsim.entities import GroundBaseStation, Satellite


def test_a2g_los_probability_and_path_loss_are_geometry_sensitive() -> None:
    user_position = np.array([0.0, 0.0, 0.0])
    near_uav_position = np.array([0.0, 0.0, 100.0])
    far_uav_position = np.array([500.0, 0.0, 100.0])

    near_elevation_deg = elevation_angle_deg(near_uav_position, user_position)
    far_elevation_deg = elevation_angle_deg(far_uav_position, user_position)

    assert near_elevation_deg > far_elevation_deg
    assert los_probability(near_elevation_deg) > los_probability(far_elevation_deg)
    assert average_path_loss_db(near_uav_position, user_position) < average_path_loss_db(far_uav_position, user_position)


def test_a2g_rate_is_positive_and_interference_reduces_sinr() -> None:
    uav_position = np.array([0.0, 0.0, 100.0])
    user_position = np.array([50.0, 0.0, 0.0])

    clean_sinr_linear = a2g_sinr_linear(uav_position, user_position, interference_power_w=0.0)
    interfered_sinr_linear = a2g_sinr_linear(uav_position, user_position, interference_power_w=1e-8)
    rate_bps = a2g_subchannel_rate_bps(uav_position, user_position)

    assert clean_sinr_linear > interfered_sinr_linear
    assert rate_bps > 0.0


def test_a2a_capacity_decreases_with_distance() -> None:
    anchor_position = np.array([0.0, 0.0, 100.0])
    near_member_position = np.array([10.0, 0.0, 100.0])
    far_member_position = np.array([200.0, 0.0, 100.0])

    near_capacity_bps = a2a_capacity_bps(anchor_position, near_member_position)
    far_capacity_bps = a2a_capacity_bps(anchor_position, far_member_position)

    assert near_capacity_bps > far_capacity_bps
    assert a2a_snr_db(anchor_position, near_member_position) > a2a_snr_db(anchor_position, far_member_position)
    assert isinstance(a2a_link_is_active(anchor_position, near_member_position), bool)


def test_a2a_link_is_active_respects_max_range_cap() -> None:
    source_position = np.array([0.0, 0.0, 100.0])
    target_position = np.array([config.MAX_RELAY_RANGE_M + 50.0, 0.0, 100.0])

    assert a2a_snr_db(source_position, target_position) > config.GAMMA_TH_DB
    assert a2a_link_is_active(source_position, target_position) is False


def test_satellite_and_gbs_backhaul_entities_are_usable() -> None:
    anchor_position = np.array([100.0, 100.0, config.UAV_HEIGHT])
    satellite = Satellite(id=0)
    gbs = GroundBaseStation(id=0)

    sat_capacity_bps = satellite_backhaul_capacity_bps(anchor_position, satellite.position)
    gbs_capacity_bps = gbs_backhaul_capacity_bps(anchor_position, gbs.position)

    assert sat_capacity_bps > 0.0
    assert gbs_capacity_bps > 0.0
    assert math.isclose(backhaul_capacity_bps(anchor_position, satellite), sat_capacity_bps)
    assert math.isclose(backhaul_capacity_bps(anchor_position, gbs), gbs_capacity_bps)


def test_energy_models_return_finite_positive_values() -> None:
    simplified_energy_model = SimplifiedEnergyModel()
    zeng_energy_model = Zeng2019EnergyModel()

    assert math.isclose(
        simplified_energy_model.step_energy_j(speed_mps=10.0, delta_t_s=1.0, distance_m=10.0),
        config.E_HOVER + config.E_FLY * 10.0,
    )
    assert zeng_energy_model.power_consumption_w(0.0) > 0.0
    assert zeng_energy_model.power_consumption_w(15.0) > 0.0
