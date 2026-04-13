from __future__ import annotations

import math

import numpy as np
import pytest

from rl_uavnetsim import config
from rl_uavnetsim.entities import GroundBaseStation, GroundUser, UAV
from rl_uavnetsim.environment import SimEnv
from rl_uavnetsim.network import compute_routing_table, execute_backhaul_service, execute_relay_service


def test_compute_routing_prefers_less_loaded_gateway_when_capacity_ties() -> None:
    gateway_busy = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
        relay_queue_bits_by_user={99: 500.0},
    )
    gateway_idle = UAV(
        id=1,
        position=np.array([10.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    source = UAV(
        id=2,
        position=np.array([20.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    capacity_matrix_bps = np.array(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0],
            [10.0, 10.0, 0.0],
        ],
        dtype=float,
    )

    routing_table = compute_routing_table(
        uavs=[gateway_busy, gateway_idle, source],
        active_gateway_uav_ids=[gateway_busy.id, gateway_idle.id],
        capacity_matrix_bps=capacity_matrix_bps,
        backhaul_capacity_bps_by_gateway={
            gateway_busy.id: 10.0,
            gateway_idle.id: 10.0,
        },
    )

    assert routing_table[source.id].selected_gateway_uav_id == gateway_idle.id
    assert routing_table[source.id].next_hop_uav_id == gateway_idle.id


def test_relay_service_only_moves_bits_one_hop_per_step() -> None:
    gateway = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    relay = UAV(
        id=1,
        position=np.array([10.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    source = UAV(
        id=2,
        position=np.array([20.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    source.enqueue_relay_bits(7, 1000.0, count_as_access_ingress=False)
    capacity_matrix_bps = np.array(
        [
            [0.0, 1000.0, 0.0],
            [1000.0, 0.0, 1000.0],
            [0.0, 1000.0, 0.0],
        ],
        dtype=float,
    )

    relay_result = execute_relay_service(
        [gateway, relay, source],
        active_gateway_uav_ids=[gateway.id],
        delta_t_s=1.0,
        capacity_matrix_bps=capacity_matrix_bps,
        backhaul_capacity_bps_by_gateway={gateway.id: 1000.0},
    )

    assert relay_result.relay_out_bits_by_uav[source.id] == 1000.0
    assert math.isclose(relay.relay_queue_bits_by_user[7], 1000.0)
    assert math.isclose(gateway.relay_queue_total_bits, 0.0)


def test_backhaul_service_aggregates_deliveries_from_multiple_gateways() -> None:
    gateway_a = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    gateway_b = UAV(
        id=1,
        position=np.array([10.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    user = GroundUser(
        id=3,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
    )
    gateway_a.enqueue_relay_bits(user.id, 600.0, count_as_access_ingress=False)
    gateway_b.enqueue_relay_bits(user.id, 400.0, count_as_access_ingress=False)

    backhaul_result = execute_backhaul_service(
        gateway_uavs=[gateway_a, gateway_b],
        users=[user],
        backhaul_capacity_bps_by_gateway={
            gateway_a.id: 600.0,
            gateway_b.id: 400.0,
        },
        delta_t_s=1.0,
    )

    assert math.isclose(backhaul_result.backhaul_out_bits, 1000.0)
    assert math.isclose(user.delivered_bits_step, 1000.0)
    assert math.isclose(gateway_a.relay_queue_total_bits, 0.0)
    assert math.isclose(gateway_b.relay_queue_total_bits, 0.0)


def test_sim_env_gbs_only_activates_gateway_within_range() -> None:
    gateway_near = UAV(
        id=0,
        position=np.array(config.GBS_POSITIONS[0], dtype=float) + np.array([50.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    gateway_far = UAV(
        id=1,
        position=np.array([config.MAP_LENGTH, config.MAP_WIDTH, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    relay_only = UAV(
        id=2,
        position=np.array([config.MAP_LENGTH / 2.0, config.MAP_WIDTH / 2.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    env = SimEnv(
        uavs=[gateway_near, gateway_far, relay_only],
        users=[],
        ground_base_stations=[GroundBaseStation(id=0)],
        gateway_capable_uav_ids=[gateway_near.id, gateway_far.id],
        backhaul_type="gbs",
    )

    step_result = env.step(
        relay_capacity_matrix_bps=np.zeros((3, 3), dtype=float),
    )

    assert step_result.env_state.active_gateway_uav_ids == (gateway_near.id,)
    assert step_result.env_state.backhaul_capacity_bps_by_gateway[gateway_near.id] > 0.0
    assert step_result.env_state.backhaul_capacity_bps_by_gateway[gateway_far.id] == 0.0


def test_sim_env_rejects_scalar_backhaul_override_with_multiple_gateways() -> None:
    gateway_a = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    gateway_b = UAV(
        id=1,
        position=np.array([100.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    env = SimEnv(
        uavs=[gateway_a, gateway_b],
        users=[],
        ground_base_stations=[GroundBaseStation(id=0)],
        gateway_capable_uav_ids=[gateway_a.id, gateway_b.id],
        backhaul_type="gbs",
    )

    with pytest.raises(ValueError):
        env.step(backhaul_capacity_bps_override=1000.0)
