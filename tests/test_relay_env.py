from __future__ import annotations

import math

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.entities import GroundUser, Satellite, UAV
from rl_uavnetsim.environment import SimEnv
from rl_uavnetsim.network import execute_backhaul_service, execute_relay_service, find_widest_path_to_anchor


def test_relay_service_moves_bits_from_member_to_anchor() -> None:
    anchor = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_anchor=True,
    )
    member = UAV(
        id=1,
        position=np.array([100.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    member.enqueue_relay_bits(9, 1000.0, count_as_access_ingress=False)

    relay_capacity_matrix_bps = np.array([[0.0, 1000.0], [1000.0, 0.0]], dtype=float)
    relay_result = execute_relay_service(
        [anchor, member],
        anchor_uav_id=anchor.id,
        delta_t_s=1.0,
        capacity_matrix_bps=relay_capacity_matrix_bps,
    )

    assert relay_result.relay_out_bits_by_uav[member.id] == 1000.0
    assert anchor.relay_queue_bits_by_user[9] == 1000.0
    assert member.relay_queue_total_bits == 0.0
    assert relay_result.lambda2 > 0.0


def test_backhaul_service_delivers_anchor_queue_bits_to_users() -> None:
    anchor = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_anchor=True,
    )
    user = GroundUser(
        id=15,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
    )
    anchor.enqueue_relay_bits(user.id, 1500.0, count_as_access_ingress=False)

    backhaul_result = execute_backhaul_service(anchor, [user], backhaul_capacity_bps=1200.0, delta_t_s=1.0)

    assert backhaul_result.backhaul_out_bits == 1200.0
    assert math.isclose(user.delivered_bits_step, 1200.0)
    assert math.isclose(user.final_rate_bps, 1200.0)
    assert math.isclose(anchor.relay_queue_total_bits, 300.0)


def test_widest_path_prefers_higher_bottleneck_route() -> None:
    uavs = [
        UAV(id=0, position=np.array([0.0, 0.0, config.UAV_HEIGHT]), velocity=np.zeros(2), speed=0.0, direction=0.0, is_anchor=True),
        UAV(id=1, position=np.array([1.0, 0.0, config.UAV_HEIGHT]), velocity=np.zeros(2), speed=0.0, direction=0.0),
        UAV(id=2, position=np.array([2.0, 0.0, config.UAV_HEIGHT]), velocity=np.zeros(2), speed=0.0, direction=0.0),
    ]
    capacity_matrix_bps = np.array(
        [
            [0.0, 5.0, 7.0],
            [5.0, 0.0, 5.0],
            [7.0, 5.0, 0.0],
        ],
        dtype=float,
    )

    relay_path = find_widest_path_to_anchor(2, 0, uavs, capacity_matrix_bps)

    assert relay_path.path_uav_ids == [2, 0]
    assert relay_path.bottleneck_capacity_bps == 7.0


def test_sim_env_step_preserves_plan35_flow_conservation() -> None:
    anchor = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_anchor=True,
    )
    member = UAV(
        id=1,
        position=np.array([50.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    member.enqueue_relay_bits(30, 2000.0, count_as_access_ingress=False)
    anchor.enqueue_relay_bits(30, 1000.0, count_as_access_ingress=False)

    user = GroundUser(
        id=30,
        position=np.array([55.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        demand_rate_bps=100.0,
        user_access_backlog_bits=500.0,
    )
    satellite = Satellite(id=0)
    env = SimEnv(
        uavs=[anchor, member],
        users=[user],
        satellites=[satellite],
        anchor_uav_id=anchor.id,
        backhaul_type="satellite",
    )

    relay_capacity_matrix_bps = np.array([[0.0, 1000.0], [1000.0, 0.0]], dtype=float)
    step_result = env.step(
        relay_capacity_matrix_bps=relay_capacity_matrix_bps,
        backhaul_capacity_bps_override=1200.0,
    )

    accounting = step_result.accounting
    access_uploaded_bits = step_result.access_step_result.total_access_uploaded_bits_by_user[user.id]
    relay_out_bits = step_result.relay_service_result.relay_out_bits_by_uav[member.id]
    backhaul_out_bits = step_result.backhaul_service_result.backhaul_out_bits

    assert math.isclose(
        accounting.user_access_backlog_next_bits_by_user[user.id],
        accounting.user_access_backlog_prev_bits_by_user[user.id] + accounting.arrived_bits_by_user[user.id] - access_uploaded_bits,
    )
    assert math.isclose(
        accounting.member_queue_next_bits_by_uav[member.id],
        accounting.member_queue_prev_bits_by_uav[member.id]
        + step_result.access_step_result.total_access_ingress_bits_by_uav[member.id]
        - relay_out_bits,
    )
    assert math.isclose(
        accounting.anchor_queue_next_bits,
        accounting.anchor_queue_prev_bits
        + accounting.anchor_access_ingress_bits
        + step_result.relay_service_result.total_relay_in_bits_to_anchor
        - backhaul_out_bits,
    )
    assert math.isclose(env.users[0].final_rate_bps, env.users[0].delivered_bits_step / config.DELTA_T)
    assert step_result.env_state.total_delivered_bits_step == env.users[0].delivered_bits_step
