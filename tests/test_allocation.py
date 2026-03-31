from __future__ import annotations

import math

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.allocation import associate_users_to_uavs, run_access_pf_step, run_pf_slot
from rl_uavnetsim.entities import GroundUser, UAV


def test_user_association_picks_strongest_feasible_uav() -> None:
    uav_near = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    uav_far = UAV(
        id=1,
        position=np.array([500.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    user = GroundUser(
        id=10,
        position=np.array([20.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
    )

    association_result = associate_users_to_uavs([user], [uav_near, uav_far])

    assert association_result.associated_uav_id_by_user[user.id] == uav_near.id
    assert user.associated_uav_id == uav_near.id
    assert user.id in uav_near.associated_user_ids
    assert association_result.upper_bound_rate_bps_by_user_and_uav[(user.id, uav_near.id)] > association_result.upper_bound_rate_bps_by_user_and_uav[(user.id, uav_far.id)]


def test_user_association_can_leave_user_unassociated() -> None:
    uav = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    user = GroundUser(
        id=11,
        position=np.array([1000.0, 1000.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
    )

    association_result = associate_users_to_uavs([user], [uav], min_rate_bps=1e12)

    assert association_result.associated_uav_id_by_user[user.id] == -1
    assert user.associated_uav_id == -1
    assert uav.associated_user_ids == []


def test_pf_slot_respects_backlog_and_updates_ema() -> None:
    uav = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    user = GroundUser(
        id=20,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        user_access_backlog_bits=5_000.0,
        avg_throughput_bps=100.0,
    )
    user.associated_uav_id = uav.id
    uav.associated_user_ids = [user.id]

    slot_result = run_pf_slot(
        uav=uav,
        users=[user],
        slot_duration_s=0.1,
        num_subchannels=4,
        alpha=1.0,
    )

    assert slot_result.total_served_bits <= 5_000.0
    assert sum(slot_result.assigned_user_id_by_subchannel.values()) == user.id * len(slot_result.assigned_user_id_by_subchannel)
    assert slot_result.served_bits_by_user[user.id] == 5_000.0
    assert slot_result.updated_avg_throughput_bps_by_user[user.id] > user.avg_throughput_bps


def test_access_resource_manager_moves_bits_to_relay_queue() -> None:
    uav = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    user = GroundUser(
        id=21,
        position=np.array([10.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        user_access_backlog_bits=8_000.0,
    )
    user.associated_uav_id = uav.id
    uav.associated_user_ids = [user.id]

    access_step_result = run_access_pf_step(
        uavs=[uav],
        users=[user],
        num_slots_per_step=1,
        num_subchannels=2,
        slot_duration_s=0.1,
    )

    assert math.isclose(access_step_result.total_access_uploaded_bits_by_user[user.id], 8_000.0)
    assert math.isclose(access_step_result.total_access_ingress_bits_by_uav[uav.id], 8_000.0)
    assert math.isclose(user.user_access_backlog_bits, 0.0)
    assert math.isclose(user.access_uploaded_bits_step, 8_000.0)
    assert math.isclose(uav.access_ingress_bits_step, 8_000.0)
    assert math.isclose(uav.relay_queue_bits_by_user[user.id], 8_000.0)
    assert math.isclose(uav.relay_queue_total_bits, 8_000.0)
