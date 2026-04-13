from __future__ import annotations

import math

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.allocation import associate_users_to_uavs, run_access_pf_step
from rl_uavnetsim.entities import GroundUser, UAV


def test_access_step_applies_cross_uav_cochannel_interference() -> None:
    uav_left = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    uav_right = UAV(
        id=1,
        position=np.array([50.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    user_left = GroundUser(
        id=0,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        user_access_backlog_bits=1_000_000.0,
    )
    user_right = GroundUser(
        id=1,
        position=np.array([50.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        user_access_backlog_bits=1_000_000.0,
    )

    user_left.associated_uav_id = uav_left.id
    user_right.associated_uav_id = uav_right.id
    uav_left.associated_user_ids = [user_left.id]
    uav_right.associated_user_ids = [user_right.id]

    access_step_result = run_access_pf_step(
        uavs=[uav_left, uav_right],
        users=[user_left, user_right],
        num_slots_per_step=1,
        num_subchannels=1,
        slot_duration_s=0.1,
    )

    assert access_step_result.total_access_uploaded_bits < 2_000_000.0
    assert user_left.user_access_backlog_bits > 0.0 or user_right.user_access_backlog_bits > 0.0


def test_association_uses_load_aware_proxy_to_split_equal_users() -> None:
    uav_left = UAV(
        id=0,
        position=np.array([200.0, 200.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    uav_right = UAV(
        id=1,
        position=np.array([200.0, 200.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    user_first = GroundUser(
        id=0,
        position=np.array([210.0, 200.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        user_access_backlog_bits=20_000.0,
    )
    user_second = GroundUser(
        id=1,
        position=np.array([210.0, 200.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        user_access_backlog_bits=10_000.0,
    )

    association_result = associate_users_to_uavs(
        users=[user_first, user_second],
        uavs=[uav_left, uav_right],
    )

    assert association_result.associated_uav_id_by_user[user_first.id] == uav_left.id
    assert association_result.associated_uav_id_by_user[user_second.id] == uav_right.id
    assert uav_left.associated_user_ids == [user_first.id]
    assert uav_right.associated_user_ids == [user_second.id]
