from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rl_uavnetsim import config
from rl_uavnetsim.channel.a2g_channel import a2g_upper_bound_rate_bps
from rl_uavnetsim.entities.ground_user import GroundUser
from rl_uavnetsim.entities.uav import UAV


@dataclass
class AssociationResult:
    associated_uav_id_by_user: dict[int, int]
    upper_bound_rate_bps_by_user_and_uav: dict[tuple[int, int], float]


def select_strongest_feasible_uav(
    user: GroundUser,
    uavs: Sequence[UAV],
    *,
    min_rate_bps: float = config.R_MIN,
    num_subchannels: int = config.NUM_SUBCHANNELS,
) -> tuple[int, dict[int, float]]:
    best_uav_id = -1
    best_upper_bound_rate_bps = -1.0
    upper_bound_rate_bps_by_uav_id: dict[int, float] = {}

    for uav in uavs:
        upper_bound_rate_bps = a2g_upper_bound_rate_bps(
            uav_position=uav.position,
            user_position=user.position,
            num_subchannels=num_subchannels,
        )
        upper_bound_rate_bps_by_uav_id[uav.id] = upper_bound_rate_bps

        if upper_bound_rate_bps < float(min_rate_bps):
            continue

        if upper_bound_rate_bps > best_upper_bound_rate_bps:
            best_uav_id = uav.id
            best_upper_bound_rate_bps = upper_bound_rate_bps

    return best_uav_id, upper_bound_rate_bps_by_uav_id


def associate_users_to_uavs(
    users: Sequence[GroundUser],
    uavs: Sequence[UAV],
    *,
    min_rate_bps: float = config.R_MIN,
    num_subchannels: int = config.NUM_SUBCHANNELS,
) -> AssociationResult:
    associated_uav_id_by_user: dict[int, int] = {}
    upper_bound_rate_bps_by_user_and_uav: dict[tuple[int, int], float] = {}

    for uav in uavs:
        uav.associated_user_ids = []

    for user in users:
        selected_uav_id, upper_bound_rate_bps_by_uav_id = select_strongest_feasible_uav(
            user=user,
            uavs=uavs,
            min_rate_bps=min_rate_bps,
            num_subchannels=num_subchannels,
        )
        user.associated_uav_id = selected_uav_id
        associated_uav_id_by_user[user.id] = selected_uav_id

        for uav_id, upper_bound_rate_bps in upper_bound_rate_bps_by_uav_id.items():
            upper_bound_rate_bps_by_user_and_uav[(user.id, uav_id)] = upper_bound_rate_bps

        if selected_uav_id >= 0:
            for uav in uavs:
                if uav.id == selected_uav_id:
                    uav.associated_user_ids.append(user.id)
                    break

    return AssociationResult(
        associated_uav_id_by_user=associated_uav_id_by_user,
        upper_bound_rate_bps_by_user_and_uav=upper_bound_rate_bps_by_user_and_uav,
    )
