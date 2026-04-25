from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rl_uavnetsim import config
from rl_uavnetsim.channel.a2g_channel import a2g_upper_bound_rate_bps
from rl_uavnetsim.entities.ground_user import GroundUser
from rl_uavnetsim.entities.uav import UAV
from rl_uavnetsim.utils.helpers import euclidean_distance_2d


@dataclass
class AssociationResult:
    associated_uav_id_by_user: dict[int, int]
    upper_bound_rate_bps_by_user_and_uav: dict[tuple[int, int], float]
    proxy_rate_bps_by_user_and_uav: dict[tuple[int, int], float]
    projected_load_by_user_and_uav: dict[tuple[int, int], int]


def select_strongest_feasible_uav(
    user: GroundUser,
    uavs: Sequence[UAV],
    *,
    current_load_by_uav_id: dict[int, int] | None = None,
    min_rate_bps: float = config.R_MIN,
    max_access_range_m: float | None = None,
    num_subchannels: int = config.NUM_SUBCHANNELS,
) -> tuple[int, dict[int, float], dict[int, float], dict[int, int]]:
    best_uav_id = -1
    best_proxy_rate_bps = -1.0
    upper_bound_rate_bps_by_uav_id: dict[int, float] = {}
    proxy_rate_bps_by_uav_id: dict[int, float] = {}
    projected_load_by_uav_id: dict[int, int] = {}

    for uav in uavs:
        upper_bound_rate_bps = a2g_upper_bound_rate_bps(
            uav_position=uav.position,
            user_position=user.position,
            num_subchannels=num_subchannels,
        )
        upper_bound_rate_bps_by_uav_id[uav.id] = upper_bound_rate_bps
        projected_load = int((current_load_by_uav_id or {}).get(uav.id, 0)) + 1
        projected_load_by_uav_id[uav.id] = projected_load
        proxy_rate_bps = upper_bound_rate_bps / max(projected_load, 1)
        proxy_rate_bps_by_uav_id[uav.id] = proxy_rate_bps

        if max_access_range_m is not None and euclidean_distance_2d(uav.position, user.position) > float(max_access_range_m):
            continue

        if proxy_rate_bps < float(min_rate_bps):
            continue

        if proxy_rate_bps > best_proxy_rate_bps:
            best_uav_id = uav.id
            best_proxy_rate_bps = proxy_rate_bps

    return best_uav_id, upper_bound_rate_bps_by_uav_id, proxy_rate_bps_by_uav_id, projected_load_by_uav_id


def associate_users_to_uavs(
    users: Sequence[GroundUser],
    uavs: Sequence[UAV],
    *,
    min_rate_bps: float = config.R_MIN,
    max_access_range_m: float | None = None,
    num_subchannels: int = config.NUM_SUBCHANNELS,
) -> AssociationResult:
    associated_uav_id_by_user: dict[int, int] = {}
    upper_bound_rate_bps_by_user_and_uav: dict[tuple[int, int], float] = {}
    proxy_rate_bps_by_user_and_uav: dict[tuple[int, int], float] = {}
    projected_load_by_user_and_uav: dict[tuple[int, int], int] = {}

    for uav in uavs:
        uav.associated_user_ids = []

    current_load_by_uav_id = {uav.id: 0 for uav in uavs}
    ordered_users = sorted(
        users,
        key=lambda user: (-float(user.user_access_backlog_bits), user.id),
    )

    for user in ordered_users:
        (
            selected_uav_id,
            upper_bound_rate_bps_by_uav_id,
            proxy_rate_bps_by_uav_id,
            projected_load_by_uav_id,
        ) = select_strongest_feasible_uav(
            user=user,
            uavs=uavs,
            current_load_by_uav_id=current_load_by_uav_id,
            min_rate_bps=min_rate_bps,
            max_access_range_m=max_access_range_m,
            num_subchannels=num_subchannels,
        )
        user.associated_uav_id = selected_uav_id
        associated_uav_id_by_user[user.id] = selected_uav_id

        for uav_id, upper_bound_rate_bps in upper_bound_rate_bps_by_uav_id.items():
            upper_bound_rate_bps_by_user_and_uav[(user.id, uav_id)] = upper_bound_rate_bps
        for uav_id, proxy_rate_bps in proxy_rate_bps_by_uav_id.items():
            proxy_rate_bps_by_user_and_uav[(user.id, uav_id)] = proxy_rate_bps
        for uav_id, projected_load in projected_load_by_uav_id.items():
            projected_load_by_user_and_uav[(user.id, uav_id)] = projected_load

        if selected_uav_id >= 0:
            for uav in uavs:
                if uav.id == selected_uav_id:
                    uav.associated_user_ids.append(user.id)
                    current_load_by_uav_id[uav.id] += 1
                    break

    return AssociationResult(
        associated_uav_id_by_user=associated_uav_id_by_user,
        upper_bound_rate_bps_by_user_and_uav=upper_bound_rate_bps_by_user_and_uav,
        proxy_rate_bps_by_user_and_uav=proxy_rate_bps_by_user_and_uav,
        projected_load_by_user_and_uav=projected_load_by_user_and_uav,
    )
