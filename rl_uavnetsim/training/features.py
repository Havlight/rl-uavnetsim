from __future__ import annotations

from typing import Sequence

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.entities import GroundUser, UAV
from rl_uavnetsim.rl_interface.mdp import _normalize_uav_velocity, _safe_norm
from rl_uavnetsim.scenario import ScenarioGeometry
from rl_uavnetsim.utils.helpers import euclidean_distance_2d


def compact_observation_dim(num_uavs: int, max_obs_users: int) -> int:
    return 7 + 5 * max(int(num_uavs) - 1, 0) + 2 * int(max_obs_users)


def compact_v2_observation_dim(num_uavs: int, max_obs_users: int) -> int:
    return 8 + 5 * max(int(num_uavs) - 1, 0) + 4 * int(max_obs_users)


def compact_state_dim(num_uavs: int, num_users: int) -> int:
    return 7 * int(num_uavs) + 2 * int(num_users)


def build_compact_local_observation(
    uav: UAV,
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
    *,
    max_obs_users: int = 15,
    obs_radius_m: float = config.OBS_RADIUS,
    geometry: ScenarioGeometry | None = None,
) -> np.ndarray:
    geometry = geometry or ScenarioGeometry()
    self_features = np.array(
        [
            *geometry.normalize_uav_position(uav.position).tolist(),
            *_normalize_uav_velocity(uav.velocity).tolist(),
            float(uav.is_gateway_capable),
            _safe_norm(uav.relay_queue_total_bits, config.RELAY_QUEUE_REF_BITS),
        ],
        dtype=np.float32,
    )

    other_uav_features: list[float] = []
    for other_uav in sorted(uavs, key=lambda item: item.id):
        if other_uav.id == uav.id:
            continue
        other_uav_features.extend(geometry.normalize_uav_position(other_uav.position).tolist())
        other_uav_features.extend(_normalize_uav_velocity(other_uav.velocity).tolist())

    visible_users = sorted(
        [
            user
            for user in users
            if euclidean_distance_2d(uav.position, user.position) <= float(obs_radius_m)
        ],
        key=lambda user: (euclidean_distance_2d(uav.position, user.position), user.id),
    )[: int(max_obs_users)]

    visible_user_features: list[float] = []
    for user in visible_users:
        visible_user_features.extend(geometry.normalize_user_position(user.position).tolist())

    num_padding_users = int(max_obs_users) - len(visible_users)
    if num_padding_users > 0:
        visible_user_features.extend([0.0] * (2 * num_padding_users))

    return np.concatenate(
        [
            self_features,
            np.asarray(other_uav_features, dtype=np.float32),
            np.asarray(visible_user_features, dtype=np.float32),
        ],
        dtype=np.float32,
    )


def build_compact_v2_local_observation(
    uav: UAV,
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
    *,
    max_obs_users: int = 15,
    obs_radius_m: float = config.OBS_RADIUS,
    geometry: ScenarioGeometry | None = None,
) -> np.ndarray:
    geometry = geometry or ScenarioGeometry()
    self_features = np.array(
        [
            *geometry.normalize_uav_position(uav.position).tolist(),
            *_normalize_uav_velocity(uav.velocity).tolist(),
            float(uav.is_gateway_capable),
            _safe_norm(uav.relay_queue_total_bits, config.RELAY_QUEUE_REF_BITS),
            _safe_norm(len(uav.associated_user_ids), max(len(users), 1)),
        ],
        dtype=np.float32,
    )

    other_uav_features: list[float] = []
    for other_uav in sorted(uavs, key=lambda item: item.id):
        if other_uav.id == uav.id:
            continue
        other_uav_features.extend(geometry.normalize_uav_position(other_uav.position).tolist())
        other_uav_features.extend(_normalize_uav_velocity(other_uav.velocity).tolist())

    visible_users = sorted(
        [
            user
            for user in users
            if euclidean_distance_2d(uav.position, user.position) <= float(obs_radius_m)
        ],
        key=lambda user: (euclidean_distance_2d(uav.position, user.position), user.id),
    )[: int(max_obs_users)]

    visible_user_features: list[float] = []
    for user in visible_users:
        visible_user_features.extend(geometry.normalize_user_position(user.position).tolist())
        visible_user_features.append(_safe_norm(user.user_access_backlog_bits, config.ACCESS_BACKLOG_REF_BITS))
        visible_user_features.append(float(user.associated_uav_id == uav.id))

    num_padding_users = int(max_obs_users) - len(visible_users)
    if num_padding_users > 0:
        visible_user_features.extend([0.0] * (4 * num_padding_users))

    return np.concatenate(
        [
            self_features,
            np.asarray(other_uav_features, dtype=np.float32),
            np.asarray(visible_user_features, dtype=np.float32),
        ],
        dtype=np.float32,
    )


def build_compact_state(
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
    *,
    geometry: ScenarioGeometry | None = None,
) -> np.ndarray:
    geometry = geometry or ScenarioGeometry()
    ordered_uavs = sorted(uavs, key=lambda item: item.id)

    uav_geometry: list[float] = []
    for uav in ordered_uavs:
        uav_geometry.extend(geometry.normalize_uav_position(uav.position).tolist())
        uav_geometry.extend(_normalize_uav_velocity(uav.velocity).tolist())

    gateway_capable_mask = [float(uav.is_gateway_capable) for uav in ordered_uavs]
    relay_queue_norms = [_safe_norm(uav.relay_queue_total_bits, config.RELAY_QUEUE_REF_BITS) for uav in ordered_uavs]

    user_positions: list[float] = []
    for user in sorted(users, key=lambda item: item.id):
        user_positions.extend(geometry.normalize_user_position(user.position).tolist())

    return np.asarray(
        [*uav_geometry, *gateway_capable_mask, *relay_queue_norms, *user_positions],
        dtype=np.float32,
    )
