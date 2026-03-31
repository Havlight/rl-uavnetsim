from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from rl_uavnetsim import config
from rl_uavnetsim.channel.a2g_channel import a2g_sinr_linear, a2g_subchannel_rate_bps
from rl_uavnetsim.entities.ground_user import GroundUser
from rl_uavnetsim.entities.uav import UAV
from rl_uavnetsim.utils.helpers import linear_to_db

InterferencePowerProvider = Callable[[UAV, GroundUser, int], float]


@dataclass
class PFSlotResult:
    uav_id: int
    slot_index: int
    alpha: float
    assigned_user_id_by_subchannel: dict[int, int] = field(default_factory=dict)
    sinr_linear_by_user_and_subchannel: dict[tuple[int, int], float] = field(default_factory=dict)
    raw_rate_bps_by_user: dict[int, float] = field(default_factory=dict)
    served_rate_bps_by_user: dict[int, float] = field(default_factory=dict)
    served_bits_by_user: dict[int, float] = field(default_factory=dict)
    updated_avg_throughput_bps_by_user: dict[int, float] = field(default_factory=dict)

    @property
    def total_served_bits(self) -> float:
        return float(sum(self.served_bits_by_user.values()))


def _normalize_user_mapping(users: Sequence[GroundUser] | Mapping[int, GroundUser]) -> dict[int, GroundUser]:
    if isinstance(users, Mapping):
        return dict(users)
    return {user.id: user for user in users}


def run_pf_slot(
    uav: UAV,
    users: Sequence[GroundUser] | Mapping[int, GroundUser],
    *,
    slot_index: int = 0,
    alpha: float = config.PF_ALPHA_DEFAULT,
    slot_duration_s: float = config.SLOT_DURATION,
    num_subchannels: int = config.NUM_SUBCHANNELS,
    subchannel_bandwidth_hz: float = config.SUBCHANNEL_BW,
    sinr_threshold_db: float = config.SINR_THRESHOLD_DB,
    beta_pf: float = config.PF_BETA,
    epsilon: float = config.PF_EPSILON,
    interference_power_provider: InterferencePowerProvider | None = None,
) -> PFSlotResult:
    users_by_id = _normalize_user_mapping(users)
    associated_users = [
        users_by_id[user_id]
        for user_id in uav.associated_user_ids
        if user_id in users_by_id and users_by_id[user_id].associated_uav_id == uav.id
    ]

    result = PFSlotResult(uav_id=uav.id, slot_index=slot_index, alpha=float(alpha))
    if not associated_users:
        return result

    remaining_backlog_bits_by_user = {
        user.id: max(0.0, user.user_access_backlog_bits) for user in associated_users
    }
    raw_rate_bps_by_user = {user.id: 0.0 for user in associated_users}
    served_rate_bps_by_user = {user.id: 0.0 for user in associated_users}
    served_bits_by_user = {user.id: 0.0 for user in associated_users}

    for subchannel_index in range(int(num_subchannels)):
        best_user: GroundUser | None = None
        best_score = -1.0
        best_raw_rate_bps = 0.0
        best_sinr_linear = 0.0

        for user in associated_users:
            if remaining_backlog_bits_by_user[user.id] <= config.EPSILON:
                continue

            interference_power_w = 0.0
            if interference_power_provider is not None:
                interference_power_w = max(0.0, float(interference_power_provider(uav, user, subchannel_index)))

            sinr_linear_value = a2g_sinr_linear(
                uav_position=uav.position,
                user_position=user.position,
                interference_power_w=interference_power_w,
                bandwidth_hz=subchannel_bandwidth_hz,
            )
            if linear_to_db(sinr_linear_value) < float(sinr_threshold_db):
                continue

            instantaneous_rate_bps = a2g_subchannel_rate_bps(
                uav_position=uav.position,
                user_position=user.position,
                interference_power_w=interference_power_w,
                bandwidth_hz=subchannel_bandwidth_hz,
            )
            denominator = max(user.avg_throughput_bps + float(epsilon), float(epsilon)) ** float(alpha)
            score = instantaneous_rate_bps / denominator

            if best_user is None:
                should_select = True
            elif score > best_score:
                should_select = True
            elif math.isclose(score, best_score) and instantaneous_rate_bps > best_raw_rate_bps:
                should_select = True
            elif (
                math.isclose(score, best_score)
                and math.isclose(instantaneous_rate_bps, best_raw_rate_bps)
                and user.id < best_user.id
            ):
                should_select = True
            else:
                should_select = False

            if should_select:
                best_user = user
                best_score = score
                best_raw_rate_bps = instantaneous_rate_bps
                best_sinr_linear = sinr_linear_value

        if best_user is None:
            continue

        raw_slot_bits = best_raw_rate_bps * float(slot_duration_s)
        served_bits = min(raw_slot_bits, remaining_backlog_bits_by_user[best_user.id])
        if served_bits <= config.EPSILON:
            continue

        remaining_backlog_bits_by_user[best_user.id] -= served_bits
        raw_rate_bps_by_user[best_user.id] += best_raw_rate_bps
        served_bits_by_user[best_user.id] += served_bits
        served_rate_bps_by_user[best_user.id] += served_bits / float(slot_duration_s)
        result.assigned_user_id_by_subchannel[subchannel_index] = best_user.id
        result.sinr_linear_by_user_and_subchannel[(best_user.id, subchannel_index)] = best_sinr_linear

    for user in associated_users:
        served_rate_bps = served_rate_bps_by_user[user.id]
        result.updated_avg_throughput_bps_by_user[user.id] = (
            (1.0 - float(beta_pf)) * user.avg_throughput_bps + float(beta_pf) * served_rate_bps
        )

    result.raw_rate_bps_by_user = raw_rate_bps_by_user
    result.served_rate_bps_by_user = served_rate_bps_by_user
    result.served_bits_by_user = served_bits_by_user
    return result
