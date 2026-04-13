from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rl_uavnetsim import config
from rl_uavnetsim.allocation.pf_scheduler import InterferencePowerProvider, PFSlotResult, run_pf_slot
from rl_uavnetsim.channel.a2g_channel import a2g_sinr_linear, a2g_subchannel_rate_bps, channel_gain_linear
from rl_uavnetsim.utils.helpers import linear_to_db
from rl_uavnetsim.entities.ground_user import GroundUser
from rl_uavnetsim.entities.uav import UAV


@dataclass
class AccessStepResult:
    slot_results: list[PFSlotResult] = field(default_factory=list)
    alpha_by_uav: dict[int, float] = field(default_factory=dict)
    total_access_uploaded_bits_by_user: dict[int, float] = field(default_factory=dict)
    total_access_ingress_bits_by_uav: dict[int, float] = field(default_factory=dict)

    @property
    def total_access_uploaded_bits(self) -> float:
        return float(sum(self.total_access_uploaded_bits_by_user.values()))


def _processing_order_for_slot(uavs: Sequence[UAV], slot_index: int) -> list[UAV]:
    ordered_uavs = sorted(uavs, key=lambda uav: uav.id)
    if not ordered_uavs:
        return []
    offset = int(slot_index) % len(ordered_uavs)
    return ordered_uavs[offset:] + ordered_uavs[:offset]


def _interference_power_w(
    *,
    target_user: GroundUser,
    interfering_uavs: Sequence[UAV],
) -> float:
    return float(
        sum(
            float(config.P_TX_RF) * channel_gain_linear(interfering_uav.position, target_user.position)
            for interfering_uav in interfering_uavs
        )
    )


def _select_pf_alpha(
    uav: UAV,
    *,
    alpha_by_uav: Mapping[int, float] | None = None,
    linucb_controllers: Mapping[int, Any] | None = None,
    context_by_uav: Mapping[int, Any] | None = None,
) -> float:
    if alpha_by_uav is not None and uav.id in alpha_by_uav:
        return float(alpha_by_uav[uav.id])

    if linucb_controllers is not None and uav.id in linucb_controllers:
        context = None if context_by_uav is None else context_by_uav.get(uav.id)
        return float(linucb_controllers[uav.id].select_alpha(context))

    return float(config.PF_ALPHA_DEFAULT)


def run_access_pf_step(
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
    *,
    num_slots_per_step: int = config.NUM_SLOTS_PER_STEP,
    slot_duration_s: float = config.SLOT_DURATION,
    num_subchannels: int = config.NUM_SUBCHANNELS,
    subchannel_bandwidth_hz: float = config.SUBCHANNEL_BW,
    sinr_threshold_db: float = config.SINR_THRESHOLD_DB,
    beta_pf: float = config.PF_BETA,
    epsilon: float = config.PF_EPSILON,
    alpha_by_uav: Mapping[int, float] | None = None,
    linucb_controllers: Mapping[int, Any] | None = None,
    context_by_uav: Mapping[int, Any] | None = None,
    interference_power_provider: InterferencePowerProvider | None = None,
) -> AccessStepResult:
    users_by_id = {user.id: user for user in users}
    uavs_by_id = {uav.id: uav for uav in uavs}
    result = AccessStepResult(
        total_access_uploaded_bits_by_user={user.id: 0.0 for user in users},
        total_access_ingress_bits_by_uav={uav.id: 0.0 for uav in uavs},
    )

    for slot_index in range(int(num_slots_per_step)):
        tentative_remaining_backlog_bits_by_user = {
            user.id: max(0.0, float(user.user_access_backlog_bits)) for user in users
        }
        tentative_assignments_by_uav_id: dict[int, dict[int, int]] = {
            uav.id: {} for uav in uavs
        }
        ordered_uavs = _processing_order_for_slot(uavs, slot_index)

        for uav in ordered_uavs:
            alpha = _select_pf_alpha(
                uav=uav,
                alpha_by_uav=alpha_by_uav,
                linucb_controllers=linucb_controllers,
                context_by_uav=context_by_uav,
            )
            result.alpha_by_uav[uav.id] = alpha
            associated_users = [
                users_by_id[user_id]
                for user_id in uav.associated_user_ids
                if user_id in users_by_id and users_by_id[user_id].associated_uav_id == uav.id
            ]

            for subchannel_index in range(int(num_subchannels)):
                best_user: GroundUser | None = None
                best_score = -1.0
                best_rate_bps = 0.0

                interfering_uavs = [
                    uavs_by_id[other_uav_id]
                    for other_uav_id, assignment_by_subchannel in tentative_assignments_by_uav_id.items()
                    if other_uav_id != uav.id and subchannel_index in assignment_by_subchannel
                ]

                for user in associated_users:
                    if tentative_remaining_backlog_bits_by_user[user.id] <= config.EPSILON:
                        continue

                    interference_power_w = _interference_power_w(
                        target_user=user,
                        interfering_uavs=interfering_uavs,
                    )
                    if interference_power_provider is not None:
                        interference_power_w += max(
                            0.0,
                            float(interference_power_provider(uav, user, subchannel_index)),
                        )

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
                    if score > best_score or (
                        abs(score - best_score) <= config.EPSILON
                        and (
                            instantaneous_rate_bps > best_rate_bps
                            or (
                                abs(instantaneous_rate_bps - best_rate_bps) <= config.EPSILON
                                and best_user is not None
                                and user.id < best_user.id
                            )
                        )
                    ):
                        best_user = user
                        best_score = score
                        best_rate_bps = instantaneous_rate_bps

                if best_user is None:
                    continue

                tentative_assignments_by_uav_id[uav.id][subchannel_index] = best_user.id
                tentative_raw_slot_bits = best_rate_bps * float(slot_duration_s)
                tentative_remaining_backlog_bits_by_user[best_user.id] = max(
                    0.0,
                    tentative_remaining_backlog_bits_by_user[best_user.id] - tentative_raw_slot_bits,
                )

        slot_results_by_uav_id: dict[int, PFSlotResult] = {}
        for uav in uavs:
            alpha = result.alpha_by_uav.get(uav.id, _select_pf_alpha(
                uav=uav,
                alpha_by_uav=alpha_by_uav,
                linucb_controllers=linucb_controllers,
                context_by_uav=context_by_uav,
            ))
            slot_result = PFSlotResult(uav_id=uav.id, slot_index=slot_index, alpha=float(alpha))
            slot_results_by_uav_id[uav.id] = slot_result

            for subchannel_index, user_id in tentative_assignments_by_uav_id[uav.id].items():
                user = users_by_id[user_id]
                interfering_uavs = [
                    uavs_by_id[other_uav_id]
                    for other_uav_id, assignment_by_subchannel in tentative_assignments_by_uav_id.items()
                    if other_uav_id != uav.id and subchannel_index in assignment_by_subchannel
                ]
                interference_power_w = _interference_power_w(
                    target_user=user,
                    interfering_uavs=interfering_uavs,
                )
                if interference_power_provider is not None:
                    interference_power_w += max(
                        0.0,
                        float(interference_power_provider(uav, user, subchannel_index)),
                    )
                sinr_linear_value = a2g_sinr_linear(
                    uav_position=uav.position,
                    user_position=user.position,
                    interference_power_w=interference_power_w,
                    bandwidth_hz=subchannel_bandwidth_hz,
                )
                slot_result.assigned_user_id_by_subchannel[subchannel_index] = user_id
                slot_result.sinr_linear_by_user_and_subchannel[(user_id, subchannel_index)] = sinr_linear_value
                if linear_to_db(sinr_linear_value) < float(sinr_threshold_db):
                    continue
                slot_result.raw_rate_bps_by_user[user_id] = slot_result.raw_rate_bps_by_user.get(user_id, 0.0) + a2g_subchannel_rate_bps(
                    uav_position=uav.position,
                    user_position=user.position,
                    interference_power_w=interference_power_w,
                    bandwidth_hz=subchannel_bandwidth_hz,
                )

            for user_id, raw_rate_bps in slot_result.raw_rate_bps_by_user.items():
                served_bits = min(
                    float(users_by_id[user_id].user_access_backlog_bits),
                    raw_rate_bps * float(slot_duration_s),
                )
                slot_result.served_bits_by_user[user_id] = served_bits
                slot_result.served_rate_bps_by_user[user_id] = served_bits / float(slot_duration_s)

            for user_id in uav.associated_user_ids:
                if user_id not in users_by_id or users_by_id[user_id].associated_uav_id != uav.id:
                    continue
                served_rate_bps = slot_result.served_rate_bps_by_user.get(user_id, 0.0)
                slot_result.updated_avg_throughput_bps_by_user[user_id] = (
                    (1.0 - float(beta_pf)) * users_by_id[user_id].avg_throughput_bps
                    + float(beta_pf) * served_rate_bps
                )

        for uav in uavs:
            slot_result = slot_results_by_uav_id[uav.id]
            result.slot_results.append(slot_result)
            for user_id, updated_avg_throughput_bps in slot_result.updated_avg_throughput_bps_by_user.items():
                users_by_id[user_id].avg_throughput_bps = updated_avg_throughput_bps
            for user_id, served_bits in slot_result.served_bits_by_user.items():
                if served_bits <= config.EPSILON:
                    continue
                uploaded_bits = users_by_id[user_id].consume_access_bits(served_bits)
                uav.enqueue_relay_bits(user_id, uploaded_bits, count_as_access_ingress=True)
                result.total_access_uploaded_bits_by_user[user_id] += uploaded_bits
                result.total_access_ingress_bits_by_uav[uav.id] += uploaded_bits

    return result
