from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rl_uavnetsim import config
from rl_uavnetsim.allocation.pf_scheduler import InterferencePowerProvider, PFSlotResult, run_pf_slot
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
    result = AccessStepResult(
        total_access_uploaded_bits_by_user={user.id: 0.0 for user in users},
        total_access_ingress_bits_by_uav={uav.id: 0.0 for uav in uavs},
    )

    for slot_index in range(int(num_slots_per_step)):
        for uav in uavs:
            alpha = _select_pf_alpha(
                uav=uav,
                alpha_by_uav=alpha_by_uav,
                linucb_controllers=linucb_controllers,
                context_by_uav=context_by_uav,
            )
            result.alpha_by_uav[uav.id] = alpha
            slot_result = run_pf_slot(
                uav=uav,
                users=users_by_id,
                slot_index=slot_index,
                alpha=alpha,
                slot_duration_s=slot_duration_s,
                num_subchannels=num_subchannels,
                subchannel_bandwidth_hz=subchannel_bandwidth_hz,
                sinr_threshold_db=sinr_threshold_db,
                beta_pf=beta_pf,
                epsilon=epsilon,
                interference_power_provider=interference_power_provider,
            )
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
