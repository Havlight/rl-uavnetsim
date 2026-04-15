from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.allocation import associate_users_to_uavs
from rl_uavnetsim.channel.a2g_channel import a2g_sinr_linear
from rl_uavnetsim.entities import GroundUser, UAV
from rl_uavnetsim.environment import EnvState, SimEnv, SimStepResult
from rl_uavnetsim.network import (
    algebraic_connectivity_lambda2,
    build_a2a_capacity_matrix_bps,
    build_adjacency_matrix,
    compute_routing_table,
)
from rl_uavnetsim.rl_interface.linucb_stub import LinUCBStub
from rl_uavnetsim.utils.helpers import euclidean_distance_2d, euclidean_distance_3d


@dataclass(frozen=True)
class RewardReferenceScales:
    throughput_ref_bits: float = config.THROUGHPUT_REF_BITS
    energy_ref_j: float = config.ENERGY_REF_J
    access_backlog_ref_bits: float = config.ACCESS_BACKLOG_REF_BITS
    relay_queue_ref_bits: float = config.RELAY_QUEUE_REF_BITS


def _safe_norm(value: float, reference: float) -> float:
    return float(value) / max(float(reference), config.EPSILON)


def _bps_reference() -> float:
    return max(config.THROUGHPUT_REF_BITS / config.DELTA_T, config.EPSILON)


def _normalize_uav_position(position: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            _safe_norm(float(position[0]), config.MAP_LENGTH),
            _safe_norm(float(position[1]), config.MAP_WIDTH),
            _safe_norm(float(position[2]), config.UAV_HEIGHT),
        ],
        dtype=float,
    )


def _normalize_uav_velocity(velocity: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            _safe_norm(float(velocity[0]), config.V_MAX),
            _safe_norm(float(velocity[1]), config.V_MAX),
        ],
        dtype=float,
    )


def _normalize_user_position(position: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            _safe_norm(float(position[0]), config.MAP_LENGTH),
            _safe_norm(float(position[1]), config.MAP_WIDTH),
        ],
        dtype=float,
    )


def count_safety_violations(uavs: Sequence[UAV], d_safe_m: float = config.D_SAFE) -> int:
    violations = 0
    for source_index, source_uav in enumerate(uavs):
        for target_index in range(source_index + 1, len(uavs)):
            target_uav = uavs[target_index]
            if euclidean_distance_3d(source_uav.position, target_uav.position) < float(d_safe_m):
                violations += 1
    return violations


def build_reward_reference_scales(
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
) -> RewardReferenceScales:
    total_arrival_bits_per_step = float(sum(user.demand_rate_bps for user in users)) * float(config.DELTA_T)
    return RewardReferenceScales(
        throughput_ref_bits=max(total_arrival_bits_per_step, config.EPSILON),
        energy_ref_j=max(float(len(uavs)) * float(config.E_HOVER) * float(config.DELTA_T), config.EPSILON),
        access_backlog_ref_bits=max(total_arrival_bits_per_step, config.EPSILON),
        relay_queue_ref_bits=max(total_arrival_bits_per_step, config.EPSILON),
    )


def build_linucb_context(uav: UAV, uavs: Sequence[UAV], users: Sequence[GroundUser]) -> np.ndarray:
    associated_users = [user for user in users if user.associated_uav_id == uav.id]
    normalized_user_load = _safe_norm(len(associated_users), max(config.NUM_USERS, 1))

    low_sinr_user_count = 0
    for user in associated_users:
        reference_sinr_linear = a2g_sinr_linear(uav.position, user.position, interference_power_w=0.0)
        if 10.0 * np.log10(max(reference_sinr_linear, config.EPSILON)) < config.SINR_THRESHOLD_DB:
            low_sinr_user_count += 1
    edge_user_ratio = low_sinr_user_count / max(len(associated_users), 1)

    interference_proxy_raw = 0.0
    for other_uav in uavs:
        if other_uav.id == uav.id:
            continue
        distance_m = max(euclidean_distance_3d(uav.position, other_uav.position), 1.0)
        interference_proxy_raw += 1.0 / (distance_m ** 2)
    interference_proxy_norm = interference_proxy_raw / (1.0 + interference_proxy_raw)

    normalized_speed = _safe_norm(uav.speed, config.V_MAX)
    return np.array(
        [1.0, normalized_user_load, edge_user_ratio, interference_proxy_norm, normalized_speed],
        dtype=float,
    )


def build_global_state(
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
    env_state: EnvState,
) -> dict[str, np.ndarray | float]:
    uav_positions_norm = np.asarray([_normalize_uav_position(uav.position) for uav in uavs], dtype=float)
    uav_velocities_norm = np.asarray([_normalize_uav_velocity(uav.velocity) for uav in uavs], dtype=float)
    active_gateway_mask = np.asarray(
        [float(uav.id in env_state.active_gateway_uav_ids) for uav in uavs],
        dtype=float,
    )
    gateway_capable_mask = np.asarray([float(uav.is_gateway_capable) for uav in uavs], dtype=float)
    backhaul_capacity_by_uav = np.asarray(
        [env_state.backhaul_capacity_bps_by_gateway.get(uav.id, 0.0) for uav in uavs],
        dtype=float,
    )
    reachable_gateway_counts_norm = np.asarray(
        [
            _safe_norm(
                env_state.reachable_gateway_count_by_uav.get(uav.id, 0),
                max(len(env_state.active_gateway_uav_ids), 1),
            )
            for uav in uavs
        ],
        dtype=float,
    )
    user_positions_norm = np.asarray([_normalize_user_position(user.position) for user in users], dtype=float).reshape((-1, 2))
    user_access_backlogs_norm = np.asarray(
        [_safe_norm(user.user_access_backlog_bits, config.ACCESS_BACKLOG_REF_BITS) for user in users],
        dtype=float,
    )
    return {
        "uav_positions_norm": uav_positions_norm,
        "uav_velocities_norm": uav_velocities_norm,
        "uav_energies_norm": np.asarray(
            [_safe_norm(uav.residual_energy_j, config.E_INITIAL) for uav in uavs],
            dtype=float,
        ),
        "uav_queue_totals_norm": np.asarray(
            [_safe_norm(uav.relay_queue_total_bits, config.RELAY_QUEUE_REF_BITS) for uav in uavs],
            dtype=float,
        ),
        "gateway_capable_mask": gateway_capable_mask,
        "user_positions_norm": user_positions_norm,
        "user_access_backlogs_norm": user_access_backlogs_norm,
        "associations": np.asarray([user.associated_uav_id for user in users], dtype=int),
        "connectivity_matrix": np.asarray(env_state.adjacency_matrix, dtype=int),
        "active_gateway_mask": active_gateway_mask,
        "reachable_gateway_counts_norm": reachable_gateway_counts_norm,
        "backhaul_capacity_by_uav_norm": np.asarray(
            [_safe_norm(value, _bps_reference()) for value in backhaul_capacity_by_uav],
            dtype=float,
        ),
        "backhaul_capacity_norm": np.asarray(
            [_safe_norm(env_state.backhaul_capacity_bps, _bps_reference())],
            dtype=float,
        ),
    }


def build_local_observation(
    uav: UAV,
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
    env_state: EnvState,
    *,
    relay_capacity_matrix_bps: np.ndarray | None = None,
    max_obs_users_pad: int = config.MAX_OBS_USERS_PAD,
    obs_radius_m: float = config.OBS_RADIUS,
) -> np.ndarray:
    # Observation layout:
    # [0:3] self normalized absolute position xyz
    # [3:5] self normalized absolute velocity xy
    # [5] self energy norm
    # [6] self is gateway-capable
    # [7] self relay queue norm
    # [8] self associated-user-count norm
    # [9] reachable gateway count norm
    # [10] best gateway path capacity norm
    # [11] best gateway backhaul capacity norm
    # [12:12+6*(N-1)] other UAV normalized absolute position xyz + velocity xy + link-active
    # final 3*MAX_OBS_USERS_PAD entries: visible user normalized absolute xy + backlog norm
    if relay_capacity_matrix_bps is not None:
        np.asarray(relay_capacity_matrix_bps, dtype=float)
    id_to_index = {other_uav.id: index for index, other_uav in enumerate(uavs)}
    uav_index = id_to_index[uav.id]
    reachable_gateway_count = env_state.reachable_gateway_count_by_uav.get(uav.id, 0)
    best_gateway_path_capacity_bps = env_state.best_gateway_path_capacity_bps_by_uav.get(uav.id, 0.0)
    best_gateway_backhaul_capacity_bps = env_state.best_gateway_backhaul_capacity_bps_by_uav.get(uav.id, 0.0)

    self_features = np.array(
        [
            _safe_norm(uav.position[0], config.MAP_LENGTH),
            _safe_norm(uav.position[1], config.MAP_WIDTH),
            _safe_norm(uav.position[2], config.UAV_HEIGHT),
            _safe_norm(uav.velocity[0], config.V_MAX),
            _safe_norm(uav.velocity[1], config.V_MAX),
            _safe_norm(uav.residual_energy_j, config.E_INITIAL),
            float(uav.is_gateway_capable),
            _safe_norm(uav.relay_queue_total_bits, config.RELAY_QUEUE_REF_BITS),
            _safe_norm(len(uav.associated_user_ids), max(config.NUM_USERS, 1)),
            _safe_norm(reachable_gateway_count, max(len(env_state.active_gateway_uav_ids), 1)),
            _safe_norm(best_gateway_path_capacity_bps, _bps_reference()),
            _safe_norm(best_gateway_backhaul_capacity_bps, _bps_reference()),
        ],
        dtype=float,
    )

    other_uav_features: list[float] = []
    for other_uav in sorted(uavs, key=lambda item: item.id):
        if other_uav.id == uav.id:
            continue
        other_index = id_to_index[other_uav.id]
        link_active = float(env_state.adjacency_matrix[uav_index, other_index] > 0)
        other_uav_features.extend(_normalize_uav_position(other_uav.position).tolist())
        other_uav_features.extend(_normalize_uav_velocity(other_uav.velocity).tolist())
        other_uav_features.append(link_active)

    visible_users = sorted(
        [
            user
            for user in users
            if euclidean_distance_2d(uav.position, user.position) <= float(obs_radius_m)
        ],
        key=lambda user: euclidean_distance_2d(uav.position, user.position),
    )[: int(max_obs_users_pad)]

    visible_user_features: list[float] = []
    for user in visible_users:
        visible_user_features.extend(_normalize_user_position(user.position).tolist())
        visible_user_features.append(_safe_norm(user.user_access_backlog_bits, config.ACCESS_BACKLOG_REF_BITS))

    num_padding_users = int(max_obs_users_pad) - len(visible_users)
    if num_padding_users > 0:
        visible_user_features.extend([0.0] * (3 * num_padding_users))

    return np.concatenate(
        [
            self_features,
            np.asarray(other_uav_features, dtype=float),
            np.asarray(visible_user_features, dtype=float),
        ]
    )


def compute_team_reward(
    step_result: SimStepResult,
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
    *,
    reward_reference_scales: RewardReferenceScales | None = None,
) -> float:
    reward_reference_scales = reward_reference_scales or RewardReferenceScales()
    throughput_norm = _safe_norm(step_result.env_state.total_delivered_bits_step, reward_reference_scales.throughput_ref_bits)
    total_energy_step_j = float(sum(step_result.accounting.energy_used_j_by_uav.values()))
    energy_norm = _safe_norm(total_energy_step_j, reward_reference_scales.energy_ref_j)
    outage_ratio = float(sum(user.final_rate_bps < config.R_MIN for user in users)) / max(len(users), 1)
    access_backlog_norm = _safe_norm(
        sum(user.user_access_backlog_bits for user in users),
        reward_reference_scales.access_backlog_ref_bits,
    )
    relay_queue_norm = _safe_norm(
        sum(uav.relay_queue_total_bits for uav in uavs),
        reward_reference_scales.relay_queue_ref_bits,
    )
    num_safety_violations = count_safety_violations(uavs)

    return (
        throughput_norm
        - config.ETA * energy_norm
        - config.MU * outage_ratio
        - config.BETA_ACCESS * access_backlog_norm
        - config.BETA_RELAY * relay_queue_norm
        - config.LAMBDA_CONN * float(step_result.env_state.lambda2 <= 0.0)
        - config.LAMBDA_SAFE * float(num_safety_violations)
    )


@dataclass
class MultiAgentStep:
    observations_by_agent: dict[int, np.ndarray]
    rewards_by_agent: dict[int, float]
    terminated_by_agent: dict[int, bool]
    truncated_by_agent: dict[int, bool]
    info: dict[str, Any]


class MultiAgentUavNetEnv:
    def __init__(
        self,
        sim_env: SimEnv,
        *,
        max_steps: int = config.SIM_STEPS,
        alpha_controllers: Mapping[int, LinUCBStub] | None = None,
    ) -> None:
        self.sim_env = sim_env
        self.max_steps = int(max_steps)
        self.agent_ids = [uav.id for uav in self.sim_env.uavs]
        self.reward_reference_scales = build_reward_reference_scales(self.sim_env.uavs, self.sim_env.users)
        self.alpha_controllers = dict(alpha_controllers or {
            agent_id: LinUCBStub(fixed_alpha=config.PF_ALPHA_DEFAULT) for agent_id in self.agent_ids
        })

    def reset(self) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
        self.reward_reference_scales = build_reward_reference_scales(self.sim_env.uavs, self.sim_env.users)
        env_state = self.sim_env.reset()
        associate_users_to_uavs(
            self.sim_env.users,
            self.sim_env.uavs,
            min_rate_bps=self.sim_env.association_min_rate_bps,
        )
        relay_capacity_matrix_bps = build_a2a_capacity_matrix_bps(self.sim_env.uavs)
        adjacency_matrix = build_adjacency_matrix(relay_capacity_matrix_bps)
        lambda2 = algebraic_connectivity_lambda2(adjacency_matrix)
        try:
            backhaul_capacity_bps_by_gateway = self.sim_env._resolve_backhaul_capacity_bps_by_gateway(
                backhaul_capacity_bps_override=None,
                backhaul_capacity_bps_override_by_gateway=None,
            )
        except ValueError:
            backhaul_capacity_bps_by_gateway = {}
        active_gateway_uav_ids = tuple(
            sorted(gateway_uav_id for gateway_uav_id, capacity_bps in backhaul_capacity_bps_by_gateway.items() if capacity_bps > 0.0)
        )
        routing_table = compute_routing_table(
            uavs=self.sim_env.uavs,
            active_gateway_uav_ids=active_gateway_uav_ids,
            capacity_matrix_bps=relay_capacity_matrix_bps,
            backhaul_capacity_bps_by_gateway=backhaul_capacity_bps_by_gateway,
        )
        env_state = EnvState(
            current_step=env_state.current_step,
            adjacency_matrix=adjacency_matrix,
            lambda2=lambda2,
            backhaul_capacity_bps=float(sum(backhaul_capacity_bps_by_gateway.values())),
            total_delivered_bits_step=0.0,
            active_gateway_uav_ids=active_gateway_uav_ids,
            routing_next_hop_by_uav={
                uav_id: decision.next_hop_uav_id for uav_id, decision in routing_table.items()
            },
            reachable_gateway_count_by_uav={
                uav_id: decision.reachable_gateway_count for uav_id, decision in routing_table.items()
            },
            backhaul_capacity_bps_by_gateway=dict(backhaul_capacity_bps_by_gateway),
            best_gateway_path_capacity_bps_by_uav={
                uav_id: decision.effective_path_capacity_bps for uav_id, decision in routing_table.items()
            },
            best_gateway_backhaul_capacity_bps_by_uav={
                uav_id: decision.gateway_backhaul_capacity_bps for uav_id, decision in routing_table.items()
            },
        )
        observations_by_agent = {
            uav.id: build_local_observation(
                uav,
                self.sim_env.uavs,
                self.sim_env.users,
                env_state,
                relay_capacity_matrix_bps=relay_capacity_matrix_bps,
            )
            for uav in self.sim_env.uavs
        }
        info = {
            "env_state": env_state,
            "global_state": build_global_state(self.sim_env.uavs, self.sim_env.users, env_state),
        }
        return observations_by_agent, info

    def step_struct(
        self,
        actions_by_agent: Mapping[int, Mapping[str, float]],
        **sim_step_kwargs: Any,
    ) -> MultiAgentStep:
        associate_users_to_uavs(self.sim_env.users, self.sim_env.uavs)
        context_by_uav = {
            uav.id: build_linucb_context(uav, self.sim_env.uavs, self.sim_env.users)
            for uav in self.sim_env.uavs
        }
        step_result = self.sim_env.step(
            actions_by_uav_id=actions_by_agent,
            alpha_controllers=self.alpha_controllers,
            context_by_uav=context_by_uav,
            **sim_step_kwargs,
        )

        team_reward = compute_team_reward(
            step_result,
            self.sim_env.uavs,
            self.sim_env.users,
            reward_reference_scales=self.reward_reference_scales,
        )
        for agent_id, alpha_controller in self.alpha_controllers.items():
            alpha_controller.update(context_by_uav.get(agent_id), team_reward)

        observations_by_agent = {
            uav.id: build_local_observation(
                uav,
                self.sim_env.uavs,
                self.sim_env.users,
                step_result.env_state,
                relay_capacity_matrix_bps=step_result.relay_service_result.capacity_matrix_bps,
            )
            for uav in self.sim_env.uavs
        }

        terminated = self.sim_env.current_step >= self.max_steps
        rewards_by_agent = {agent_id: team_reward for agent_id in self.agent_ids}
        terminated_by_agent = {agent_id: terminated for agent_id in self.agent_ids}
        terminated_by_agent["__all__"] = terminated
        truncated_by_agent = {agent_id: False for agent_id in self.agent_ids}
        truncated_by_agent["__all__"] = False

        info = {
            "team_reward": team_reward,
            "env_state": step_result.env_state,
            "global_state": build_global_state(self.sim_env.uavs, self.sim_env.users, step_result.env_state),
            "step_result": step_result,
            "alpha_by_uav": step_result.access_step_result.alpha_by_uav,
        }
        return MultiAgentStep(
            observations_by_agent=observations_by_agent,
            rewards_by_agent=rewards_by_agent,
            terminated_by_agent=terminated_by_agent,
            truncated_by_agent=truncated_by_agent,
            info=info,
        )

    def step(
        self,
        actions_by_agent: Mapping[int, Mapping[str, float]],
        **sim_step_kwargs: Any,
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, float],
        dict[int | str, bool],
        dict[int | str, bool],
        dict[str, Any],
    ]:
        step = self.step_struct(actions_by_agent, **sim_step_kwargs)
        return (
            step.observations_by_agent,
            step.rewards_by_agent,
            step.terminated_by_agent,
            step.truncated_by_agent,
            step.info,
        )
