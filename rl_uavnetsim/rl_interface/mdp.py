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
    find_widest_path_to_anchor,
)
from rl_uavnetsim.rl_interface.linucb_stub import LinUCBStub
from rl_uavnetsim.utils.helpers import euclidean_distance_2d, euclidean_distance_3d


def _safe_norm(value: float, reference: float) -> float:
    return float(value) / max(float(reference), config.EPSILON)


def _bps_reference() -> float:
    return max(config.THROUGHPUT_REF_BITS / config.DELTA_T, config.EPSILON)


def count_safety_violations(uavs: Sequence[UAV], d_safe_m: float = config.D_SAFE) -> int:
    violations = 0
    for source_index, source_uav in enumerate(uavs):
        for target_index in range(source_index + 1, len(uavs)):
            target_uav = uavs[target_index]
            if euclidean_distance_3d(source_uav.position, target_uav.position) < float(d_safe_m):
                violations += 1
    return violations


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
    return {
        "uav_positions": np.asarray([uav.position for uav in uavs], dtype=float),
        "uav_energies": np.asarray([uav.residual_energy_j for uav in uavs], dtype=float),
        "uav_queue_totals": np.asarray([uav.relay_queue_total_bits for uav in uavs], dtype=float),
        "user_positions": np.asarray([user.position[:2] for user in users], dtype=float),
        "user_access_backlogs": np.asarray([user.user_access_backlog_bits for user in users], dtype=float),
        "associations": np.asarray([user.associated_uav_id for user in users], dtype=int),
        "connectivity_matrix": np.asarray(env_state.adjacency_matrix, dtype=int),
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
    last_known_positions_by_uav_id: Mapping[int, np.ndarray] | None = None,
    max_obs_users_pad: int = config.MAX_OBS_USERS_PAD,
    obs_radius_m: float = config.OBS_RADIUS,
) -> np.ndarray:
    relay_capacity_matrix_bps = (
        build_a2a_capacity_matrix_bps(uavs) if relay_capacity_matrix_bps is None else np.asarray(relay_capacity_matrix_bps, dtype=float)
    )
    id_to_index = {other_uav.id: index for index, other_uav in enumerate(uavs)}
    uav_index = id_to_index[uav.id]

    self_features = np.array(
        [
            uav.position[0],
            uav.position[1],
            uav.position[2],
            uav.velocity[0],
            uav.velocity[1],
            _safe_norm(uav.residual_energy_j, config.E_INITIAL),
            float(uav.is_anchor),
            _safe_norm(uav.relay_queue_total_bits, config.RELAY_QUEUE_REF_BITS),
            _safe_norm(len(uav.associated_user_ids), max(config.NUM_USERS, 1)),
        ],
        dtype=float,
    )

    if uav.is_anchor:
        est_relay_capacity_bps = env_state.backhaul_capacity_bps
    else:
        relay_path = find_widest_path_to_anchor(
            source_uav_id=uav.id,
            anchor_uav_id=config.ANCHOR_UAV_ID,
            uavs=uavs,
            capacity_matrix_bps=relay_capacity_matrix_bps,
        )
        est_relay_capacity_bps = relay_path.bottleneck_capacity_bps
    self_features = np.concatenate(
        [
            self_features,
            np.array([_safe_norm(est_relay_capacity_bps, _bps_reference())], dtype=float),
        ]
    )

    other_relative_positions: list[float] = []
    other_link_active: list[float] = []
    for other_uav in sorted(uavs, key=lambda item: item.id):
        if other_uav.id == uav.id:
            continue
        other_index = id_to_index[other_uav.id]
        link_active = float(env_state.adjacency_matrix[uav_index, other_index] > 0)
        reference_position = other_uav.position
        if link_active <= 0.0 and last_known_positions_by_uav_id is not None and other_uav.id in last_known_positions_by_uav_id:
            reference_position = np.asarray(last_known_positions_by_uav_id[other_uav.id], dtype=float)
        relative_position = np.asarray(reference_position, dtype=float) - uav.position
        other_relative_positions.extend(relative_position.tolist())
        other_link_active.append(link_active)

    visible_users = sorted(
        [
            user
            for user in users
            if euclidean_distance_2d(uav.position, user.position) <= float(obs_radius_m)
        ],
        key=lambda user: euclidean_distance_2d(uav.position, user.position),
    )[: int(max_obs_users_pad)]

    visible_relative_positions: list[float] = []
    visible_user_backlog_norm: list[float] = []
    for user in visible_users:
        relative_position_2d = user.position[:2] - uav.position[:2]
        visible_relative_positions.extend(relative_position_2d.tolist())
        visible_user_backlog_norm.append(_safe_norm(user.user_access_backlog_bits, config.ACCESS_BACKLOG_REF_BITS))

    num_padding_users = int(max_obs_users_pad) - len(visible_users)
    if num_padding_users > 0:
        visible_relative_positions.extend([0.0] * (2 * num_padding_users))
        visible_user_backlog_norm.extend([0.0] * num_padding_users)

    return np.concatenate(
        [
            self_features,
            np.asarray(other_relative_positions, dtype=float),
            np.asarray(other_link_active, dtype=float),
            np.asarray(visible_relative_positions, dtype=float),
            np.asarray(visible_user_backlog_norm, dtype=float),
        ]
    )


def compute_team_reward(step_result: SimStepResult, uavs: Sequence[UAV], users: Sequence[GroundUser]) -> float:
    throughput_norm = _safe_norm(step_result.env_state.total_delivered_bits_step, config.THROUGHPUT_REF_BITS)
    total_energy_step_j = float(sum(step_result.accounting.energy_used_j_by_uav.values()))
    energy_norm = _safe_norm(total_energy_step_j, config.ENERGY_REF_J)
    outage_ratio = float(sum(user.final_rate_bps < config.R_MIN for user in users)) / max(len(users), 1)
    access_backlog_norm = _safe_norm(sum(user.user_access_backlog_bits for user in users), config.ACCESS_BACKLOG_REF_BITS)
    relay_queue_norm = _safe_norm(sum(uav.relay_queue_total_bits for uav in uavs), config.RELAY_QUEUE_REF_BITS)
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
        linucb_controllers: Mapping[int, LinUCBStub] | None = None,
    ) -> None:
        self.sim_env = sim_env
        self.max_steps = int(max_steps)
        self.agent_ids = [uav.id for uav in self.sim_env.uavs]
        self.linucb_controllers = dict(linucb_controllers or {
            agent_id: LinUCBStub(fixed_alpha=config.PF_ALPHA_DEFAULT) for agent_id in self.agent_ids
        })
        self.last_known_positions_by_uav_id = {
            uav.id: np.asarray(uav.position, dtype=float).copy() for uav in self.sim_env.uavs
        }

    def reset(self) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
        env_state = self.sim_env.reset()
        associate_users_to_uavs(self.sim_env.users, self.sim_env.uavs)
        relay_capacity_matrix_bps = build_a2a_capacity_matrix_bps(self.sim_env.uavs)
        adjacency_matrix = build_adjacency_matrix(relay_capacity_matrix_bps)
        lambda2 = algebraic_connectivity_lambda2(adjacency_matrix)
        try:
            backhaul_capacity_bps_value = self.sim_env._resolve_backhaul_capacity_bps()
        except ValueError:
            backhaul_capacity_bps_value = 0.0
        env_state = EnvState(
            current_step=env_state.current_step,
            adjacency_matrix=adjacency_matrix,
            lambda2=lambda2,
            backhaul_capacity_bps=backhaul_capacity_bps_value,
            total_delivered_bits_step=0.0,
        )
        self.last_known_positions_by_uav_id = {
            uav.id: np.asarray(uav.position, dtype=float).copy() for uav in self.sim_env.uavs
        }
        observations_by_agent = {
            uav.id: build_local_observation(
                uav,
                self.sim_env.uavs,
                self.sim_env.users,
                env_state,
                relay_capacity_matrix_bps=relay_capacity_matrix_bps,
                last_known_positions_by_uav_id=self.last_known_positions_by_uav_id,
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
        previous_known_positions_by_uav_id = {
            uav_id: position.copy() for uav_id, position in self.last_known_positions_by_uav_id.items()
        }
        step_result = self.sim_env.step(
            actions_by_uav_id=actions_by_agent,
            linucb_controllers=self.linucb_controllers,
            context_by_uav=context_by_uav,
            **sim_step_kwargs,
        )

        team_reward = compute_team_reward(step_result, self.sim_env.uavs, self.sim_env.users)
        for agent_id, linucb_controller in self.linucb_controllers.items():
            linucb_controller.update(context_by_uav.get(agent_id), team_reward)

        self.last_known_positions_by_uav_id = {
            uav.id: np.asarray(uav.position, dtype=float).copy() for uav in self.sim_env.uavs
        }
        observations_by_agent = {
            uav.id: build_local_observation(
                uav,
                self.sim_env.uavs,
                self.sim_env.users,
                step_result.env_state,
                relay_capacity_matrix_bps=step_result.relay_service_result.capacity_matrix_bps,
                last_known_positions_by_uav_id=previous_known_positions_by_uav_id,
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
