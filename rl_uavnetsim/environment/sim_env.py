from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.allocation import AssociationResult, AccessStepResult, associate_users_to_uavs, run_access_pf_step
from rl_uavnetsim.channel.backhaul_channel import backhaul_capacity_bps, gbs_backhaul_snr_db
from rl_uavnetsim.entities import GroundBaseStation, GroundUser, Satellite, UAV
from rl_uavnetsim.network import (
    BackhaulServiceResult,
    RelayServiceResult,
    build_a2a_capacity_matrix_bps,
    compute_routing_table,
    execute_backhaul_service,
    execute_relay_service,
)
from rl_uavnetsim.utils.helpers import euclidean_distance_3d


@dataclass
class EnvState:
    current_step: int
    adjacency_matrix: np.ndarray
    lambda2: float
    backhaul_capacity_bps: float
    total_delivered_bits_step: float
    active_gateway_uav_ids: tuple[int, ...] = ()
    routing_next_hop_by_uav: dict[int, int | None] = field(default_factory=dict)
    reachable_gateway_count_by_uav: dict[int, int] = field(default_factory=dict)
    backhaul_capacity_bps_by_gateway: dict[int, float] = field(default_factory=dict)
    best_gateway_path_capacity_bps_by_uav: dict[int, float] = field(default_factory=dict)
    best_gateway_backhaul_capacity_bps_by_uav: dict[int, float] = field(default_factory=dict)


@dataclass
class SimStepAccounting:
    user_access_backlog_prev_bits_by_user: dict[int, float] = field(default_factory=dict)
    user_access_backlog_next_bits_by_user: dict[int, float] = field(default_factory=dict)
    arrived_bits_by_user: dict[int, float] = field(default_factory=dict)
    relay_queue_prev_bits_by_uav: dict[int, float] = field(default_factory=dict)
    relay_queue_next_bits_by_uav: dict[int, float] = field(default_factory=dict)
    access_ingress_bits_by_uav: dict[int, float] = field(default_factory=dict)
    relay_in_bits_by_uav: dict[int, float] = field(default_factory=dict)
    backhaul_out_bits_by_gateway: dict[int, float] = field(default_factory=dict)
    energy_used_j_by_uav: dict[int, float] = field(default_factory=dict)


@dataclass
class SimStepResult:
    env_state: EnvState
    association_result: AssociationResult
    access_step_result: AccessStepResult
    relay_service_result: RelayServiceResult
    backhaul_service_result: BackhaulServiceResult
    accounting: SimStepAccounting


class SimEnv:
    def __init__(
        self,
        *,
        uavs: Sequence[UAV],
        users: Sequence[GroundUser],
        satellites: Sequence[Satellite] | None = None,
        ground_base_stations: Sequence[GroundBaseStation] | None = None,
        gateway_capable_uav_ids: Sequence[int] | None = None,
        backhaul_type: str = config.BACKHAUL_TYPE,
        association_min_rate_bps: float = config.R_MIN,
        map_length_m: float = config.MAP_LENGTH,
        map_width_m: float = config.MAP_WIDTH,
        rng: np.random.Generator | None = None,
    ) -> None:
        resolved_gateway_capable_uav_ids = self._resolve_gateway_capable_uav_ids(
            uavs=uavs,
            gateway_capable_uav_ids=gateway_capable_uav_ids,
        )
        self.gateway_capable_uav_ids = tuple(resolved_gateway_capable_uav_ids)
        self.backhaul_type = str(backhaul_type)
        self.association_min_rate_bps = float(association_min_rate_bps)
        self.map_length_m = float(map_length_m)
        self.map_width_m = float(map_width_m)
        if self.map_length_m <= 0.0 or self.map_width_m <= 0.0:
            raise ValueError("SimEnv map dimensions must be positive.")
        self.rng = rng or np.random.default_rng()

        self._initial_uavs = copy.deepcopy(list(uavs))
        for uav in self._initial_uavs:
            uav.is_gateway_capable = uav.id in self.gateway_capable_uav_ids
        self._initial_users = copy.deepcopy(list(users))
        self._initial_satellites = copy.deepcopy(list(satellites or []))
        self._initial_ground_base_stations = copy.deepcopy(list(ground_base_stations or []))

        self.current_step = 0
        self.uavs: list[UAV] = []
        self.users: list[GroundUser] = []
        self.satellites: list[Satellite] = []
        self.ground_base_stations: list[GroundBaseStation] = []
        self.reset()

    def reset(self) -> EnvState:
        self.current_step = 0
        self.uavs = copy.deepcopy(self._initial_uavs)
        for uav in self.uavs:
            uav.is_gateway_capable = uav.id in self.gateway_capable_uav_ids
        self.users = copy.deepcopy(self._initial_users)
        self.satellites = copy.deepcopy(self._initial_satellites)
        self.ground_base_stations = copy.deepcopy(self._initial_ground_base_stations)
        return self._build_env_state(
            adjacency_matrix=np.zeros((len(self.uavs), len(self.uavs)), dtype=int),
            lambda2=0.0,
            backhaul_capacity_bps_by_gateway={},
            total_delivered_bits_step=0.0,
            routing_next_hop_by_uav={uav.id: None for uav in self.uavs},
            reachable_gateway_count_by_uav={uav.id: 0 for uav in self.uavs},
            active_gateway_uav_ids=(),
            best_gateway_path_capacity_bps_by_uav={uav.id: 0.0 for uav in self.uavs},
            best_gateway_backhaul_capacity_bps_by_uav={uav.id: 0.0 for uav in self.uavs},
        )

    def step(
        self,
        *,
        actions_by_uav_id: Mapping[int, Mapping[str, float]] | None = None,
        alpha_by_uav: Mapping[int, float] | None = None,
        alpha_controllers: Mapping[int, Any] | None = None,
        context_by_uav: Mapping[int, Any] | None = None,
        relay_capacity_matrix_bps: np.ndarray | None = None,
        backhaul_capacity_bps_override: float | None = None,
        backhaul_capacity_bps_override_by_gateway: Mapping[int, float] | None = None,
    ) -> SimStepResult:
        if backhaul_capacity_bps_override is not None and backhaul_capacity_bps_override_by_gateway is not None:
            raise ValueError("Pass either backhaul_capacity_bps_override or backhaul_capacity_bps_override_by_gateway, not both.")
        if backhaul_capacity_bps_override is not None and len(self.gateway_capable_uav_ids) != 1:
            raise ValueError("Scalar backhaul override is only valid when exactly one gateway-capable UAV is configured.")

        accounting = SimStepAccounting(
            user_access_backlog_prev_bits_by_user={
                user.id: user.user_access_backlog_bits for user in self.users
            },
            relay_queue_prev_bits_by_uav={
                uav.id: uav.relay_queue_total_bits for uav in self.uavs
            },
        )

        self._reset_step_counters()
        accounting.energy_used_j_by_uav = self._apply_uav_actions(actions_by_uav_id=actions_by_uav_id)
        self._move_ground_users()

        accounting.arrived_bits_by_user = {
            user.id: user.add_demand_bits(delta_t_s=config.DELTA_T) for user in self.users
        }

        association_result = associate_users_to_uavs(
            self.users,
            self.uavs,
            min_rate_bps=self.association_min_rate_bps,
        )
        access_step_result = run_access_pf_step(
            uavs=self.uavs,
            users=self.users,
            alpha_by_uav=alpha_by_uav,
            alpha_controllers=alpha_controllers,
            context_by_uav=context_by_uav,
        )
        accounting.access_ingress_bits_by_uav = dict(access_step_result.total_access_ingress_bits_by_uav)

        backhaul_capacity_bps_by_gateway = self._resolve_backhaul_capacity_bps_by_gateway(
            backhaul_capacity_bps_override=backhaul_capacity_bps_override,
            backhaul_capacity_bps_override_by_gateway=backhaul_capacity_bps_override_by_gateway,
        )
        active_gateway_uav_ids = tuple(
            sorted(
                gateway_uav_id
                for gateway_uav_id, capacity_bps in backhaul_capacity_bps_by_gateway.items()
                if capacity_bps > 0.0
            )
        )
        if relay_capacity_matrix_bps is None:
            relay_capacity_matrix_bps = build_a2a_capacity_matrix_bps(self.uavs)
        routing_table = compute_routing_table(
            uavs=self.uavs,
            active_gateway_uav_ids=active_gateway_uav_ids,
            capacity_matrix_bps=relay_capacity_matrix_bps,
            backhaul_capacity_bps_by_gateway=backhaul_capacity_bps_by_gateway,
        )
        relay_service_result = execute_relay_service(
            self.uavs,
            active_gateway_uav_ids=active_gateway_uav_ids,
            delta_t_s=config.DELTA_T,
            capacity_matrix_bps=relay_capacity_matrix_bps,
            backhaul_capacity_bps_by_gateway=backhaul_capacity_bps_by_gateway,
            routing_table=routing_table,
        )
        accounting.relay_in_bits_by_uav = dict(relay_service_result.relay_in_bits_by_uav)

        active_gateway_uavs = [uav for uav in self.uavs if uav.id in active_gateway_uav_ids]
        backhaul_service_result = execute_backhaul_service(
            gateway_uavs=active_gateway_uavs,
            users=self.users,
            backhaul_capacity_bps_by_gateway=backhaul_capacity_bps_by_gateway,
            delta_t_s=config.DELTA_T,
        )
        accounting.backhaul_out_bits_by_gateway = dict(backhaul_service_result.backhaul_out_bits_by_gateway)

        for user in self.users:
            if user.delivered_bits_step <= config.EPSILON:
                user.final_rate_bps = 0.0

        accounting.user_access_backlog_next_bits_by_user = {
            user.id: user.user_access_backlog_bits for user in self.users
        }
        accounting.relay_queue_next_bits_by_uav = {
            uav.id: uav.relay_queue_total_bits for uav in self.uavs
        }

        total_delivered_bits_step = float(sum(user.delivered_bits_step for user in self.users))
        self.current_step += 1
        env_state = self._build_env_state(
            adjacency_matrix=relay_service_result.adjacency_matrix,
            lambda2=relay_service_result.lambda2,
            backhaul_capacity_bps_by_gateway=backhaul_capacity_bps_by_gateway,
            total_delivered_bits_step=total_delivered_bits_step,
            routing_next_hop_by_uav={
                uav_id: decision.next_hop_uav_id for uav_id, decision in routing_table.items()
            },
            reachable_gateway_count_by_uav={
                uav_id: decision.reachable_gateway_count for uav_id, decision in routing_table.items()
            },
            active_gateway_uav_ids=active_gateway_uav_ids,
            best_gateway_path_capacity_bps_by_uav={
                uav_id: decision.effective_path_capacity_bps for uav_id, decision in routing_table.items()
            },
            best_gateway_backhaul_capacity_bps_by_uav={
                uav_id: decision.gateway_backhaul_capacity_bps for uav_id, decision in routing_table.items()
            },
        )

        return SimStepResult(
            env_state=env_state,
            association_result=association_result,
            access_step_result=access_step_result,
            relay_service_result=relay_service_result,
            backhaul_service_result=backhaul_service_result,
            accounting=accounting,
        )

    def _resolve_gateway_capable_uav_ids(
        self,
        *,
        uavs: Sequence[UAV],
        gateway_capable_uav_ids: Sequence[int] | None,
    ) -> list[int]:
        if gateway_capable_uav_ids is not None:
            return sorted({int(uav_id) for uav_id in gateway_capable_uav_ids})
        inferred_gateway_capable_uav_ids = [uav.id for uav in uavs if uav.is_gateway_capable]
        if inferred_gateway_capable_uav_ids:
            return sorted({int(uav_id) for uav_id in inferred_gateway_capable_uav_ids})
        return [config.DEFAULT_GATEWAY_UAV_ID]

    def _reset_step_counters(self) -> None:
        for user in self.users:
            user.reset_step_counters()
        for uav in self.uavs:
            uav.reset_step_counters()

    def _apply_uav_actions(self, actions_by_uav_id: Mapping[int, Mapping[str, float]] | None) -> dict[int, float]:
        energy_used_j_by_uav = {uav.id: 0.0 for uav in self.uavs}
        if actions_by_uav_id is None:
            return energy_used_j_by_uav
        for uav in self.uavs:
            action = actions_by_uav_id.get(uav.id)
            if action is None:
                continue
            _, _, energy_used_j = uav.move_by_action(
                rho_norm=float(action.get("rho", 0.0)),
                psi_rad=float(action.get("psi", uav.direction)),
                delta_t_s=config.DELTA_T,
                x_bounds_m=(0.0, self.map_length_m),
                y_bounds_m=(0.0, self.map_width_m),
            )
            energy_used_j_by_uav[uav.id] = energy_used_j
        return energy_used_j_by_uav

    def _move_ground_users(self) -> None:
        for user in self.users:
            user.move(delta_t_s=config.DELTA_T, rng=self.rng)

    def _resolve_backhaul_capacity_bps_by_gateway(
        self,
        *,
        backhaul_capacity_bps_override: float | None,
        backhaul_capacity_bps_override_by_gateway: Mapping[int, float] | None,
    ) -> dict[int, float]:
        if backhaul_capacity_bps_override_by_gateway is not None:
            return {
                int(gateway_uav_id): float(capacity_bps)
                for gateway_uav_id, capacity_bps in backhaul_capacity_bps_override_by_gateway.items()
            }
        if backhaul_capacity_bps_override is not None:
            gateway_uav_id = self.gateway_capable_uav_ids[0]
            return {gateway_uav_id: float(backhaul_capacity_bps_override)}

        if self.backhaul_type == "satellite":
            if len(self.gateway_capable_uav_ids) != 1:
                raise NotImplementedError("Satellite mode currently supports exactly one configured gateway-capable UAV.")
            if not self.satellites:
                raise ValueError("Satellite backhaul requested but no satellite entity is available.")
            gateway_uav = self._uav_by_id(self.gateway_capable_uav_ids[0])
            return {
                gateway_uav.id: backhaul_capacity_bps(gateway_uav.position, self.satellites[0], backhaul_type="satellite")
            }

        if self.backhaul_type == "gbs":
            if not self.ground_base_stations:
                raise ValueError("GBS backhaul requested but no ground base station entity is available.")
            gbs = self.ground_base_stations[0]
            capacity_bps_by_gateway: dict[int, float] = {}
            for gateway_uav_id in self.gateway_capable_uav_ids:
                gateway_uav = self._uav_by_id(gateway_uav_id)
                distance_m = euclidean_distance_3d(gateway_uav.position, gbs.position)
                if distance_m > float(config.MAX_GBS_RANGE_M):
                    capacity_bps_by_gateway[gateway_uav_id] = 0.0
                    continue
                gateway_snr_db = gbs_backhaul_snr_db(gateway_uav.position, gbs.position)
                if gateway_snr_db < float(config.GBS_BACKHAUL_SNR_THRESHOLD_DB):
                    capacity_bps_by_gateway[gateway_uav_id] = 0.0
                    continue
                capacity_bps_by_gateway[gateway_uav_id] = backhaul_capacity_bps(
                    gateway_uav.position,
                    gbs,
                    backhaul_type="gbs",
                )
            return capacity_bps_by_gateway

        raise ValueError(f"Unsupported backhaul_type: {self.backhaul_type}")

    def _uav_by_id(self, uav_id: int) -> UAV:
        for uav in self.uavs:
            if uav.id == int(uav_id):
                return uav
        raise ValueError(f"UAV id {uav_id} was not found.")

    def _build_env_state(
        self,
        *,
        adjacency_matrix: np.ndarray,
        lambda2: float,
        backhaul_capacity_bps_by_gateway: Mapping[int, float],
        total_delivered_bits_step: float,
        routing_next_hop_by_uav: Mapping[int, int | None],
        reachable_gateway_count_by_uav: Mapping[int, int],
        active_gateway_uav_ids: Sequence[int],
        best_gateway_path_capacity_bps_by_uav: Mapping[int, float],
        best_gateway_backhaul_capacity_bps_by_uav: Mapping[int, float],
    ) -> EnvState:
        return EnvState(
            current_step=self.current_step,
            adjacency_matrix=np.asarray(adjacency_matrix, dtype=int),
            lambda2=float(lambda2),
            backhaul_capacity_bps=float(sum(backhaul_capacity_bps_by_gateway.values())),
            total_delivered_bits_step=float(total_delivered_bits_step),
            active_gateway_uav_ids=tuple(int(gateway_uav_id) for gateway_uav_id in active_gateway_uav_ids),
            routing_next_hop_by_uav={int(uav_id): next_hop_uav_id for uav_id, next_hop_uav_id in routing_next_hop_by_uav.items()},
            reachable_gateway_count_by_uav={int(uav_id): int(count) for uav_id, count in reachable_gateway_count_by_uav.items()},
            backhaul_capacity_bps_by_gateway={int(gateway_uav_id): float(capacity_bps) for gateway_uav_id, capacity_bps in backhaul_capacity_bps_by_gateway.items()},
            best_gateway_path_capacity_bps_by_uav={int(uav_id): float(capacity_bps) for uav_id, capacity_bps in best_gateway_path_capacity_bps_by_uav.items()},
            best_gateway_backhaul_capacity_bps_by_uav={int(uav_id): float(capacity_bps) for uav_id, capacity_bps in best_gateway_backhaul_capacity_bps_by_uav.items()},
        )
