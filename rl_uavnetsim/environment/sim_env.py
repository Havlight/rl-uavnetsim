from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.allocation import AssociationResult, AccessStepResult, associate_users_to_uavs, run_access_pf_step
from rl_uavnetsim.channel.backhaul_channel import backhaul_capacity_bps
from rl_uavnetsim.entities import GroundBaseStation, GroundUser, Satellite, UAV
from rl_uavnetsim.network import BackhaulServiceResult, RelayServiceResult, execute_backhaul_service, execute_relay_service


@dataclass
class EnvState:
    current_step: int
    adjacency_matrix: np.ndarray
    lambda2: float
    backhaul_capacity_bps: float
    total_delivered_bits_step: float


@dataclass
class SimStepAccounting:
    user_access_backlog_prev_bits_by_user: dict[int, float] = field(default_factory=dict)
    user_access_backlog_next_bits_by_user: dict[int, float] = field(default_factory=dict)
    arrived_bits_by_user: dict[int, float] = field(default_factory=dict)
    member_queue_prev_bits_by_uav: dict[int, float] = field(default_factory=dict)
    member_queue_next_bits_by_uav: dict[int, float] = field(default_factory=dict)
    anchor_queue_prev_bits: float = 0.0
    anchor_queue_next_bits: float = 0.0
    anchor_access_ingress_bits: float = 0.0
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
        anchor_uav_id: int = config.ANCHOR_UAV_ID,
        backhaul_type: str = config.BACKHAUL_TYPE,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.anchor_uav_id = int(anchor_uav_id)
        self.backhaul_type = str(backhaul_type)
        self.rng = rng or np.random.default_rng()

        self._initial_uavs = copy.deepcopy(list(uavs))
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
        self.users = copy.deepcopy(self._initial_users)
        self.satellites = copy.deepcopy(self._initial_satellites)
        self.ground_base_stations = copy.deepcopy(self._initial_ground_base_stations)
        return self._build_env_state(backhaul_capacity_bps_value=0.0, total_delivered_bits_step=0.0)

    def step(
        self,
        *,
        actions_by_uav_id: Mapping[int, Mapping[str, float]] | None = None,
        alpha_by_uav: Mapping[int, float] | None = None,
        linucb_controllers: Mapping[int, Any] | None = None,
        context_by_uav: Mapping[int, Any] | None = None,
        relay_capacity_matrix_bps: np.ndarray | None = None,
        backhaul_capacity_bps_override: float | None = None,
    ) -> SimStepResult:
        accounting = SimStepAccounting(
            user_access_backlog_prev_bits_by_user={
                user.id: user.user_access_backlog_bits for user in self.users
            },
        )
        anchor_uav = self._anchor_uav()
        accounting.anchor_queue_prev_bits = anchor_uav.refresh_relay_queue_total_bits()
        accounting.member_queue_prev_bits_by_uav = {
            uav.id: uav.refresh_relay_queue_total_bits()
            for uav in self.uavs
            if uav.id != self.anchor_uav_id
        }

        self._reset_step_counters()
        accounting.energy_used_j_by_uav = self._apply_uav_actions(actions_by_uav_id=actions_by_uav_id)
        self._move_ground_users()

        accounting.arrived_bits_by_user = {
            user.id: user.add_demand_bits(delta_t_s=config.DELTA_T) for user in self.users
        }

        association_result = associate_users_to_uavs(self.users, self.uavs)
        access_step_result = run_access_pf_step(
            uavs=self.uavs,
            users=self.users,
            alpha_by_uav=alpha_by_uav,
            linucb_controllers=linucb_controllers,
            context_by_uav=context_by_uav,
        )

        relay_service_result = execute_relay_service(
            self.uavs,
            anchor_uav_id=self.anchor_uav_id,
            delta_t_s=config.DELTA_T,
            capacity_matrix_bps=relay_capacity_matrix_bps,
        )

        backhaul_capacity_bps_value = (
            float(backhaul_capacity_bps_override)
            if backhaul_capacity_bps_override is not None
            else self._resolve_backhaul_capacity_bps()
        )
        backhaul_service_result = execute_backhaul_service(
            anchor_uav=self._anchor_uav(),
            users=self.users,
            backhaul_capacity_bps=backhaul_capacity_bps_value,
            delta_t_s=config.DELTA_T,
        )

        for user in self.users:
            if user.delivered_bits_step <= config.EPSILON:
                user.final_rate_bps = 0.0

        accounting.user_access_backlog_next_bits_by_user = {
            user.id: user.user_access_backlog_bits for user in self.users
        }
        accounting.member_queue_next_bits_by_uav = {
            uav.id: uav.refresh_relay_queue_total_bits()
            for uav in self.uavs
            if uav.id != self.anchor_uav_id
        }
        accounting.anchor_access_ingress_bits = access_step_result.total_access_ingress_bits_by_uav.get(self.anchor_uav_id, 0.0)
        accounting.anchor_queue_next_bits = self._anchor_uav().refresh_relay_queue_total_bits()

        total_delivered_bits_step = float(sum(user.delivered_bits_step for user in self.users))
        self.current_step += 1
        env_state = EnvState(
            current_step=self.current_step,
            adjacency_matrix=relay_service_result.adjacency_matrix,
            lambda2=relay_service_result.lambda2,
            backhaul_capacity_bps=backhaul_capacity_bps_value,
            total_delivered_bits_step=total_delivered_bits_step,
        )

        return SimStepResult(
            env_state=env_state,
            association_result=association_result,
            access_step_result=access_step_result,
            relay_service_result=relay_service_result,
            backhaul_service_result=backhaul_service_result,
            accounting=accounting,
        )

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
            )
            energy_used_j_by_uav[uav.id] = energy_used_j
        return energy_used_j_by_uav

    def _move_ground_users(self) -> None:
        for user in self.users:
            user.move(delta_t_s=config.DELTA_T, rng=self.rng)

    def _resolve_backhaul_capacity_bps(self) -> float:
        anchor_uav = self._anchor_uav()
        if self.backhaul_type == "satellite":
            if not self.satellites:
                raise ValueError("Satellite backhaul requested but no satellite entity is available.")
            return backhaul_capacity_bps(anchor_uav.position, self.satellites[0], backhaul_type="satellite")
        if self.backhaul_type == "gbs":
            if not self.ground_base_stations:
                raise ValueError("GBS backhaul requested but no ground base station entity is available.")
            return backhaul_capacity_bps(anchor_uav.position, self.ground_base_stations[0], backhaul_type="gbs")
        raise ValueError(f"Unsupported backhaul_type: {self.backhaul_type}")

    def _anchor_uav(self) -> UAV:
        for uav in self.uavs:
            if uav.id == self.anchor_uav_id:
                return uav
        raise ValueError(f"Anchor UAV id {self.anchor_uav_id} was not found.")

    def _build_env_state(self, *, backhaul_capacity_bps_value: float, total_delivered_bits_step: float) -> EnvState:
        num_uavs = len(self.uavs)
        adjacency_matrix = np.zeros((num_uavs, num_uavs), dtype=int)
        return EnvState(
            current_step=self.current_step,
            adjacency_matrix=adjacency_matrix,
            lambda2=0.0,
            backhaul_capacity_bps=float(backhaul_capacity_bps_value),
            total_delivered_bits_step=float(total_delivered_bits_step),
        )
