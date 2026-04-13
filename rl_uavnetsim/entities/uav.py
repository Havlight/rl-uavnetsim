from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.energy.energy_model import EnergyModelBase
from rl_uavnetsim.utils.helpers import clamp, ensure_2d_velocity, ensure_3d_position


@dataclass
class UAV:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    speed: float
    direction: float
    is_gateway_capable: bool = False
    residual_energy_j: float = config.E_INITIAL
    associated_user_ids: list[int] = field(default_factory=list)
    access_ingress_bits_step: float = 0.0
    relay_queue_bits_by_user: dict[int, float] = field(default_factory=dict)
    relay_forwarded_bits_step: float = 0.0
    backhaul_forwarded_bits_step: float = 0.0
    energy_model: Optional[EnergyModelBase] = None

    def __post_init__(self) -> None:
        self.position = ensure_3d_position(self.position, default_z=config.UAV_HEIGHT)
        self.velocity = ensure_2d_velocity(self.velocity)
        self.speed = float(self.speed)
        self.direction = float(self.direction)
        self.residual_energy_j = float(self.residual_energy_j)

    def reset_step_counters(self) -> None:
        self.associated_user_ids = []
        self.access_ingress_bits_step = 0.0
        self.relay_forwarded_bits_step = 0.0
        self.backhaul_forwarded_bits_step = 0.0

    @property
    def relay_queue_total_bits(self) -> float:
        return float(sum(self.relay_queue_bits_by_user.values()))

    def refresh_relay_queue_total_bits(self) -> float:
        return self.relay_queue_total_bits

    def enqueue_relay_bits(
        self,
        user_id: int,
        bits: float,
        *,
        count_as_access_ingress: bool = True,
    ) -> float:
        bits = max(0.0, float(bits))
        if bits == 0.0:
            return 0.0
        self.relay_queue_bits_by_user[user_id] = self.relay_queue_bits_by_user.get(user_id, 0.0) + bits
        if count_as_access_ingress:
            self.access_ingress_bits_step += bits
        return bits

    def proportional_dequeue(self, service_budget_bits: float) -> dict[int, float]:
        queue_total_bits = self.refresh_relay_queue_total_bits()
        if queue_total_bits <= 0.0:
            return {}

        service_budget_bits = max(0.0, min(float(service_budget_bits), queue_total_bits))
        forwarded_bits_by_user: dict[int, float] = {}
        for user_id, queued_bits in list(self.relay_queue_bits_by_user.items()):
            if queued_bits <= 0.0:
                continue
            share_norm = queued_bits / queue_total_bits
            forwarded_bits = share_norm * service_budget_bits
            remaining_bits = max(0.0, queued_bits - forwarded_bits)
            if remaining_bits <= config.EPSILON:
                self.relay_queue_bits_by_user.pop(user_id, None)
            else:
                self.relay_queue_bits_by_user[user_id] = remaining_bits
            forwarded_bits_by_user[user_id] = forwarded_bits

        return forwarded_bits_by_user

    def register_relay_forwarded_bits(self, forwarded_bits: float) -> float:
        forwarded_bits = max(0.0, float(forwarded_bits))
        self.relay_forwarded_bits_step += forwarded_bits
        return forwarded_bits

    def register_backhaul_forwarded_bits(self, forwarded_bits: float) -> float:
        forwarded_bits = max(0.0, float(forwarded_bits))
        self.backhaul_forwarded_bits_step += forwarded_bits
        return forwarded_bits

    def move_by_action(
        self,
        rho_norm: float,
        psi_rad: float,
        delta_t_s: float = config.DELTA_T,
        v_max_mps: float = config.V_MAX,
    ) -> tuple[np.ndarray, float, float]:
        delta_t_s = float(delta_t_s)
        rho_norm = clamp(float(rho_norm), 0.0, 1.0)
        requested_distance_m = rho_norm * float(v_max_mps) * delta_t_s

        dx_m = requested_distance_m * math.cos(float(psi_rad))
        dy_m = requested_distance_m * math.sin(float(psi_rad))

        next_position = self.position.copy()
        next_position[0] = clamp(next_position[0] + dx_m, 0.0, config.MAP_LENGTH)
        next_position[1] = clamp(next_position[1] + dy_m, 0.0, config.MAP_WIDTH)
        next_position[2] = self.position[2]

        actual_distance_m = float(np.linalg.norm(next_position[:2] - self.position[:2]))
        actual_speed_mps = actual_distance_m / delta_t_s if delta_t_s > 0.0 else 0.0

        self.position = next_position
        self.velocity = np.array(
            [
                actual_speed_mps * math.cos(float(psi_rad)),
                actual_speed_mps * math.sin(float(psi_rad)),
            ],
            dtype=float,
        )
        self.speed = actual_speed_mps
        self.direction = float(psi_rad)

        energy_used_j = 0.0
        if self.energy_model is not None:
            energy_used_j = self.energy_model.step_energy_j(
                speed_mps=self.speed,
                delta_t_s=delta_t_s,
                distance_m=actual_distance_m,
            )
            self.residual_energy_j = max(0.0, self.residual_energy_j - energy_used_j)

        return self.position, actual_distance_m, energy_used_j
