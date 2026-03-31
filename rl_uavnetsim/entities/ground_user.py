from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.mobility.base_mobility import BaseMobilityModel
from rl_uavnetsim.utils.helpers import ensure_2d_velocity, ensure_3d_position


@dataclass
class GroundUser:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    speed: float
    associated_uav_id: int = -1
    demand_rate_bps: float = config.USER_DEMAND_RATE_BPS
    user_access_backlog_bits: float = 0.0
    arrived_bits_step: float = 0.0
    access_uploaded_bits_step: float = 0.0
    delivered_bits_step: float = 0.0
    final_rate_bps: float = 0.0
    avg_throughput_bps: float = 0.0
    mobility_model: Optional[BaseMobilityModel] = None

    def __post_init__(self) -> None:
        self.position = ensure_3d_position(self.position, default_z=0.0)
        self.velocity = ensure_2d_velocity(self.velocity)
        self.speed = float(self.speed)
        self.demand_rate_bps = float(self.demand_rate_bps)
        self.user_access_backlog_bits = float(self.user_access_backlog_bits)
        self.arrived_bits_step = float(self.arrived_bits_step)
        self.access_uploaded_bits_step = float(self.access_uploaded_bits_step)
        self.delivered_bits_step = float(self.delivered_bits_step)
        self.final_rate_bps = float(self.final_rate_bps)
        self.avg_throughput_bps = float(self.avg_throughput_bps)

    def reset_step_counters(self) -> None:
        self.arrived_bits_step = 0.0
        self.access_uploaded_bits_step = 0.0
        self.delivered_bits_step = 0.0
        self.final_rate_bps = 0.0

    def add_demand_bits(self, delta_t_s: float = config.DELTA_T) -> float:
        arrived_bits = self.demand_rate_bps * float(delta_t_s)
        self.arrived_bits_step += arrived_bits
        self.user_access_backlog_bits += arrived_bits
        return arrived_bits

    def consume_access_bits(self, uploaded_bits: float) -> float:
        uploaded_bits = max(0.0, min(float(uploaded_bits), self.user_access_backlog_bits))
        self.user_access_backlog_bits -= uploaded_bits
        self.access_uploaded_bits_step += uploaded_bits
        return uploaded_bits

    def add_delivered_bits(self, delivered_bits: float, delta_t_s: float = config.DELTA_T) -> float:
        delivered_bits = max(0.0, float(delivered_bits))
        self.delivered_bits_step += delivered_bits
        self.final_rate_bps = self.delivered_bits_step / float(delta_t_s)
        return delivered_bits

    def move(self, delta_t_s: float = config.DELTA_T, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if self.mobility_model is None:
            self.position[:2] = self.position[:2] + self.velocity * float(delta_t_s)
            self.speed = float(np.linalg.norm(self.velocity))
            return self.position

        mobility_state = self.mobility_model.step(
            position=self.position,
            velocity=self.velocity,
            speed=self.speed,
            delta_t_s=delta_t_s,
            rng=rng,
        )
        self.position = mobility_state.position
        self.velocity = mobility_state.velocity
        self.speed = mobility_state.speed
        return self.position
