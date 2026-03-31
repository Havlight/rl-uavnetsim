from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rl_uavnetsim import config


class EnergyModelBase(ABC):
    @abstractmethod
    def power_consumption_w(self, speed_mps: float) -> float:
        """Return propulsion power in watts for the given speed."""

    def step_energy_j(
        self,
        speed_mps: float,
        delta_t_s: float = config.DELTA_T,
        distance_m: float | None = None,
    ) -> float:
        return self.power_consumption_w(speed_mps=float(speed_mps)) * float(delta_t_s)


@dataclass
class SimplifiedEnergyModel(EnergyModelBase):
    hover_energy_per_s_j: float = config.E_HOVER
    fly_energy_per_m_j: float = config.E_FLY

    def power_consumption_w(self, speed_mps: float) -> float:
        return float(self.hover_energy_per_s_j) + float(self.fly_energy_per_m_j) * max(0.0, float(speed_mps))

    def step_energy_j(
        self,
        speed_mps: float,
        delta_t_s: float = config.DELTA_T,
        distance_m: float | None = None,
    ) -> float:
        if distance_m is None:
            distance_m = max(0.0, float(speed_mps)) * float(delta_t_s)
        return float(self.hover_energy_per_s_j) * float(delta_t_s) + float(self.fly_energy_per_m_j) * max(0.0, float(distance_m))


@dataclass
class Zeng2019EnergyModel(EnergyModelBase):
    delta: float = config.PROFILE_DRAG_COEFFICIENT
    rho: float = config.AIR_DENSITY
    s: float = config.ROTOR_SOLIDITY
    a: float = config.ROTOR_DISC_AREA
    omega: float = config.BLADE_ANGULAR_VELOCITY
    r: float = config.ROTOR_RADIUS
    k: float = config.INCREMENTAL_CORRECTION_FACTOR
    w: float = config.AIRCRAFT_WEIGHT
    u_tip: float = config.ROTOR_BLADE_TIP_SPEED
    v0: float = config.MEAN_ROTOR_VELOCITY
    d0: float = config.FUSELAGE_DRAG_RATIO

    def power_consumption_w(self, speed_mps: float) -> float:
        speed_mps = max(0.0, float(speed_mps))
        p0 = (self.delta / 8.0) * self.rho * self.s * self.a * (self.omega ** 3) * (self.r ** 3)
        pi = (1.0 + self.k) * (self.w ** 1.5) / math.sqrt(2.0 * self.rho * self.a)
        blade_profile_w = p0 * (1.0 + (3.0 * speed_mps ** 2) / (self.u_tip ** 2))
        induced_w = pi * math.sqrt(math.sqrt(1.0 + speed_mps ** 4 / (4.0 * self.v0 ** 4)) - speed_mps ** 2 / (2.0 * self.v0 ** 2))
        parasite_w = 0.5 * self.d0 * self.rho * self.s * self.a * speed_mps ** 3
        return blade_profile_w + induced_w + parasite_w
