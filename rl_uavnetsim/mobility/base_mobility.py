from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MobilityState:
    position: np.ndarray
    velocity: np.ndarray
    speed: float


class BaseMobilityModel(ABC):
    @abstractmethod
    def step(
        self,
        *,
        position: np.ndarray,
        velocity: np.ndarray,
        speed: float,
        delta_t_s: float,
        rng: Optional[np.random.Generator] = None,
    ) -> MobilityState:
        """Advance one mobility step and return the updated state."""
