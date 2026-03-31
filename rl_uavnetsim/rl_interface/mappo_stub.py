from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass
class MAPPOStub:
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.rng = self.rng or np.random.default_rng()

    def act(
        self,
        observations_by_agent: Mapping[int, np.ndarray],
        *,
        deterministic: bool = False,
    ) -> dict[int, dict[str, float]]:
        actions_by_agent: dict[int, dict[str, float]] = {}
        for agent_id in observations_by_agent:
            if deterministic:
                rho_norm = 0.0
                psi_rad = 0.0
            else:
                rho_norm = float(self.rng.uniform(0.0, 1.0))
                psi_rad = float(self.rng.uniform(-math.pi, math.pi))
            actions_by_agent[agent_id] = {"rho": rho_norm, "psi": psi_rad}
        return actions_by_agent

    def update(self, *args, **kwargs) -> dict[str, float]:
        return {"policy_loss": 0.0, "value_loss": 0.0}
