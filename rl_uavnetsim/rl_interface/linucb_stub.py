from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rl_uavnetsim import config


@dataclass
class LinUCBStub:
    alpha_candidates: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    fixed_alpha: float = config.PF_ALPHA_DEFAULT
    last_context: np.ndarray | None = None
    reward_history: list[float] = field(default_factory=list)

    def select_alpha(self, context: np.ndarray | None) -> float:
        self.last_context = None if context is None else np.asarray(context, dtype=float)
        if float(self.fixed_alpha) in self.alpha_candidates:
            return float(self.fixed_alpha)
        return min(self.alpha_candidates, key=lambda candidate: abs(candidate - float(self.fixed_alpha)))

    def update(self, context: np.ndarray | None, reward: float) -> None:
        self.last_context = None if context is None else np.asarray(context, dtype=float)
        self.reward_history.append(float(reward))
