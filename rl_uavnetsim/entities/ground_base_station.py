from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.utils.helpers import ensure_3d_position


@dataclass
class GroundBaseStation:
    id: int
    position: np.ndarray = None  # type: ignore[assignment]
    bandwidth_hz: float = config.GBS_BW
    rx_gain_db: float = config.G_RX_GBS_DB
    atmospheric_loss_db: float = 0.0
    noise_figure_db: float = config.NF_GBS

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = np.asarray(config.GBS_POSITIONS[0], dtype=float)
        self.position = ensure_3d_position(self.position, default_z=0.0)
