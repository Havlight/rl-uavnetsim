from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.utils.helpers import ensure_3d_position


@dataclass
class Satellite:
    id: int
    position: np.ndarray = None  # type: ignore[assignment]
    bandwidth_hz: float = config.B_SAT
    rx_gain_db: float = config.G_RX_SAT_DB
    atmospheric_loss_db: float = config.L_ATM_DB
    noise_figure_db: float = config.NF_SAT

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = np.asarray(config.SAT_POSITION, dtype=float)
        self.position = ensure_3d_position(self.position, default_z=config.SAT_ALTITUDE)
