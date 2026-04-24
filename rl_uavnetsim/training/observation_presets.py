from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from rl_uavnetsim.entities import GroundUser, UAV
from rl_uavnetsim.scenario import ScenarioGeometry
from rl_uavnetsim.training.features import (
    build_compact_local_observation,
    build_compact_state,
    build_compact_v2_local_observation,
    compact_observation_dim,
    compact_state_dim,
    compact_v2_observation_dim,
)


ObservationBuilder = Callable[
    [UAV, Sequence[UAV], Sequence[GroundUser], int, float, ScenarioGeometry],
    np.ndarray,
]
StateBuilder = Callable[[Sequence[UAV], Sequence[GroundUser], ScenarioGeometry], np.ndarray]


@dataclass(frozen=True)
class ObservationPreset:
    name: str
    observation_dim: Callable[[int, int], int]
    observation_builder: ObservationBuilder
    state_dim: Callable[[int, int], int]
    state_builder: StateBuilder


def _compact_v1_observation_builder(
    uav: UAV,
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
    max_obs_users: int,
    obs_radius_m: float,
    geometry: ScenarioGeometry,
) -> np.ndarray:
    return build_compact_local_observation(
        uav,
        uavs,
        users,
        max_obs_users=max_obs_users,
        obs_radius_m=obs_radius_m,
        geometry=geometry,
    )


def _compact_v2_observation_builder(
    uav: UAV,
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
    max_obs_users: int,
    obs_radius_m: float,
    geometry: ScenarioGeometry,
) -> np.ndarray:
    return build_compact_v2_local_observation(
        uav,
        uavs,
        users,
        max_obs_users=max_obs_users,
        obs_radius_m=obs_radius_m,
        geometry=geometry,
    )


def _compact_state_builder(
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
    geometry: ScenarioGeometry,
) -> np.ndarray:
    return build_compact_state(uavs, users, geometry=geometry)


OBSERVATION_PRESETS: dict[str, ObservationPreset] = {
    "compact_v1": ObservationPreset(
        name="compact_v1",
        observation_dim=compact_observation_dim,
        observation_builder=_compact_v1_observation_builder,
        state_dim=compact_state_dim,
        state_builder=_compact_state_builder,
    ),
    "compact_v2": ObservationPreset(
        name="compact_v2",
        observation_dim=compact_v2_observation_dim,
        observation_builder=_compact_v2_observation_builder,
        state_dim=compact_state_dim,
        state_builder=_compact_state_builder,
    ),
}


def get_observation_preset(name: str) -> ObservationPreset:
    try:
        return OBSERVATION_PRESETS[str(name)]
    except KeyError as exc:
        supported = ", ".join(sorted(OBSERVATION_PRESETS))
        raise ValueError(f"Unsupported observation preset '{name}'. Supported presets: {supported}.") from exc
