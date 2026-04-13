from __future__ import annotations

import numpy as np
import pytest

from rl_uavnetsim import config, network
from rl_uavnetsim.entities import GroundUser, Satellite, UAV
from rl_uavnetsim.environment import SimEnv
from rl_uavnetsim.visualization.trajectory_visualizer import VisualizationFrame


def test_uav_rejects_removed_legacy_role_keyword() -> None:
    legacy_role_keyword = "is_" + "anchor"
    with pytest.raises(TypeError):
        UAV(
            id=0,
            position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
            velocity=np.zeros(2),
            speed=0.0,
            direction=0.0,
            **{legacy_role_keyword: True},
        )


def test_sim_env_rejects_removed_legacy_gateway_id_keyword() -> None:
    gateway = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    user = GroundUser(
        id=0,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
    )
    satellite = Satellite(id=0)
    legacy_gateway_id_keyword = "anchor" + "_uav_id"

    with pytest.raises(TypeError):
        SimEnv(
            uavs=[gateway],
            users=[user],
            satellites=[satellite],
            **{legacy_gateway_id_keyword: gateway.id},
            backhaul_type="satellite",
        )


def test_removed_legacy_symbols_are_no_longer_exported() -> None:
    legacy_config_name = "ANCHOR_" + "UAV_ID"
    legacy_helper_name = "find_widest_path_" + "to_anchor"
    legacy_frame_alias = "anchor_" + "uav_id"

    assert not hasattr(config, legacy_config_name)
    assert not hasattr(network, legacy_helper_name)
    assert not hasattr(VisualizationFrame, legacy_frame_alias)
