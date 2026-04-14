from __future__ import annotations

import math

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.entities import GroundUser, Satellite, UAV
from rl_uavnetsim.environment import SimEnv
from rl_uavnetsim.rl_interface import LinUCBStub, MAPPOStub, MultiAgentUavNetEnv


def test_linucb_stub_returns_configured_alpha() -> None:
    linucb_stub = LinUCBStub(fixed_alpha=2.0)

    selected_alpha = linucb_stub.select_alpha(np.array([1.0, 0.2, 0.3]))
    linucb_stub.update(np.array([1.0, 0.2, 0.3]), 0.75)

    assert selected_alpha == 2.0
    assert linucb_stub.last_context is not None
    assert linucb_stub.reward_history == [0.75]


def test_mappo_stub_generates_valid_action_ranges() -> None:
    mappo_stub = MAPPOStub(rng=np.random.default_rng(1))

    actions = mappo_stub.act({0: np.zeros(4), 1: np.ones(4)})

    assert set(actions) == {0, 1}
    for action in actions.values():
        assert 0.0 <= action["rho"] <= 1.0
        assert -math.pi <= action["psi"] <= math.pi


def test_multi_agent_env_reset_and_step_share_team_reward() -> None:
    gateway = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    relay = UAV(
        id=1,
        position=np.array([50.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    user = GroundUser(
        id=5,
        position=np.array([55.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        demand_rate_bps=1000.0,
        user_access_backlog_bits=1000.0,
    )
    sim_env = SimEnv(
        uavs=[gateway, relay],
        users=[user],
        satellites=[Satellite(id=0)],
        gateway_capable_uav_ids=[gateway.id],
        backhaul_type="satellite",
    )
    marl_env = MultiAgentUavNetEnv(
        sim_env,
        max_steps=2,
        alpha_controllers={
            0: LinUCBStub(fixed_alpha=1.0),
            1: LinUCBStub(fixed_alpha=0.5),
        },
    )

    observations_by_agent, info = marl_env.reset()
    next_observations_by_agent, rewards_by_agent, terminated_by_agent, truncated_by_agent, step_info = marl_env.step(
        {
            0: {"rho": 0.0, "psi": 0.0},
            1: {"rho": 0.0, "psi": 0.0},
        }
    )

    assert set(observations_by_agent) == {0, 1}
    assert observations_by_agent[0].shape == observations_by_agent[1].shape
    assert "global_state" in info
    assert set(next_observations_by_agent) == {0, 1}
    assert rewards_by_agent[0] == rewards_by_agent[1]
    assert step_info["team_reward"] == rewards_by_agent[0]
    assert terminated_by_agent["__all__"] is False
    assert truncated_by_agent["__all__"] is False
    assert set(step_info["alpha_by_uav"]) == {0, 1}
