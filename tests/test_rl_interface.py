from __future__ import annotations

import math

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.entities import GroundUser, Satellite, UAV
from rl_uavnetsim.environment import EnvState, SimEnv
from rl_uavnetsim.rl_interface import (
    LinUCBStub,
    MAPPOStub,
    MultiAgentUavNetEnv,
    build_global_state,
    build_local_observation,
)


def _make_env_state(*, adjacency_matrix: np.ndarray, active_gateway_uav_ids: tuple[int, ...] = ()) -> EnvState:
    num_uavs = int(adjacency_matrix.shape[0])
    return EnvState(
        current_step=0,
        adjacency_matrix=np.asarray(adjacency_matrix, dtype=int),
        lambda2=1.0,
        backhaul_capacity_bps=8.0e6,
        total_delivered_bits_step=0.0,
        active_gateway_uav_ids=active_gateway_uav_ids,
        routing_next_hop_by_uav={uav_id: None for uav_id in range(num_uavs)},
        reachable_gateway_count_by_uav={uav_id: len(active_gateway_uav_ids) for uav_id in range(num_uavs)},
        backhaul_capacity_bps_by_gateway={uav_id: 8.0e6 for uav_id in active_gateway_uav_ids},
        best_gateway_path_capacity_bps_by_uav={uav_id: 6.0e6 for uav_id in range(num_uavs)},
        best_gateway_backhaul_capacity_bps_by_uav={uav_id: 8.0e6 for uav_id in range(num_uavs)},
    )


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


def test_build_local_observation_uses_normalized_absolute_geometry() -> None:
    gateway = UAV(
        id=0,
        position=np.array([500.0, 1000.0, config.UAV_HEIGHT]),
        velocity=np.array([10.0, -5.0]),
        speed=float(np.linalg.norm([10.0, -5.0])),
        direction=0.0,
        is_gateway_capable=True,
        residual_energy_j=0.5 * config.E_INITIAL,
        associated_user_ids=[10, 11],
        relay_queue_bits_by_user={10: 0.4 * config.RELAY_QUEUE_REF_BITS},
    )
    relay = UAV(
        id=1,
        position=np.array([1500.0, 250.0, config.UAV_HEIGHT]),
        velocity=np.array([-20.0, 5.0]),
        speed=float(np.linalg.norm([-20.0, 5.0])),
        direction=0.0,
    )
    near_user = GroundUser(
        id=10,
        position=np.array([600.0, 900.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        associated_uav_id=gateway.id,
        user_access_backlog_bits=0.25 * config.ACCESS_BACKLOG_REF_BITS,
    )
    far_user = GroundUser(
        id=11,
        position=np.array([700.0, 1100.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        associated_uav_id=gateway.id,
        user_access_backlog_bits=0.5 * config.ACCESS_BACKLOG_REF_BITS,
    )
    hidden_user = GroundUser(
        id=12,
        position=np.array([1800.0, 1800.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        associated_uav_id=-1,
        user_access_backlog_bits=0.9 * config.ACCESS_BACKLOG_REF_BITS,
    )
    env_state = _make_env_state(adjacency_matrix=np.array([[0, 1], [1, 0]], dtype=int), active_gateway_uav_ids=(0,))

    observation = build_local_observation(
        gateway,
        [gateway, relay],
        [far_user, hidden_user, near_user],
        env_state,
        max_obs_users_pad=2,
        obs_radius_m=400.0,
    )

    expected_prefix = np.array(
        [
            500.0 / config.MAP_LENGTH,
            1000.0 / config.MAP_WIDTH,
            config.UAV_HEIGHT / config.UAV_HEIGHT,
            10.0 / config.V_MAX,
            -5.0 / config.V_MAX,
            0.5,
            1.0,
            0.4,
            2.0 / config.NUM_USERS,
            1.0,
            6.0e6 / max(config.THROUGHPUT_REF_BITS / config.DELTA_T, config.EPSILON),
            8.0e6 / max(config.THROUGHPUT_REF_BITS / config.DELTA_T, config.EPSILON),
            1500.0 / config.MAP_LENGTH,
            250.0 / config.MAP_WIDTH,
            config.UAV_HEIGHT / config.UAV_HEIGHT,
            -20.0 / config.V_MAX,
            5.0 / config.V_MAX,
            1.0,
            600.0 / config.MAP_LENGTH,
            900.0 / config.MAP_WIDTH,
            0.25,
            700.0 / config.MAP_LENGTH,
            1100.0 / config.MAP_WIDTH,
            0.5,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(observation, expected_prefix)


def test_build_global_state_returns_normalized_geometry_and_gateway_masks() -> None:
    gateway = UAV(
        id=0,
        position=np.array([1000.0, 500.0, config.UAV_HEIGHT]),
        velocity=np.array([5.0, 10.0]),
        speed=float(np.linalg.norm([5.0, 10.0])),
        direction=0.0,
        is_gateway_capable=True,
        residual_energy_j=0.8 * config.E_INITIAL,
        relay_queue_bits_by_user={7: 0.2 * config.RELAY_QUEUE_REF_BITS},
    )
    relay = UAV(
        id=1,
        position=np.array([250.0, 1500.0, config.UAV_HEIGHT]),
        velocity=np.array([-10.0, 0.0]),
        speed=10.0,
        direction=0.0,
        is_gateway_capable=False,
        residual_energy_j=0.4 * config.E_INITIAL,
        relay_queue_bits_by_user={8: 0.6 * config.RELAY_QUEUE_REF_BITS},
    )
    users = [
        GroundUser(
            id=7,
            position=np.array([400.0, 600.0, 0.0]),
            velocity=np.zeros(2),
            speed=0.0,
            associated_uav_id=0,
            user_access_backlog_bits=0.3 * config.ACCESS_BACKLOG_REF_BITS,
        ),
        GroundUser(
            id=8,
            position=np.array([1600.0, 1400.0, 0.0]),
            velocity=np.zeros(2),
            speed=0.0,
            associated_uav_id=1,
            user_access_backlog_bits=0.7 * config.ACCESS_BACKLOG_REF_BITS,
        ),
    ]
    env_state = _make_env_state(adjacency_matrix=np.array([[0, 0], [0, 0]], dtype=int), active_gateway_uav_ids=(0,))

    global_state = build_global_state([gateway, relay], users, env_state)

    assert set(global_state) == {
        "uav_positions_norm",
        "uav_velocities_norm",
        "uav_energies_norm",
        "uav_queue_totals_norm",
        "gateway_capable_mask",
        "active_gateway_mask",
        "reachable_gateway_counts_norm",
        "backhaul_capacity_by_uav_norm",
        "user_positions_norm",
        "user_access_backlogs_norm",
        "associations",
        "connectivity_matrix",
        "backhaul_capacity_norm",
    }
    np.testing.assert_allclose(
        global_state["uav_positions_norm"],
        np.array(
            [
                [1000.0 / config.MAP_LENGTH, 500.0 / config.MAP_WIDTH, 1.0],
                [250.0 / config.MAP_LENGTH, 1500.0 / config.MAP_WIDTH, 1.0],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(
        global_state["uav_velocities_norm"],
        np.array(
            [
                [5.0 / config.V_MAX, 10.0 / config.V_MAX],
                [-10.0 / config.V_MAX, 0.0],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(global_state["gateway_capable_mask"], np.array([1.0, 0.0], dtype=float))
    np.testing.assert_allclose(global_state["active_gateway_mask"], np.array([1.0, 0.0], dtype=float))
    np.testing.assert_allclose(
        global_state["user_positions_norm"],
        np.array(
            [
                [400.0 / config.MAP_LENGTH, 600.0 / config.MAP_WIDTH],
                [1600.0 / config.MAP_LENGTH, 1400.0 / config.MAP_WIDTH],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(
        global_state["user_access_backlogs_norm"],
        np.array([0.3, 0.7], dtype=float),
    )


def test_default_observation_dim_matches_absolute_geometry_contract() -> None:
    assert config.OBS_DIM == 126
