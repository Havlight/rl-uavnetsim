from __future__ import annotations

import math

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.entities import GroundUser, UAV
from rl_uavnetsim.environment import SimEnv
from rl_uavnetsim.main import build_demo_entities
from rl_uavnetsim.training.features import (
    build_compact_local_observation,
    build_compact_state,
    compact_observation_dim,
    compact_state_dim,
)
from rl_uavnetsim.training.configuration import EnvConfig, EvalConfig, ModelConfig, ObservationConfig, OutputConfig, RunConfig, TrainerConfig
from rl_uavnetsim.training.mappo_trainer import run_torchrl_spike
from rl_uavnetsim.training.pettingzoo_env import PettingZooUavNetEnv, decode_movement_action


def test_compact_observation_dim_matches_v1_contract() -> None:
    assert compact_observation_dim(num_uavs=5, max_obs_users=15) == 57


def test_compact_state_dim_matches_v1_contract() -> None:
    assert compact_state_dim(num_uavs=5, num_users=60) == 155


def test_build_compact_local_observation_uses_geometry_first_layout() -> None:
    observed_uav = UAV(
        id=0,
        position=np.array([100.0, 400.0, config.UAV_HEIGHT]),
        velocity=np.array([10.0, -5.0]),
        speed=float(np.linalg.norm([10.0, -5.0])),
        direction=0.0,
        is_gateway_capable=True,
        relay_queue_bits_by_user={7: 0.4 * config.RELAY_QUEUE_REF_BITS},
    )
    peer_uav = UAV(
        id=1,
        position=np.array([1500.0, 1000.0, config.UAV_HEIGHT]),
        velocity=np.array([-8.0, 6.0]),
        speed=10.0,
        direction=0.0,
    )
    near_user = GroundUser(
        id=0,
        position=np.array([150.0, 450.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        user_access_backlog_bits=999.0,
    )
    far_user = GroundUser(
        id=1,
        position=np.array([300.0, 700.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        user_access_backlog_bits=123.0,
    )
    hidden_user = GroundUser(
        id=2,
        position=np.array([1200.0, 1200.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
    )

    observation = build_compact_local_observation(
        observed_uav,
        [observed_uav, peer_uav],
        [near_user, far_user, hidden_user],
        max_obs_users=2,
        obs_radius_m=400.0,
    )

    expected = np.array(
        [
            100.0 / config.MAP_LENGTH,
            400.0 / config.MAP_WIDTH,
            1.0,
            10.0 / config.V_MAX,
            -5.0 / config.V_MAX,
            1.0,
            0.4,
            1500.0 / config.MAP_LENGTH,
            1000.0 / config.MAP_WIDTH,
            1.0,
            -8.0 / config.V_MAX,
            6.0 / config.V_MAX,
            150.0 / config.MAP_LENGTH,
            450.0 / config.MAP_WIDTH,
            300.0 / config.MAP_LENGTH,
            700.0 / config.MAP_WIDTH,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(observation, expected)


def test_build_compact_state_flattens_uav_and_user_geometry() -> None:
    gateway_uav = UAV(
        id=0,
        position=np.array([0.0, 2000.0, config.UAV_HEIGHT]),
        velocity=np.array([20.0, 0.0]),
        speed=20.0,
        direction=0.0,
        is_gateway_capable=True,
        relay_queue_bits_by_user={1: 0.5 * config.RELAY_QUEUE_REF_BITS},
    )
    relay_uav = UAV(
        id=1,
        position=np.array([2000.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.array([0.0, -10.0]),
        speed=10.0,
        direction=0.0,
        is_gateway_capable=False,
        relay_queue_bits_by_user={},
    )
    user_a = GroundUser(
        id=10,
        position=np.array([500.0, 1000.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
    )
    user_b = GroundUser(
        id=11,
        position=np.array([1500.0, 500.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
    )

    compact_state = build_compact_state([gateway_uav, relay_uav], [user_b, user_a])

    expected = np.array(
        [
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            -0.5,
            1.0,
            0.0,
            0.5,
            0.0,
            0.25,
            0.5,
            0.75,
            0.25,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(compact_state, expected)


def test_decode_movement_action_maps_unit_box_to_rho_and_psi() -> None:
    rho_norm, psi_rad = decode_movement_action(np.array([-1.0, 1.0], dtype=np.float32))
    assert rho_norm == 0.0
    assert psi_rad == math.pi


def test_pettingzoo_wrapper_exposes_compact_spaces_and_agent_ordering() -> None:
    uavs, users, satellites, ground_base_stations = build_demo_entities(
        num_uavs=2,
        num_users=3,
        seed=3,
        backhaul_type="satellite",
    )
    env = PettingZooUavNetEnv(
        SimEnv(
            uavs=uavs,
            users=users,
            satellites=satellites,
            ground_base_stations=ground_base_stations,
            gateway_capable_uav_ids=[0],
            backhaul_type="satellite",
            rng=np.random.default_rng(9),
        ),
        max_steps=5,
        max_obs_users=2,
    )

    observations, info = env.reset(seed=11)

    assert env.possible_agents == ["uav_0", "uav_1"]
    assert env.agents == ["uav_0", "uav_1"]
    assert list(observations) == ["uav_0", "uav_1"]
    assert observations["uav_0"].shape == (compact_observation_dim(2, 2),)
    assert env.observation_space("uav_0").shape == (compact_observation_dim(2, 2),)
    assert env.action_space("uav_0").shape == (2,)
    assert info["uav_0"] == {}
    assert env.latest_env_state is not None
    assert env.latest_env_state.current_step == 0
    assert env.state().shape == (compact_state_dim(2, 3),)
    assert env.state_space.shape == (compact_state_dim(2, 3),)


def test_pettingzoo_wrapper_step_decodes_actions_and_returns_shared_team_reward() -> None:
    uavs, users, satellites, ground_base_stations = build_demo_entities(
        num_uavs=2,
        num_users=2,
        seed=5,
        backhaul_type="satellite",
    )
    env = PettingZooUavNetEnv(
        SimEnv(
            uavs=uavs,
            users=users,
            satellites=satellites,
            ground_base_stations=ground_base_stations,
            gateway_capable_uav_ids=[0],
            backhaul_type="satellite",
            rng=np.random.default_rng(13),
        ),
        max_steps=3,
        max_obs_users=2,
    )
    env.reset(seed=17)

    observations, rewards, terminations, truncations, infos = env.step(
        {
            "uav_0": np.array([-1.0, 1.0], dtype=np.float32),
            "uav_1": np.array([1.0, -1.0], dtype=np.float32),
        }
    )

    assert set(observations) == {"uav_0", "uav_1"}
    assert rewards["uav_0"] == rewards["uav_1"]
    assert terminations["uav_0"] is False
    assert truncations["uav_0"] is False
    assert infos["uav_0"] == {}
    assert env.latest_team_reward == rewards["uav_0"]
    np.testing.assert_allclose(env.last_actions_by_uav_id[0]["rho"], 0.0)
    np.testing.assert_allclose(env.last_actions_by_uav_id[0]["psi"], math.pi)
    np.testing.assert_allclose(env.last_actions_by_uav_id[1]["rho"], 1.0)
    np.testing.assert_allclose(env.last_actions_by_uav_id[1]["psi"], -math.pi)


def test_torchrl_spike_runs_against_pettingzoo_wrapper() -> None:
    run_config = RunConfig(
        seed=23,
        env=EnvConfig(num_steps=5, num_uavs=2, num_users=3, backhaul_type="satellite", demo_mode="default"),
        observation=ObservationConfig(preset="compact_v1", max_obs_users=2, obs_radius_m=500.0),
        trainer=TrainerConfig(total_frames=8, frames_per_batch=4),
        model=ModelConfig(actor_hidden_dims=(16,), critic_hidden_dims=(16,), activation="tanh"),
        eval=EvalConfig(num_eval_episodes=1, deterministic_policy=True),
        output=OutputConfig(root_dir="runs/test", run_name="spike"),
    )

    spike_summary = run_torchrl_spike(run_config)

    assert spike_summary["obs_dim"] == compact_observation_dim(2, 2)
    assert spike_summary["state_dim"] == compact_state_dim(2, 3)
    assert spike_summary["actor_output_dim"] == 2
    assert spike_summary["critic_output_dim"] == 1
