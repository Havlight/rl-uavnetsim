from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.entities import GroundUser, UAV
from rl_uavnetsim.environment import SimEnv
from rl_uavnetsim.main import build_demo_entities
from rl_uavnetsim.training.features import (
    build_compact_local_observation,
    build_compact_state,
    build_compact_v2_local_observation,
    compact_observation_dim,
    compact_state_dim,
    compact_v2_observation_dim,
)
from rl_uavnetsim.training.configuration import EnvConfig, EvalConfig, ModelConfig, ObservationConfig, OutputConfig, RunConfig, TrainerConfig, load_run_config
from rl_uavnetsim.training.mappo_trainer import (
    _build_progress_postfix,
    _format_best_checkpoint_message,
    build_training_env,
    run_torchrl_spike,
)
from rl_uavnetsim.training.observation_presets import get_observation_preset
from rl_uavnetsim.training.pettingzoo_env import PettingZooUavNetEnv, decode_movement_action


def test_compact_observation_dim_matches_v1_contract() -> None:
    assert compact_observation_dim(num_uavs=5, max_obs_users=15) == 57


def test_compact_v2_observation_dim_adds_backlog_association_and_self_load() -> None:
    assert compact_v2_observation_dim(num_uavs=5, max_obs_users=15) == 88


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


def test_build_compact_v2_local_observation_adds_urgency_and_association_features() -> None:
    observed_uav = UAV(
        id=0,
        position=np.array([100.0, 400.0, config.UAV_HEIGHT]),
        velocity=np.array([10.0, -5.0]),
        speed=float(np.linalg.norm([10.0, -5.0])),
        direction=0.0,
        is_gateway_capable=True,
        associated_user_ids=[0],
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
        associated_uav_id=0,
        user_access_backlog_bits=0.25 * config.ACCESS_BACKLOG_REF_BITS,
    )
    far_user = GroundUser(
        id=1,
        position=np.array([300.0, 700.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        associated_uav_id=1,
        user_access_backlog_bits=0.5 * config.ACCESS_BACKLOG_REF_BITS,
    )

    observation = build_compact_v2_local_observation(
        observed_uav,
        [observed_uav, peer_uav],
        [near_user, far_user],
        max_obs_users=2,
        obs_radius_m=400.0,
    )

    assert observation.shape == (compact_v2_observation_dim(2, 2),)
    np.testing.assert_allclose(observation[7], 0.5)
    user_block = observation[13:]
    np.testing.assert_allclose(
        user_block,
        np.array(
            [
                150.0 / config.MAP_LENGTH,
                450.0 / config.MAP_WIDTH,
                0.25,
                1.0,
                300.0 / config.MAP_LENGTH,
                700.0 / config.MAP_WIDTH,
                0.5,
                0.0,
            ],
            dtype=np.float32,
        ),
    )


def test_observation_preset_registry_owns_observation_and_state_contracts() -> None:
    compact_v1 = get_observation_preset("compact_v1")
    compact_v2 = get_observation_preset("compact_v2")

    assert compact_v1.observation_dim(5, 15) == 57
    assert compact_v2.observation_dim(5, 15) == 88
    assert compact_v1.state_dim(5, 60) == compact_v2.state_dim(5, 60) == 155


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


def test_build_progress_postfix_includes_core_training_metrics() -> None:
    postfix = _build_progress_postfix(
        update_index=2,
        total_frames=512,
        batch_reward_mean=1.23456,
        update_stats={
            "policy_loss": -0.125,
            "value_loss": 0.75,
        },
    )

    assert postfix == {
        "upd": "3",
        "frames": "512",
        "reward": "1.235",
        "pi": "-0.125",
        "v": "0.750",
        "eval": "-",
    }


def test_build_progress_postfix_includes_eval_mean_reward_when_available() -> None:
    postfix = _build_progress_postfix(
        update_index=0,
        total_frames=64,
        batch_reward_mean=0.5,
        update_stats={
            "policy_loss": 0.25,
            "value_loss": 1.5,
        },
        eval_mean_team_reward=2.3456,
    )

    assert postfix == {
        "upd": "1",
        "frames": "64",
        "reward": "0.500",
        "pi": "0.250",
        "v": "1.500",
        "eval": "2.346",
    }


def test_format_best_checkpoint_message_includes_reward_and_path() -> None:
    message = _format_best_checkpoint_message(
        update_index=4,
        total_frames=320,
        mean_team_reward=12.3456,
        checkpoint_path="runs/demo/checkpoints/best.pt",
    )

    assert message == (
        "[best checkpoint improved] "
        "update=5 "
        "frames=320 "
        "eval_mean_reward=12.346 "
        "path=runs/demo/checkpoints/best.pt"
    )


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


def test_load_run_config_accepts_explicit_training_scenario_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "explicit_training_scenario.yaml"
    config_path.write_text(
        """
seed: 42

env:
  num_steps: 60
  num_uavs: 3
  num_users: 30
  backhaul_type: satellite
  user_demand_rate_bps: 2.0e6
  orbit_radius_m: 600.0
  user_speed_mean_mps: 3.5
  user_distribution: hotspots
  spawn_margin: 0.1
  association_min_rate_bps: 0.5e6
  max_access_range_m: 800.0

observation:
  preset: compact_v1
  max_obs_users: 15
  obs_radius_m: 500.0

trainer:
  frames_per_batch: 300
  total_frames: 14400
  ppo_epochs: 4
  minibatch_size: 128
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coef: 0.0
  value_coef: 0.5
  lr: 0.0003
  device: cuda
  checkpoint_interval: 2
  eval_interval: 2

model:
  actor_hidden_dims: [128, 128, 128]
  critic_hidden_dims: [256, 256, 256]
  activation: tanh

eval:
  num_eval_episodes: 5
  deterministic_policy: true

output:
  root_dir: runs
  run_name: mappo_satellite_3uav_medium
""",
        encoding="utf-8",
    )
    run_config = load_run_config(config_path)

    assert run_config.env.user_demand_rate_bps == 2.0e6
    assert run_config.env.orbit_radius_m == 600.0
    assert run_config.env.user_speed_mean_mps == 3.5
    assert run_config.env.user_distribution == "hotspots"
    assert run_config.env.spawn_margin == 0.1
    assert run_config.env.association_min_rate_bps == config.R_MIN
    assert run_config.env.max_access_range_m == 800.0


def test_build_training_env_prefers_explicit_scenario_fields_over_demo_mode_preset() -> None:
    run_config = RunConfig(
        seed=31,
        env=EnvConfig(
            num_steps=5,
            num_uavs=3,
            num_users=6,
            backhaul_type="satellite",
            demo_mode="default",
            user_demand_rate_bps=1.75e6,
            orbit_radius_m=444.0,
            user_speed_mean_mps=1.25,
            user_distribution="hotspots",
            spawn_margin=0.02,
            association_min_rate_bps=9.0e6,
            max_access_range_m=555.0,
        ),
        observation=ObservationConfig(preset="compact_v1", max_obs_users=4, obs_radius_m=500.0),
        trainer=TrainerConfig(total_frames=8, frames_per_batch=4),
        model=ModelConfig(actor_hidden_dims=(16,), critic_hidden_dims=(16,), activation="tanh"),
        eval=EvalConfig(num_eval_episodes=1, deterministic_policy=True),
        output=OutputConfig(root_dir="runs/test", run_name="explicit_scenario"),
    )

    env = build_training_env(run_config, seed=run_config.seed)

    np.testing.assert_allclose([user.demand_rate_bps for user in env.sim_env.users], 1.75e6)
    np.testing.assert_allclose(
        [user.mobility_model.speed_mean_mps for user in env.sim_env.users if user.mobility_model is not None],
        1.25,
    )
    orbit_radius_m = float(np.linalg.norm(env.sim_env.uavs[1].position[:2] - env.sim_env.uavs[0].position[:2]))
    np.testing.assert_allclose(orbit_radius_m, 444.0)
    assert env.sim_env.association_min_rate_bps == 9.0e6
    assert env.sim_env.max_access_range_m == 555.0
    xs = np.asarray([user.position[0] for user in env.sim_env.users], dtype=float)
    ys = np.asarray([user.position[1] for user in env.sim_env.users], dtype=float)
    assert xs.min() >= 0.02 * config.MAP_LENGTH - 1e-6
    assert ys.min() >= 0.02 * config.MAP_WIDTH - 1e-6


def test_build_training_env_applies_scenario_map_size_to_entities_and_normalization() -> None:
    run_config = RunConfig(
        seed=41,
        env=EnvConfig(
            num_steps=5,
            num_uavs=3,
            num_users=8,
            backhaul_type="satellite",
            map_length_m=3000.0,
            map_width_m=2500.0,
            user_distribution="uniform",
            spawn_margin=0.02,
        ),
        observation=ObservationConfig(preset="compact_v2", max_obs_users=4, obs_radius_m=800.0),
        trainer=TrainerConfig(total_frames=8, frames_per_batch=4),
        model=ModelConfig(actor_hidden_dims=(16,), critic_hidden_dims=(16,), activation="tanh"),
        eval=EvalConfig(num_eval_episodes=1, deterministic_policy=True),
        output=OutputConfig(root_dir="runs/test", run_name="large_map"),
    )

    env = build_training_env(run_config, seed=run_config.seed)

    np.testing.assert_allclose(env.sim_env.uavs[0].position[:2], [1500.0, 1250.0])
    np.testing.assert_allclose(env.sim_env.satellites[0].position[:2], [1500.0, 1250.0])
    assert env.sim_env.map_length_m == 3000.0
    assert env.sim_env.map_width_m == 2500.0
    observations, _ = env.reset(seed=run_config.seed)
    assert observations["uav_0"].shape == (compact_v2_observation_dim(3, 4),)
    np.testing.assert_allclose(observations["uav_0"][:2], [0.5, 0.5])
    for user in env.sim_env.users:
        assert user.mobility_model.x_bounds_m == (0.0, 3000.0)
        assert user.mobility_model.y_bounds_m == (0.0, 2500.0)
