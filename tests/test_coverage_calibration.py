from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.allocation import associate_users_to_uavs
from rl_uavnetsim.entities import GroundUser, Satellite, UAV
from rl_uavnetsim.environment import SimEnv
from rl_uavnetsim.main import build_demo_entities
from rl_uavnetsim.rl_interface import MultiAgentUavNetEnv
from rl_uavnetsim.training.configuration import load_run_config


def test_load_run_config_accepts_spawn_margin_and_association_min_rate(tmp_path: Path) -> None:
    config_path = tmp_path / "coverage.yaml"
    config_path.write_text(
        """
seed: 1
env:
  num_steps: 20
  num_uavs: 3
  num_users: 30
  backhaul_type: satellite
  user_demand_rate_bps: 2.0e6
  orbit_radius_m: 600.0
  user_speed_mean_mps: 3.5
  user_distribution: uniform
  spawn_margin: 0.02
  association_min_rate_bps: 3.0e6
observation:
  preset: compact_v1
  max_obs_users: 15
  obs_radius_m: 500.0
trainer:
  frames_per_batch: 300
  total_frames: 600
  ppo_epochs: 2
  minibatch_size: 128
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coef: 0.0
  value_coef: 0.5
  lr: 0.0003
  device: cpu
  checkpoint_interval: 1
  eval_interval: 1
model:
  actor_hidden_dims: [64, 64]
  critic_hidden_dims: [128, 128]
  activation: tanh
eval:
  num_eval_episodes: 1
  deterministic_policy: true
output:
  root_dir: runs
  run_name: coverage_test
""",
        encoding="utf-8",
    )

    run_config = load_run_config(config_path)

    assert run_config.env.spawn_margin == 0.02
    assert run_config.env.association_min_rate_bps == 3.0e6


def test_build_demo_entities_respects_spawn_margin_for_uniform_users() -> None:
    _, users, _, _ = build_demo_entities(
        num_uavs=3,
        num_users=200,
        seed=7,
        backhaul_type="satellite",
        user_distribution="uniform",
        spawn_margin=0.02,
    )

    xs = np.asarray([user.position[0] for user in users], dtype=float)
    ys = np.asarray([user.position[1] for user in users], dtype=float)

    assert xs.min() < 0.1 * config.MAP_LENGTH
    assert ys.min() < 0.1 * config.MAP_WIDTH
    assert xs.max() > 0.9 * config.MAP_LENGTH
    assert ys.max() > 0.9 * config.MAP_WIDTH


def test_sim_env_step_respects_association_min_rate_bps() -> None:
    uav = UAV(
        id=0,
        position=np.array([config.MAP_LENGTH / 2.0, config.MAP_WIDTH / 2.0, config.UAV_HEIGHT], dtype=float),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    user = GroundUser(
        id=0,
        position=np.array([0.0, 0.0, 0.0], dtype=float),
        velocity=np.zeros(2),
        speed=0.0,
        user_access_backlog_bits=1.0,
    )
    sim_env = SimEnv(
        uavs=[uav],
        users=[user],
        satellites=[Satellite(id=0)],
        ground_base_stations=[],
        gateway_capable_uav_ids=[0],
        backhaul_type="satellite",
        association_min_rate_bps=20.0e6,
        rng=np.random.default_rng(3),
    )

    step_result = sim_env.step()

    assert step_result.association_result.associated_uav_id_by_user[0] == -1
    assert sim_env.users[0].associated_uav_id == -1


def test_multi_agent_reset_uses_sim_env_association_min_rate_bps() -> None:
    uav = UAV(
        id=0,
        position=np.array([config.MAP_LENGTH / 2.0, config.MAP_WIDTH / 2.0, config.UAV_HEIGHT], dtype=float),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_gateway_capable=True,
    )
    user = GroundUser(
        id=0,
        position=np.array([0.0, 0.0, 0.0], dtype=float),
        velocity=np.zeros(2),
        speed=0.0,
    )
    sim_env = SimEnv(
        uavs=[uav],
        users=[user],
        satellites=[],
        ground_base_stations=[],
        gateway_capable_uav_ids=[0],
        backhaul_type="satellite",
        association_min_rate_bps=20.0e6,
        rng=np.random.default_rng(5),
    )
    marl_env = MultiAgentUavNetEnv(sim_env, max_steps=5)

    marl_env.reset()

    assert marl_env.sim_env.users[0].associated_uav_id == -1


def test_uniform_coverage_calibration_scenario_can_drop_coverage_below_one() -> None:
    uavs, users, _, _ = build_demo_entities(
        num_uavs=3,
        num_users=30,
        seed=42,
        backhaul_type="satellite",
        user_demand_rate_bps=2.0e6,
        orbit_radius_m=600.0,
        user_speed_mean_mps=3.5,
        user_distribution="uniform",
        spawn_margin=0.02,
    )

    association_result = associate_users_to_uavs(users, uavs, min_rate_bps=10.0e6)
    coverage_ratio = sum(
        associated_uav_id >= 0
        for associated_uav_id in association_result.associated_uav_id_by_user.values()
    ) / len(users)

    assert 0.8 <= coverage_ratio < 1.0
