from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_uavnetsim.training.configuration import (
    EnvConfig,
    EvalConfig,
    ModelConfig,
    ObservationConfig,
    OutputConfig,
    RunConfig,
    TrainerConfig,
    merge_eval_config,
    run_config_from_dict,
)
from rl_uavnetsim.training.mappo_trainer import StaticMovementPolicy, evaluate_policy


def _tiny_run_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        seed=7,
        env=EnvConfig(
            num_steps=2,
            num_uavs=2,
            num_users=3,
            backhaul_type="satellite",
            user_demand_rate_bps=1.0e6,
            orbit_radius_m=200.0,
            user_speed_mean_mps=1.0,
            user_distribution="uniform",
            spawn_margin=0.1,
            association_min_rate_bps=0.5e6,
            max_access_range_m=900.0,
        ),
        observation=ObservationConfig(preset="compact_v2", max_obs_users=2, obs_radius_m=800.0),
        trainer=TrainerConfig(total_frames=4, frames_per_batch=2),
        model=ModelConfig(actor_hidden_dims=(16,), critic_hidden_dims=(16,), activation="tanh"),
        eval=EvalConfig(num_eval_episodes=2, deterministic_policy=True, run_static_baseline=True, write_static_artifacts=False),
        output=OutputConfig(root_dir=str(tmp_path), run_name="eval_schema"),
    )


def test_evaluate_policy_writes_schema_v2_and_cleans_stale_episode_dirs(tmp_path: Path) -> None:
    run_config = _tiny_run_config(tmp_path)
    output_dir = tmp_path / "eval"
    stale_episode = output_dir / "episodes" / "episode_999"
    stale_episode.mkdir(parents=True)
    (stale_episode / "stale.txt").write_text("old", encoding="utf-8")
    policy = StaticMovementPolicy(agent_names=["uav_0", "uav_1"])

    artifacts = evaluate_policy(
        policy,
        run_config,
        output_dir=output_dir,
        run_static_baseline=True,
        write_static_artifacts=False,
    )

    summary = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == 2
    assert "trained" in summary["policies"]
    assert "static" in summary["policies"]
    assert "trained_minus_static_reward" in summary["comparison"]
    assert not stale_episode.exists()
    assert (output_dir / "episodes" / "episode_000" / "summary.json").exists()
    assert (output_dir / "episodes" / "episode_000" / "metrics_history.json").exists()
    assert not (output_dir / "static" / "episodes").exists()
    assert (output_dir / "plots" / "throughput.png").exists()
    assert (output_dir / "plots" / "effective_coverage.png").exists()


def test_eval_config_merge_allows_compatible_runtime_overrides() -> None:
    base_config = _tiny_run_config(Path("runs/test"))
    override_payload = {
        "seed": 99,
        "env": {
            "num_steps": 3,
            "map_length_m": 3000.0,
            "map_width_m": 3000.0,
            "association_min_rate_bps": 14.0e6,
            "max_access_range_m": 700.0,
        },
        "eval": {"num_eval_episodes": 1, "deterministic_policy": True},
        "reward": {"outage_coef": 3.0, "target_coverage": 0.9, "target_effective_coverage": 0.8},
        "output": {"root_dir": "runs/eval", "run_name": "manual_eval"},
    }
    overrides = run_config_from_dict(override_payload)

    merged = merge_eval_config(base_config, overrides, override_payload=override_payload)

    assert merged.seed == 99
    assert merged.env.num_steps == 3
    assert merged.env.map_length_m == 3000.0
    assert merged.env.association_min_rate_bps == 14.0e6
    assert merged.env.max_access_range_m == 700.0
    assert merged.env.num_uavs == base_config.env.num_uavs
    assert merged.observation == base_config.observation
    assert merged.reward.outage_coef == 3.0
    assert merged.reward.target_effective_coverage == 0.8


def test_eval_config_merge_rejects_incompatible_shape_overrides() -> None:
    base_config = _tiny_run_config(Path("runs/test"))
    override_payload = {
        "observation": {"preset": "compact_v1"},
    }
    overrides = run_config_from_dict(override_payload)

    with pytest.raises(ValueError, match="observation.preset"):
        merge_eval_config(base_config, overrides, override_payload=override_payload)
