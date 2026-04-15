from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from rl_uavnetsim import config

try:  # pragma: no cover - exercised in runtime when [marl] extras are installed
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for MARL config loading. Install with `pip install -e .[marl]`.")


def _tuple_of_ints(values: list[int] | tuple[int, ...] | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class EnvConfig:
    num_steps: int = 50
    num_uavs: int = config.NUM_UAVS
    num_users: int = config.NUM_USERS
    backhaul_type: str = "satellite"
    demo_mode: str = "default"
    user_demand_rate_bps: float | None = None
    orbit_radius_m: float | None = None
    user_speed_mean_mps: float | None = None
    user_distribution: str | None = None


@dataclass(frozen=True)
class ObservationConfig:
    preset: str = "compact_v1"
    max_obs_users: int = 15
    obs_radius_m: float = config.OBS_RADIUS


@dataclass(frozen=True)
class TrainerConfig:
    frames_per_batch: int = 128
    total_frames: int = 1024
    ppo_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    lr: float = 3.0e-4
    device: str = "cpu"
    checkpoint_interval: int = 1
    eval_interval: int = 1


@dataclass(frozen=True)
class ModelConfig:
    actor_hidden_dims: tuple[int, ...] = (128, 128)
    critic_hidden_dims: tuple[int, ...] = (128, 128)
    activation: str = "tanh"


@dataclass(frozen=True)
class EvalConfig:
    num_eval_episodes: int = 2
    deterministic_policy: bool = True


@dataclass(frozen=True)
class OutputConfig:
    root_dir: str = "runs"
    run_name: str = "mappo_satellite"


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_run_config(path: str | Path) -> RunConfig:
    _require_yaml()
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return run_config_from_dict(payload)


def run_config_from_dict(payload: dict[str, Any]) -> RunConfig:
    env_payload = payload.get("env", {})
    observation_payload = payload.get("observation", {})
    trainer_payload = payload.get("trainer", {})
    model_payload = payload.get("model", {})
    eval_payload = payload.get("eval", {})
    output_payload = payload.get("output", {})

    return RunConfig(
        seed=int(payload.get("seed", 0)),
        env=EnvConfig(
            num_steps=int(env_payload.get("num_steps", EnvConfig.num_steps)),
            num_uavs=int(env_payload.get("num_uavs", EnvConfig.num_uavs)),
            num_users=int(env_payload.get("num_users", EnvConfig.num_users)),
            backhaul_type=str(env_payload.get("backhaul_type", EnvConfig.backhaul_type)),
            demo_mode=str(env_payload.get("demo_mode", EnvConfig.demo_mode)),
            user_demand_rate_bps=(
                float(env_payload["user_demand_rate_bps"])
                if "user_demand_rate_bps" in env_payload
                else EnvConfig.user_demand_rate_bps
            ),
            orbit_radius_m=(
                float(env_payload["orbit_radius_m"])
                if "orbit_radius_m" in env_payload
                else EnvConfig.orbit_radius_m
            ),
            user_speed_mean_mps=(
                float(env_payload["user_speed_mean_mps"])
                if "user_speed_mean_mps" in env_payload
                else EnvConfig.user_speed_mean_mps
            ),
            user_distribution=(
                str(env_payload["user_distribution"])
                if "user_distribution" in env_payload
                else EnvConfig.user_distribution
            ),
        ),
        observation=ObservationConfig(
            preset=str(observation_payload.get("preset", ObservationConfig.preset)),
            max_obs_users=int(observation_payload.get("max_obs_users", ObservationConfig.max_obs_users)),
            obs_radius_m=float(observation_payload.get("obs_radius_m", ObservationConfig.obs_radius_m)),
        ),
        trainer=TrainerConfig(
            frames_per_batch=int(trainer_payload.get("frames_per_batch", TrainerConfig.frames_per_batch)),
            total_frames=int(trainer_payload.get("total_frames", TrainerConfig.total_frames)),
            ppo_epochs=int(trainer_payload.get("ppo_epochs", TrainerConfig.ppo_epochs)),
            minibatch_size=int(trainer_payload.get("minibatch_size", TrainerConfig.minibatch_size)),
            gamma=float(trainer_payload.get("gamma", TrainerConfig.gamma)),
            gae_lambda=float(trainer_payload.get("gae_lambda", TrainerConfig.gae_lambda)),
            clip_epsilon=float(trainer_payload.get("clip_epsilon", TrainerConfig.clip_epsilon)),
            entropy_coef=float(trainer_payload.get("entropy_coef", TrainerConfig.entropy_coef)),
            value_coef=float(trainer_payload.get("value_coef", TrainerConfig.value_coef)),
            lr=float(trainer_payload.get("lr", TrainerConfig.lr)),
            device=str(trainer_payload.get("device", TrainerConfig.device)),
            checkpoint_interval=int(trainer_payload.get("checkpoint_interval", TrainerConfig.checkpoint_interval)),
            eval_interval=int(trainer_payload.get("eval_interval", TrainerConfig.eval_interval)),
        ),
        model=ModelConfig(
            actor_hidden_dims=_tuple_of_ints(model_payload.get("actor_hidden_dims"), default=ModelConfig.actor_hidden_dims),
            critic_hidden_dims=_tuple_of_ints(model_payload.get("critic_hidden_dims"), default=ModelConfig.critic_hidden_dims),
            activation=str(model_payload.get("activation", ModelConfig.activation)),
        ),
        eval=EvalConfig(
            num_eval_episodes=int(eval_payload.get("num_eval_episodes", EvalConfig.num_eval_episodes)),
            deterministic_policy=bool(eval_payload.get("deterministic_policy", EvalConfig.deterministic_policy)),
        ),
        output=OutputConfig(
            root_dir=str(output_payload.get("root_dir", OutputConfig.root_dir)),
            run_name=str(output_payload.get("run_name", OutputConfig.run_name)),
        ),
    )


def merge_eval_config(base_config: RunConfig, eval_overrides: RunConfig) -> RunConfig:
    return replace(
        base_config,
        seed=eval_overrides.seed,
        eval=eval_overrides.eval,
        output=eval_overrides.output,
    )


def run_config_to_dict(run_config: RunConfig) -> dict[str, Any]:
    return asdict(run_config)


def save_run_config(path: str | Path, run_config: RunConfig) -> Path:
    _require_yaml()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_config_to_dict(run_config), handle, sort_keys=False)
    return path
