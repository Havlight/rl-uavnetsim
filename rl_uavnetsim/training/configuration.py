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
    map_length_m: float = config.MAP_LENGTH
    map_width_m: float = config.MAP_WIDTH
    demo_mode: str = "default"
    user_demand_rate_bps: float | None = None
    orbit_radius_m: float | None = None
    user_speed_mean_mps: float | None = None
    user_distribution: str | None = None
    spawn_margin: float | None = None
    association_min_rate_bps: float | None = None
    max_access_range_m: float | None = None


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
    run_static_baseline: bool = True
    write_static_artifacts: bool = True


@dataclass(frozen=True)
class RewardConfig:
    energy_coef: float = config.ETA
    outage_coef: float = config.MU
    access_backlog_coef: float = config.BETA_ACCESS
    relay_queue_coef: float = config.BETA_RELAY
    connectivity_coef: float = config.LAMBDA_CONN
    safety_coef: float = config.LAMBDA_SAFE
    outage_threshold_bps: float = config.R_MIN
    target_coverage: float = 0.0
    coverage_gap_coef: float = 0.0
    target_effective_coverage: float = 0.0
    effective_coverage_gap_coef: float = 0.0
    target_fairness: float = 0.0
    fairness_gap_coef: float = 0.0


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
    reward: RewardConfig = field(default_factory=RewardConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_run_config(path: str | Path) -> RunConfig:
    _require_yaml()
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return run_config_from_dict(payload)


def load_run_config_payload(path: str | Path) -> dict[str, Any]:
    _require_yaml()
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_config_from_dict(payload: dict[str, Any]) -> RunConfig:
    env_payload = payload.get("env", {})
    observation_payload = payload.get("observation", {})
    trainer_payload = payload.get("trainer", {})
    model_payload = payload.get("model", {})
    eval_payload = payload.get("eval", {})
    reward_payload = payload.get("reward", {})
    output_payload = payload.get("output", {})

    return RunConfig(
        seed=int(payload.get("seed", 0)),
        env=EnvConfig(
            num_steps=int(env_payload.get("num_steps", EnvConfig.num_steps)),
            num_uavs=int(env_payload.get("num_uavs", EnvConfig.num_uavs)),
            num_users=int(env_payload.get("num_users", EnvConfig.num_users)),
            backhaul_type=str(env_payload.get("backhaul_type", EnvConfig.backhaul_type)),
            map_length_m=float(env_payload.get("map_length_m", EnvConfig.map_length_m)),
            map_width_m=float(env_payload.get("map_width_m", EnvConfig.map_width_m)),
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
            spawn_margin=(
                float(env_payload["spawn_margin"])
                if "spawn_margin" in env_payload
                else EnvConfig.spawn_margin
            ),
            association_min_rate_bps=(
                float(env_payload["association_min_rate_bps"])
                if "association_min_rate_bps" in env_payload
                else EnvConfig.association_min_rate_bps
            ),
            max_access_range_m=(
                float(env_payload["max_access_range_m"])
                if "max_access_range_m" in env_payload
                else EnvConfig.max_access_range_m
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
            run_static_baseline=bool(eval_payload.get("run_static_baseline", EvalConfig.run_static_baseline)),
            write_static_artifacts=bool(eval_payload.get("write_static_artifacts", EvalConfig.write_static_artifacts)),
        ),
        reward=RewardConfig(
            energy_coef=float(reward_payload.get("energy_coef", RewardConfig.energy_coef)),
            outage_coef=float(reward_payload.get("outage_coef", RewardConfig.outage_coef)),
            access_backlog_coef=float(reward_payload.get("access_backlog_coef", RewardConfig.access_backlog_coef)),
            relay_queue_coef=float(reward_payload.get("relay_queue_coef", RewardConfig.relay_queue_coef)),
            connectivity_coef=float(reward_payload.get("connectivity_coef", RewardConfig.connectivity_coef)),
            safety_coef=float(reward_payload.get("safety_coef", RewardConfig.safety_coef)),
            outage_threshold_bps=float(reward_payload.get("outage_threshold_bps", RewardConfig.outage_threshold_bps)),
            target_coverage=float(reward_payload.get("target_coverage", RewardConfig.target_coverage)),
            coverage_gap_coef=float(reward_payload.get("coverage_gap_coef", RewardConfig.coverage_gap_coef)),
            target_effective_coverage=float(
                reward_payload.get("target_effective_coverage", RewardConfig.target_effective_coverage)
            ),
            effective_coverage_gap_coef=float(
                reward_payload.get("effective_coverage_gap_coef", RewardConfig.effective_coverage_gap_coef)
            ),
            target_fairness=float(reward_payload.get("target_fairness", RewardConfig.target_fairness)),
            fairness_gap_coef=float(reward_payload.get("fairness_gap_coef", RewardConfig.fairness_gap_coef)),
        ),
        output=OutputConfig(
            root_dir=str(output_payload.get("root_dir", OutputConfig.root_dir)),
            run_name=str(output_payload.get("run_name", OutputConfig.run_name)),
        ),
    )


_COMPATIBLE_EVAL_ENV_FIELDS = {
    "num_steps",
    "user_demand_rate_bps",
    "orbit_radius_m",
    "user_speed_mean_mps",
    "user_distribution",
    "spawn_margin",
    "association_min_rate_bps",
    "max_access_range_m",
    "map_length_m",
    "map_width_m",
}
_INCOMPATIBLE_EVAL_FIELDS = {
    "env.num_uavs",
    "observation.preset",
    "observation.max_obs_users",
    "model",
}


def _field_value(config_object: Any, dotted_field: str) -> Any:
    value = config_object
    for part in dotted_field.split("."):
        value = getattr(value, part)
    return value


def _assert_eval_override_compatible(
    *,
    base_config: RunConfig,
    override_payload: dict[str, Any],
    eval_overrides: RunConfig,
) -> None:
    for field_name in ("observation.preset", "observation.max_obs_users"):
        section_name, key_name = field_name.split(".")
        if key_name in override_payload.get(section_name, {}):
            if _field_value(base_config, field_name) != _field_value(eval_overrides, field_name):
                raise ValueError(f"Evaluation config cannot override incompatible field '{field_name}'.")

    if "model" in override_payload:
        if base_config.model != eval_overrides.model:
            raise ValueError("Evaluation config cannot override incompatible field 'model'.")

    env_payload = override_payload.get("env", {})
    if "num_uavs" in env_payload and int(env_payload["num_uavs"]) != base_config.env.num_uavs:
        raise ValueError("Evaluation config cannot override incompatible field 'env.num_uavs'.")

    allowed_env_fields = set(_COMPATIBLE_EVAL_ENV_FIELDS) | {"num_uavs", "num_users", "backhaul_type", "demo_mode"}
    for key_name in env_payload:
        if key_name not in allowed_env_fields:
            raise ValueError(f"Evaluation config cannot override unsupported env field 'env.{key_name}'.")
        if key_name in {"num_users", "backhaul_type", "demo_mode"} and getattr(eval_overrides.env, key_name) != getattr(base_config.env, key_name):
            raise ValueError(f"Evaluation config cannot override incompatible field 'env.{key_name}'.")


def merge_eval_config(
    base_config: RunConfig,
    eval_overrides: RunConfig,
    *,
    override_payload: dict[str, Any] | None = None,
) -> RunConfig:
    if override_payload is not None:
        _assert_eval_override_compatible(
            base_config=base_config,
            override_payload=override_payload,
            eval_overrides=eval_overrides,
        )
        env_updates = {
            key: getattr(eval_overrides.env, key)
            for key in _COMPATIBLE_EVAL_ENV_FIELDS
            if key in override_payload.get("env", {})
        }
        eval_updates = {
            key: getattr(eval_overrides.eval, key)
            for key in override_payload.get("eval", {})
        }
        reward_updates = {
            key: getattr(eval_overrides.reward, key)
            for key in override_payload.get("reward", {})
        }
        output_updates = {
            key: getattr(eval_overrides.output, key)
            for key in override_payload.get("output", {})
        }
        return replace(
            base_config,
            seed=eval_overrides.seed if "seed" in override_payload else base_config.seed,
            env=replace(base_config.env, **env_updates),
            eval=replace(base_config.eval, **eval_updates),
            reward=replace(base_config.reward, **reward_updates),
            output=replace(base_config.output, **output_updates),
        )

    return replace(
        base_config,
        seed=eval_overrides.seed,
        eval=eval_overrides.eval,
        reward=eval_overrides.reward,
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
