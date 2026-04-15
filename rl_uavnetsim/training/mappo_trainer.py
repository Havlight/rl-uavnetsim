from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.environment import SimEnv
from rl_uavnetsim.main import build_demo_entities, get_demo_mode_config
from rl_uavnetsim.metrics import MetricsCollector
from rl_uavnetsim.training.configuration import RunConfig, run_config_from_dict, run_config_to_dict, save_run_config
from rl_uavnetsim.training.pettingzoo_env import PettingZooUavNetEnv
from rl_uavnetsim.visualization import MetricsPlotter, TrajectoryVisualizer, build_visualization_frame

try:  # pragma: no cover - runtime dependency gate
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.distributions import Normal
    from torch.utils.tensorboard import SummaryWriter
    from torchrl.envs.libs.pettingzoo import PettingZooWrapper
    from torchrl.modules import MultiAgentMLP, TanhNormal
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    torch = None
    F = None
    nn = None
    Normal = None
    SummaryWriter = None
    PettingZooWrapper = None
    MultiAgentMLP = None
    TanhNormal = None
    tqdm = None


def _require_marl_dependencies() -> None:
    if (
        torch is None
        or SummaryWriter is None
        or PettingZooWrapper is None
        or MultiAgentMLP is None
        or TanhNormal is None
        or tqdm is None
    ):
        raise RuntimeError(
            "TorchRL MAPPO training requires the optional MARL dependencies. "
            "Install them with `pip install -e .[marl]`."
        )


def _activation_class(name: str) -> type[nn.Module]:
    _require_marl_dependencies()
    normalized_name = name.lower()
    if normalized_name == "relu":
        return nn.ReLU
    if normalized_name == "elu":
        return nn.ELU
    return nn.Tanh


def set_global_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    _require_marl_dependencies()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_training_env(run_config: RunConfig, *, seed: int) -> PettingZooUavNetEnv:
    if run_config.env.backhaul_type != "satellite":
        raise ValueError("MAPPO v1 only supports satellite backhaul.")
    if run_config.observation.preset != "compact_v1":
        raise ValueError("MAPPO v1 only supports the compact_v1 observation preset.")

    demo_mode_config = get_demo_mode_config(run_config.env.demo_mode)
    user_demand_rate_bps = (
        float(run_config.env.user_demand_rate_bps)
        if run_config.env.user_demand_rate_bps is not None
        else float(demo_mode_config.user_demand_rate_bps)
    )
    orbit_radius_m = (
        float(run_config.env.orbit_radius_m)
        if run_config.env.orbit_radius_m is not None
        else float(demo_mode_config.orbit_radius_m)
    )
    user_speed_mean_mps = (
        float(run_config.env.user_speed_mean_mps)
        if run_config.env.user_speed_mean_mps is not None
        else float(demo_mode_config.user_speed_mean_mps)
    )
    user_distribution = (
        str(run_config.env.user_distribution)
        if run_config.env.user_distribution is not None
        else str(demo_mode_config.user_distribution)
    )
    spawn_margin = (
        float(run_config.env.spawn_margin)
        if run_config.env.spawn_margin is not None
        else float(demo_mode_config.spawn_margin)
    )
    association_min_rate_bps = (
        float(run_config.env.association_min_rate_bps)
        if run_config.env.association_min_rate_bps is not None
        else float(demo_mode_config.association_min_rate_bps)
    )
    uavs, users, satellites, ground_base_stations = build_demo_entities(
        num_uavs=run_config.env.num_uavs,
        num_users=run_config.env.num_users,
        seed=seed,
        backhaul_type=run_config.env.backhaul_type,
        user_demand_rate_bps=user_demand_rate_bps,
        orbit_radius_m=orbit_radius_m,
        user_speed_mean_mps=user_speed_mean_mps,
        user_distribution=user_distribution,
        spawn_margin=spawn_margin,
    )
    sim_env = SimEnv(
        uavs=uavs,
        users=users,
        satellites=satellites,
        ground_base_stations=ground_base_stations,
        gateway_capable_uav_ids=[0],
        backhaul_type=run_config.env.backhaul_type,
        association_min_rate_bps=association_min_rate_bps,
        rng=np.random.default_rng(seed),
    )
    return PettingZooUavNetEnv(
        sim_env,
        max_steps=run_config.env.num_steps,
        max_obs_users=run_config.observation.max_obs_users,
        obs_radius_m=run_config.observation.obs_radius_m,
    )


@dataclass
class EvaluationArtifacts:
    output_dir: Path
    summary_json_path: Path
    metrics_history_json_path: Path
    trajectory_png_path: Path
    trajectory_gif_path: Path
    metric_png_paths: dict[str, Path]
    mean_team_reward: float


@dataclass
class TrainingArtifacts:
    run_dir: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    config_path: Path
    eval_artifacts: EvaluationArtifacts


class SharedGaussianActor(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dims: tuple[int, ...],
        activation_class: type[nn.Module],
    ) -> None:
        super().__init__()
        self.mean_net = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=action_dim,
            n_agents=n_agents,
            centralized=False,
            share_params=True,
            num_cells=list(hidden_dims),
            activation_class=activation_class,
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        means = self.mean_net(observations)
        log_std = self.log_std.clamp(-5.0, 2.0)
        std = torch.exp(log_std).view(*([1] * (means.ndim - 1)), -1).expand_as(means)
        return means, std

    def distribution(self, observations: torch.Tensor) -> TanhNormal:
        means, std = self.forward(observations)
        return TanhNormal(loc=means, scale=std, low=-1.0, high=1.0, event_dims=1)

    def deterministic_action(self, observations: torch.Tensor) -> torch.Tensor:
        means, _ = self.forward(observations)
        return torch.tanh(means)

    def gaussian_entropy(self, observations: torch.Tensor) -> torch.Tensor:
        means, std = self.forward(observations)
        base_dist = Normal(loc=means, scale=std)
        return base_dist.entropy().sum(dim=-1)


class CentralizedValueNet(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        n_agents: int,
        hidden_dims: tuple[int, ...],
        activation_class: type[nn.Module],
    ) -> None:
        super().__init__()
        self.value_net = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=n_agents,
            centralized=True,
            share_params=True,
            num_cells=list(hidden_dims),
            activation_class=activation_class,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        values = self.value_net(observations)
        return values.mean(dim=-2).squeeze(-1)


class TorchRLMappoPolicy:
    def __init__(self, actor: SharedGaussianActor, *, agent_names: list[str], device: torch.device) -> None:
        self.actor = actor
        self.agent_names = list(agent_names)
        self.device = device

    @torch.no_grad()
    def act(
        self,
        observations_by_agent: dict[str, np.ndarray],
        *,
        deterministic: bool = False,
    ) -> dict[str, np.ndarray]:
        observation_tensor = torch.as_tensor(
            np.asarray([observations_by_agent[agent_name] for agent_name in self.agent_names], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        if deterministic:
            action_tensor = self.actor.deterministic_action(observation_tensor)
        else:
            action_tensor = self.actor.distribution(observation_tensor).sample()
        action_tensor = action_tensor.detach().cpu().numpy().astype(np.float32)
        return {
            agent_name: action_tensor[index]
            for index, agent_name in enumerate(self.agent_names)
        }


def run_torchrl_spike(run_config: RunConfig) -> dict[str, int]:
    _require_marl_dependencies()
    base_env = build_training_env(run_config, seed=run_config.seed)
    torchrl_env = PettingZooWrapper(env=base_env, return_state=True)
    reset_td = torchrl_env.reset()
    observations = reset_td["uav", "observation"]
    actor = MultiAgentMLP(
        n_agent_inputs=observations.shape[-1],
        n_agent_outputs=2,
        n_agents=observations.shape[-2],
        centralized=False,
        share_params=True,
        num_cells=list(run_config.model.actor_hidden_dims),
        activation_class=_activation_class(run_config.model.activation),
    )
    critic = MultiAgentMLP(
        n_agent_inputs=observations.shape[-1],
        n_agent_outputs=1,
        n_agents=observations.shape[-2],
        centralized=True,
        share_params=True,
        num_cells=list(run_config.model.critic_hidden_dims),
        activation_class=_activation_class(run_config.model.activation),
    )
    actor_output = actor(observations)
    critic_output = critic(observations)
    zero_action = observations.new_zeros((*observations.shape[:-1], 2))
    torchrl_env.step(reset_td.set(("uav", "action"), zero_action))
    torchrl_env.close()
    return {
        "obs_dim": int(observations.shape[-1]),
        "state_dim": int(reset_td["state"].shape[-1]),
        "actor_output_dim": int(actor_output.shape[-1]),
        "critic_output_dim": int(critic_output.shape[-1]),
    }


def _stack_observations(
    observations_by_agent: dict[str, np.ndarray],
    *,
    agent_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray([observations_by_agent[agent_name] for agent_name in agent_names], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )


def _compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    *,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_advantage = 0.0
    next_non_terminal = 1.0 - float(dones[-1]) if dones else 1.0
    next_state_value = float(next_value)
    for index in reversed(range(len(rewards))):
        if index < len(rewards) - 1:
            next_non_terminal = 1.0 - float(dones[index])
            next_state_value = float(values[index + 1])
        delta = float(rewards[index]) + gamma * next_state_value * next_non_terminal - float(values[index])
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[index] = last_advantage
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def _collect_batch(
    env: PettingZooUavNetEnv,
    actor: SharedGaussianActor,
    critic: CentralizedValueNet,
    *,
    frames_per_batch: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    seed: int,
    update_index: int,
) -> dict[str, Any]:
    batch_obs: list[torch.Tensor] = []
    batch_actions: list[torch.Tensor] = []
    batch_log_probs: list[torch.Tensor] = []
    batch_advantages: list[float] = []
    batch_returns: list[float] = []
    batch_values: list[float] = []

    collected_frames = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_index = 0

    while collected_frames < frames_per_batch:
        observations_by_agent, _ = env.reset(seed=seed + update_index * 1000 + episode_index)
        episode_obs: list[torch.Tensor] = []
        episode_actions: list[torch.Tensor] = []
        episode_log_probs: list[torch.Tensor] = []
        episode_values: list[float] = []
        episode_rewards_raw: list[float] = []
        episode_dones: list[bool] = []
        cumulative_episode_reward = 0.0
        episode_length = 0

        while True:
            observation_tensor = _stack_observations(
                observations_by_agent,
                agent_names=env.possible_agents,
                device=device,
            )
            with torch.no_grad():
                action_distribution = actor.distribution(observation_tensor)
                action_tensor = action_distribution.rsample()
                log_prob_tensor = action_distribution.log_prob(action_tensor)
                value_scalar = float(critic(observation_tensor).item())

            next_observations_by_agent, rewards, terminations, truncations, _ = env.step(
                {
                    agent_name: action_tensor[index].detach().cpu().numpy().astype(np.float32)
                    for index, agent_name in enumerate(env.possible_agents)
                }
            )
            reward_scalar = float(rewards[env.possible_agents[0]])
            done = all(terminations.values()) or all(truncations.values())

            episode_obs.append(observation_tensor.detach().cpu())
            episode_actions.append(action_tensor.detach().cpu())
            episode_log_probs.append(log_prob_tensor.detach().cpu())
            episode_values.append(value_scalar)
            episode_rewards_raw.append(reward_scalar)
            episode_dones.append(done)

            observations_by_agent = next_observations_by_agent
            cumulative_episode_reward += reward_scalar
            episode_length += 1
            collected_frames += 1

            if done:
                break

        next_value = 0.0
        advantages, returns = _compute_gae(
            episode_rewards_raw,
            episode_values,
            episode_dones,
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        batch_obs.extend(episode_obs)
        batch_actions.extend(episode_actions)
        batch_log_probs.extend(episode_log_probs)
        batch_advantages.extend(float(value) for value in advantages)
        batch_returns.extend(float(value) for value in returns)
        batch_values.extend(episode_values)
        episode_rewards.append(cumulative_episode_reward)
        episode_lengths.append(episode_length)
        episode_index += 1

    observations = torch.stack(batch_obs).to(device)
    actions = torch.stack(batch_actions).to(device)
    log_probs = torch.stack(batch_log_probs).to(device)
    advantages = torch.as_tensor(batch_advantages, dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch_returns, dtype=torch.float32, device=device)
    values = torch.as_tensor(batch_values, dtype=torch.float32, device=device)

    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

    return {
        "observations": observations,
        "actions": actions,
        "log_probs": log_probs,
        "advantages": advantages,
        "returns": returns,
        "values": values,
        "frames_collected": collected_frames,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def _ppo_update(
    actor: SharedGaussianActor,
    critic: CentralizedValueNet,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    *,
    ppo_epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    entropy_coef: float,
    value_coef: float,
) -> dict[str, float]:
    observations = batch["observations"]
    actions = batch["actions"]
    old_log_probs = batch["log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    num_samples = observations.shape[0]
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
        "num_updates": 0,
    }

    for _ in range(int(ppo_epochs)):
        permutation = torch.randperm(num_samples, device=observations.device)
        for start_index in range(0, num_samples, int(minibatch_size)):
            batch_indices = permutation[start_index : start_index + int(minibatch_size)]
            minibatch_observations = observations[batch_indices]
            minibatch_actions = actions[batch_indices]
            minibatch_old_log_probs = old_log_probs[batch_indices]
            minibatch_advantages = advantages[batch_indices].unsqueeze(-1)
            minibatch_returns = returns[batch_indices]

            action_distribution = actor.distribution(minibatch_observations)
            new_log_probs = action_distribution.log_prob(minibatch_actions)
            log_ratio = new_log_probs - minibatch_old_log_probs
            ratio = torch.exp(log_ratio)

            surrogate_1 = ratio * minibatch_advantages
            surrogate_2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * minibatch_advantages
            policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()

            values = critic(minibatch_observations)
            value_loss = F.mse_loss(values, minibatch_returns)
            entropy = actor.gaussian_entropy(minibatch_observations).mean()
            total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_norm=1.0)
            optimizer.step()

            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["total_loss"] += float(total_loss.item())
            stats["num_updates"] += 1

    divisor = max(stats["num_updates"], 1)
    return {
        "policy_loss": stats["policy_loss"] / divisor,
        "value_loss": stats["value_loss"] / divisor,
        "entropy": stats["entropy"] / divisor,
        "total_loss": stats["total_loss"] / divisor,
    }


def _timestamped_run_dir(output_root: str | Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(output_root) / run_name / timestamp


def _write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _build_progress_postfix(
    *,
    update_index: int,
    total_frames: int,
    batch_reward_mean: float,
    update_stats: dict[str, float],
    eval_mean_team_reward: float | None = None,
) -> dict[str, str]:
    postfix = {
        "upd": str(int(update_index) + 1),
        "frames": str(int(total_frames)),
        "reward": f"{float(batch_reward_mean):.3f}",
        "pi": f"{float(update_stats['policy_loss']):.3f}",
        "v": f"{float(update_stats['value_loss']):.3f}",
        "eval": f"{float(eval_mean_team_reward):.3f}" if eval_mean_team_reward is not None else "-",
    }
    return postfix


def _format_best_checkpoint_message(
    *,
    update_index: int,
    total_frames: int,
    mean_team_reward: float,
    checkpoint_path: str | Path,
) -> str:
    return (
        f"[best checkpoint improved] "
        f"update={int(update_index) + 1} "
        f"frames={int(total_frames)} "
        f"eval_mean_reward={float(mean_team_reward):.3f} "
        f"path={checkpoint_path}"
    )


def _checkpoint_payload(
    *,
    actor: SharedGaussianActor,
    critic: CentralizedValueNet,
    optimizer: torch.optim.Optimizer,
    run_config: RunConfig,
    total_frames: int,
    update_index: int,
    best_eval_reward: float,
) -> dict[str, Any]:
    return {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "run_config": run_config_to_dict(run_config),
        "total_frames": int(total_frames),
        "update_index": int(update_index),
        "best_eval_reward": float(best_eval_reward),
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _build_backhaul_node_position(env: PettingZooUavNetEnv) -> np.ndarray | None:
    if env.sim_env.backhaul_type == "satellite" and env.sim_env.satellites:
        return env.sim_env.satellites[0].position
    return None


def evaluate_policy(
    policy: TorchRLMappoPolicy,
    run_config: RunConfig,
    *,
    output_dir: str | Path,
) -> EvaluationArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_collector = MetricsCollector()
    metrics_plotter = MetricsPlotter()
    trajectory_visualizer = TrajectoryVisualizer()
    team_rewards: list[float] = []
    visualization_frames = []

    for episode_index in range(int(run_config.eval.num_eval_episodes)):
        env = build_training_env(run_config, seed=run_config.seed + 100_000 + episode_index)
        observations_by_agent, _ = env.reset(seed=run_config.seed + 100_000 + episode_index)
        episode_team_reward = 0.0
        episode_frames: list[Any] = []
        backhaul_node_position = _build_backhaul_node_position(env)

        while env.agents:
            actions_by_agent = policy.act(
                observations_by_agent,
                deterministic=run_config.eval.deterministic_policy,
            )
            observations_by_agent, rewards, terminations, truncations, _ = env.step(actions_by_agent)
            if env.latest_step_info is None or env.latest_env_state is None:
                raise RuntimeError("Evaluation step did not expose step information.")
            step_result = env.latest_step_info["step_result"]
            metrics_collector.record(step_result, env.sim_env.uavs, env.sim_env.users)
            episode_team_reward += float(rewards[env.possible_agents[0]])
            episode_frames.append(
                build_visualization_frame(
                    step_result,
                    env.sim_env.uavs,
                    env.sim_env.users,
                    gateway_uav_ids=env.latest_env_state.active_gateway_uav_ids,
                    backhaul_type=env.sim_env.backhaul_type,
                    backhaul_node_position=backhaul_node_position,
                )
            )
            if all(terminations.values()) or all(truncations.values()):
                break

        if episode_index == 0:
            visualization_frames = episode_frames
        team_rewards.append(episode_team_reward)
        env.close()

    if not visualization_frames:
        raise RuntimeError("Evaluation did not produce any visualization frames.")

    metrics_summary = metrics_collector.summarize()
    summary_payload = {
        **asdict(metrics_summary),
        "mean_team_reward": float(np.mean(team_rewards)) if team_rewards else 0.0,
        "episode_team_rewards": [float(value) for value in team_rewards],
    }
    summary_json_path = output_dir / "summary.json"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    metrics_history_json_path = output_dir / "metrics_history.json"
    with metrics_history_json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_collector.export_history(), handle, indent=2)

    metric_png_paths = metrics_plotter.plot_metric_set(metrics_collector.step_records, output_dir / "plots")
    trajectory_png_path = trajectory_visualizer.render_frame(
        visualization_frames,
        frame_index=len(visualization_frames) - 1,
        output_png_path=output_dir / "trajectory_final.png",
    )
    trajectory_gif_path = trajectory_visualizer.create_gif(
        visualization_frames,
        output_gif_path=output_dir / "trajectory.gif",
    )

    return EvaluationArtifacts(
        output_dir=output_dir,
        summary_json_path=summary_json_path,
        metrics_history_json_path=metrics_history_json_path,
        trajectory_png_path=trajectory_png_path,
        trajectory_gif_path=trajectory_gif_path,
        metric_png_paths=metric_png_paths,
        mean_team_reward=float(summary_payload["mean_team_reward"]),
    )


def train_mappo(run_config: RunConfig) -> TrainingArtifacts:
    _require_marl_dependencies()
    set_global_seeds(run_config.seed)

    run_dir = _timestamped_run_dir(run_config.output.root_dir, run_config.output.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_run_config(run_dir / "config_resolved.yaml", run_config)
    writer = SummaryWriter(log_dir=run_dir / "tensorboard")

    spike_summary = run_torchrl_spike(run_config)
    for key, value in spike_summary.items():
        writer.add_scalar(f"spike/{key}", value, 0)

    env = build_training_env(run_config, seed=run_config.seed)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    n_agents = len(env.possible_agents)
    device = torch.device(run_config.trainer.device)
    activation_class = _activation_class(run_config.model.activation)

    actor = SharedGaussianActor(
        obs_dim=obs_dim,
        action_dim=2,
        n_agents=n_agents,
        hidden_dims=run_config.model.actor_hidden_dims,
        activation_class=activation_class,
    ).to(device)
    critic = CentralizedValueNet(
        obs_dim=obs_dim,
        n_agents=n_agents,
        hidden_dims=run_config.model.critic_hidden_dims,
        activation_class=activation_class,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=run_config.trainer.lr,
    )

    total_frames = 0
    update_index = 0
    best_eval_reward = -float("inf")
    latest_checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    train_metrics_path = run_dir / "train_metrics.jsonl"
    progress_bar = tqdm(
        total=int(run_config.trainer.total_frames),
        desc="MAPPO train",
        unit="frame",
        dynamic_ncols=True,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    eval_artifacts: EvaluationArtifacts | None = None
    try:
        while total_frames < run_config.trainer.total_frames:
            previous_total_frames = total_frames
            batch = _collect_batch(
                env,
                actor,
                critic,
                frames_per_batch=run_config.trainer.frames_per_batch,
                gamma=run_config.trainer.gamma,
                gae_lambda=run_config.trainer.gae_lambda,
                device=device,
                seed=run_config.seed,
                update_index=update_index,
            )
            total_frames += int(batch["frames_collected"])
            update_stats = _ppo_update(
                actor,
                critic,
                optimizer,
                batch,
                ppo_epochs=run_config.trainer.ppo_epochs,
                minibatch_size=run_config.trainer.minibatch_size,
                clip_epsilon=run_config.trainer.clip_epsilon,
                entropy_coef=run_config.trainer.entropy_coef,
                value_coef=run_config.trainer.value_coef,
            )
            batch_reward_mean = float(np.mean(batch["episode_rewards"])) if batch["episode_rewards"] else 0.0
            batch_episode_length_mean = float(np.mean(batch["episode_lengths"])) if batch["episode_lengths"] else 0.0

            writer.add_scalar("train/policy_loss", update_stats["policy_loss"], update_index)
            writer.add_scalar("train/value_loss", update_stats["value_loss"], update_index)
            writer.add_scalar("train/entropy", update_stats["entropy"], update_index)
            writer.add_scalar("train/batch_reward_mean", batch_reward_mean, update_index)
            writer.add_scalar("train/batch_episode_length_mean", batch_episode_length_mean, update_index)
            writer.add_scalar("train/total_frames", total_frames, update_index)

            record = {
                "update_index": update_index,
                "total_frames": total_frames,
                "policy_loss": update_stats["policy_loss"],
                "value_loss": update_stats["value_loss"],
                "entropy": update_stats["entropy"],
                "batch_reward_mean": batch_reward_mean,
                "batch_episode_length_mean": batch_episode_length_mean,
            }
            _write_jsonl_record(train_metrics_path, record)

            if (update_index + 1) % max(run_config.trainer.checkpoint_interval, 1) == 0:
                _save_checkpoint(
                    latest_checkpoint_path,
                    _checkpoint_payload(
                        actor=actor,
                        critic=critic,
                        optimizer=optimizer,
                        run_config=run_config,
                        total_frames=total_frames,
                        update_index=update_index,
                        best_eval_reward=best_eval_reward,
                    ),
                )

            eval_mean_team_reward: float | None = None
            if (update_index + 1) % max(run_config.trainer.eval_interval, 1) == 0:
                policy = TorchRLMappoPolicy(actor, agent_names=env.possible_agents, device=device)
                eval_artifacts = evaluate_policy(policy, run_config, output_dir=run_dir / "eval")
                eval_mean_team_reward = eval_artifacts.mean_team_reward
                writer.add_scalar("eval/mean_team_reward", eval_artifacts.mean_team_reward, update_index)
                if eval_artifacts.mean_team_reward > best_eval_reward:
                    best_eval_reward = eval_artifacts.mean_team_reward
                    _save_checkpoint(
                        best_checkpoint_path,
                        _checkpoint_payload(
                            actor=actor,
                            critic=critic,
                            optimizer=optimizer,
                            run_config=run_config,
                            total_frames=total_frames,
                            update_index=update_index,
                            best_eval_reward=best_eval_reward,
                        ),
                    )
                    tqdm.write(
                        _format_best_checkpoint_message(
                            update_index=update_index,
                            total_frames=total_frames,
                            mean_team_reward=best_eval_reward,
                            checkpoint_path=best_checkpoint_path,
                        )
                    )

            displayed_increment = min(
                total_frames,
                int(run_config.trainer.total_frames),
            ) - previous_total_frames
            displayed_total_frames = min(
                total_frames,
                int(run_config.trainer.total_frames),
            )
            if displayed_increment > 0:
                progress_bar.update(displayed_increment)
            progress_bar.set_postfix(
                _build_progress_postfix(
                    update_index=update_index,
                    total_frames=displayed_total_frames,
                    batch_reward_mean=batch_reward_mean,
                    update_stats=update_stats,
                    eval_mean_team_reward=eval_mean_team_reward,
                ),
                refresh=False,
            )

            update_index += 1

        if eval_artifacts is None:
            policy = TorchRLMappoPolicy(actor, agent_names=env.possible_agents, device=device)
            eval_artifacts = evaluate_policy(policy, run_config, output_dir=run_dir / "eval")
            best_eval_reward = eval_artifacts.mean_team_reward
            _save_checkpoint(
                best_checkpoint_path,
                _checkpoint_payload(
                    actor=actor,
                    critic=critic,
                    optimizer=optimizer,
                    run_config=run_config,
                    total_frames=total_frames,
                    update_index=update_index,
                    best_eval_reward=best_eval_reward,
                ),
            )

        _save_checkpoint(
            latest_checkpoint_path,
            _checkpoint_payload(
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                run_config=run_config,
                total_frames=total_frames,
                update_index=update_index,
                best_eval_reward=best_eval_reward,
            ),
        )
    finally:
        progress_bar.close()
        writer.close()
        env.close()

    return TrainingArtifacts(
        run_dir=run_dir,
        latest_checkpoint_path=latest_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        config_path=config_path,
        eval_artifacts=eval_artifacts,
    )


def load_policy_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[TorchRLMappoPolicy, RunConfig]:
    _require_marl_dependencies()
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    run_config = run_config_from_dict(checkpoint["run_config"])
    activation_class = _activation_class(run_config.model.activation)
    env = build_training_env(run_config, seed=run_config.seed)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    actor = SharedGaussianActor(
        obs_dim=obs_dim,
        action_dim=2,
        n_agents=len(env.possible_agents),
        hidden_dims=run_config.model.actor_hidden_dims,
        activation_class=activation_class,
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    resolved_device = torch.device(device or run_config.trainer.device)
    actor = actor.to(resolved_device)
    policy = TorchRLMappoPolicy(actor, agent_names=env.possible_agents, device=resolved_device)
    env.close()
    return policy, run_config
