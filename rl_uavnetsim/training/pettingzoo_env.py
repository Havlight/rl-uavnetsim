from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.environment import EnvState, SimEnv
from rl_uavnetsim.rl_interface import MultiAgentUavNetEnv
from rl_uavnetsim.training.features import build_compact_local_observation, build_compact_state, compact_observation_dim, compact_state_dim

try:  # pragma: no cover - exercised when optional MARL deps are installed
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback keeps core tests runnable without [marl]
    spaces = None

try:  # pragma: no cover - exercised when optional MARL deps are installed
    from pettingzoo.utils.env import ParallelEnv as PettingZooParallelEnv
except ImportError:  # pragma: no cover - fallback keeps core tests runnable without [marl]
    PettingZooParallelEnv = object


@dataclass(frozen=True)
class _FallbackBox:
    low: float
    high: float
    shape: tuple[int, ...]
    dtype: np.dtype


def decode_movement_action(action: np.ndarray | Iterable[float]) -> tuple[float, float]:
    action_array = np.asarray(action, dtype=float).reshape(-1)
    if action_array.size != 2:
        raise ValueError(f"Expected 2 action values for rho/psi, got shape {action_array.shape}.")
    rho_norm = float(np.clip((action_array[0] + 1.0) / 2.0, 0.0, 1.0))
    psi_rad = float(np.clip(action_array[1], -1.0, 1.0) * math.pi)
    return rho_norm, psi_rad


def _box(low: float, high: float, shape: tuple[int, ...], dtype: np.dtype) -> Any:
    if spaces is not None:
        return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return _FallbackBox(low=low, high=high, shape=shape, dtype=np.dtype(dtype))


class PettingZooUavNetEnv(PettingZooParallelEnv):
    metadata = {"name": "rl_uavnetsim_parallel_v0", "is_parallelizable": True}

    def __init__(
        self,
        sim_env: SimEnv,
        *,
        max_steps: int = config.SIM_STEPS,
        max_obs_users: int = 15,
        obs_radius_m: float = config.OBS_RADIUS,
    ) -> None:
        self.sim_env = sim_env
        self.max_steps = int(max_steps)
        self.max_obs_users = int(max_obs_users)
        self.obs_radius_m = float(obs_radius_m)
        self.agent_name_mapping = {uav.id: f"uav_{uav.id}" for uav in self.sim_env.uavs}
        self.possible_agents = [self.agent_name_mapping[uav.id] for uav in sorted(self.sim_env.uavs, key=lambda item: item.id)]
        self.agents = list(self.possible_agents)
        self.agent_ids = [uav.id for uav in sorted(self.sim_env.uavs, key=lambda item: item.id)]
        self._marl_env = MultiAgentUavNetEnv(self.sim_env, max_steps=max_steps)
        self._observation_dim = compact_observation_dim(len(self.agent_ids), self.max_obs_users)
        self._state_dim = compact_state_dim(len(self.agent_ids), len(self.sim_env.users))
        self._observation_space = _box(-np.inf, np.inf, (self._observation_dim,), np.float32)
        self._action_space = _box(-1.0, 1.0, (2,), np.float32)
        self.state_space = _box(-np.inf, np.inf, (self._state_dim,), np.float32)
        self.last_actions_by_uav_id: dict[int, dict[str, float]] = {}
        self.latest_env_state: EnvState | None = None
        self.latest_global_state: dict[str, Any] | None = None
        self.latest_step_info: dict[str, Any] | None = None
        self.latest_team_reward: float | None = None

    def observation_space(self, agent: str) -> Any:
        if agent not in self.possible_agents:
            raise KeyError(f"Unknown agent '{agent}'.")
        return self._observation_space

    def action_space(self, agent: str) -> Any:
        if agent not in self.possible_agents:
            raise KeyError(f"Unknown agent '{agent}'.")
        return self._action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        del options
        if seed is not None:
            self.sim_env.rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        _, info = self._marl_env.reset()
        self.latest_env_state = info["env_state"]
        self.latest_global_state = info["global_state"]
        self.latest_step_info = None
        self.latest_team_reward = None
        observations = self._build_observations(self.latest_env_state)
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(
        self,
        actions: dict[str, np.ndarray | Iterable[float]],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            raise RuntimeError("step() called on an environment with no active agents. Call reset() first.")

        actions_by_uav_id: dict[int, dict[str, float]] = {}
        for agent_name in self.agents:
            uav_id = int(agent_name.split("_", maxsplit=1)[1])
            rho_norm, psi_rad = decode_movement_action(actions[agent_name])
            actions_by_uav_id[uav_id] = {"rho": rho_norm, "psi": psi_rad}
        self.last_actions_by_uav_id = actions_by_uav_id

        _, rewards_by_agent_id, terminated_by_agent_id, truncated_by_agent_id, info = self._marl_env.step(actions_by_uav_id)
        self.latest_env_state = info["env_state"]
        self.latest_global_state = info["global_state"]
        self.latest_step_info = info
        self.latest_team_reward = float(info["team_reward"])

        observations = self._build_observations(self.latest_env_state)
        rewards = {self.agent_name_mapping[agent_id]: float(rewards_by_agent_id[agent_id]) for agent_id in self.agent_ids}
        terminations = {self.agent_name_mapping[agent_id]: bool(terminated_by_agent_id[agent_id]) for agent_id in self.agent_ids}
        truncations = {self.agent_name_mapping[agent_id]: bool(truncated_by_agent_id[agent_id]) for agent_id in self.agent_ids}
        infos = {self.agent_name_mapping[agent_id]: {} for agent_id in self.agent_ids}

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        return build_compact_state(self.sim_env.uavs, self.sim_env.users).astype(np.float32, copy=False)

    def close(self) -> None:
        self.agents = []

    def _build_observations(self, env_state: EnvState) -> dict[str, np.ndarray]:
        del env_state
        return {
            self.agent_name_mapping[uav.id]: build_compact_local_observation(
                uav,
                self.sim_env.uavs,
                self.sim_env.users,
                max_obs_users=self.max_obs_users,
                obs_radius_m=self.obs_radius_m,
            ).astype(np.float32, copy=False)
            for uav in sorted(self.sim_env.uavs, key=lambda item: item.id)
        }
