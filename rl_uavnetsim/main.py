from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.entities import GroundBaseStation, GroundUser, Satellite, UAV
from rl_uavnetsim.energy import SimplifiedEnergyModel
from rl_uavnetsim.environment import SimEnv
from rl_uavnetsim.metrics import EpisodeMetricsSummary, MetricsCollector
from rl_uavnetsim.mobility import RandomWalkMobility
from rl_uavnetsim.rl_interface import MAPPOStub, MultiAgentUavNetEnv
from rl_uavnetsim.network import build_a2a_capacity_matrix_bps
from rl_uavnetsim.visualization import MetricsPlotter, TrajectoryVisualizer, build_visualization_frame


@dataclass
class DemoArtifacts:
    output_dir: Path
    summary_json_path: Path
    metrics_history_json_path: Path
    metric_png_paths: dict[str, Path]
    trajectory_png_path: Path
    trajectory_gif_path: Path
    metrics_summary: EpisodeMetricsSummary


@dataclass(frozen=True)
class DemoModeConfig:
    mode: str
    num_uavs: int
    num_users: int
    user_demand_rate_bps: float
    orbit_radius_m: float
    user_speed_mean_mps: float
    backhaul_capacity_bps_override: float | None
    relay_capacity_scale: float
    relay_capacity_cap_bps: float | None
    user_distribution: str


def get_demo_mode_config(mode: str) -> DemoModeConfig:
    normalized_mode = mode.lower()
    if normalized_mode == "stress":
        return DemoModeConfig(
            mode="stress",
            num_uavs=4,
            num_users=config.NUM_USERS,
            user_demand_rate_bps=2.0e6,
            orbit_radius_m=600.0,
            user_speed_mean_mps=3.5,
            backhaul_capacity_bps_override=None,
            relay_capacity_scale=0.35,
            relay_capacity_cap_bps=None,
            user_distribution="hotspots",
        )
    return DemoModeConfig(
        mode="default",
        num_uavs=config.NUM_UAVS,
        num_users=min(config.NUM_USERS, 20),
        user_demand_rate_bps=config.USER_DEMAND_RATE_BPS,
        orbit_radius_m=220.0,
        user_speed_mean_mps=config.USER_SPEED_MEAN,
        backhaul_capacity_bps_override=None,
        relay_capacity_scale=1.0,
        relay_capacity_cap_bps=None,
        user_distribution="uniform",
    )


def build_demo_entities(
    *,
    num_uavs: int = config.NUM_UAVS,
    num_users: int = config.NUM_USERS,
    seed: int = 0,
    backhaul_type: str = config.BACKHAUL_TYPE,
    user_demand_rate_bps: float = config.USER_DEMAND_RATE_BPS,
    orbit_radius_m: float = 220.0,
    user_speed_mean_mps: float = config.USER_SPEED_MEAN,
    user_distribution: str = "uniform",
) -> tuple[list[UAV], list[GroundUser], list[Satellite], list[GroundBaseStation]]:
    rng = np.random.default_rng(seed)

    gateway_position = np.array([config.MAP_LENGTH / 2.0, config.MAP_WIDTH / 2.0, config.UAV_HEIGHT], dtype=float)
    uavs: list[UAV] = [
        UAV(
            id=0,
            position=gateway_position,
            velocity=np.zeros(2),
            speed=0.0,
            direction=0.0,
            is_gateway_capable=True,
            energy_model=SimplifiedEnergyModel(),
        )
    ]

    for uav_id in range(1, int(num_uavs)):
        angle_rad = 2.0 * np.pi * (uav_id - 1) / max(int(num_uavs) - 1, 1)
        position = np.array(
                [
                    gateway_position[0] + orbit_radius_m * np.cos(angle_rad),
                    gateway_position[1] + orbit_radius_m * np.sin(angle_rad),
                config.UAV_HEIGHT,
            ],
            dtype=float,
        )
        position[0] = np.clip(position[0], 0.0, config.MAP_LENGTH)
        position[1] = np.clip(position[1], 0.0, config.MAP_WIDTH)
        uavs.append(
            UAV(
                id=uav_id,
                position=position,
                velocity=np.zeros(2),
                speed=0.0,
                direction=0.0,
                energy_model=SimplifiedEnergyModel(),
            )
        )

    users: list[GroundUser] = []
    hotspot_centers = [
        np.array([0.25 * config.MAP_LENGTH, 0.30 * config.MAP_WIDTH, 0.0], dtype=float),
        np.array([0.72 * config.MAP_LENGTH, 0.32 * config.MAP_WIDTH, 0.0], dtype=float),
        np.array([0.58 * config.MAP_LENGTH, 0.72 * config.MAP_WIDTH, 0.0], dtype=float),
    ]
    hotspot_spread_m = 180.0
    for user_id in range(int(num_users)):
        if user_distribution == "hotspots":
            center = hotspot_centers[user_id % len(hotspot_centers)]
            position = center + np.array(
                [
                    rng.normal(0.0, hotspot_spread_m),
                    rng.normal(0.0, hotspot_spread_m),
                    0.0,
                ],
                dtype=float,
            )
            position[0] = np.clip(position[0], 0.0, config.MAP_LENGTH)
            position[1] = np.clip(position[1], 0.0, config.MAP_WIDTH)
        else:
            position = np.array(
                [
                    rng.uniform(0.1 * config.MAP_LENGTH, 0.9 * config.MAP_LENGTH),
                    rng.uniform(0.1 * config.MAP_WIDTH, 0.9 * config.MAP_WIDTH),
                    0.0,
                ],
                dtype=float,
            )
        speed_mps = float(rng.uniform(0.0, user_speed_mean_mps))
        direction_rad = float(rng.uniform(-np.pi, np.pi))
        velocity = np.array([speed_mps * np.cos(direction_rad), speed_mps * np.sin(direction_rad)], dtype=float)
        users.append(
            GroundUser(
                id=user_id,
                position=position,
                velocity=velocity,
                speed=speed_mps,
                demand_rate_bps=user_demand_rate_bps,
                mobility_model=RandomWalkMobility(speed_mean_mps=user_speed_mean_mps),
            )
        )

    satellites = [Satellite(id=0)] if backhaul_type == "satellite" else []
    ground_base_stations = [GroundBaseStation(id=0)] if backhaul_type == "gbs" else []
    return uavs, users, satellites, ground_base_stations


def run_demo_episode(
    *,
    output_dir: str | Path,
    num_steps: int = 20,
    seed: int = 0,
    num_uavs: int | None = None,
    num_users: int | None = None,
    backhaul_type: str = config.BACKHAUL_TYPE,
    deterministic_policy: bool = False,
    demo_mode: str = "default",
) -> DemoArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_config = get_demo_mode_config(demo_mode)
    num_uavs = mode_config.num_uavs if num_uavs is None else int(num_uavs)
    num_users = mode_config.num_users if num_users is None else int(num_users)

    uavs, users, satellites, ground_base_stations = build_demo_entities(
        num_uavs=num_uavs,
        num_users=num_users,
        seed=seed,
        backhaul_type=backhaul_type,
        user_demand_rate_bps=mode_config.user_demand_rate_bps,
        orbit_radius_m=mode_config.orbit_radius_m,
        user_speed_mean_mps=mode_config.user_speed_mean_mps,
        user_distribution=mode_config.user_distribution,
    )
    sim_env = SimEnv(
        uavs=uavs,
        users=users,
        satellites=satellites,
        ground_base_stations=ground_base_stations,
        gateway_capable_uav_ids=[0],
        backhaul_type=backhaul_type,
        rng=np.random.default_rng(seed),
    )
    marl_env = MultiAgentUavNetEnv(sim_env, max_steps=num_steps)
    policy = MAPPOStub(rng=np.random.default_rng(seed + 1))
    metrics_collector = MetricsCollector()
    trajectory_visualizer = TrajectoryVisualizer()
    metrics_plotter = MetricsPlotter()

    observations_by_agent, _ = marl_env.reset()
    visualization_frames = []
    backhaul_node_position = None
    if backhaul_type == "satellite" and sim_env.satellites:
        backhaul_node_position = sim_env.satellites[0].position
    elif backhaul_type == "gbs" and sim_env.ground_base_stations:
        backhaul_node_position = sim_env.ground_base_stations[0].position

    for _ in range(int(num_steps)):
        actions_by_agent = policy.act(observations_by_agent, deterministic=deterministic_policy)
        relay_capacity_matrix_bps = build_a2a_capacity_matrix_bps(sim_env.uavs)
        relay_capacity_matrix_bps = relay_capacity_matrix_bps * mode_config.relay_capacity_scale
        if mode_config.relay_capacity_cap_bps is not None:
            relay_capacity_matrix_bps = np.minimum(relay_capacity_matrix_bps, mode_config.relay_capacity_cap_bps)
        observations_by_agent, _, terminated_by_agent, truncated_by_agent, step_info = marl_env.step(
            actions_by_agent,
            relay_capacity_matrix_bps=relay_capacity_matrix_bps,
            backhaul_capacity_bps_override=mode_config.backhaul_capacity_bps_override,
        )
        step_result = step_info["step_result"]
        metrics_collector.record(step_result, sim_env.uavs, sim_env.users)
        visualization_frames.append(
            build_visualization_frame(
                step_result,
                sim_env.uavs,
                sim_env.users,
                gateway_uav_ids=step_result.env_state.active_gateway_uav_ids,
                backhaul_type=backhaul_type,
                backhaul_node_position=backhaul_node_position,
            )
        )
        if terminated_by_agent["__all__"] or truncated_by_agent["__all__"]:
            break

    if not visualization_frames:
        raise RuntimeError("The demo runner did not produce any visualization frames.")

    metrics_summary = metrics_collector.summarize()
    summary_json_path = output_dir / "summary.json"
    metrics_history_json_path = output_dir / "metrics_history.json"
    metric_png_paths = metrics_plotter.plot_metric_set(metrics_collector.step_records, output_dir / "plots")
    trajectory_png_path = trajectory_visualizer.render_frame(
        visualization_frames,
        frame_index=len(visualization_frames) - 1,
        output_png_path=output_dir / "trajectory_final.png",
    )
    trajectory_gif_path = trajectory_visualizer.create_gif(
        visualization_frames,
        output_gif_path=output_dir / "trajectory.gif",
        frame_duration_ms=250,
    )

    summary_json_path.write_text(json.dumps(asdict(metrics_summary), indent=2), encoding="utf-8")
    metrics_history_json_path.write_text(json.dumps(metrics_collector.export_history(), indent=2), encoding="utf-8")

    return DemoArtifacts(
        output_dir=output_dir,
        summary_json_path=summary_json_path,
        metrics_history_json_path=metrics_history_json_path,
        metric_png_paths=metric_png_paths,
        trajectory_png_path=trajectory_png_path,
        trajectory_gif_path=trajectory_gif_path,
        metrics_summary=metrics_summary,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal rl_uavnetsim demo episode.")
    parser.add_argument("--output-dir", default="demo_outputs", help="Directory for summary, png, and gif outputs.")
    parser.add_argument("--steps", type=int, default=12, help="Number of environment steps to run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num-uavs", type=int, default=None, help="Number of UAVs. Defaults depend on demo mode.")
    parser.add_argument("--num-users", type=int, default=None, help="Number of users. Defaults depend on demo mode.")
    parser.add_argument("--backhaul-type", choices=["satellite", "gbs"], default=config.BACKHAUL_TYPE)
    parser.add_argument("--deterministic-policy", action="store_true", help="Use zero-motion stub actions.")
    parser.add_argument("--demo-mode", choices=["default", "stress"], default="default")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifacts = run_demo_episode(
        output_dir=args.output_dir,
        num_steps=args.steps,
        seed=args.seed,
        num_uavs=args.num_uavs,
        num_users=args.num_users,
        backhaul_type=args.backhaul_type,
        deterministic_policy=args.deterministic_policy,
        demo_mode=args.demo_mode,
    )
    print(f"Demo complete. Outputs written to: {artifacts.output_dir}")
    print(f"Summary: {artifacts.summary_json_path}")
    print("Metric plots:")
    for metric_name, metric_path in artifacts.metric_png_paths.items():
        print(f"  {metric_name}: {metric_path}")
    print(f"Trajectory PNG: {artifacts.trajectory_png_path}")
    print(f"Trajectory GIF: {artifacts.trajectory_gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
