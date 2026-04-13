from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.entities import GroundUser, Satellite, UAV
from rl_uavnetsim.environment import SimEnv
from rl_uavnetsim.metrics import MetricsCollector
from rl_uavnetsim.visualization import MetricsPlotter, TrajectoryVisualizer, build_visualization_frame


def test_metrics_collector_and_plotters_generate_outputs(tmp_path: Path) -> None:
    anchor = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        is_anchor=True,
    )
    member = UAV(
        id=1,
        position=np.array([50.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    user = GroundUser(
        id=7,
        position=np.array([55.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        demand_rate_bps=2_000.0,
        user_access_backlog_bits=1_000.0,
    )
    satellite = Satellite(id=0)
    env = SimEnv(
        uavs=[anchor, member],
        users=[user],
        satellites=[satellite],
        anchor_uav_id=anchor.id,
        backhaul_type="satellite",
    )

    metrics_collector = MetricsCollector()
    visualization_frames = []
    for _ in range(2):
        step_result = env.step(
            relay_capacity_matrix_bps=np.array([[0.0, 5000.0], [5000.0, 0.0]], dtype=float),
            backhaul_capacity_bps_override=4000.0,
        )
        metrics_collector.record(step_result, env.uavs, env.users)
        visualization_frames.append(
            build_visualization_frame(
                step_result,
                env.uavs,
                env.users,
                anchor_uav_id=anchor.id,
                backhaul_type="satellite",
                backhaul_node_position=satellite.position,
            )
        )

    summary = metrics_collector.summarize()
    history = metrics_collector.export_history()

    assert summary.num_steps == 2
    assert len(history) == 2
    assert "sum_throughput_bps" in history[0]
    assert summary.cumulative_arrived_bits > 0.0

    metrics_png_path = MetricsPlotter().plot_step_metrics(
        metrics_collector.step_records,
        tmp_path / "metrics.png",
    )
    frame_png_path = TrajectoryVisualizer().render_frame(
        visualization_frames,
        frame_index=1,
        output_png_path=tmp_path / "frame.png",
    )
    gif_path = TrajectoryVisualizer().create_gif(
        visualization_frames,
        output_gif_path=tmp_path / "episode.gif",
        frame_duration_ms=150,
    )

    assert metrics_png_path.exists() and metrics_png_path.stat().st_size > 0
    assert frame_png_path.exists() and frame_png_path.stat().st_size > 0
    assert gif_path.exists() and gif_path.stat().st_size > 0


def test_visualization_frame_tracks_gateway_ids_from_env_state() -> None:
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
        id=9,
        position=np.array([40.0, 0.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
        demand_rate_bps=1_000.0,
        user_access_backlog_bits=1_000.0,
    )
    satellite = Satellite(id=0)
    env = SimEnv(
        uavs=[gateway, relay],
        users=[user],
        satellites=[satellite],
        gateway_capable_uav_ids=[gateway.id],
        backhaul_type="satellite",
    )
    step_result = env.step(
        relay_capacity_matrix_bps=np.array([[0.0, 2000.0], [2000.0, 0.0]], dtype=float),
        backhaul_capacity_bps_override=1000.0,
    )

    frame = build_visualization_frame(
        step_result,
        env.uavs,
        env.users,
        gateway_uav_ids=step_result.env_state.active_gateway_uav_ids,
        backhaul_type="satellite",
        backhaul_node_position=satellite.position,
    )

    assert frame.gateway_uav_ids == step_result.env_state.active_gateway_uav_ids
