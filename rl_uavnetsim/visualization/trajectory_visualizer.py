from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rl_uavnetsim import config
from rl_uavnetsim.entities import GroundUser, UAV
from rl_uavnetsim.environment import SimStepResult


@dataclass
class VisualizationFrame:
    current_step: int
    uav_ids: list[int]
    user_ids: list[int]
    uav_positions: np.ndarray
    user_positions: np.ndarray
    associations: np.ndarray
    adjacency_matrix: np.ndarray
    user_backlog_bits: np.ndarray
    uav_queue_bits: np.ndarray
    access_uploaded_bits_by_user: np.ndarray
    final_rate_bps_by_user: np.ndarray
    relay_out_bits_by_uav: np.ndarray
    backhaul_capacity_bps: float
    total_delivered_bits_step: float
    gateway_uav_ids: tuple[int, ...]
    gateway_capable_uav_ids: tuple[int, ...]
    backhaul_type: str | None = None
    backhaul_node_position: np.ndarray | None = None


def build_visualization_frame(
    step_result: SimStepResult,
    uavs: Sequence[UAV],
    users: Sequence[GroundUser],
    *,
    gateway_uav_ids: Sequence[int] | None = None,
    backhaul_type: str | None = None,
    backhaul_node_position: np.ndarray | None = None,
) -> VisualizationFrame:
    resolved_gateway_uav_ids = tuple(
        int(gateway_uav_id)
        for gateway_uav_id in (
            gateway_uav_ids
            if gateway_uav_ids is not None
            else step_result.env_state.active_gateway_uav_ids
        )
    )
    uav_ids = [uav.id for uav in uavs]
    user_ids = [user.id for user in users]
    access_uploaded_bits_by_user = np.asarray(
        [step_result.access_step_result.total_access_uploaded_bits_by_user.get(user.id, 0.0) for user in users],
        dtype=float,
    )
    relay_out_bits_by_uav = np.asarray(
        [step_result.relay_service_result.relay_out_bits_by_uav.get(uav.id, 0.0) for uav in uavs],
        dtype=float,
    )
    return VisualizationFrame(
        current_step=step_result.env_state.current_step,
        uav_ids=uav_ids,
        user_ids=user_ids,
        uav_positions=np.asarray([uav.position for uav in uavs], dtype=float),
        user_positions=np.asarray([user.position for user in users], dtype=float),
        associations=np.asarray([user.associated_uav_id for user in users], dtype=int),
        adjacency_matrix=np.asarray(step_result.env_state.adjacency_matrix, dtype=int),
        user_backlog_bits=np.asarray([user.user_access_backlog_bits for user in users], dtype=float),
        uav_queue_bits=np.asarray([uav.relay_queue_total_bits for uav in uavs], dtype=float),
        access_uploaded_bits_by_user=access_uploaded_bits_by_user,
        final_rate_bps_by_user=np.asarray([user.final_rate_bps for user in users], dtype=float),
        relay_out_bits_by_uav=relay_out_bits_by_uav,
        backhaul_capacity_bps=float(step_result.env_state.backhaul_capacity_bps),
        total_delivered_bits_step=float(step_result.env_state.total_delivered_bits_step),
        gateway_uav_ids=resolved_gateway_uav_ids,
        gateway_capable_uav_ids=tuple(
            sorted(uav.id for uav in uavs if uav.is_gateway_capable)
        ),
        backhaul_type=backhaul_type,
        backhaul_node_position=None if backhaul_node_position is None else np.asarray(backhaul_node_position, dtype=float),
    )


class TrajectoryVisualizer:
    def render_frame(
        self,
        frames: Sequence[VisualizationFrame],
        frame_index: int,
        output_png_path: str | Path,
    ) -> Path:
        output_png_path = Path(output_png_path)
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        figure = self._build_figure(frames, frame_index)
        figure.savefig(output_png_path, dpi=160)
        plt.close(figure)
        return output_png_path

    def create_gif(
        self,
        frames: Sequence[VisualizationFrame],
        output_gif_path: str | Path,
        *,
        frame_duration_ms: int = 350,
    ) -> Path:
        output_gif_path = Path(output_gif_path)
        output_gif_path.parent.mkdir(parents=True, exist_ok=True)

        images: list[Image.Image] = []
        for frame_index in range(len(frames)):
            figure = self._build_figure(frames, frame_index)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=120)
            plt.close(figure)
            buffer.seek(0)
            images.append(Image.open(buffer).convert("P"))

        if not images:
            raise ValueError("At least one visualization frame is required to create a GIF.")

        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=int(frame_duration_ms),
            loop=0,
        )
        return output_gif_path

    def _build_figure(self, frames: Sequence[VisualizationFrame], frame_index: int):
        current_frame = frames[frame_index]
        figure = plt.figure(figsize=(10, 7))
        axis = figure.add_subplot(111, projection="3d")
        axis.set_title(
            f"Step {current_frame.current_step} | delivered={current_frame.total_delivered_bits_step:.1f} bits"
        )
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Z (m)")
        axis.set_xlim(0, config.MAP_LENGTH)
        axis.set_ylim(0, config.MAP_WIDTH)
        axis.set_zlim(0, max(config.UAV_HEIGHT * 1.5, 150.0))

        trajectory_by_uav_id: dict[int, np.ndarray] = {}
        for uav_index, uav_id in enumerate(current_frame.uav_ids):
            trajectory_by_uav_id[uav_id] = np.asarray(
                [frame.uav_positions[uav_index] for frame in frames[: frame_index + 1]],
                dtype=float,
            )

        user_backlog_norm = current_frame.user_backlog_bits / max(config.ACCESS_BACKLOG_REF_BITS, config.EPSILON)
        uav_queue_norm = current_frame.uav_queue_bits / max(config.RELAY_QUEUE_REF_BITS, config.EPSILON)

        for uav_index, uav_id in enumerate(current_frame.uav_ids):
            trajectory = trajectory_by_uav_id[uav_id]
            is_active_gateway = uav_id in current_frame.gateway_uav_ids
            is_gateway_capable = uav_id in current_frame.gateway_capable_uav_ids
            if is_active_gateway:
                color = "crimson"
            elif is_gateway_capable:
                color = "darkorange"
            else:
                color = plt.cm.cividis(min(uav_queue_norm[uav_index], 1.0))
            axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, linewidth=2.0, alpha=0.85)
            axis.scatter(
                [trajectory[-1, 0]],
                [trajectory[-1, 1]],
                [trajectory[-1, 2]],
                s=90 + 180 * min(uav_queue_norm[uav_index], 1.0),
                c=[color],
                edgecolors="black",
            )

        user_colors = plt.cm.YlOrRd(np.clip(user_backlog_norm, 0.0, 1.0))
        axis.scatter(
            current_frame.user_positions[:, 0],
            current_frame.user_positions[:, 1],
            current_frame.user_positions[:, 2],
            s=25 + 120 * np.clip(user_backlog_norm, 0.0, 1.0),
            c=user_colors,
            alpha=0.85,
        )

        uav_id_to_index = {uav_id: index for index, uav_id in enumerate(current_frame.uav_ids)}
        for user_index, associated_uav_id in enumerate(current_frame.associations):
            if associated_uav_id < 0 or associated_uav_id not in uav_id_to_index:
                continue
            uav_index = uav_id_to_index[int(associated_uav_id)]
            access_norm = current_frame.access_uploaded_bits_by_user[user_index] / max(config.ACCESS_BACKLOG_REF_BITS, config.EPSILON)
            line_width = 0.75 + 6.0 * min(access_norm, 1.0)
            axis.plot(
                [current_frame.user_positions[user_index, 0], current_frame.uav_positions[uav_index, 0]],
                [current_frame.user_positions[user_index, 1], current_frame.uav_positions[uav_index, 1]],
                [current_frame.user_positions[user_index, 2], current_frame.uav_positions[uav_index, 2]],
                color="tab:blue",
                linewidth=line_width,
                alpha=0.35,
            )

        for source_index in range(len(current_frame.uav_ids)):
            for target_index in range(source_index + 1, len(current_frame.uav_ids)):
                if current_frame.adjacency_matrix[source_index, target_index] <= 0 and current_frame.adjacency_matrix[target_index, source_index] <= 0:
                    continue
                relay_norm = max(
                    current_frame.relay_out_bits_by_uav[source_index],
                    current_frame.relay_out_bits_by_uav[target_index],
                ) / max(config.RELAY_QUEUE_REF_BITS, config.EPSILON)
                axis.plot(
                    [current_frame.uav_positions[source_index, 0], current_frame.uav_positions[target_index, 0]],
                    [current_frame.uav_positions[source_index, 1], current_frame.uav_positions[target_index, 1]],
                    [current_frame.uav_positions[source_index, 2], current_frame.uav_positions[target_index, 2]],
                    color="tab:green",
                    linewidth=1.0 + 4.0 * min(relay_norm, 1.0),
                    alpha=0.5,
                )

        if current_frame.backhaul_node_position is not None:
            backhaul_norm = current_frame.backhaul_capacity_bps / max(config.THROUGHPUT_REF_BITS / config.DELTA_T, config.EPSILON)
            for gateway_uav_id in current_frame.gateway_uav_ids:
                if gateway_uav_id not in uav_id_to_index:
                    continue
                gateway_index = uav_id_to_index[gateway_uav_id]
                axis.plot(
                    [current_frame.uav_positions[gateway_index, 0], current_frame.backhaul_node_position[0]],
                    [current_frame.uav_positions[gateway_index, 1], current_frame.backhaul_node_position[1]],
                    [current_frame.uav_positions[gateway_index, 2], current_frame.backhaul_node_position[2]],
                    color="tab:purple",
                    linewidth=1.0 + 4.0 * min(backhaul_norm, 1.0),
                    alpha=0.6,
                )

        return figure
