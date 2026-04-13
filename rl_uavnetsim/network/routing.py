from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rl_uavnetsim.entities.uav import UAV


@dataclass
class RouteDecision:
    source_uav_id: int
    selected_gateway_uav_id: int | None
    next_hop_uav_id: int | None
    path_uav_ids: list[int]
    reachable_gateway_uav_ids: list[int]
    path_bottleneck_capacity_bps: float
    gateway_backhaul_capacity_bps: float
    effective_path_capacity_bps: float
    downstream_queue_pressure: float
    hop_count: int

    @property
    def is_reachable(self) -> bool:
        return self.selected_gateway_uav_id is not None and self.effective_path_capacity_bps > 0.0

    @property
    def reachable_gateway_count(self) -> int:
        return len(self.reachable_gateway_uav_ids)


def _id_to_index_map(uavs: Sequence[UAV]) -> dict[int, int]:
    return {uav.id: index for index, uav in enumerate(uavs)}


def _widest_path_indices(
    source_index: int,
    target_index: int,
    capacity_matrix_bps: np.ndarray,
) -> tuple[list[int], float]:
    num_nodes = capacity_matrix_bps.shape[0]
    best_bottleneck_by_index = np.zeros(num_nodes, dtype=float)
    predecessor_by_index = np.full(num_nodes, -1, dtype=int)
    best_bottleneck_by_index[source_index] = np.inf

    heap: list[tuple[float, int]] = [(-np.inf, source_index)]
    while heap:
        negative_bottleneck, current_index = heapq.heappop(heap)
        current_bottleneck = -negative_bottleneck
        if current_index == target_index:
            break
        if current_bottleneck < best_bottleneck_by_index[current_index]:
            continue
        for neighbor_index in range(num_nodes):
            edge_capacity_bps = float(capacity_matrix_bps[current_index, neighbor_index])
            if edge_capacity_bps <= 0.0:
                continue
            candidate_bottleneck = edge_capacity_bps
            if np.isfinite(current_bottleneck):
                candidate_bottleneck = min(current_bottleneck, edge_capacity_bps)
            if candidate_bottleneck > best_bottleneck_by_index[neighbor_index]:
                best_bottleneck_by_index[neighbor_index] = candidate_bottleneck
                predecessor_by_index[neighbor_index] = current_index
                heapq.heappush(heap, (-candidate_bottleneck, neighbor_index))

    bottleneck_capacity_bps = float(best_bottleneck_by_index[target_index])
    if bottleneck_capacity_bps <= 0.0:
        return [], 0.0

    path_indices = [target_index]
    current_index = target_index
    while current_index != source_index:
        current_index = int(predecessor_by_index[current_index])
        if current_index < 0:
            return [], 0.0
        path_indices.append(current_index)
    path_indices.reverse()
    return path_indices, bottleneck_capacity_bps


def compute_routing_table(
    *,
    uavs: Sequence[UAV],
    active_gateway_uav_ids: Sequence[int],
    capacity_matrix_bps: np.ndarray,
    backhaul_capacity_bps_by_gateway: dict[int, float],
) -> dict[int, RouteDecision]:
    active_gateway_uav_ids = sorted({int(gateway_uav_id) for gateway_uav_id in active_gateway_uav_ids})
    id_to_index = _id_to_index_map(uavs)
    capacity_matrix_bps = np.asarray(capacity_matrix_bps, dtype=float)
    routing_table: dict[int, RouteDecision] = {}

    for source_uav in sorted(uavs, key=lambda uav: uav.id):
        candidate_decisions: list[tuple[tuple[float, float, int, int], RouteDecision]] = []
        reachable_gateway_uav_ids: list[int] = []

        for gateway_uav_id in active_gateway_uav_ids:
            gateway_backhaul_capacity_bps = max(
                0.0,
                float(backhaul_capacity_bps_by_gateway.get(gateway_uav_id, 0.0)),
            )
            if gateway_backhaul_capacity_bps <= 0.0:
                continue

            if source_uav.id == gateway_uav_id:
                path_uav_ids = [source_uav.id]
                path_bottleneck_capacity_bps = np.inf
            else:
                path_indices, path_bottleneck_capacity_bps = _widest_path_indices(
                    source_index=id_to_index[source_uav.id],
                    target_index=id_to_index[gateway_uav_id],
                    capacity_matrix_bps=capacity_matrix_bps,
                )
                path_uav_ids = [uavs[index].id for index in path_indices]
            if not path_uav_ids or path_bottleneck_capacity_bps <= 0.0:
                continue

            effective_path_capacity_bps = min(path_bottleneck_capacity_bps, gateway_backhaul_capacity_bps)
            if effective_path_capacity_bps <= 0.0:
                continue

            reachable_gateway_uav_ids.append(gateway_uav_id)
            downstream_queue_pressure = float(
                sum(
                    uav.relay_queue_total_bits
                    for uav in uavs
                    if uav.id in path_uav_ids[1:]
                )
            )
            hop_count = max(0, len(path_uav_ids) - 1)
            decision = RouteDecision(
                source_uav_id=source_uav.id,
                selected_gateway_uav_id=gateway_uav_id,
                next_hop_uav_id=path_uav_ids[1] if len(path_uav_ids) > 1 else None,
                path_uav_ids=path_uav_ids,
                reachable_gateway_uav_ids=[],
                path_bottleneck_capacity_bps=float(path_bottleneck_capacity_bps),
                gateway_backhaul_capacity_bps=gateway_backhaul_capacity_bps,
                effective_path_capacity_bps=float(effective_path_capacity_bps),
                downstream_queue_pressure=downstream_queue_pressure,
                hop_count=hop_count,
            )
            candidate_decisions.append(
                (
                    (
                        -float(effective_path_capacity_bps),
                        downstream_queue_pressure,
                        hop_count,
                        gateway_uav_id,
                    ),
                    decision,
                )
            )

        if candidate_decisions:
            _, best_decision = min(candidate_decisions, key=lambda item: item[0])
            best_decision.reachable_gateway_uav_ids = sorted(reachable_gateway_uav_ids)
            routing_table[source_uav.id] = best_decision
        else:
            routing_table[source_uav.id] = RouteDecision(
                source_uav_id=source_uav.id,
                selected_gateway_uav_id=None,
                next_hop_uav_id=None,
                path_uav_ids=[],
                reachable_gateway_uav_ids=[],
                path_bottleneck_capacity_bps=0.0,
                gateway_backhaul_capacity_bps=0.0,
                effective_path_capacity_bps=0.0,
                downstream_queue_pressure=0.0,
                hop_count=0,
            )

    return routing_table
