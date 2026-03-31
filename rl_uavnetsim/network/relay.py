from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.channel.a2a_channel import a2a_capacity_bps, a2a_link_is_active
from rl_uavnetsim.entities.ground_user import GroundUser
from rl_uavnetsim.entities.uav import UAV


@dataclass
class RelayPath:
    source_uav_id: int
    anchor_uav_id: int
    path_uav_ids: list[int]
    bottleneck_capacity_bps: float

    @property
    def is_reachable(self) -> bool:
        return len(self.path_uav_ids) > 0 and self.bottleneck_capacity_bps > 0.0


@dataclass
class RelayServiceResult:
    relay_path_by_uav: dict[int, RelayPath] = field(default_factory=dict)
    relay_budget_bits_by_uav: dict[int, float] = field(default_factory=dict)
    relay_out_bits_by_uav: dict[int, float] = field(default_factory=dict)
    relay_in_bits_to_anchor_by_user: dict[int, float] = field(default_factory=dict)
    adjacency_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    capacity_matrix_bps: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    lambda2: float = 0.0

    @property
    def total_relay_in_bits_to_anchor(self) -> float:
        return float(sum(self.relay_in_bits_to_anchor_by_user.values()))


@dataclass
class BackhaulServiceResult:
    backhaul_capacity_bps: float
    backhaul_budget_bits: float
    backhaul_out_bits: float
    delivered_bits_by_user: dict[int, float]


def build_a2a_capacity_matrix_bps(
    uavs: Sequence[UAV],
    *,
    threshold_db: float = config.GAMMA_TH_DB,
) -> np.ndarray:
    num_uavs = len(uavs)
    capacity_matrix_bps = np.zeros((num_uavs, num_uavs), dtype=float)
    for source_index, source_uav in enumerate(uavs):
        for target_index, target_uav in enumerate(uavs):
            if source_index == target_index:
                continue
            if not a2a_link_is_active(source_uav.position, target_uav.position, threshold_db=threshold_db):
                continue
            capacity_matrix_bps[source_index, target_index] = a2a_capacity_bps(
                source_uav.position,
                target_uav.position,
            )
    return capacity_matrix_bps


def build_adjacency_matrix(capacity_matrix_bps: np.ndarray) -> np.ndarray:
    return (np.asarray(capacity_matrix_bps, dtype=float) > 0.0).astype(int)


def algebraic_connectivity_lambda2(adjacency_matrix: np.ndarray) -> float:
    adjacency_matrix = np.asarray(adjacency_matrix, dtype=float)
    if adjacency_matrix.size == 0:
        return 0.0
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    eigenvalues = np.linalg.eigvalsh(laplacian_matrix)
    if eigenvalues.size < 2:
        return 0.0
    return float(sorted(eigenvalues)[1])


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


def find_widest_path_to_anchor(
    source_uav_id: int,
    anchor_uav_id: int,
    uavs: Sequence[UAV],
    capacity_matrix_bps: np.ndarray,
) -> RelayPath:
    if source_uav_id == anchor_uav_id:
        return RelayPath(
            source_uav_id=source_uav_id,
            anchor_uav_id=anchor_uav_id,
            path_uav_ids=[anchor_uav_id],
            bottleneck_capacity_bps=np.inf,
        )

    id_to_index = _id_to_index_map(uavs)
    path_indices, bottleneck_capacity_bps = _widest_path_indices(
        source_index=id_to_index[source_uav_id],
        target_index=id_to_index[anchor_uav_id],
        capacity_matrix_bps=np.asarray(capacity_matrix_bps, dtype=float),
    )
    path_uav_ids = [uavs[index].id for index in path_indices]
    return RelayPath(
        source_uav_id=source_uav_id,
        anchor_uav_id=anchor_uav_id,
        path_uav_ids=path_uav_ids,
        bottleneck_capacity_bps=bottleneck_capacity_bps,
    )


def execute_relay_service(
    uavs: Sequence[UAV],
    *,
    anchor_uav_id: int = config.ANCHOR_UAV_ID,
    delta_t_s: float = config.DELTA_T,
    capacity_matrix_bps: np.ndarray | None = None,
) -> RelayServiceResult:
    if capacity_matrix_bps is None:
        capacity_matrix_bps = build_a2a_capacity_matrix_bps(uavs)
    capacity_matrix_bps = np.asarray(capacity_matrix_bps, dtype=float)
    adjacency_matrix = build_adjacency_matrix(capacity_matrix_bps)
    lambda2 = algebraic_connectivity_lambda2(adjacency_matrix)

    uavs_by_id = {uav.id: uav for uav in uavs}
    anchor_uav = uavs_by_id[anchor_uav_id]
    result = RelayServiceResult(
        adjacency_matrix=adjacency_matrix,
        capacity_matrix_bps=capacity_matrix_bps,
        lambda2=lambda2,
    )

    for uav in uavs:
        if uav.id == anchor_uav_id:
            continue
        relay_path = find_widest_path_to_anchor(
            source_uav_id=uav.id,
            anchor_uav_id=anchor_uav_id,
            uavs=uavs,
            capacity_matrix_bps=capacity_matrix_bps,
        )
        result.relay_path_by_uav[uav.id] = relay_path
        relay_budget_bits = relay_path.bottleneck_capacity_bps * float(delta_t_s) if relay_path.is_reachable else 0.0
        queue_total_bits = uav.refresh_relay_queue_total_bits()
        relay_out_bits = min(queue_total_bits, relay_budget_bits)
        result.relay_budget_bits_by_uav[uav.id] = relay_budget_bits
        result.relay_out_bits_by_uav[uav.id] = relay_out_bits

        forwarded_bits_by_user = uav.proportional_dequeue(relay_out_bits)
        uav.register_relay_forwarded_bits(sum(forwarded_bits_by_user.values()))
        for user_id, forwarded_bits in forwarded_bits_by_user.items():
            anchor_uav.enqueue_relay_bits(user_id, forwarded_bits, count_as_access_ingress=False)
            result.relay_in_bits_to_anchor_by_user[user_id] = (
                result.relay_in_bits_to_anchor_by_user.get(user_id, 0.0) + forwarded_bits
            )

    return result


def execute_backhaul_service(
    anchor_uav: UAV,
    users: Sequence[GroundUser] | Mapping[int, GroundUser],
    *,
    backhaul_capacity_bps: float,
    delta_t_s: float = config.DELTA_T,
) -> BackhaulServiceResult:
    users_by_id = {user.id: user for user in users} if not isinstance(users, Mapping) else dict(users)
    backhaul_budget_bits = max(0.0, float(backhaul_capacity_bps)) * float(delta_t_s)
    anchor_queue_total_bits = anchor_uav.refresh_relay_queue_total_bits()
    backhaul_out_bits = min(anchor_queue_total_bits, backhaul_budget_bits)

    delivered_bits_by_user = anchor_uav.proportional_dequeue(backhaul_out_bits)
    anchor_uav.register_backhaul_forwarded_bits(sum(delivered_bits_by_user.values()))

    for user_id, delivered_bits in delivered_bits_by_user.items():
        users_by_id[user_id].add_delivered_bits(delivered_bits, delta_t_s=delta_t_s)

    return BackhaulServiceResult(
        backhaul_capacity_bps=float(backhaul_capacity_bps),
        backhaul_budget_bits=backhaul_budget_bits,
        backhaul_out_bits=backhaul_out_bits,
        delivered_bits_by_user=delivered_bits_by_user,
    )
