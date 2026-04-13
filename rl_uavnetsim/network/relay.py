from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.channel.a2a_channel import a2a_capacity_bps, a2a_link_is_active
from rl_uavnetsim.entities.ground_user import GroundUser
from rl_uavnetsim.entities.uav import UAV
from rl_uavnetsim.network.routing import RouteDecision, compute_routing_table


@dataclass
class RelayServiceResult:
    route_decision_by_uav: dict[int, RouteDecision] = field(default_factory=dict)
    relay_budget_bits_by_uav: dict[int, float] = field(default_factory=dict)
    relay_out_bits_by_uav: dict[int, float] = field(default_factory=dict)
    relay_in_bits_by_uav: dict[int, float] = field(default_factory=dict)
    relay_in_bits_by_uav_and_user: dict[tuple[int, int], float] = field(default_factory=dict)
    active_gateway_uav_ids: list[int] = field(default_factory=list)
    adjacency_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    capacity_matrix_bps: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    lambda2: float = 0.0

    @property
    def total_relay_in_bits(self) -> float:
        return float(sum(self.relay_in_bits_by_uav.values()))


@dataclass
class BackhaulServiceResult:
    backhaul_capacity_bps_by_gateway: dict[int, float]
    backhaul_budget_bits_by_gateway: dict[int, float]
    backhaul_out_bits_by_gateway: dict[int, float]
    delivered_bits_by_user: dict[int, float]

    @property
    def backhaul_capacity_bps(self) -> float:
        return float(sum(self.backhaul_capacity_bps_by_gateway.values()))

    @property
    def backhaul_budget_bits(self) -> float:
        return float(sum(self.backhaul_budget_bits_by_gateway.values()))

    @property
    def backhaul_out_bits(self) -> float:
        return float(sum(self.backhaul_out_bits_by_gateway.values()))


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


def _proportional_dequeue_from_queue(
    queue_bits_by_user: Mapping[int, float],
    service_budget_bits: float,
) -> tuple[dict[int, float], dict[int, float]]:
    queue_total_bits = float(sum(max(0.0, float(bits)) for bits in queue_bits_by_user.values()))
    if queue_total_bits <= 0.0:
        return dict(queue_bits_by_user), {}

    service_budget_bits = max(0.0, min(float(service_budget_bits), queue_total_bits))
    remaining_bits_by_user = {int(user_id): max(0.0, float(bits)) for user_id, bits in queue_bits_by_user.items() if bits > 0.0}
    forwarded_bits_by_user: dict[int, float] = {}
    for user_id, queued_bits in list(remaining_bits_by_user.items()):
        share_norm = queued_bits / queue_total_bits
        forwarded_bits = share_norm * service_budget_bits
        remaining_bits = max(0.0, queued_bits - forwarded_bits)
        if remaining_bits <= config.EPSILON:
            remaining_bits_by_user.pop(user_id, None)
        else:
            remaining_bits_by_user[user_id] = remaining_bits
        forwarded_bits_by_user[user_id] = forwarded_bits

    return remaining_bits_by_user, forwarded_bits_by_user


def execute_relay_service(
    uavs: Sequence[UAV],
    *,
    active_gateway_uav_ids: Sequence[int],
    delta_t_s: float = config.DELTA_T,
    capacity_matrix_bps: np.ndarray | None = None,
    backhaul_capacity_bps_by_gateway: Mapping[int, float] | None = None,
    routing_table: Mapping[int, RouteDecision] | None = None,
) -> RelayServiceResult:
    if capacity_matrix_bps is None:
        capacity_matrix_bps = build_a2a_capacity_matrix_bps(uavs)
    capacity_matrix_bps = np.asarray(capacity_matrix_bps, dtype=float)
    adjacency_matrix = build_adjacency_matrix(capacity_matrix_bps)
    lambda2 = algebraic_connectivity_lambda2(adjacency_matrix)

    gateway_capacity_lookup = {
        int(gateway_uav_id): max(0.0, float(capacity_bps))
        for gateway_uav_id, capacity_bps in (backhaul_capacity_bps_by_gateway or {}).items()
    }
    for gateway_uav_id in active_gateway_uav_ids:
        gateway_capacity_lookup.setdefault(int(gateway_uav_id), float(np.inf))
    if routing_table is None:
        routing_table = compute_routing_table(
            uavs=uavs,
            active_gateway_uav_ids=active_gateway_uav_ids,
            capacity_matrix_bps=capacity_matrix_bps,
            backhaul_capacity_bps_by_gateway=gateway_capacity_lookup,
        )

    result = RelayServiceResult(
        route_decision_by_uav=dict(routing_table),
        active_gateway_uav_ids=sorted(int(gateway_uav_id) for gateway_uav_id in active_gateway_uav_ids),
        adjacency_matrix=adjacency_matrix,
        capacity_matrix_bps=capacity_matrix_bps,
        lambda2=lambda2,
        relay_in_bits_by_uav={uav.id: 0.0 for uav in uavs},
    )

    snapshot_queue_by_uav = {
        uav.id: {
            user_id: float(bits)
            for user_id, bits in uav.relay_queue_bits_by_user.items()
            if bits > config.EPSILON
        }
        for uav in uavs
    }
    next_queue_by_uav = {
        uav_id: dict(queue_bits_by_user) for uav_id, queue_bits_by_user in snapshot_queue_by_uav.items()
    }
    staging_buffer_by_uav = {uav.id: {} for uav in uavs}

    for uav in sorted(uavs, key=lambda uav: uav.id):
        decision = routing_table.get(uav.id)
        if decision is None or decision.next_hop_uav_id is None or not decision.is_reachable:
            result.relay_budget_bits_by_uav[uav.id] = 0.0
            result.relay_out_bits_by_uav[uav.id] = 0.0
            continue

        relay_budget_bits = decision.effective_path_capacity_bps * float(delta_t_s)
        queue_bits_by_user = snapshot_queue_by_uav.get(uav.id, {})
        queue_total_bits = float(sum(queue_bits_by_user.values()))
        relay_out_bits = min(queue_total_bits, relay_budget_bits)
        result.relay_budget_bits_by_uav[uav.id] = relay_budget_bits
        result.relay_out_bits_by_uav[uav.id] = relay_out_bits

        remaining_bits_by_user, forwarded_bits_by_user = _proportional_dequeue_from_queue(
            queue_bits_by_user=queue_bits_by_user,
            service_budget_bits=relay_out_bits,
        )
        next_queue_by_uav[uav.id] = remaining_bits_by_user
        uav.register_relay_forwarded_bits(sum(forwarded_bits_by_user.values()))

        for user_id, forwarded_bits in forwarded_bits_by_user.items():
            staging_buffer = staging_buffer_by_uav[decision.next_hop_uav_id]
            staging_buffer[user_id] = staging_buffer.get(user_id, 0.0) + forwarded_bits
            result.relay_in_bits_by_uav[decision.next_hop_uav_id] += forwarded_bits
            result.relay_in_bits_by_uav_and_user[(decision.next_hop_uav_id, user_id)] = (
                result.relay_in_bits_by_uav_and_user.get((decision.next_hop_uav_id, user_id), 0.0) + forwarded_bits
            )

    for uav in uavs:
        merged_queue_bits_by_user = dict(next_queue_by_uav[uav.id])
        for user_id, forwarded_bits in staging_buffer_by_uav[uav.id].items():
            merged_queue_bits_by_user[user_id] = merged_queue_bits_by_user.get(user_id, 0.0) + forwarded_bits
        uav.relay_queue_bits_by_user = {
            user_id: bits for user_id, bits in merged_queue_bits_by_user.items() if bits > config.EPSILON
        }

    return result


def execute_backhaul_service(
    gateway_uavs: Sequence[UAV] | Mapping[int, UAV] | UAV,
    users: Sequence[GroundUser] | Mapping[int, GroundUser],
    *,
    backhaul_capacity_bps_by_gateway: Mapping[int, float] | None = None,
    backhaul_capacity_bps: float | None = None,
    delta_t_s: float = config.DELTA_T,
) -> BackhaulServiceResult:
    if isinstance(gateway_uavs, UAV):
        gateway_uavs_by_id = {gateway_uavs.id: gateway_uavs}
    elif isinstance(gateway_uavs, Mapping):
        gateway_uavs_by_id = dict(gateway_uavs)
    else:
        gateway_uavs_by_id = {gateway_uav.id: gateway_uav for gateway_uav in gateway_uavs}
    users_by_id = {user.id: user for user in users} if not isinstance(users, Mapping) else dict(users)

    if backhaul_capacity_bps_by_gateway is not None and backhaul_capacity_bps is not None:
        raise ValueError("Pass either backhaul_capacity_bps_by_gateway or backhaul_capacity_bps, not both.")

    if backhaul_capacity_bps_by_gateway is None:
        if backhaul_capacity_bps is None:
            raise ValueError("A backhaul capacity override must be provided.")
        if len(gateway_uavs_by_id) != 1:
            raise ValueError("Scalar backhaul capacity is only valid when exactly one gateway UAV is provided.")
        gateway_uav_id = next(iter(gateway_uavs_by_id))
        backhaul_capacity_bps_by_gateway = {gateway_uav_id: float(backhaul_capacity_bps)}

    backhaul_budget_bits_by_gateway: dict[int, float] = {}
    backhaul_out_bits_by_gateway: dict[int, float] = {}
    delivered_bits_by_user: dict[int, float] = {}

    for gateway_uav_id, gateway_uav in gateway_uavs_by_id.items():
        gateway_capacity_bps = max(0.0, float(backhaul_capacity_bps_by_gateway.get(gateway_uav_id, 0.0)))
        backhaul_budget_bits = gateway_capacity_bps * float(delta_t_s)
        backhaul_budget_bits_by_gateway[gateway_uav_id] = backhaul_budget_bits

        queue_total_bits = gateway_uav.relay_queue_total_bits
        backhaul_out_bits = min(queue_total_bits, backhaul_budget_bits)
        backhaul_out_bits_by_gateway[gateway_uav_id] = backhaul_out_bits

        delivered_bits_from_gateway = gateway_uav.proportional_dequeue(backhaul_out_bits)
        gateway_uav.register_backhaul_forwarded_bits(sum(delivered_bits_from_gateway.values()))
        for user_id, delivered_bits in delivered_bits_from_gateway.items():
            delivered_bits_by_user[user_id] = delivered_bits_by_user.get(user_id, 0.0) + delivered_bits

    for user_id, delivered_bits in delivered_bits_by_user.items():
        users_by_id[user_id].add_delivered_bits(delivered_bits, delta_t_s=delta_t_s)

    return BackhaulServiceResult(
        backhaul_capacity_bps_by_gateway={
            int(gateway_uav_id): float(capacity_bps)
            for gateway_uav_id, capacity_bps in backhaul_capacity_bps_by_gateway.items()
        },
        backhaul_budget_bits_by_gateway=backhaul_budget_bits_by_gateway,
        backhaul_out_bits_by_gateway=backhaul_out_bits_by_gateway,
        delivered_bits_by_user=delivered_bits_by_user,
    )
