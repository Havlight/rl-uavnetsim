from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from rl_uavnetsim import config
from rl_uavnetsim.entities import GroundUser, UAV
from rl_uavnetsim.environment import SimStepResult


@dataclass
class StepMetricsRecord:
    current_step: int
    sum_throughput_bps: float
    coverage_ratio: float
    outage_ratio: float
    jain_fairness: float
    total_energy_j: float
    energy_efficiency_bits_per_j: float
    lambda2: float
    total_user_access_backlog_bits: float
    total_uav_relay_queue_bits: float
    demand_satisfaction_ratio: float
    total_delivered_bits_step: float
    total_arrived_bits_step: float


@dataclass
class EpisodeMetricsSummary:
    num_steps: int
    mean_sum_throughput_bps: float
    mean_coverage_ratio: float
    mean_outage_ratio: float
    mean_jain_fairness: float
    total_energy_j: float
    energy_efficiency_bits_per_j: float
    mean_lambda2: float
    final_total_user_access_backlog_bits: float
    final_total_uav_relay_queue_bits: float
    demand_satisfaction_ratio: float
    cumulative_delivered_bits: float
    cumulative_arrived_bits: float


def jain_fairness_index(values: Sequence[float]) -> float:
    values = [max(0.0, float(value)) for value in values]
    if not values:
        return 0.0
    numerator = sum(values) ** 2
    denominator = len(values) * sum(value ** 2 for value in values)
    if denominator <= config.EPSILON:
        return 0.0
    return numerator / denominator


class MetricsCollector:
    def __init__(self, *, outage_threshold_bps: float = config.R_MIN) -> None:
        self.outage_threshold_bps = float(outage_threshold_bps)
        self.step_records: list[StepMetricsRecord] = []

    def reset(self) -> None:
        self.step_records = []

    def record(
        self,
        step_result: SimStepResult,
        uavs: Sequence[UAV],
        users: Sequence[GroundUser],
    ) -> StepMetricsRecord:
        total_delivered_bits_step = float(sum(user.delivered_bits_step for user in users))
        total_arrived_bits_step = float(sum(user.arrived_bits_step for user in users))
        total_energy_j = float(sum(step_result.accounting.energy_used_j_by_uav.values()))
        total_user_access_backlog_bits = float(sum(user.user_access_backlog_bits for user in users))
        total_uav_relay_queue_bits = float(sum(uav.relay_queue_total_bits for uav in uavs))
        associated_user_count = sum(user.associated_uav_id >= 0 for user in users)
        outage_user_count = sum(user.final_rate_bps < self.outage_threshold_bps for user in users)

        record = StepMetricsRecord(
            current_step=step_result.env_state.current_step,
            sum_throughput_bps=total_delivered_bits_step / config.DELTA_T,
            coverage_ratio=associated_user_count / max(len(users), 1),
            outage_ratio=outage_user_count / max(len(users), 1),
            jain_fairness=jain_fairness_index([user.final_rate_bps for user in users]),
            total_energy_j=total_energy_j,
            energy_efficiency_bits_per_j=total_delivered_bits_step / max(total_energy_j, config.EPSILON),
            lambda2=float(step_result.env_state.lambda2),
            total_user_access_backlog_bits=total_user_access_backlog_bits,
            total_uav_relay_queue_bits=total_uav_relay_queue_bits,
            demand_satisfaction_ratio=total_delivered_bits_step / max(total_arrived_bits_step, config.EPSILON),
            total_delivered_bits_step=total_delivered_bits_step,
            total_arrived_bits_step=total_arrived_bits_step,
        )
        self.step_records.append(record)
        return record

    def export_history(self) -> list[dict[str, float | int]]:
        return [asdict(record) for record in self.step_records]

    def summarize(self) -> EpisodeMetricsSummary:
        if not self.step_records:
            return EpisodeMetricsSummary(
                num_steps=0,
                mean_sum_throughput_bps=0.0,
                mean_coverage_ratio=0.0,
                mean_outage_ratio=0.0,
                mean_jain_fairness=0.0,
                total_energy_j=0.0,
                energy_efficiency_bits_per_j=0.0,
                mean_lambda2=0.0,
                final_total_user_access_backlog_bits=0.0,
                final_total_uav_relay_queue_bits=0.0,
                demand_satisfaction_ratio=0.0,
                cumulative_delivered_bits=0.0,
                cumulative_arrived_bits=0.0,
            )

        num_steps = len(self.step_records)
        cumulative_delivered_bits = sum(record.total_delivered_bits_step for record in self.step_records)
        cumulative_arrived_bits = sum(record.total_arrived_bits_step for record in self.step_records)
        total_energy_j = sum(record.total_energy_j for record in self.step_records)

        return EpisodeMetricsSummary(
            num_steps=num_steps,
            mean_sum_throughput_bps=sum(record.sum_throughput_bps for record in self.step_records) / num_steps,
            mean_coverage_ratio=sum(record.coverage_ratio for record in self.step_records) / num_steps,
            mean_outage_ratio=sum(record.outage_ratio for record in self.step_records) / num_steps,
            mean_jain_fairness=sum(record.jain_fairness for record in self.step_records) / num_steps,
            total_energy_j=total_energy_j,
            energy_efficiency_bits_per_j=cumulative_delivered_bits / max(total_energy_j, config.EPSILON),
            mean_lambda2=sum(record.lambda2 for record in self.step_records) / num_steps,
            final_total_user_access_backlog_bits=self.step_records[-1].total_user_access_backlog_bits,
            final_total_uav_relay_queue_bits=self.step_records[-1].total_uav_relay_queue_bits,
            demand_satisfaction_ratio=cumulative_delivered_bits / max(cumulative_arrived_bits, config.EPSILON),
            cumulative_delivered_bits=cumulative_delivered_bits,
            cumulative_arrived_bits=cumulative_arrived_bits,
        )
