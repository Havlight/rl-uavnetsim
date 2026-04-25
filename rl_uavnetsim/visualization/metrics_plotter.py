from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_uavnetsim.metrics import StepMetricsRecord


class MetricsPlotter:
    def plot_metric_set(
        self,
        step_records: Sequence[StepMetricsRecord],
        output_dir: str | Path,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        steps = [record.current_step for record in step_records]

        outputs: dict[str, Path] = {}
        outputs["throughput"] = self._plot_single_metric(
            steps,
            [record.sum_throughput_bps for record in step_records],
            output_dir / "throughput.png",
            title="Sum Throughput",
            ylabel="bps",
        )
        outputs["outage"] = self._plot_single_metric(
            steps,
            [record.outage_ratio for record in step_records],
            output_dir / "outage_ratio.png",
            title="Outage Ratio",
            ylabel="ratio",
        )
        outputs["backlog_queue"] = self._plot_two_metrics(
            steps,
            [record.total_user_access_backlog_bits for record in step_records],
            [record.total_uav_relay_queue_bits for record in step_records],
            output_dir / "backlog_queue.png",
            title="Access Backlog / Relay Queue",
            labels=("Access backlog (bits)", "Relay queue (bits)"),
        )
        outputs["fairness"] = self._plot_single_metric(
            steps,
            [record.jain_fairness for record in step_records],
            output_dir / "jain_fairness.png",
            title="Jain Fairness",
            ylabel="index",
        )
        outputs["coverage"] = self._plot_single_metric(
            steps,
            [record.coverage_ratio for record in step_records],
            output_dir / "coverage_ratio.png",
            title="Association Coverage Ratio",
            ylabel="ratio",
        )
        outputs["effective_coverage"] = self._plot_single_metric(
            steps,
            [record.effective_coverage_ratio for record in step_records],
            output_dir / "effective_coverage_ratio.png",
            title="End-to-End Effective Coverage Ratio",
            ylabel="ratio",
        )
        outputs["lambda2"] = self._plot_single_metric(
            steps,
            [record.lambda2 for record in step_records],
            output_dir / "lambda2.png",
            title="Algebraic Connectivity",
            ylabel="lambda2",
        )
        outputs["energy"] = self._plot_single_metric(
            steps,
            [record.total_energy_j for record in step_records],
            output_dir / "energy_j.png",
            title="Total Energy",
            ylabel="J",
        )
        outputs["demand_satisfaction"] = self._plot_single_metric(
            steps,
            [record.demand_satisfaction_ratio for record in step_records],
            output_dir / "demand_satisfaction_ratio.png",
            title="Demand Satisfaction Ratio",
            ylabel="ratio",
        )
        outputs["energy_efficiency"] = self._plot_single_metric(
            steps,
            [record.energy_efficiency_bits_per_j for record in step_records],
            output_dir / "energy_efficiency.png",
            title="Energy Efficiency",
            ylabel="bits/J",
        )
        return outputs

    def plot_episode_metric_set(
        self,
        episode_histories: Sequence[Sequence[dict[str, float | int]]],
        output_dir: str | Path,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metric_specs = {
            "throughput": ("sum_throughput_bps", "Sum Throughput", "Mbps", 1.0e-6, "tab:blue"),
            "outage": ("outage_ratio", "Outage Ratio", "ratio", 1.0, "tab:red"),
            "fairness": ("jain_fairness", "Jain Fairness", "index", 1.0, "tab:green"),
            "coverage": ("coverage_ratio", "Association Coverage Ratio", "ratio", 1.0, "tab:purple"),
            "effective_coverage": (
                "effective_coverage_ratio",
                "End-to-End Effective Coverage Ratio",
                "ratio",
                1.0,
                "tab:cyan",
            ),
            "lambda2": ("lambda2", "Algebraic Connectivity", "lambda2", 1.0, "tab:olive"),
            "demand_satisfaction": (
                "demand_satisfaction_ratio",
                "Demand Satisfaction Ratio",
                "ratio",
                1.0,
                "tab:brown",
            ),
            "energy_efficiency": (
                "energy_efficiency_bits_per_j",
                "Energy Efficiency",
                "bits/J",
                1.0,
                "tab:cyan",
            ),
        }
        outputs: dict[str, Path] = {}
        for output_name, (metric_key, title, ylabel, scale, color) in metric_specs.items():
            outputs[output_name] = self._plot_episode_metric(
                episode_histories,
                metric_key,
                output_dir / f"{output_name}.png",
                title=title,
                ylabel=ylabel,
                scale=scale,
                color=color,
            )

        outputs["backlog_queue"] = self._plot_episode_two_metrics(
            episode_histories,
            "total_user_access_backlog_bits",
            "total_uav_relay_queue_bits",
            output_dir / "backlog_queue.png",
            title="Access Backlog / Relay Queue",
            ylabel="Gbits",
            scale=1.0e-9,
            labels=("Access backlog", "Relay queue"),
            colors=("tab:orange", "tab:gray"),
        )
        outputs["energy"] = self._plot_episode_metric(
            episode_histories,
            "total_energy_j",
            output_dir / "energy_j.png",
            title="Total Energy",
            ylabel="J",
            scale=1.0,
            color="tab:pink",
        )
        return outputs

    def plot_step_metrics(
        self,
        step_records: Sequence[StepMetricsRecord],
        output_png_path: str | Path,
    ) -> Path:
        output_png_path = Path(output_png_path)
        output_png_path.parent.mkdir(parents=True, exist_ok=True)

        steps = [record.current_step for record in step_records]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        axes[0, 0].plot(steps, [record.sum_throughput_bps for record in step_records], label="Throughput (bps)")
        axes[0, 0].plot(steps, [record.outage_ratio for record in step_records], label="Outage")
        axes[0, 0].set_title("Throughput / Outage")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].legend()

        axes[0, 1].plot(steps, [record.total_user_access_backlog_bits for record in step_records], label="Access backlog")
        axes[0, 1].plot(steps, [record.total_uav_relay_queue_bits for record in step_records], label="Relay queue")
        axes[0, 1].set_title("Backlog / Queue")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].legend()

        axes[1, 0].plot(steps, [record.jain_fairness for record in step_records], label="Jain fairness")
        axes[1, 0].plot(steps, [record.coverage_ratio for record in step_records], label="Association coverage")
        axes[1, 0].plot(
            steps,
            [record.effective_coverage_ratio for record in step_records],
            label="Effective coverage",
        )
        axes[1, 0].plot(steps, [record.lambda2 for record in step_records], label="Lambda2")
        axes[1, 0].set_title("Fairness / Coverage / Connectivity")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].legend()

        axes[1, 1].plot(steps, [record.total_energy_j for record in step_records], label="Energy (J)")
        axes[1, 1].plot(
            steps,
            [record.demand_satisfaction_ratio for record in step_records],
            label="Demand satisfaction",
        )
        axes[1, 1].set_title("Energy / Demand Satisfaction")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].legend()

        for axis in axes.flat:
            axis.grid(True, alpha=0.3)

        fig.savefig(output_png_path, dpi=160)
        plt.close(fig)
        return output_png_path

    def _plot_single_metric(
        self,
        steps: Sequence[int],
        values: Sequence[float],
        output_png_path: Path,
        *,
        title: str,
        ylabel: str,
    ) -> Path:
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.plot(steps, values, linewidth=2.0)
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        fig.savefig(output_png_path, dpi=160)
        plt.close(fig)
        return output_png_path

    def _plot_episode_metric(
        self,
        episode_histories: Sequence[Sequence[dict[str, float | int]]],
        metric_key: str,
        output_png_path: Path,
        *,
        title: str,
        ylabel: str,
        scale: float,
        color: str,
    ) -> Path:
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        max_len = max((len(history) for history in episode_histories), default=0)
        value_matrix = np.full((len(episode_histories), max_len), np.nan, dtype=float)
        for episode_index, history in enumerate(episode_histories):
            steps = [int(record["current_step"]) for record in history]
            values = [float(record[metric_key]) * scale for record in history]
            axis.plot(steps, values, color=color, alpha=0.22, linewidth=1.0)
            value_matrix[episode_index, : len(values)] = values
        if max_len > 0:
            mean_values = np.nanmean(value_matrix, axis=0)
            axis.plot(range(1, max_len + 1), mean_values, color=color, linewidth=2.6, label="episode mean")
            axis.legend()
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        fig.savefig(output_png_path, dpi=160)
        plt.close(fig)
        return output_png_path

    def _plot_episode_two_metrics(
        self,
        episode_histories: Sequence[Sequence[dict[str, float | int]]],
        metric_key_a: str,
        metric_key_b: str,
        output_png_path: Path,
        *,
        title: str,
        ylabel: str,
        scale: float,
        labels: tuple[str, str],
        colors: tuple[str, str],
    ) -> Path:
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for metric_key, label, color in (
            (metric_key_a, labels[0], colors[0]),
            (metric_key_b, labels[1], colors[1]),
        ):
            max_len = max((len(history) for history in episode_histories), default=0)
            value_matrix = np.full((len(episode_histories), max_len), np.nan, dtype=float)
            for episode_index, history in enumerate(episode_histories):
                steps = [int(record["current_step"]) for record in history]
                values = [float(record[metric_key]) * scale for record in history]
                axis.plot(steps, values, color=color, alpha=0.20, linewidth=1.0)
                value_matrix[episode_index, : len(values)] = values
            if max_len > 0:
                mean_values = np.nanmean(value_matrix, axis=0)
                axis.plot(range(1, max_len + 1), mean_values, color=color, linewidth=2.6, label=f"{label} mean")
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.savefig(output_png_path, dpi=160)
        plt.close(fig)
        return output_png_path

    def _plot_two_metrics(
        self,
        steps: Sequence[int],
        values_a: Sequence[float],
        values_b: Sequence[float],
        output_png_path: Path,
        *,
        title: str,
        labels: tuple[str, str],
    ) -> Path:
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.plot(steps, values_a, linewidth=2.0, label=labels[0])
        axis.plot(steps, values_b, linewidth=2.0, label=labels[1])
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel("bits")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.savefig(output_png_path, dpi=160)
        plt.close(fig)
        return output_png_path
