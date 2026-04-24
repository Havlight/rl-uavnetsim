from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_value(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_schema_v2(summary: dict[str, Any], label: str) -> None:
    trained = summary.get("policies", {}).get("trained") or {}
    static = summary.get("policies", {}).get("static") or {}
    comparison = summary.get("comparison") or {}
    print(f"{label}: schema_version=2")
    print(f"  trained mean reward: {_format_value(trained.get('mean_team_reward'))}")
    print(f"  static mean reward: {_format_value(static.get('mean_team_reward') if static else None)}")
    print(f"  trained - static: {_format_value(comparison.get('trained_minus_static_reward'))}")
    print(f"  static not beaten: {_format_value(comparison.get('static_baseline_not_beaten'))}")
    movement = trained.get("movement") or {}
    print(f"  trained path length mean: {_format_value((movement.get('mean_path_length_m') or {}).get('mean'))} m")
    print(f"  trained net displacement mean: {_format_value((movement.get('mean_net_displacement_m') or {}).get('mean'))} m")
    print(f"  trained rho mean: {_format_value((movement.get('rho') or {}).get('mean'))}")


def _print_legacy_summary(summary: dict[str, Any], label: str) -> None:
    print(f"{label}: schema_version=1/legacy")
    print(f"  mean reward: {_format_value(summary.get('mean_team_reward'))}")
    print(f"  throughput: {_format_value(summary.get('mean_sum_throughput_bps'))} bps")
    print(f"  coverage: {_format_value(summary.get('mean_coverage_ratio'))}")
    print(f"  outage: {_format_value(summary.get('mean_outage_ratio'))}")


def analyze_run(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    print(f"Run: {run_dir}")
    for checkpoint_name in ("latest.pt", "best.pt"):
        checkpoint_path = run_dir / "checkpoints" / checkpoint_name
        print(f"Checkpoint {checkpoint_name}: {'present' if checkpoint_path.exists() else 'missing'}")

    for label, summary_path in (
        ("eval/latest", run_dir / "eval" / "latest" / "summary.json"),
        ("eval/best", run_dir / "eval" / "best" / "summary.json"),
        ("eval", run_dir / "eval" / "summary.json"),
    ):
        summary = _load_json(summary_path)
        if summary is None:
            continue
        if int(summary.get("schema_version", 1)) == 2:
            _print_schema_v2(summary, label)
        else:
            _print_legacy_summary(summary, label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MAPPO training/evaluation artifacts.")
    parser.add_argument("--run-dir", required=True, help="Path to a training run directory.")
    args = parser.parse_args()
    analyze_run(args.run_dir)


if __name__ == "__main__":
    main()
