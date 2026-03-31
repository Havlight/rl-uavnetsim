from __future__ import annotations

import json
from pathlib import Path

from rl_uavnetsim.main import run_demo_episode


def test_demo_runner_creates_expected_artifacts(tmp_path: Path) -> None:
    artifacts = run_demo_episode(
        output_dir=tmp_path / "demo_run",
        num_steps=2,
        seed=3,
        num_uavs=2,
        num_users=3,
        deterministic_policy=True,
    )

    assert artifacts.summary_json_path.exists()
    assert artifacts.metrics_history_json_path.exists()
    assert artifacts.metric_png_paths
    assert all(path.exists() for path in artifacts.metric_png_paths.values())
    assert artifacts.trajectory_png_path.exists()
    assert artifacts.trajectory_gif_path.exists()

    summary = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    history = json.loads(artifacts.metrics_history_json_path.read_text(encoding="utf-8"))

    assert summary["num_steps"] >= 1
    assert len(history) >= 1
