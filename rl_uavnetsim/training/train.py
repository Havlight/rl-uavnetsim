from __future__ import annotations

import argparse

from rl_uavnetsim.training.configuration import load_run_config
from rl_uavnetsim.training.mappo_trainer import train_mappo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact-observation MAPPO policy for satellite rl_uavnetsim.")
    parser.add_argument("--config", required=True, help="Path to the YAML training config.")
    args = parser.parse_args()

    run_config = load_run_config(args.config)
    artifacts = train_mappo(run_config)
    print(f"Run directory: {artifacts.run_dir}")
    print(f"Latest checkpoint: {artifacts.latest_checkpoint_path}")
    print(f"Best checkpoint: {artifacts.best_checkpoint_path}")
    print(f"Evaluation summary: {artifacts.eval_artifacts.summary_json_path}")


if __name__ == "__main__":
    main()
