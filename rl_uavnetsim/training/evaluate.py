from __future__ import annotations

import argparse

from rl_uavnetsim.training.configuration import load_run_config, merge_eval_config
from rl_uavnetsim.training.mappo_trainer import evaluate_policy, load_policy_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained MAPPO checkpoint on rl_uavnetsim.")
    parser.add_argument("--config", required=True, help="Path to the YAML evaluation config.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint to evaluate.")
    args = parser.parse_args()

    policy, checkpoint_run_config = load_policy_from_checkpoint(args.checkpoint)
    eval_overrides = load_run_config(args.config)
    run_config = merge_eval_config(checkpoint_run_config, eval_overrides)
    artifacts = evaluate_policy(policy, run_config, output_dir=run_config.output.root_dir)
    print(f"Evaluation output directory: {artifacts.output_dir}")
    print(f"Evaluation summary: {artifacts.summary_json_path}")


if __name__ == "__main__":
    main()
