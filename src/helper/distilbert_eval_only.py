"""
distilbert_eval_only.py
=======================
Safe wrapper for running DistilBERT experiment evaluation without retraining.

What it does:
  1. Loads the saved experiment configuration from `experiment_state.json`
  2. Verifies that the saved splits, Model A checkpoint, Model B checkpoint,
     and Model B cache already exist locally
  3. Runs `run_distilbert_experiment(...)` using the saved config so the main
     experiment logic reuses those artifacts instead of resetting state

Run:
    python3 src/distilbert_eval_only.py

Optional:
    python3 src/distilbert_eval_only.py --force-eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run saved DistilBERT evaluations without retraining."
    )
    parser.add_argument(
        "--output-dir",
        default="results/distilbert_experiment",
        help="Directory containing experiment_state.json and saved artifacts.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Clear saved evaluation markers so the four evaluation CSVs are recomputed.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    """Raise a clear validation error for missing experiment artifacts."""
    if not condition:
        raise RuntimeError(message)


def validate_saved_artifacts(output_dir: Path, state: dict):
    """Ensure eval-only mode cannot silently fall back to training."""
    from helper.distilbert_experiment import (
        DistilBERTExperimentConfig,
        cache_artifact_exists,
        checkpoint_artifact_exists,
    )

    saved_config = state.get("config")
    require(
        isinstance(saved_config, dict) and saved_config,
        f"Missing saved config in {output_dir / 'experiment_state.json'}.",
    )

    split_dir = output_dir / "splits"
    require(state.get("splits_complete"), "Saved train/test splits are not marked complete.")
    require((split_dir / "train_split.csv").exists(), "Missing saved train split CSV.")
    require((split_dir / "test_split.csv").exists(), "Missing saved test split CSV.")
    require((split_dir / "mutated_data_split.csv").exists(), "Missing saved mutated_data split CSV.")

    baseline_checkpoint = state.get("baseline_checkpoint_path")
    require(state.get("baseline_complete"), "Model A is not marked complete in experiment state.")
    require(
        checkpoint_artifact_exists(baseline_checkpoint),
        f"Missing Model A checkpoint: {baseline_checkpoint}",
    )

    model_b_checkpoint = state.get("adversarial_checkpoint_path")
    model_b_cache = state.get("adversarial_cache_path")
    require(state.get("adversarial_complete"), "Model B is not marked complete in experiment state.")
    require(
        checkpoint_artifact_exists(model_b_checkpoint),
        f"Missing Model B checkpoint: {model_b_checkpoint}",
    )
    require(
        cache_artifact_exists(model_b_cache),
        f"Missing Model B cache artifact: {model_b_cache}",
    )

    return DistilBERTExperimentConfig(**saved_config)


def clear_evaluation_markers(state_path: Path, state: dict) -> None:
    """Reset only evaluation outputs, keeping trained artifacts intact."""
    state["evaluations"] = {}
    state["summary_complete"] = False
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    state_path = output_dir / "experiment_state.json"
    try:
        from helper.distilbert_experiment import load_json, run_distilbert_experiment
    except ModuleNotFoundError as exc:
        missing_package = exc.name or "a required dependency"
        raise RuntimeError(
            "distilbert_eval_only.py needs the project dependencies installed "
            f"before it can run. Missing module: {missing_package}"
        ) from exc

    state = load_json(state_path, None)
    require(isinstance(state, dict), f"Could not load experiment state from {state_path}.")

    config = validate_saved_artifacts(output_dir, state)

    if args.force_eval:
        clear_evaluation_markers(state_path, state)

    results = run_distilbert_experiment(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
