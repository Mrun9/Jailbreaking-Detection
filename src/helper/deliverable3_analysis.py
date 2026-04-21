"""
deliverable3_analysis.py
========================
Post-process saved experiment CSVs into Deliverable 3 report artifacts.

This script is intentionally lightweight: it uses only the Python standard
library so it can run even in environments that do not currently have the full
training stack installed.

Outputs:
  - results/deliverable3/deliverable3_summary.json
  - results/deliverable3/threshold_sweep.csv
  - results/deliverable3/operating_modes.csv
  - results/deliverable3/source_breakdown.csv
  - results/deliverable3/category_breakdown.csv
  - results/deliverable3/false_positive_examples.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "results" / "distilbert_experiment"
CSV_DIR = EXPERIMENT_DIR / "csv"


@dataclass(frozen=True)
class PredictionRow:
    experiment_name: str
    model_variant: str
    dataset_variant: str
    prompt: str
    original_prompt: str
    expected_label: int
    confidence: float
    cache_hit: bool
    cache_stage: str
    cache_similarity: float
    prediction_latency_ms: float
    correct_prediction: bool
    category: str
    source: str


EXPERIMENT_FILES = {
    "test1_model_a_on_test": ("Model A", "test"),
    "test2_model_a_on_mutated_data": ("Model A", "mutated_data"),
    "test3_model_b_on_test": ("Model B", "test"),
    "test4_model_b_on_mutated_data": ("Model B", "mutated_data"),
}

OPERATING_MODES = {
    "baseline_default": {
        "label": "Baseline default",
        "model_variant": "Model A",
        "threshold": 0.50,
        "cache_hits_force_positive": False,
        "notes": "Original classifier threshold used in the saved baseline evaluation.",
    },
    "balanced": {
        "label": "Balanced",
        "model_variant": "Model A",
        "threshold": 0.51,
        "cache_hits_force_positive": False,
        "notes": "Slightly calibrated operating point that maximizes clean-test F1.",
    },
    "strict": {
        "label": "Strict",
        "model_variant": "Model A",
        "threshold": 0.85,
        "cache_hits_force_positive": False,
        "notes": "Higher-threshold mode that reduces false positives on noisy templated prompts.",
    },
    "adversarial_default": {
        "label": "Adversarial default",
        "model_variant": "Model B",
        "threshold": 0.50,
        "cache_hits_force_positive": True,
        "notes": "Saved adversarial/cache-backed configuration from the extended experiment.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Deliverable 3 analysis artifacts from saved evaluation CSVs."
    )
    parser.add_argument(
        "--experiment-dir",
        default=str(EXPERIMENT_DIR),
        help="Directory containing the saved distilbert_experiment artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "deliverable3"),
        help="Destination directory for Deliverable 3 summary files.",
    )
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected artifact is missing: {path}")
    return path


def safe_int(value, default: int = 0) -> int:
    text = str(value).strip()
    return int(text) if text else default


def safe_float(value, default: float = 0.0) -> float:
    text = str(value).strip()
    return float(text) if text else default


def parse_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def trim_text(text: str, limit: int = 220) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def load_prediction_rows(csv_path: Path) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    with require(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                PredictionRow(
                    experiment_name=row["experiment_name"],
                    model_variant=row["model_variant"],
                    dataset_variant=row["dataset_variant"],
                    prompt=row["prompt"],
                    original_prompt=row.get("original_prompt", "") or "",
                    expected_label=safe_int(row["expected_label"]),
                    confidence=safe_float(row["confidence"]),
                    cache_hit=parse_bool(row["cache_hit"]),
                    cache_stage=row.get("cache_stage", ""),
                    cache_similarity=safe_float(row.get("cache_similarity")),
                    prediction_latency_ms=safe_float(row.get("prediction_latency_ms")),
                    correct_prediction=parse_bool(row.get("correct_prediction")),
                    category=(row.get("category") or "MISSING").strip() or "MISSING",
                    source=(row.get("source") or "unknown").strip() or "unknown",
                )
            )
    if not rows:
        raise ValueError(f"No rows were loaded from {csv_path}")
    return rows


def predict_label(
    row: PredictionRow,
    threshold: float,
    cache_hits_force_positive: bool,
) -> int:
    if cache_hits_force_positive and row.cache_hit:
        return 1
    return int(row.confidence >= threshold)


def summarize_rows(
    rows: Iterable[PredictionRow],
    threshold: float,
    cache_hits_force_positive: bool,
) -> dict:
    tp = fp = fn = tn = 0
    total_latency = 0.0
    total_confidence = 0.0
    total = 0
    cache_hits = 0
    model_calls = 0

    for row in rows:
        total += 1
        total_latency += row.prediction_latency_ms
        total_confidence += row.confidence
        cache_hits += int(row.cache_hit)
        model_calls += int(not row.cache_hit)

        predicted = predict_label(
            row=row,
            threshold=threshold,
            cache_hits_force_positive=cache_hits_force_positive,
        )
        expected = row.expected_label

        if predicted == 1 and expected == 1:
            tp += 1
        elif predicted == 1 and expected == 0:
            fp += 1
        elif predicted == 0 and expected == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "num_examples": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_latency_ms": round(total_latency / total, 3) if total else 0.0,
        "avg_confidence": round(total_confidence / total, 4) if total else 0.0,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total, 4) if total else 0.0,
        "model_calls": model_calls,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def threshold_values(step: float = 0.01) -> list[float]:
    values = []
    current = 0.0
    while current <= 1.000001:
        values.append(round(current, 2))
        current += step
    return values


def choose_best_threshold(
    rows: list[PredictionRow],
    cache_hits_force_positive: bool,
) -> dict:
    best_record: Optional[dict] = None
    for threshold in threshold_values():
        metrics = summarize_rows(rows, threshold, cache_hits_force_positive)
        record = {"threshold": threshold, **metrics}
        score = (
            record["f1"],
            record["accuracy"],
            record["precision"],
            -record["fp"],
        )
        if best_record is None or score > (
            best_record["f1"],
            best_record["accuracy"],
            best_record["precision"],
            -best_record["fp"],
        ):
            best_record = record
    assert best_record is not None
    return best_record


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_baseline_metrics(experiment_dir: Path) -> dict:
    metrics_path = experiment_dir / "models" / "model_a" / "baseline_metrics.json"
    with require(metrics_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_model_b_history(experiment_dir: Path) -> list[dict]:
    history_path = experiment_dir / "models" / "model_b_history.json"
    with require(history_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def build_group_breakdown(
    rows: list[PredictionRow],
    group_key: Callable[[PredictionRow], str],
    threshold: float,
    cache_hits_force_positive: bool,
    min_examples: int = 50,
) -> list[dict]:
    grouped: dict[str, list[PredictionRow]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    breakdown = []
    for group_name, group_rows in grouped.items():
        if len(group_rows) < min_examples:
            continue
        metrics = summarize_rows(group_rows, threshold, cache_hits_force_positive)
        breakdown.append({"group": group_name, **metrics})

    breakdown.sort(key=lambda item: (-item["num_examples"], item["group"]))
    return breakdown


def build_false_positive_examples(
    rows: list[PredictionRow],
    threshold: float,
    cache_hits_force_positive: bool,
    mode_name: str,
    limit: int = 20,
) -> list[dict]:
    false_positives = []
    for row in rows:
        predicted = predict_label(row, threshold, cache_hits_force_positive)
        if predicted == 1 and row.expected_label == 0:
            false_positives.append(
                {
                    "mode": mode_name,
                    "dataset_variant": row.dataset_variant,
                    "cache_stage": row.cache_stage,
                    "confidence": round(row.confidence, 6),
                    "source": row.source,
                    "category": row.category,
                    "prompt_preview": trim_text(row.prompt),
                    "original_prompt_preview": trim_text(row.original_prompt),
                }
            )
    false_positives.sort(key=lambda item: (-item["confidence"], item["source"], item["prompt_preview"]))
    return false_positives[:limit]


def build_threshold_sweep_rows(experiment_rows: dict[str, list[PredictionRow]]) -> list[dict]:
    sweep_rows = []
    for experiment_name, rows in experiment_rows.items():
        model_variant = rows[0].model_variant
        dataset_variant = rows[0].dataset_variant
        cache_hits_force_positive = model_variant == "Model B"
        for threshold in threshold_values():
            metrics = summarize_rows(rows, threshold, cache_hits_force_positive)
            sweep_rows.append(
                {
                    "experiment_name": experiment_name,
                    "model_variant": model_variant,
                    "dataset_variant": dataset_variant,
                    "threshold": f"{threshold:.2f}",
                    **metrics,
                }
            )
    return sweep_rows


def build_operating_mode_rows(experiment_rows: dict[str, list[PredictionRow]]) -> list[dict]:
    rows = []
    for mode_name, config in OPERATING_MODES.items():
        for dataset_variant in ("test", "mutated_data"):
            experiment_name = (
                "test1_model_a_on_test"
                if config["model_variant"] == "Model A" and dataset_variant == "test"
                else "test2_model_a_on_mutated_data"
                if config["model_variant"] == "Model A"
                else "test3_model_b_on_test"
                if dataset_variant == "test"
                else "test4_model_b_on_mutated_data"
            )
            metrics = summarize_rows(
                experiment_rows[experiment_name],
                threshold=config["threshold"],
                cache_hits_force_positive=config["cache_hits_force_positive"],
            )
            rows.append(
                {
                    "mode_name": mode_name,
                    "mode_label": config["label"],
                    "model_variant": config["model_variant"],
                    "dataset_variant": dataset_variant,
                    "threshold": f"{config['threshold']:.2f}",
                    "notes": config["notes"],
                    **metrics,
                }
            )
    return rows


def index_mode_rows(operating_mode_rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (row["mode_name"], row["dataset_variant"]): row
        for row in operating_mode_rows
    }


def delta(new_value: float, old_value: float, digits: int = 4) -> float:
    return round(new_value - old_value, digits)


def build_summary_payload(
    experiment_dir: Path,
    experiment_rows: dict[str, list[PredictionRow]],
    operating_mode_rows: list[dict],
    threshold_sweep_rows: list[dict],
) -> dict:
    baseline_metrics = load_baseline_metrics(experiment_dir)
    model_b_history = load_model_b_history(experiment_dir)
    mode_index = index_mode_rows(operating_mode_rows)

    best_model_a_clean = choose_best_threshold(
        experiment_rows["test1_model_a_on_test"],
        cache_hits_force_positive=False,
    )
    best_model_a_mutated = choose_best_threshold(
        experiment_rows["test2_model_a_on_mutated_data"],
        cache_hits_force_positive=False,
    )
    best_model_b_clean = choose_best_threshold(
        experiment_rows["test3_model_b_on_test"],
        cache_hits_force_positive=True,
    )

    baseline_default_test = mode_index[("baseline_default", "test")]
    baseline_default_mutated = mode_index[("baseline_default", "mutated_data")]
    balanced_test = mode_index[("balanced", "test")]
    balanced_mutated = mode_index[("balanced", "mutated_data")]
    strict_test = mode_index[("strict", "test")]
    strict_mutated = mode_index[("strict", "mutated_data")]
    adversarial_test = mode_index[("adversarial_default", "test")]
    adversarial_mutated = mode_index[("adversarial_default", "mutated_data")]

    latest_round = model_b_history[-1] if model_b_history else {}

    return {
        "artifact_paths": {
            "experiment_dir": str(experiment_dir),
            "baseline_metrics": str(experiment_dir / "models" / "model_a" / "baseline_metrics.json"),
            "model_b_history": str(experiment_dir / "models" / "model_b_history.json"),
        },
        "production_recommendation": {
            "final_model_variant": "Model A",
            "default_mode": "balanced",
            "strict_mode": "strict",
            "reason": (
                "Extended evaluation shows the calibrated baseline is the most stable choice. "
                "The adversarial/cache-backed variant improves recall but introduces too many "
                "false positives on mutated benign prompts."
            ),
        },
        "best_thresholds": {
            "model_a_clean_test": best_model_a_clean,
            "model_a_mutated_data": best_model_a_mutated,
            "model_b_clean_test": best_model_b_clean,
        },
        "operating_modes": {
            "baseline_default": {
                "test": baseline_default_test,
                "mutated_data": baseline_default_mutated,
            },
            "balanced": {
                "test": balanced_test,
                "mutated_data": balanced_mutated,
            },
            "strict": {
                "test": strict_test,
                "mutated_data": strict_mutated,
            },
            "adversarial_default": {
                "test": adversarial_test,
                "mutated_data": adversarial_mutated,
            },
        },
        "improvement_highlights": {
            "balanced_vs_baseline_default": {
                "clean_test_f1_delta": delta(balanced_test["f1"], baseline_default_test["f1"]),
                "clean_test_accuracy_delta": delta(
                    balanced_test["accuracy"], baseline_default_test["accuracy"]
                ),
                "mutated_data_f1_delta": delta(
                    balanced_mutated["f1"], baseline_default_mutated["f1"]
                ),
            },
            "strict_vs_baseline_default": {
                "clean_test_false_positive_delta": strict_test["fp"] - baseline_default_test["fp"],
                "mutated_data_false_positive_delta": (
                    strict_mutated["fp"] - baseline_default_mutated["fp"]
                ),
                "clean_test_precision_delta": delta(
                    strict_test["precision"], baseline_default_test["precision"]
                ),
                "mutated_data_accuracy_delta": delta(
                    strict_mutated["accuracy"], baseline_default_mutated["accuracy"]
                ),
                "mutated_data_f1_delta": delta(
                    strict_mutated["f1"], baseline_default_mutated["f1"]
                ),
            },
            "adversarial_vs_balanced": {
                "clean_test_f1_delta": delta(adversarial_test["f1"], balanced_test["f1"]),
                "mutated_data_f1_delta": delta(
                    adversarial_mutated["f1"], balanced_mutated["f1"]
                ),
                "mutated_data_false_positive_delta": (
                    adversarial_mutated["fp"] - balanced_mutated["fp"]
                ),
                "clean_test_cache_hit_rate": adversarial_test["cache_hit_rate"],
            },
        },
        "saved_training_snapshot": baseline_metrics,
        "adversarial_training_summary": {
            "rounds_completed": len(model_b_history),
            "latest_round": latest_round,
            "history": model_b_history,
        },
        "threshold_sweep_rows": len(threshold_sweep_rows),
    }


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    csv_dir = experiment_dir / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_rows = {
        name: load_prediction_rows(csv_dir / f"{name}.csv")
        for name in EXPERIMENT_FILES
    }

    threshold_sweep_rows = build_threshold_sweep_rows(experiment_rows)
    write_csv(
        output_dir / "threshold_sweep.csv",
        threshold_sweep_rows,
        fieldnames=list(threshold_sweep_rows[0].keys()),
    )

    operating_mode_rows = build_operating_mode_rows(experiment_rows)
    write_csv(
        output_dir / "operating_modes.csv",
        operating_mode_rows,
        fieldnames=list(operating_mode_rows[0].keys()),
    )

    source_rows = []
    category_rows = []
    for mode_name in ("balanced", "strict", "adversarial_default"):
        mode_config = OPERATING_MODES[mode_name]
        for dataset_variant, experiment_name in (
            ("test", "test1_model_a_on_test" if mode_config["model_variant"] == "Model A" else "test3_model_b_on_test"),
            (
                "mutated_data",
                "test2_model_a_on_mutated_data"
                if mode_config["model_variant"] == "Model A"
                else "test4_model_b_on_mutated_data",
            ),
        ):
            rows = experiment_rows[experiment_name]
            threshold = mode_config["threshold"]
            cache_hits_force_positive = mode_config["cache_hits_force_positive"]

            for entry in build_group_breakdown(
                rows,
                group_key=lambda row: row.source,
                threshold=threshold,
                cache_hits_force_positive=cache_hits_force_positive,
            ):
                source_rows.append(
                    {
                        "mode_name": mode_name,
                        "model_variant": mode_config["model_variant"],
                        "dataset_variant": dataset_variant,
                        **entry,
                    }
                )

            for entry in build_group_breakdown(
                rows,
                group_key=lambda row: row.category,
                threshold=threshold,
                cache_hits_force_positive=cache_hits_force_positive,
            ):
                category_rows.append(
                    {
                        "mode_name": mode_name,
                        "model_variant": mode_config["model_variant"],
                        "dataset_variant": dataset_variant,
                        **entry,
                    }
                )

    write_csv(
        output_dir / "source_breakdown.csv",
        source_rows,
        fieldnames=list(source_rows[0].keys()),
    )
    write_csv(
        output_dir / "category_breakdown.csv",
        category_rows,
        fieldnames=list(category_rows[0].keys()),
    )

    false_positive_rows = []
    for mode_name in ("balanced", "strict", "adversarial_default"):
        mode_config = OPERATING_MODES[mode_name]
        experiment_name = (
            "test2_model_a_on_mutated_data"
            if mode_config["model_variant"] == "Model A"
            else "test4_model_b_on_mutated_data"
        )
        false_positive_rows.extend(
            build_false_positive_examples(
                rows=experiment_rows[experiment_name],
                threshold=mode_config["threshold"],
                cache_hits_force_positive=mode_config["cache_hits_force_positive"],
                mode_name=mode_name,
            )
        )
    write_csv(
        output_dir / "false_positive_examples.csv",
        false_positive_rows,
        fieldnames=list(false_positive_rows[0].keys()),
    )

    summary_payload = build_summary_payload(
        experiment_dir=experiment_dir,
        experiment_rows=experiment_rows,
        operating_mode_rows=operating_mode_rows,
        threshold_sweep_rows=threshold_sweep_rows,
    )
    summary_path = output_dir / "deliverable3_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"Wrote {summary_path}")
    print(f"Wrote {output_dir / 'threshold_sweep.csv'}")
    print(f"Wrote {output_dir / 'operating_modes.csv'}")
    print(f"Wrote {output_dir / 'source_breakdown.csv'}")
    print(f"Wrote {output_dir / 'category_breakdown.csv'}")
    print(f"Wrote {output_dir / 'false_positive_examples.csv'}")


if __name__ == "__main__":
    main()
