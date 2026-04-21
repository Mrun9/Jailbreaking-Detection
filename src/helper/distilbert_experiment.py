"""
distilbert_experiment.py
========================
End-to-end experiment runner for the DistilBERT-only project scope.

What it does:
  1. Loads `results/collected_prompts.csv`
  2. Creates a reproducible 80/20 train/test split
  3. Builds `mutated_data` from the held-out test split
  4. Trains:
       - Model A: baseline DistilBERT on train only
       - Model B: adversarially trained DistilBERT on train only
  5. Evaluates four experiments and writes one CSV per experiment:
       - Test 1: Model A on Test
       - Test 2: Model A on Mutated_data
       - Test 3: Model B on Test
       - Test 4: Model B on Mutated_data
  6. Writes summary artifacts so you can answer:
       - Did adversarial training help?
       - Did the cache reduce latency?
       - Did the mutator create challenging prompts?

Run:
    python3 src/distilbert_experiment.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from detector import TwoStageDetector
from mutator import JailbreakMutator
from train_loop import AdversarialTrainer, TrainingConfig


@dataclass
class PromptRecord:
    prompt: str
    label: int
    category: str = "MISSING"
    source: str = "unknown"
    prompt_len: int = 0
    original_prompt: Optional[str] = None
    generated_by_mutator: bool = False
    prompt_changed: bool = False
    variant_index: int = 0


@dataclass
class DistilBERTExperimentConfig:
    dataset_csv: str = "results/collected_prompts.csv"
    output_dir: str = "results/distilbert_experiment"
    base_model_name: str = "distilbert/distilbert-base-uncased"

    train_fraction: float = 0.80
    test_fraction: float = 0.20
    seed: int = 42

    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 256
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    val_split: float = 0.15

    num_rounds: int = 4
    variants_per_seed: int = 5
    mutated_variants_per_prompt: int = 3
    max_mutation_prompt_chars: int = 1000
    min_fool_confidence: float = 0.4
    detector_model_threshold: float = 0.5
    cache_threshold: float = 0.72
    training_mutator_strategies: list = field(
        default_factory=lambda: [
            "wordnet",
            "bert",
            "t5",
            "backtranslate",
            "structural",
        ]
    )
    mutated_data_mutator_strategies: list = field(
        default_factory=lambda: [
            "wordnet",
            "bert",
            "t5",
            "backtranslate",
            "roleplay",
            "structural",
        ]
    )
    mutator_combine: bool = True
    mutate_only_jailbreak: bool = False

    verbose: bool = True
    show_progress_bars: bool = True
    dry_run: bool = False


class NullCache:
    """Cheap no-op cache for Model A when you want cache-hit columns but no cache."""

    def query(self, prompt: str) -> dict:
        return {"hit": False, "similarity": 0.0, "matched_prompt": None}

    def save(self, path: str) -> None:
        return None

    def __len__(self) -> int:
        return 0


def experiment_log(message: str) -> None:
    """Compact logger for experiment orchestration messages."""
    print(f"[experiment] {message}")

def normalize_label(value) -> int:
    """Normalize mixed label formats into binary integers."""
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        if value in (0, 1):
            return value
        raise ValueError(f"Invalid integer label: {value}")

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "jailbreak", "jb"}:
        return 1
    if text in {"0", "false", "no", "n", "benign", "safe"}:
        return 0
    raise ValueError(f"Unsupported label value: {value}")


def parse_bool(value) -> bool:
    """Parse common CSV/JSON truthy strings."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def load_json(path: Path, default):
    """Load JSON from disk, falling back to a default value."""
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    """Write JSON to disk with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_prompt_records(csv_path: str) -> list[PromptRecord]:
    """Load dataset rows from CSV into prompt records."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset CSV at '{csv_path}'.")

    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"prompt", "jailbreak"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset CSV missing required columns: {sorted(missing)}")

        for row in reader:
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                continue

            label = normalize_label(row.get("jailbreak"))
            prompt_len = row.get("prompt_len")
            prompt_len_value = int(prompt_len) if str(prompt_len).strip() else len(prompt)
            records.append(
                PromptRecord(
                    prompt=prompt,
                    label=label,
                    category=(row.get("category") or "MISSING").strip() or "MISSING",
                    source=(row.get("source") or "unknown").strip() or "unknown",
                    prompt_len=prompt_len_value,
                    original_prompt=prompt,
                )
            )

    if not records:
        raise ValueError("No usable prompts were loaded from the dataset CSV.")
    return records


def load_records_csv(path: Path) -> list[PromptRecord]:
    """Reload previously saved split artifacts."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find split CSV at '{path}'.")

    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                PromptRecord(
                    prompt=row["prompt"],
                    label=normalize_label(row["jailbreak"]),
                    category=(row.get("category") or "MISSING").strip() or "MISSING",
                    source=(row.get("source") or "unknown").strip() or "unknown",
                    prompt_len=int(row.get("prompt_len") or len(row["prompt"])),
                    original_prompt=row.get("original_prompt") or None,
                    generated_by_mutator=parse_bool(row.get("generated_by_mutator", False)),
                    prompt_changed=parse_bool(row.get("prompt_changed", False)),
                    variant_index=int(row.get("variant_index") or 0),
                )
            )
    return records


def split_records(
    records: list[PromptRecord],
    test_fraction: float,
    seed: int,
) -> tuple[list[PromptRecord], list[PromptRecord]]:
    """Create a reproducible stratified train/test split."""
    labels = [record.label for record in records]
    train_records, test_records = train_test_split(
        records,
        test_size=test_fraction,
        stratify=labels,
        random_state=seed,
    )
    return list(train_records), list(test_records)


def clone_record(record: PromptRecord, **updates) -> PromptRecord:
    """Copy a prompt record while updating selected fields."""
    data = asdict(record)
    data.update(updates)
    return PromptRecord(**data)


def build_mutated_data_records(
    test_records: list[PromptRecord],
    mutator: JailbreakMutator,
    variants_per_prompt: int,
    max_prompt_chars: int,
    mutate_only_jailbreak: bool = True,
) -> list[PromptRecord]:
    """
    Build the held-out `mutated_data` split.

    By default, every held-out prompt is mutated. If you enable
    `mutate_only_jailbreak`, benign rows are copied through unchanged.
    """
    mutated_records = []
    skipped_long_prompts = 0
    iterator = tqdm(
        test_records,
        desc="Building mutated_data",
        unit="prompt",
        leave=False,
    )
    for record in iterator:
        if len(record.prompt) > max_prompt_chars:
            skipped_long_prompts += 1
            iterator.set_postfix(kept=len(mutated_records), skipped=skipped_long_prompts)
            continue

        should_mutate = (not mutate_only_jailbreak) or record.label == 1
        if not should_mutate:
            mutated_records.append(
                clone_record(
                    record,
                    generated_by_mutator=False,
                    prompt_changed=False,
                    variant_index=0,
                )
            )
            iterator.set_postfix(kept=len(mutated_records), skipped=skipped_long_prompts)
            continue

        variants = mutator.mutate(record.prompt, n=variants_per_prompt)
        if not variants:
            variants = [record.prompt]

        for variant_index, variant_text in enumerate(variants[:variants_per_prompt], start=1):
            mutated_records.append(
                clone_record(
                    record,
                    prompt=variant_text,
                    original_prompt=record.prompt,
                    prompt_len=len(variant_text),
                    generated_by_mutator=True,
                    prompt_changed=(variant_text != record.prompt),
                    variant_index=variant_index,
                )
            )
        iterator.set_postfix(kept=len(mutated_records), skipped=skipped_long_prompts)

    if skipped_long_prompts:
        experiment_log(
            f"Skipped {skipped_long_prompts} held-out prompts longer than {max_prompt_chars} chars "
            "while building mutated_data."
        )

    return mutated_records
def records_to_prompt_lists(records: list[PromptRecord]) -> tuple[list[str], list[int]]:
    return [record.prompt for record in records], [record.label for record in records]


def split_by_label(records: list[PromptRecord]) -> tuple[list[str], list[str]]:
    jailbreak_prompts = [record.prompt for record in records if record.label == 1]
    benign_prompts = [record.prompt for record in records if record.label == 0]
    return jailbreak_prompts, benign_prompts


def write_records_csv(records: list[PromptRecord], path: Path) -> None:
    """Write split artifacts for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt",
        "jailbreak",
        "category",
        "source",
        "prompt_len",
        "original_prompt",
        "generated_by_mutator",
        "prompt_changed",
        "variant_index",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "prompt": record.prompt,
                    "jailbreak": record.label,
                    "category": record.category,
                    "source": record.source,
                    "prompt_len": record.prompt_len,
                    "original_prompt": record.original_prompt,
                    "generated_by_mutator": record.generated_by_mutator,
                    "prompt_changed": record.prompt_changed,
                    "variant_index": record.variant_index,
                }
            )


def create_training_config(
    experiment_config: DistilBERTExperimentConfig,
    output_dir: Path,
    cache_save_path: Path,
    history_save_path: Path,
) -> TrainingConfig:
    """Translate experiment config into the trainer config used elsewhere."""
    return TrainingConfig(
        model_name=experiment_config.base_model_name,
        num_epochs=experiment_config.num_epochs,
        batch_size=experiment_config.batch_size,
        learning_rate=experiment_config.learning_rate,
        max_length=experiment_config.max_length,
        warmup_ratio=experiment_config.warmup_ratio,
        weight_decay=experiment_config.weight_decay,
        num_rounds=experiment_config.num_rounds,
        variants_per_seed=experiment_config.variants_per_seed,
        min_fool_confidence=experiment_config.min_fool_confidence,
        detector_model_threshold=experiment_config.detector_model_threshold,
        mutator_strategies=experiment_config.training_mutator_strategies,
        cache_threshold=experiment_config.cache_threshold,
        output_dir=str(output_dir),
        cache_save_path=str(cache_save_path),
        history_save_path=str(history_save_path),
        seed=experiment_config.seed,
        val_split=experiment_config.val_split,
        test_split=experiment_config.test_fraction,
        verbose=experiment_config.verbose,
        show_progress_bars=experiment_config.show_progress_bars,
        dry_run=experiment_config.dry_run,
    )


def label_to_text(label: int) -> str:
    return "jailbreak" if int(label) == 1 else "benign"


def build_detector_for_model_a(model_path: str, threshold: float) -> TwoStageDetector:
    """Model A is classifier-only by default, so we attach an empty cache."""
    return TwoStageDetector(
        model_path=model_path,
        cache=NullCache(),
        model_threshold=threshold,
        auto_update_cache=False,
    )


def build_detector_for_model_b(
    model_path: str,
    cache_path: Optional[str],
    cache_threshold: float,
    model_threshold: float,
    cache=None,
) -> TwoStageDetector:
    """Model B uses its saved cache plus the adversarially trained classifier."""
    return TwoStageDetector(
        model_path=model_path,
        cache=cache,
        cache_path=cache_path,
        cache_threshold=cache_threshold,
        model_threshold=model_threshold,
        auto_update_cache=False,
    )


def summarize_predictions(rows: list[dict]) -> dict:
    """Aggregate the per-prompt CSV rows into experiment-level metrics."""
    total = len(rows)
    tp = sum(row["predicted_label"] == 1 and row["expected_label"] == 1 for row in rows)
    fp = sum(row["predicted_label"] == 1 and row["expected_label"] == 0 for row in rows)
    fn = sum(row["predicted_label"] == 0 and row["expected_label"] == 1 for row in rows)
    tn = sum(row["predicted_label"] == 0 and row["expected_label"] == 0 for row in rows)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    avg_latency = sum(row["prediction_latency_ms"] for row in rows) / total if total else 0.0
    avg_confidence = sum(row["confidence"] for row in rows) / total if total else 0.0
    avg_jb_confidence = (
        sum(row["confidence"] for row in rows if row["expected_label"] == 1)
        / max(1, sum(row["expected_label"] == 1 for row in rows))
    )
    cache_hits = sum(1 for row in rows if row["cache_hit"])

    return {
        "num_examples": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_latency_ms": round(avg_latency, 3),
        "avg_confidence": round(avg_confidence, 4),
        "avg_jailbreak_confidence": round(avg_jb_confidence, 4),
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total, 4) if total else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def load_experiment_summary_from_csv(
    csv_path: Path,
    experiment_name: str,
    model_variant: str,
    dataset_variant: str,
) -> dict:
    """Recompute summary metrics from an existing experiment CSV."""
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "predicted_label": int(row["predicted_label"]),
                    "expected_label": int(row["expected_label"]),
                    "prediction_latency_ms": float(row["prediction_latency_ms"]),
                    "confidence": float(row["confidence"]),
                    "cache_hit": parse_bool(row["cache_hit"]),
                }
            )

    summary = summarize_predictions(rows)
    summary.update(
        {
            "experiment_name": experiment_name,
            "model_variant": model_variant,
            "dataset_variant": dataset_variant,
            "csv_path": str(csv_path),
        }
    )
    return summary


def checkpoint_artifact_exists(path: Optional[str | Path]) -> bool:
    """Check whether a model checkpoint directory looks usable."""
    if not path:
        return False

    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        return (
            (checkpoint_path / "model.safetensors").exists()
            and (checkpoint_path / "config.json").exists()
        )
    return checkpoint_path.exists()


def cache_artifact_exists(path: Optional[str | Path]) -> bool:
    """Check whether a saved FAISS cache artifact exists."""
    if not path:
        return False

    cache_path = Path(path)
    return cache_path.exists() or (
        Path(str(cache_path) + ".faiss").exists()
        and Path(str(cache_path) + ".meta.pkl").exists()
    )


def _first_existing_checkpoint_path(candidates: list[Path]) -> Optional[str]:
    """Pick the first checkpoint directory that exists from a candidate list."""
    for candidate in candidates:
        if checkpoint_artifact_exists(candidate):
            return str(candidate)
    return None


def _first_existing_cache_path(candidates: list[Path]) -> Optional[str]:
    """Pick the first cache artifact path that exists from a candidate list."""
    for candidate in candidates:
        if cache_artifact_exists(candidate):
            return str(candidate)
    return None


def discover_completed_model_b_artifacts(
    adversarial_output_dir: Path,
    adversarial_cache_path: Path,
    adversarial_history_path: Path,
    adversarial_resume_state_path: Path,
    expected_rounds: int,
) -> Optional[dict]:
    """
    Recover a completed imported Model B run even when top-level experiment
    state or saved resume paths still point at the original training machine.
    """
    history = load_json(adversarial_history_path, [])
    resume_state = load_json(adversarial_resume_state_path, {})

    explicit_checkpoint = (
        resume_state.get("final_checkpoint_path")
        or resume_state.get("latest_checkpoint_path")
        or resume_state.get("current_model_source")
    )
    explicit_cache = (
        resume_state.get("final_cache_path")
        or resume_state.get("latest_cache_path")
    )

    checkpoint_candidates = []
    if explicit_checkpoint:
        checkpoint_candidates.append(Path(explicit_checkpoint))
        checkpoint_candidates.append(adversarial_output_dir / Path(explicit_checkpoint).name)
    checkpoint_candidates.append(adversarial_output_dir / f"round_{expected_rounds}")

    cache_candidates = []
    if explicit_cache:
        cache_candidates.append(Path(explicit_cache))
        cache_candidates.append(adversarial_cache_path.parent / Path(explicit_cache).name)
    cache_candidates.append(adversarial_cache_path)
    cache_candidates.append(adversarial_output_dir / f"cache_round_{expected_rounds}")

    checkpoint_path = _first_existing_checkpoint_path(checkpoint_candidates)
    cache_path = _first_existing_cache_path(cache_candidates)
    history_rounds = len(history) if isinstance(history, list) else 0
    completed = bool(resume_state.get("completed")) or history_rounds >= expected_rounds

    if not (completed and checkpoint_path and cache_path):
        return None

    return {
        "checkpoint_path": checkpoint_path,
        "cache_path": cache_path,
        "history_rounds": history_rounds,
    }


def build_initial_experiment_state(config: DistilBERTExperimentConfig) -> dict:
    """Initialize the resumable top-level experiment state."""
    return {
        "config": asdict(config),
        "splits_complete": False,
        "baseline_complete": False,
        "baseline_checkpoint_path": None,
        "adversarial_complete": False,
        "adversarial_checkpoint_path": None,
        "adversarial_cache_path": None,
        "evaluations": {},
        "summary_complete": False,
    }


def run_single_experiment(
    experiment_name: str,
    model_variant: str,
    dataset_variant: str,
    detector: TwoStageDetector,
    records: list[PromptRecord],
    csv_path: Path,
) -> dict:
    """Run one of the four experiment cells and save a per-prompt CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, record in enumerate(records, start=1):
        prediction = detector.detect(record.prompt)
        predicted_label = int(prediction["is_jailbreak"])
        rows.append(
            {
                "row_id": index,
                "experiment_name": experiment_name,
                "model_variant": model_variant,
                "dataset_variant": dataset_variant,
                "prompt": record.prompt,
                "original_prompt": record.original_prompt,
                "expected_label": record.label,
                "expected_output": label_to_text(record.label),
                "predicted_label": predicted_label,
                "output": label_to_text(predicted_label),
                "correct_prediction": predicted_label == record.label,
                "confidence": round(float(prediction["confidence"]), 6),
                "generated_by_mutator": record.generated_by_mutator,
                "prompt_changed": record.prompt_changed,
                "variant_index": record.variant_index,
                "prediction_latency_ms": round(float(prediction["latency_ms"]), 6),
                "cache_hit": prediction["stage"] == "cache",
                "cache_miss": prediction["stage"] != "cache",
                "cache_stage": prediction["stage"],
                "cache_similarity": round(float(prediction["similarity"]), 6),
                "matched_prompt": prediction["matched_prompt"],
                "category": record.category,
                "source": record.source,
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_predictions(rows)
    summary.update(
        {
            "experiment_name": experiment_name,
            "model_variant": model_variant,
            "dataset_variant": dataset_variant,
            "csv_path": str(csv_path),
        }
    )
    return summary


def build_comparison_report(
    summaries: dict[str, dict],
    config: DistilBERTExperimentConfig,
    artifact_dir: Path,
) -> dict:
    """Derive the exact comparisons you described from the four experiment runs."""
    test1 = summaries["test1_model_a_on_test"]
    test2 = summaries["test2_model_a_on_mutated_data"]
    test3 = summaries["test3_model_b_on_test"]
    test4 = summaries["test4_model_b_on_mutated_data"]

    report = {
        "config": {
            "dataset_csv": config.dataset_csv,
            "base_model_name": config.base_model_name,
            "train_fraction": config.train_fraction,
            "test_fraction": config.test_fraction,
            "mutated_variants_per_prompt": config.mutated_variants_per_prompt,
            "max_mutation_prompt_chars": config.max_mutation_prompt_chars,
            "training_mutator_strategies": config.training_mutator_strategies,
            "mutated_data_mutator_strategies": config.mutated_data_mutator_strategies,
            "mutate_only_jailbreak": config.mutate_only_jailbreak,
            "mutator_combine": config.mutator_combine,
            "num_rounds": config.num_rounds,
            "variants_per_seed": config.variants_per_seed,
            "seed": config.seed,
        },
        "questions": {
            "was_adversarial_training_useful": {
                "clean_test_accuracy_delta": round(test3["accuracy"] - test1["accuracy"], 4),
                "clean_test_f1_delta": round(test3["f1"] - test1["f1"], 4),
                "mutated_data_accuracy_delta": round(test4["accuracy"] - test2["accuracy"], 4),
                "mutated_data_f1_delta": round(test4["f1"] - test2["f1"], 4),
            },
            "did_cache_reduce_latency": {
                "clean_test_latency_delta_ms": round(
                    test1["avg_latency_ms"] - test3["avg_latency_ms"], 3
                ),
                "mutated_data_latency_delta_ms": round(
                    test2["avg_latency_ms"] - test4["avg_latency_ms"], 3
                ),
                "model_b_clean_cache_hit_rate": test3["cache_hit_rate"],
                "model_b_mutated_data_cache_hit_rate": test4["cache_hit_rate"],
            },
            "did_mutator_generate_useful_prompts": {
                "model_a_clean_vs_mutated_confidence_delta": round(
                    test2["avg_confidence"] - test1["avg_confidence"], 4
                ),
                "model_b_clean_vs_mutated_confidence_delta": round(
                    test4["avg_confidence"] - test3["avg_confidence"], 4
                ),
                "model_a_mutated_vs_model_b_mutated_confidence_delta": round(
                    test4["avg_confidence"] - test2["avg_confidence"], 4
                ),
                "model_a_mutated_vs_model_b_mutated_f1_delta": round(
                    test4["f1"] - test2["f1"], 4
                ),
            },
        },
        "experiment_summaries": summaries,
        "artifact_dir": str(artifact_dir),
    }
    return report


def write_summary_csv(summaries: dict[str, dict], path: Path) -> None:
    """Write experiment-level metrics in a spreadsheet-friendly format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(summaries.values())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_distilbert_experiment(config: DistilBERTExperimentConfig) -> dict:
    """
    Execute the full DistilBERT baseline-vs-adversarial study.

    Returns a dictionary containing the four experiment summaries and the
    derived comparison report.
    """
    output_dir = Path(config.output_dir)
    split_dir = output_dir / "splits"
    csv_dir = output_dir / "csv"
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "experiment_state.json"

    state = load_json(state_path, build_initial_experiment_state(config))
    config_snapshot = asdict(config)
    if state.get("config") != config_snapshot:
        experiment_log("Configuration changed since the last run; resetting resumable stage markers.")
        state = build_initial_experiment_state(config)
        save_json(state_path, state)

    train_split_path = split_dir / "train_split.csv"
    test_split_path = split_dir / "test_split.csv"
    mutated_split_path = split_dir / "mutated_data_split.csv"

    if (
        state.get("splits_complete")
        and train_split_path.exists()
        and test_split_path.exists()
        and mutated_split_path.exists()
    ):
        experiment_log("Reusing saved train/test/mutated_data splits.")
        train_records = load_records_csv(train_split_path)
        test_records = load_records_csv(test_split_path)
        mutated_data_records = load_records_csv(mutated_split_path)
    else:
        records = load_prompt_records(config.dataset_csv)
        train_records, test_records = split_records(
            records=records,
            test_fraction=config.test_fraction,
            seed=config.seed,
        )

        mutator = JailbreakMutator(
            strategies=config.mutated_data_mutator_strategies,
            combine=config.mutator_combine,
        )
        mutated_data_records = build_mutated_data_records(
            test_records=test_records,
            mutator=mutator,
            variants_per_prompt=config.mutated_variants_per_prompt,
            max_prompt_chars=config.max_mutation_prompt_chars,
            mutate_only_jailbreak=config.mutate_only_jailbreak,
        )

        write_records_csv(train_records, train_split_path)
        write_records_csv(test_records, test_split_path)
        write_records_csv(mutated_data_records, mutated_split_path)
        state["splits_complete"] = True
        state["config"] = config_snapshot
        save_json(state_path, state)
        experiment_log("Saved train/test/mutated_data split artifacts.")

    train_jailbreak_prompts, train_benign_prompts = split_by_label(train_records)
    test_prompts, test_labels = records_to_prompt_lists(test_records)

    baseline_output_dir = model_dir / "model_a"
    baseline_cache_path = baseline_output_dir / "cache"
    baseline_history_path = baseline_output_dir / "history.json"
    baseline_checkpoint_dir = baseline_output_dir / "checkpoint"
    baseline_metrics_path = baseline_output_dir / "baseline_metrics.json"
    baseline_config = create_training_config(
        experiment_config=config,
        output_dir=baseline_output_dir,
        cache_save_path=baseline_cache_path,
        history_save_path=baseline_history_path,
    )
    if state.get("baseline_complete") and baseline_checkpoint_dir.exists():
        experiment_log("Reusing completed Model A checkpoint.")
        baseline_result = load_json(baseline_metrics_path, {})
        model_a_path = state.get("baseline_checkpoint_path") or str(baseline_checkpoint_dir)
    else:
        baseline_trainer = AdversarialTrainer(baseline_config)
        baseline_result = baseline_trainer.train_baseline(
            jailbreak_prompts=train_jailbreak_prompts,
            benign_prompts=train_benign_prompts,
            checkpoint_dir=str(baseline_checkpoint_dir),
            build_cache=False,
            test_prompts=test_prompts,
            test_labels=test_labels,
        )
        model_a_path = baseline_result["checkpoint_dir"] or baseline_trainer.current_model_source
        save_json(
            baseline_metrics_path,
            {
                "checkpoint_dir": model_a_path,
                "best_val_f1": baseline_result.get("best_val_f1"),
                "train_time_sec": baseline_result.get("train_time_sec"),
                "test_metrics": baseline_result.get("test_metrics"),
            },
        )
        state["baseline_complete"] = True
        state["baseline_checkpoint_path"] = model_a_path
        save_json(state_path, state)
        experiment_log("Saved Model A baseline checkpoint and metadata.")

    adversarial_output_dir = model_dir / "model_b_rounds"
    adversarial_cache_path = model_dir / "model_b_cache"
    adversarial_history_path = model_dir / "model_b_history.json"
    adversarial_resume_state_path = adversarial_output_dir / "resume_state.json"
    adversarial_config = create_training_config(
        experiment_config=config,
        output_dir=adversarial_output_dir,
        cache_save_path=adversarial_cache_path,
        history_save_path=adversarial_history_path,
    )
    adversarial_config.resume_state_path = str(adversarial_resume_state_path)
    discovered_model_b = discover_completed_model_b_artifacts(
        adversarial_output_dir=adversarial_output_dir,
        adversarial_cache_path=adversarial_cache_path,
        adversarial_history_path=adversarial_history_path,
        adversarial_resume_state_path=adversarial_resume_state_path,
        expected_rounds=config.num_rounds,
    )
    saved_model_b_path = state.get("adversarial_checkpoint_path")
    saved_model_b_cache_path = state.get("adversarial_cache_path") or str(adversarial_cache_path)
    if (
        state.get("adversarial_complete")
        and checkpoint_artifact_exists(saved_model_b_path)
        and cache_artifact_exists(saved_model_b_cache_path)
    ):
        experiment_log("Reusing completed Model B checkpoint and cache.")
        model_b_path = str(saved_model_b_path)
        model_b_cache_path = str(saved_model_b_cache_path)
        model_b_cache_obj = None
    elif discovered_model_b:
        model_b_path = discovered_model_b["checkpoint_path"]
        model_b_cache_path = discovered_model_b["cache_path"]
        model_b_cache_obj = None
        state["adversarial_complete"] = True
        state["adversarial_checkpoint_path"] = model_b_path
        state["adversarial_cache_path"] = model_b_cache_path
        save_json(state_path, state)
        experiment_log(
            "Recovered completed Model B artifacts from imported resume/history files."
        )
    else:
        adversarial_trainer = AdversarialTrainer(adversarial_config)
        _, model_b_cache = adversarial_trainer.run(
            jailbreak_prompts=train_jailbreak_prompts,
            benign_prompts=train_benign_prompts,
            test_prompts=test_prompts,
            test_labels=test_labels,
        )
        model_b_path = adversarial_trainer.final_checkpoint_path or adversarial_trainer.current_model_source
        model_b_cache_path = str(adversarial_cache_path) if adversarial_trainer.final_cache_path else None
        model_b_cache_obj = None if model_b_cache_path else model_b_cache
        state["adversarial_complete"] = True
        state["adversarial_checkpoint_path"] = model_b_path
        state["adversarial_cache_path"] = model_b_cache_path or str(adversarial_cache_path)
        save_json(state_path, state)
        experiment_log("Saved Model B final checkpoint, cache, and resumable round state.")

    experiment_specs = [
        (
            "test1_model_a_on_test",
            "Model A",
            "test",
            "model_a",
            test_records,
            csv_dir / "test1_model_a_on_test.csv",
        ),
        (
            "test2_model_a_on_mutated_data",
            "Model A",
            "mutated_data",
            "model_a",
            mutated_data_records,
            csv_dir / "test2_model_a_on_mutated_data.csv",
        ),
        (
            "test3_model_b_on_test",
            "Model B",
            "test",
            "model_b",
            test_records,
            csv_dir / "test3_model_b_on_test.csv",
        ),
        (
            "test4_model_b_on_mutated_data",
            "Model B",
            "mutated_data",
            "model_b",
            mutated_data_records,
            csv_dir / "test4_model_b_on_mutated_data.csv",
        ),
    ]

    summaries = {}
    state.setdefault("evaluations", {})
    for experiment_name, model_variant, dataset_variant, detector_kind, records, csv_path in experiment_specs:
        evaluation_state = state["evaluations"].get(experiment_name, {})
        if evaluation_state.get("complete") and csv_path.exists():
            experiment_log(f"Reusing saved evaluation CSV for {experiment_name}.")
            summary = load_experiment_summary_from_csv(
                csv_path=csv_path,
                experiment_name=experiment_name,
                model_variant=model_variant,
                dataset_variant=dataset_variant,
            )
        else:
            if detector_kind == "model_a":
                detector = build_detector_for_model_a(
                    model_path=model_a_path,
                    threshold=config.detector_model_threshold,
                )
            else:
                detector = build_detector_for_model_b(
                    model_path=model_b_path,
                    cache_path=model_b_cache_path,
                    cache_threshold=config.cache_threshold,
                    model_threshold=config.detector_model_threshold,
                    cache=model_b_cache_obj,
                )

            try:
                summary = run_single_experiment(
                    experiment_name=experiment_name,
                    model_variant=model_variant,
                    dataset_variant=dataset_variant,
                    detector=detector,
                    records=records,
                    csv_path=csv_path,
                )
            finally:
                del detector
                gc.collect()
            state["evaluations"][experiment_name] = {
                "complete": True,
                "csv_path": str(csv_path),
            }
            save_json(state_path, state)
            experiment_log(f"Saved evaluation CSV for {experiment_name}.")
        summaries[experiment_name] = summary

    summary_csv_path = output_dir / "experiment_summary.csv"
    write_summary_csv(summaries, summary_csv_path)

    report = build_comparison_report(
        summaries=summaries,
        config=config,
        artifact_dir=output_dir,
    )
    with (output_dir / "comparison_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    state["summary_complete"] = True
    save_json(state_path, state)
    experiment_log("Saved experiment summary and comparison report.")

    return {
        "train_size": len(train_records),
        "test_size": len(test_records),
        "mutated_data_size": len(mutated_data_records),
        "summary_csv": str(summary_csv_path),
        "comparison_report": str(output_dir / "comparison_report.json"),
        "experiments": summaries,
    }


def parse_args() -> DistilBERTExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run the DistilBERT baseline vs adversarial experiment suite."
    )
    parser.add_argument("--dataset-csv", default="results/collected_prompts.csv")
    parser.add_argument("--output-dir", default="results/distilbert_experiment")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--variants-per-seed", type=int, default=5)
    parser.add_argument("--mutated-variants-per-prompt", type=int, default=3)
    parser.add_argument("--max-mutation-prompt-chars", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mutate-only-jailbreak", action="store_true")
    args = parser.parse_args()

    return DistilBERTExperimentConfig(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        num_rounds=args.rounds,
        variants_per_seed=args.variants_per_seed,
        mutated_variants_per_prompt=args.mutated_variants_per_prompt,
        max_mutation_prompt_chars=args.max_mutation_prompt_chars,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        dry_run=args.dry_run,
        mutate_only_jailbreak=args.mutate_only_jailbreak,
    )


def main() -> None:
    config = parse_args()
    results = run_distilbert_experiment(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
