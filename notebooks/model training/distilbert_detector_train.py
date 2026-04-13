# train_deberta_jailbreak.py

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)


MODEL_NAME = os.getenv("JB_MODEL_NAME", "distilbert/distilbert-base-uncased")
CSV_PATH = "results/collected_prompts.csv"
OUTPUT_DIR = "./distilbert_jailbreak_detector"

TEXT_COL = "prompt"
LABEL_COL = "jailbreak"

MAX_LENGTH = 256
TRAIN_FRACTION = 0.90
RANDOM_SEED = 42

NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01


def offline_mode_enabled() -> bool:
    truthy = {"1", "true", "yes", "on"}
    return any(
        os.getenv(name, "").strip().lower() in truthy
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "JAILBREAK_DETECTOR_LOCAL_ONLY")
    )


def resolve_model_path(model_name_or_path: str) -> str:
    """
    Resolve a local model path or cache/download a Hugging Face snapshot.

    Loading from the snapshot directory is more reliable than passing the repo
    id directly to `from_pretrained`, especially in partially offline envs.
    """
    local_path = Path(model_name_or_path).expanduser()
    if local_path.exists():
        return str(local_path)

    try:
        return snapshot_download(
            repo_id=model_name_or_path,
            local_files_only=offline_mode_enabled(),
        )
    except Exception as exc:
        if offline_mode_enabled():
            raise RuntimeError(
                f"Model '{model_name_or_path}' is not available in the local Hugging Face cache. "
                "Disable offline mode or download it first."
            ) from exc
        raise RuntimeError(
            f"Could not resolve Hugging Face model '{model_name_or_path}': {exc}"
        ) from exc


def validate_model_artifacts(model_path: str, model_name: str) -> None:
    """
    Fail early with a clear message when the checkpoint format is incompatible
    with the installed torch/transformers combination.
    """
    model_dir = Path(model_path)
    has_safetensors = any(model_dir.glob("*.safetensors"))
    has_bin = any(model_dir.glob("pytorch_model*.bin"))

    if has_safetensors:
        return

    if has_bin:
        torch_version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
        if torch_version < (2, 6):
            raise RuntimeError(
                f"Model '{model_name}' only has PyTorch .bin weights in the resolved snapshot, "
                f"but transformers {__import__('transformers').__version__} requires torch >= 2.6 "
                "to load them safely. Use a safetensors-backed model such as "
                "'distilbert/distilbert-base-uncased', set JB_MODEL_NAME to another safe model, "
                "or upgrade torch."
            )


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(x) -> int:
    """
    Converts common label formats to 0/1.
    Accepts ints, bools, and strings like:
      0, 1, True, False, 'true', 'false', 'jailbreak', 'benign'
    """
    if pd.isna(x):
        raise ValueError("Found missing label in jailbreak column.")

    if isinstance(x, (int, np.integer)):
        if x in (0, 1):
            return int(x)
        raise ValueError(f"Invalid integer label: {x}")

    if isinstance(x, bool):
        return int(x)

    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "jailbreak", "jb"}:
        return 1
    if s in {"0", "false", "no", "n", "benign", "safe"}:
        return 0

    raise ValueError(f"Unsupported label value: {x}")


def load_dataframe(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {TEXT_COL, LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df[[TEXT_COL, LABEL_COL, "category", "source", "prompt_len"]].copy()

    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""].copy()

    df[LABEL_COL] = df[LABEL_COL].apply(normalize_label)

    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

    return df


def build_datasets(df: pd.DataFrame, train_fraction: float, seed: int):
    train_df, val_df = train_test_split(
        df,
        train_size=train_fraction,
        random_state=seed,
        stratify=df[LABEL_COL],
    )

    train_ds = Dataset.from_dict(
        {
            "text": train_df[TEXT_COL].astype(str).tolist(),
            "label": train_df[LABEL_COL].astype(int).tolist(),
        }
    )
    val_ds = Dataset.from_dict(
        {
            "text": val_df[TEXT_COL].astype(str).tolist(),
            "label": val_df[LABEL_COL].astype(int).tolist(),
        }
    )
    return train_df, val_df, train_ds, val_ds


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    seed_everything(RANDOM_SEED)

    print("Loading CSV...")
    df = load_dataframe(CSV_PATH)

    print(f"Total rows after cleaning: {len(df)}")
    print("Class distribution:")
    print(df[LABEL_COL].value_counts().sort_index())
    print()

    train_df, val_df, train_ds, val_ds = build_datasets(
        df=df,
        train_fraction=TRAIN_FRACTION,
        seed=RANDOM_SEED,
    )

    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    print()

    resolved_model_path = resolve_model_path(MODEL_NAME)
    validate_model_artifacts(resolved_model_path, MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)

    tokenized_train = train_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_val = val_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        resolved_model_path,
        num_labels=2,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Running final evaluation...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving best model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save split files for inspection
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val_split.csv"), index=False)

    print("Done.")


if __name__ == "__main__":
    main()
