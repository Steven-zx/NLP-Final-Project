"""
Fine-tune the TulongText PH transformer classifier.

This script uses a pretrained multilingual encoder with a locally trained
classification head. It does not call any third-party prediction API.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from disaster_config import (
    DATASET_PATH,
    FINAL_LABELS,
    ID_TO_LABEL,
    LABEL_DISPLAY_NAMES,
    LABEL_TO_ID,
    URGENCY_BY_FINAL_LABEL,
    simplify_category,
)
from preprocessing import TextPreprocessor


RANDOM_SEED = 42


def log_step(message: str) -> None:
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class DisasterTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def load_dataset(path: str, sample_size: int | None) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["text", "category"]).copy()
    df["final_label"] = df["category"].map(simplify_category)
    df = df[df["final_label"].isin(FINAL_LABELS)]
    df["label"] = df["final_label"].map(LABEL_TO_ID).astype(int)
    df = df[df["text"].str.strip() != ""].drop_duplicates(subset=["text", "final_label"])

    if sample_size and sample_size > 0 and len(df) > sample_size:
        per_class = max(1, sample_size // len(FINAL_LABELS))
        parts = []
        for _, group in df.groupby("final_label", group_keys=False):
            parts.append(group.sample(min(len(group), per_class), random_state=RANDOM_SEED))
        df = pd.concat(parts, ignore_index=True)

    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(df, test_size=0.15, random_state=RANDOM_SEED, stratify=df["label"])
    train, val = train_test_split(train_val, test_size=0.1765, random_state=RANDOM_SEED + 1, stratify=train_val["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def preprocess(texts: list[str], name: str) -> list[str]:
    log_step(f"Preprocessing {name} split ({len(texts):,} texts)")
    processor = TextPreprocessor(remove_stopwords=True, lemmatize=True, stem=True)
    processed = []
    for index, text in enumerate(texts, start=1):
        processed.append(processor.get_tokens_as_string(text))
        if index % 1000 == 0 or index == len(texts):
            print(f"  {name}: {index:,}/{len(texts):,} texts preprocessed", flush=True)
    return processed


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    """Create TrainingArguments across Transformers versions."""
    kwargs = {
        "output_dir": str(Path(args.output_dir) / "transformer_checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "save_strategy": "no",
        "load_best_model_at_end": False,
        "logging_steps": 100,
        "report_to": [],
        "seed": RANDOM_SEED,
    }

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def write_report(trainer: Trainer, test_dataset: DisasterTextDataset, test_df: pd.DataFrame, output_dir: Path) -> None:
    log_step("Evaluating transformer on the held-out test split")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    y_true = predictions.label_ids
    y_pred = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(FINAL_LABELS))),
        target_names=[LABEL_DISPLAY_NAMES[label] for label in FINAL_LABELS],
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(FINAL_LABELS)))).tolist(),
    }

    (output_dir / "disaster_transformer_evaluation.txt").write_text(
        "\n".join(
            [
                "TulongText PH Transformer Evaluation",
                "=" * 42,
                f"Accuracy: {metrics['accuracy']:.4f}",
                f"Precision macro: {metrics['precision_macro']:.4f}",
                f"Recall macro: {metrics['recall_macro']:.4f}",
                f"F1 macro: {metrics['f1_macro']:.4f}",
                "",
                "Classification Report",
                report,
                "",
                "Confusion Matrix",
                json.dumps(metrics["confusion_matrix"], indent=2),
            ]
        ),
        encoding="utf-8",
    )

    pred_df = test_df[["text", "final_label", "label", "source_dataset", "event", "language"]].copy()
    pred_df["predicted_label"] = [ID_TO_LABEL[int(item)] for item in y_pred]
    pred_df["predicted_label_id"] = y_pred
    pred_df["confidence"] = probs.max(axis=1)
    pred_df.to_csv(output_dir / "disaster_transformer_predictions.csv", index=False)
    log_step(f"Saved transformer evaluation files to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TulongText PH transformer model.")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--model-name", default="distilbert-base-multilingual-cased")
    parser.add_argument("--model-output", default="models/disaster_transformer")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sample-size", type=int, default=24000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    log_step(f"Loading dataset: {args.dataset}")
    df = load_dataset(args.dataset, sample_size=args.sample_size or None)
    print(f"Loaded {len(df):,} rows after sampling and cleanup", flush=True)
    print(df["final_label"].value_counts().to_string(), flush=True)

    log_step("Splitting dataset into train/validation/test")
    train_df, val_df, test_df = split_dataset(df)
    print(
        f"Train: {len(train_df):,} | Validation: {len(val_df):,} | Test: {len(test_df):,}",
        flush=True,
    )

    log_step(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(FINAL_LABELS),
        id2label={str(idx): label for idx, label in ID_TO_LABEL.items()},
        label2id=LABEL_TO_ID,
    )

    train_dataset = DisasterTextDataset(preprocess(train_df["text"].tolist(), "train"), train_df["label"].to_numpy(), tokenizer, args.max_length)
    val_dataset = DisasterTextDataset(preprocess(val_df["text"].tolist(), "validation"), val_df["label"].to_numpy(), tokenizer, args.max_length)
    test_dataset = DisasterTextDataset(preprocess(test_df["text"].tolist(), "test"), test_df["label"].to_numpy(), tokenizer, args.max_length)

    training_args = make_training_args(args)
    steps_per_epoch = max(1, int(np.ceil(len(train_dataset) / args.batch_size)))
    total_steps = int(np.ceil(steps_per_epoch * args.epochs))
    log_step(
        f"Starting training: {args.epochs:g} epoch(s), batch size {args.batch_size}, "
        f"about {steps_per_epoch:,} steps per epoch ({total_steps:,} total steps)"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    model_path = Path(args.model_output)
    model_path.mkdir(parents=True, exist_ok=True)
    log_step(f"Saving transformer model to {model_path}")
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    (model_path / "disaster_labels.json").write_text(
        json.dumps(
            {
                "labels": FINAL_LABELS,
                "label_to_id": LABEL_TO_ID,
                "id_to_label": ID_TO_LABEL,
                "display_names": LABEL_DISPLAY_NAMES,
                "urgency_by_label": URGENCY_BY_FINAL_LABEL,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(trainer, test_dataset, test_df, Path(args.output_dir))
    log_step(f"Done. Saved transformer model to {model_path}")


if __name__ == "__main__":
    main()
