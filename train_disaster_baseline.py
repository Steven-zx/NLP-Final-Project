"""
Train the RescueText PH baseline model.

Model: TF-IDF features + Logistic Regression classifier.
Dataset: dataset/processed/disaster_humanitarian_categories.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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


def load_disaster_dataset(path: str, sample_size: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "category"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "category"]).copy()
    df["final_label"] = df["category"].map(simplify_category)
    df = df[df["final_label"].isin(FINAL_LABELS)]
    df["label"] = df["final_label"].map(LABEL_TO_ID).astype(int)
    df = df[df["text"].str.strip() != ""].drop_duplicates(subset=["text", "final_label"])

    if sample_size and sample_size > 0 and len(df) > sample_size:
        per_class = max(1, sample_size // len(FINAL_LABELS))
        sampled = []
        for _, group in df.groupby("final_label", group_keys=False):
            sampled.append(group.sample(min(len(group), per_class), random_state=RANDOM_SEED))
        df = pd.concat(sampled, ignore_index=True)
        if len(df) < sample_size:
            remainder = (
                pd.read_csv(path)
                .dropna(subset=["text", "category"])
                .assign(final_label=lambda item: item["category"].map(simplify_category))
            )
            remainder["label"] = remainder["final_label"].map(LABEL_TO_ID)
            remainder = remainder.dropna(subset=["label"])
            remainder = remainder[~remainder["text"].isin(df["text"])]
            df = pd.concat(
                [df, remainder.sample(min(len(remainder), sample_size - len(df)), random_state=RANDOM_SEED)],
                ignore_index=True,
            )

    return df.reset_index(drop=True)


def preprocess_texts(texts: List[str]) -> List[str]:
    processor = TextPreprocessor(remove_stopwords=True, lemmatize=True, stem=True)
    return [processor.get_tokens_as_string(text) for text in texts]


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.1765,
        random_state=RANDOM_SEED + 1,
        stratify=train_val["label"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def train_model(texts: List[str], labels: np.ndarray) -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


def evaluate(model: Pipeline, texts: List[str], labels: np.ndarray) -> dict:
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(FINAL_LABELS))),
            target_names=[LABEL_DISPLAY_NAMES[label] for label in FINAL_LABELS],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(FINAL_LABELS)))).tolist(),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def write_outputs(
    model: Pipeline,
    eval_result: dict,
    test_df: pd.DataFrame,
    output_model: Path,
    output_dir: Path,
) -> None:
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": model,
            "labels": FINAL_LABELS,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "urgency_by_label": URGENCY_BY_FINAL_LABEL,
        },
        output_model,
    )

    report_path = output_dir / "disaster_baseline_evaluation.txt"
    report_path.write_text(
        "\n".join(
            [
                "RescueText PH Baseline Evaluation",
                "=" * 40,
                f"Accuracy: {eval_result['accuracy']:.4f}",
                f"Precision macro: {eval_result['precision_macro']:.4f}",
                f"Recall macro: {eval_result['recall_macro']:.4f}",
                f"F1 macro: {eval_result['f1_macro']:.4f}",
                "",
                "Classification Report",
                eval_result["classification_report"],
                "",
                "Confusion Matrix",
                json.dumps(eval_result["confusion_matrix"], indent=2),
            ]
        ),
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(11, 9))
        display = ConfusionMatrixDisplay(
            confusion_matrix=np.array(eval_result["confusion_matrix"]),
            display_labels=[LABEL_DISPLAY_NAMES[label] for label in FINAL_LABELS],
        )
        display.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
        ax.set_title("RescueText PH Baseline Confusion Matrix")
        fig.tight_layout()
        fig.savefig(output_dir / "disaster_confusion_matrix.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (output_dir / "disaster_confusion_matrix_NOTE.txt").write_text(
            f"Confusion matrix image was not generated because matplotlib is unavailable or failed: {exc}\n"
            "The numeric confusion matrix is included in disaster_baseline_evaluation.txt.\n",
            encoding="utf-8",
        )

    pred_path = output_dir / "disaster_baseline_predictions.csv"
    predictions = eval_result["predictions"]
    pred_df = test_df[["text", "final_label", "label", "source_dataset", "event", "language"]].copy()
    pred_df["predicted_label"] = [ID_TO_LABEL[int(item)] for item in predictions]
    pred_df["predicted_label_id"] = predictions
    pred_df["confidence"] = eval_result["probabilities"].max(axis=1)
    pred_df.to_csv(pred_path, index=False)

    summary_path = output_dir / "disaster_label_mapping.json"
    summary_path.write_text(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RescueText PH baseline model.")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--sample-size", type=int, default=0, help="Optional balanced sample size for quick runs.")
    parser.add_argument("--model-output", default="models/disaster_baseline.pkl")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    df = load_disaster_dataset(args.dataset, sample_size=args.sample_size or None)
    print(f"Loaded {len(df)} rows")
    print(df["final_label"].value_counts().to_string())

    train_df, _, test_df = split_dataset(df)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    X_train = preprocess_texts(train_df["text"].tolist())
    X_test = preprocess_texts(test_df["text"].tolist())
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    model = train_model(X_train, y_train)
    eval_result = evaluate(model, X_test, y_test)

    write_outputs(model, eval_result, test_df, Path(args.model_output), Path(args.output_dir))
    print(f"Accuracy: {eval_result['accuracy']:.4f}")
    print(f"F1 macro: {eval_result['f1_macro']:.4f}")
    print(f"Saved model to {args.model_output}")


if __name__ == "__main__":
    main()
