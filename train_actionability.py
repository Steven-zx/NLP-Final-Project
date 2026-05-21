"""
Train a binary actionability classifier for RescueText PH.

This secondary NLP task distinguishes posts that need response attention from
general or non-humanitarian posts. It complements the harder 8-class category
classifier and is useful for high-reliability triage reporting.
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from disaster_config import DATASET_PATH, simplify_category
from preprocessing import TextPreprocessor


RANDOM_SEED = 42
ACTIONABLE_LABELS = {
    "rescue_or_urgent_needs",
    "medical_or_casualties",
    "evacuation_or_displacement",
    "infrastructure_damage",
    "warnings_or_advice",
    "donation_or_volunteering",
}
ACTIONABILITY_ID_TO_LABEL = {0: "non_actionable", 1: "actionable"}
ACTIONABILITY_LABEL_TO_ID = {label: idx for idx, label in ACTIONABILITY_ID_TO_LABEL.items()}


def load_dataset(path: str, sample_size: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).dropna(subset=["text", "category"]).copy()
    if "final_label" in df.columns:
        df["final_label"] = df["final_label"].fillna(df["category"].map(simplify_category))
    else:
        df["final_label"] = df["category"].map(simplify_category)
    df = df[df["text"].str.strip() != ""].drop_duplicates(subset=["text", "final_label"])
    df["actionability"] = np.where(df["final_label"].isin(ACTIONABLE_LABELS), "actionable", "non_actionable")
    df["actionability_id"] = df["actionability"].map(ACTIONABILITY_LABEL_TO_ID).astype(int)

    if sample_size and sample_size > 0 and len(df) > sample_size:
        per_class = max(1, sample_size // 2)
        parts = []
        for _, group in df.groupby("actionability", group_keys=False):
            parts.append(group.sample(min(len(group), per_class), random_state=RANDOM_SEED))
        df = pd.concat(parts, ignore_index=True)

    return df.reset_index(drop=True)


def preprocess_texts(texts: List[str]) -> List[str]:
    processor = TextPreprocessor(remove_stopwords=True, lemmatize=True, stem=True)
    return [processor.get_tokens_as_string(text) for text in texts]


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=df["actionability_id"],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.1765,
        random_state=RANDOM_SEED + 1,
        stratify=train_val["actionability_id"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def model_candidates() -> list[tuple[str, Pipeline]]:
    return [
        (
            "actionability_word_bigram_logreg",
            Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=60000,
                            ngram_range=(1, 2),
                            min_df=2,
                            max_df=0.92,
                            sublinear_tf=True,
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="saga",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
        ),
        (
            "actionability_word_char_sgd",
            Pipeline(
                [
                    (
                        "features",
                        FeatureUnion(
                            [
                                (
                                    "word",
                                    TfidfVectorizer(
                                        analyzer="word",
                                        max_features=50000,
                                        ngram_range=(1, 2),
                                        min_df=2,
                                        max_df=0.92,
                                        sublinear_tf=True,
                                    ),
                                ),
                                (
                                    "char",
                                    TfidfVectorizer(
                                        analyzer="char_wb",
                                        max_features=40000,
                                        ngram_range=(3, 5),
                                        min_df=2,
                                        sublinear_tf=True,
                                    ),
                                ),
                            ]
                        ),
                    ),
                    (
                        "classifier",
                        SGDClassifier(
                            loss="log_loss",
                            alpha=1e-5,
                            class_weight="balanced",
                            max_iter=1500,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
        ),
        (
            "actionability_word_char_linearsvc",
            Pipeline(
                [
                    (
                        "features",
                        FeatureUnion(
                            [
                                (
                                    "word",
                                    TfidfVectorizer(
                                        analyzer="word",
                                        max_features=50000,
                                        ngram_range=(1, 3),
                                        min_df=2,
                                        max_df=0.92,
                                        sublinear_tf=True,
                                    ),
                                ),
                                (
                                    "char",
                                    TfidfVectorizer(
                                        analyzer="char_wb",
                                        max_features=40000,
                                        ngram_range=(3, 5),
                                        min_df=2,
                                        sublinear_tf=True,
                                    ),
                                ),
                            ]
                        ),
                    ),
                    (
                        "classifier",
                        LinearSVC(C=0.8, class_weight="balanced", dual="auto", random_state=RANDOM_SEED),
                    ),
                ]
            ),
        ),
    ]


def probability_like(model: Pipeline, texts: List[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)
    scores = model.decision_function(texts)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def evaluate(model: Pipeline, texts: List[str], labels: np.ndarray) -> dict:
    predictions = model.predict(texts)
    probabilities = probability_like(model, texts)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["Non-Actionable", "Actionable"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def write_outputs(
    model: Pipeline,
    model_name: str,
    eval_result: dict,
    validation_results: list[dict],
    test_df: pd.DataFrame,
    model_output: Path,
    output_dir: Path,
) -> None:
    model_output.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": model,
            "model_name": model_name,
            "id_to_label": ACTIONABILITY_ID_TO_LABEL,
            "label_to_id": ACTIONABILITY_LABEL_TO_ID,
            "actionable_category_labels": sorted(ACTIONABLE_LABELS),
            "validation_results": validation_results,
        },
        model_output,
    )

    pd.DataFrame(validation_results).sort_values("f1_macro", ascending=False).to_csv(
        output_dir / "actionability_model_comparison.csv",
        index=False,
    )

    (output_dir / "actionability_evaluation.txt").write_text(
        "\n".join(
            [
                "RescueText PH Actionability Evaluation",
                "=" * 44,
                f"Selected model: {model_name}",
                f"Accuracy: {eval_result['accuracy']:.4f}",
                f"Precision macro: {eval_result['precision_macro']:.4f}",
                f"Recall macro: {eval_result['recall_macro']:.4f}",
                f"F1 macro: {eval_result['f1_macro']:.4f}",
                f"F1 weighted: {eval_result['f1_weighted']:.4f}",
                "",
                "Validation Model Comparison",
                pd.DataFrame(validation_results)
                .sort_values("f1_macro", ascending=False)
                .to_string(index=False, float_format=lambda value: f"{value:.4f}"),
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

    pred_df = test_df[["text", "final_label", "actionability", "actionability_id", "source_dataset", "event", "language"]].copy()
    pred_df["predicted_actionability"] = [ACTIONABILITY_ID_TO_LABEL[int(item)] for item in eval_result["predictions"]]
    pred_df["confidence"] = eval_result["probabilities"].max(axis=1)
    pred_df.to_csv(output_dir / "actionability_predictions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RescueText PH binary actionability model.")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--model-output", default="models/actionability_baseline.pkl")
    parser.add_argument("--output-dir", default="outputs/actionability")
    args = parser.parse_args()

    df = load_dataset(args.dataset, sample_size=args.sample_size or None)
    print(f"Loaded {len(df)} rows")
    print(df["actionability"].value_counts().to_string())

    train_df, val_df, test_df = split_dataset(df)
    print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

    X_train = preprocess_texts(train_df["text"].tolist())
    X_val = preprocess_texts(val_df["text"].tolist())
    X_test = preprocess_texts(test_df["text"].tolist())
    y_train = train_df["actionability_id"].to_numpy()
    y_val = val_df["actionability_id"].to_numpy()
    y_test = test_df["actionability_id"].to_numpy()

    best_name = ""
    best_model = None
    best_score = -1.0
    validation_results = []
    for model_name, candidate in model_candidates():
        print(f"\nTraining candidate: {model_name}")
        candidate.fit(X_train, y_train)
        val_result = evaluate(candidate, X_val, y_val)
        row = {
            "model_name": model_name,
            "accuracy": val_result["accuracy"],
            "precision_macro": val_result["precision_macro"],
            "recall_macro": val_result["recall_macro"],
            "f1_macro": val_result["f1_macro"],
            "f1_weighted": val_result["f1_weighted"],
        }
        validation_results.append(row)
        print(f"Validation accuracy={row['accuracy']:.4f} f1_macro={row['f1_macro']:.4f}")
        if row["f1_macro"] > best_score:
            best_name = model_name
            best_model = candidate
            best_score = row["f1_macro"]

    if best_model is None:
        raise RuntimeError("No actionability candidates were trained.")

    print(f"\nSelected actionability model: {best_name} (validation macro F1={best_score:.4f})")
    eval_result = evaluate(best_model, X_test, y_test)
    write_outputs(best_model, best_name, eval_result, validation_results, test_df, Path(args.model_output), Path(args.output_dir))
    print(f"Accuracy: {eval_result['accuracy']:.4f}")
    print(f"F1 macro: {eval_result['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
