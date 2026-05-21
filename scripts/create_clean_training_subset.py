"""
Create a cleaner RescueText PH training subset.

The output keeps the original processed dataset intact and writes a separate
CSV with stricter labels, duplicate removal, and low-information text filters.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from disaster_config import FINAL_LABELS, LABEL_TO_ID, simplify_category
from preprocessing import TextPreprocessor


AMBIGUOUS_SOURCE_LABELS = {
    "terrorism_related",
}

RESPONSE_KEYWORDS = {
    "aid",
    "assist",
    "assistance",
    "donate",
    "donation",
    "relief",
    "shelter",
    "volunteer",
    "rescue",
    "response",
    "food",
    "water",
    "medical",
}

RESCUE_KEYWORDS = {
    "rescue",
    "rescued",
    "trapped",
    "stranded",
    "missing",
    "found",
    "sos",
    "help",
    "urgent",
    "save",
    "evacuate",
    "evacuation",
}

DONATION_KEYWORDS = {
    "donate",
    "donation",
    "volunteer",
    "relief",
    "aid",
    "supply",
    "supplies",
    "food",
    "water",
    "clothing",
    "blanket",
    "shelter",
}


def clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def normalized_text_key(value: object) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_set(value: object) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", str(value or "").lower()))


def remap_label(row: pd.Series) -> str | None:
    category = str(row.get("category") or "").strip()
    words = word_set(row.get("text"))

    if category in AMBIGUOUS_SOURCE_LABELS:
        return None

    if category == "affected_individuals":
        if words & {"injured", "dead", "death", "killed", "wounded", "hospital", "medical"}:
            return "medical_or_casualties"
        if words & {"evacuee", "evacuated", "displaced", "shelter", "homeless"}:
            return "evacuation_or_displacement"
        return "general_update"

    if category == "response_efforts":
        if words & RESPONSE_KEYWORDS:
            return "donation_or_volunteering"
        return "general_update"

    if category == "rescue_volunteering_or_donation_effort":
        if words & RESCUE_KEYWORDS and not words & DONATION_KEYWORDS:
            return "rescue_or_urgent_needs"
        if words & DONATION_KEYWORDS:
            return "donation_or_volunteering"
        return None

    return simplify_category(category)


def should_keep_text(text: str, processor: TextPreprocessor, min_tokens: int) -> tuple[bool, str, int]:
    cleaned = clean_text(text)
    if not cleaned:
        return False, "", 0
    if cleaned.lower().startswith("rt @"):
        return False, "", 0
    token_string = processor.get_tokens_as_string(cleaned)
    token_count = len(token_string.split())
    if token_count < min_tokens:
        return False, token_string, token_count
    return True, token_string, token_count


def build_subset(input_path: Path, output_path: Path, summary_path: Path, min_tokens: int) -> dict:
    df = pd.read_csv(input_path, low_memory=False)
    required = {"text", "category"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    before_rows = len(df)
    df = df.dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.strip() != ""].copy()

    processor = TextPreprocessor(remove_stopwords=True, lemmatize=True, stem=True)
    keep_rows = []
    for _, row in df.iterrows():
        final_label = remap_label(row)
        if final_label not in FINAL_LABELS:
            continue
        keep, token_string, token_count = should_keep_text(row["text"], processor, min_tokens)
        if not keep:
            continue
        row = row.copy()
        row["category"] = final_label
        row["final_label"] = final_label
        row["label"] = LABEL_TO_ID[final_label]
        row["normalized_text"] = normalized_text_key(row["text"])
        row["preprocessed_text"] = token_string
        row["token_count"] = token_count
        keep_rows.append(row)

    clean_df = pd.DataFrame(keep_rows)
    if clean_df.empty:
        raise ValueError("No rows survived cleaning.")

    before_dedupe = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["normalized_text"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "before_rows": int(before_rows),
        "after_required_fields": int(len(df)),
        "after_quality_filters": int(before_dedupe),
        "after_normalized_deduplication": int(len(clean_df)),
        "removed_by_normalized_deduplication": int(before_dedupe - len(clean_df)),
        "min_tokens": int(min_tokens),
        "class_distribution": clean_df["final_label"].value_counts().to_dict(),
        "source_distribution": clean_df.get("source_dataset", pd.Series(dtype=str)).value_counts().to_dict(),
        "label_mapping_notes": {
            "affected_individuals": "Remapped by text keywords; broad affected-person updates become general_update.",
            "response_efforts": "Mapped to donation_or_volunteering when response/aid terms are present; otherwise general_update.",
            "rescue_volunteering_or_donation_effort": "Split by rescue vs donation keywords; ambiguous rows are dropped.",
            "terrorism_related": "Dropped because it is too broad for disaster-response triage categories.",
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a cleaner RescueText PH training subset.")
    parser.add_argument("--input", default="dataset/processed/disaster_humanitarian_categories.csv")
    parser.add_argument("--output", default="dataset/processed/clean_training_subset.csv")
    parser.add_argument("--summary-output", default="dataset/processed/clean_training_subset_summary.json")
    parser.add_argument("--min-tokens", type=int, default=4)
    args = parser.parse_args()

    summary = build_subset(Path(args.input), Path(args.output), Path(args.summary_output), args.min_tokens)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
