"""
Import and normalize disaster-response NLP datasets for RescueText PH.

Outputs are written to dataset/processed by default:
  - disaster_posts_master.csv.gz: all imported rows with a task column
  - disaster_humanitarian_categories.csv: rows suitable for triage/category training
  - disaster_informativeness.csv: rows suitable for informative vs non-informative training
  - typhoon_yolanda_sentiment.csv: Filipino Typhoon Yolanda sentiment data
  - disaster_dataset_summary.json: row counts and label mappings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import pandas as pd
from datasets import get_dataset_config_names, load_dataset


HUMAID_REPO = "QCRI/HumAID-events"
CRISISBENCH_REPO = "QCRI/CrisisBench-all-lang"

TYPHOON_YOLANDA_URLS = {
    "train": {
        "-1": "https://raw.githubusercontent.com/imperialite/Philippine-Languages-Online-Corpora/master/Tweets/Annotated%20Yolanda/train/-1.txt",
        "0": "https://raw.githubusercontent.com/imperialite/Philippine-Languages-Online-Corpora/master/Tweets/Annotated%20Yolanda/train/0.txt",
        "1": "https://raw.githubusercontent.com/imperialite/Philippine-Languages-Online-Corpora/master/Tweets/Annotated%20Yolanda/train/1.txt",
    },
    "test": {
        "-1": "https://raw.githubusercontent.com/imperialite/Philippine-Languages-Online-Corpora/master/Tweets/Annotated%20Yolanda/test/-1.txt",
        "0": "https://raw.githubusercontent.com/imperialite/Philippine-Languages-Online-Corpora/master/Tweets/Annotated%20Yolanda/test/0.txt",
        "1": "https://raw.githubusercontent.com/imperialite/Philippine-Languages-Online-Corpora/master/Tweets/Annotated%20Yolanda/test/1.txt",
    },
}

TYPHOON_SENTIMENT_LABELS = {
    "-1": "negative",
    "0": "neutral",
    "1": "positive",
}

CATEGORY_NORMALIZATION = {
    "requests_or_urgent_needs": "requests_or_needs",
    "requests_or_needs": "requests_or_needs",
    "donation_and_volunteering": "rescue_volunteering_or_donation_effort",
    "donations_and_volunteering": "rescue_volunteering_or_donation_effort",
    "rescue_volunteering_or_donation_effort": "rescue_volunteering_or_donation_effort",
    "infrastructure_and_utilities_damage": "infrastructure_and_utility_damage",
    "infrastructure_and_utility_damage": "infrastructure_and_utility_damage",
    "infrastructure_and_utilities": "infrastructure_and_utility_damage",
    "injured_or_dead_people": "injured_or_dead_people",
    "missing_and_found_people": "missing_or_found_people",
    "missing_or_found_people": "missing_or_found_people",
    "displaced_and_evacuations": "displaced_people_and_evacuations",
    "displaced_people_and_evacuations": "displaced_people_and_evacuations",
    "affected_individual": "affected_individuals",
    "affected_individuals": "affected_individuals",
    "caution_and_advice": "caution_and_advice",
    "sympathy_and_support": "sympathy_and_support",
    "response_efforts": "response_efforts",
    "disease_related": "disease_related",
    "physical_landslide": "physical_landslide",
    "terrorism_related": "terrorism_related",
    "personal_update": "personal_update",
    "other_relevant_information": "other_relevant_information",
    "other_useful_information": "other_relevant_information",
    "not_humanitarian": "not_humanitarian",
    "not_applicable": "not_humanitarian",
    "not_labeled": "not_humanitarian",
}

URGENCY_BY_CATEGORY = {
    "requests_or_needs": "critical",
    "injured_or_dead_people": "critical",
    "missing_or_found_people": "critical",
    "affected_individuals": "high",
    "displaced_people_and_evacuations": "high",
    "infrastructure_and_utility_damage": "high",
    "caution_and_advice": "moderate",
    "response_efforts": "moderate",
    "rescue_volunteering_or_donation_effort": "moderate",
    "disease_related": "moderate",
    "physical_landslide": "moderate",
    "terrorism_related": "moderate",
    "other_relevant_information": "low",
    "sympathy_and_support": "low",
    "personal_update": "low",
    "not_humanitarian": "low",
}


def slug(value: object) -> str:
    text = str(value or "").strip().lower()
    for char in ["&", "/", "-", ",", "(", ")", "."]:
        text = text.replace(char, " ")
    return "_".join(part for part in text.split() if part)


def normalize_category(value: object) -> str:
    key = slug(value)
    return CATEGORY_NORMALIZATION.get(key, key or "unknown")


def normalize_informativeness(value: object) -> str:
    key = slug(value)
    if key in {"informative", "related_and_informative"}:
        return "informative"
    return "not_informative"


def urgency_for(category: object) -> str:
    return URGENCY_BY_CATEGORY.get(str(category), "low")


def clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def apply_limit(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42).reset_index(drop=True)


def standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "text",
        "category",
        "urgency",
        "task",
        "split",
        "source_dataset",
        "event",
        "language",
        "tweet_id",
        "original_label",
        "source_file",
    ]
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df[columns]


def load_humaid(max_per_event_split: int | None = None) -> pd.DataFrame:
    frames = []
    configs = get_dataset_config_names(HUMAID_REPO)
    for event in configs:
        dataset = load_dataset(HUMAID_REPO, event)
        for split, rows in dataset.items():
            df = rows.to_pandas()
            df = apply_limit(df, max_per_event_split)
            df = pd.DataFrame(
                {
                    "text": df["tweet_text"].map(clean_text),
                    "category": df["class_label"].map(normalize_category),
                    "task": "humanitarian_category",
                    "split": split,
                    "source_dataset": "HumAID-events",
                    "event": event,
                    "language": "en",
                    "tweet_id": df["tweet_id"].astype(str),
                    "original_label": df["class_label"].astype(str),
                    "source_file": HUMAID_REPO,
                }
            )
            df["urgency"] = df["category"].map(urgency_for)
            frames.append(df)
    return standard_columns(pd.concat(frames, ignore_index=True))


def load_crisisbench(
    task_name: str,
    languages: set[str] | None,
    max_per_split: int | None = None,
) -> pd.DataFrame:
    frames = []
    dataset = load_dataset(CRISISBENCH_REPO, task_name)
    for split, rows in dataset.items():
        df = rows.to_pandas()
        df["lang"] = df["lang"].fillna("").astype(str)
        if languages:
            df = df[df["lang"].isin(languages)]
        df = apply_limit(df, max_per_split)

        if task_name == "humanitarian":
            category = df["class_label"].map(normalize_category)
            task = "humanitarian_category"
            urgency = category.map(urgency_for)
        else:
            category = df["class_label"].map(normalize_informativeness)
            task = "informativeness"
            urgency = category.map(lambda item: "low" if item == "not_informative" else "moderate")

        frames.append(
            pd.DataFrame(
                {
                    "text": df["text"].map(clean_text),
                    "category": category,
                    "urgency": urgency,
                    "task": task,
                    "split": split,
                    "source_dataset": f"CrisisBench-all-lang/{task_name}",
                    "event": df["event"].astype(str),
                    "language": df["lang"].astype(str),
                    "tweet_id": df["id"].astype(str),
                    "original_label": df["class_label"].astype(str),
                    "source_file": CRISISBENCH_REPO,
                }
            )
        )
    return standard_columns(pd.concat(frames, ignore_index=True))


def load_typhoon_yolanda() -> pd.DataFrame:
    rows = []
    for split, label_urls in TYPHOON_YOLANDA_URLS.items():
        for label_id, url in label_urls.items():
            with urlopen(url, timeout=60) as response:
                lines = response.read().decode("utf-8", errors="replace").splitlines()
            for line_number, line in enumerate(lines, start=1):
                text = clean_text(line)
                if not text:
                    continue
                rows.append(
                    {
                        "text": text,
                        "category": TYPHOON_SENTIMENT_LABELS[label_id],
                        "urgency": "moderate" if label_id == "-1" else "low",
                        "task": "sentiment",
                        "split": split,
                        "source_dataset": "SEACrowd/typhoon_yolanda_tweets",
                        "event": "2013_typhoon_yolanda",
                        "language": "fil",
                        "tweet_id": f"yolanda-{split}-{label_id}-{line_number}",
                        "original_label": label_id,
                        "source_file": url,
                    }
                )
    return standard_columns(pd.DataFrame(rows))


def iter_crisislex_files(crisislex_dir: Path) -> Iterable[Path]:
    if not crisislex_dir.exists():
        return []
    return sorted(crisislex_dir.glob("*/*-tweets_labeled.csv"))


def load_local_crisislex(crisislex_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in iter_crisislex_files(crisislex_dir):
        df = pd.read_csv(csv_path, encoding_errors="replace")
        df.columns = df.columns.str.strip()
        if "Tweet Text" not in df.columns:
            continue

        category = df["Information Type"].map(normalize_category)
        informativeness = df["Informativeness"].map(normalize_informativeness)
        event = csv_path.parent.name

        humanitarian_df = pd.DataFrame(
            {
                "text": df["Tweet Text"].map(clean_text),
                "category": category,
                "urgency": category.map(urgency_for),
                "task": "humanitarian_category",
                "split": "all",
                "source_dataset": "CrisisLexT26",
                "event": event,
                "language": "unknown",
                "tweet_id": df["Tweet ID"].astype(str),
                "original_label": df["Information Type"].astype(str),
                "source_file": str(csv_path),
            }
        )
        informativeness_df = pd.DataFrame(
            {
                "text": df["Tweet Text"].map(clean_text),
                "category": informativeness,
                "urgency": informativeness.map(lambda item: "low" if item == "not_informative" else "moderate"),
                "task": "informativeness",
                "split": "all",
                "source_dataset": "CrisisLexT26",
                "event": event,
                "language": "unknown",
                "tweet_id": df["Tweet ID"].astype(str),
                "original_label": df["Informativeness"].astype(str),
                "source_file": str(csv_path),
            }
        )
        frames.extend([humanitarian_df, informativeness_df])

    if not frames:
        return standard_columns(pd.DataFrame())
    return standard_columns(pd.concat(frames, ignore_index=True))


def add_label_ids(df: pd.DataFrame, label_column: str = "category") -> tuple[pd.DataFrame, dict[str, int]]:
    labels = sorted(label for label in df[label_column].dropna().astype(str).unique() if label)
    mapping = {label: idx for idx, label in enumerate(labels)}
    df = df.copy()
    df["label"] = df[label_column].map(mapping).astype("Int64")
    return df, mapping


def write_outputs(all_data: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = all_data.dropna(subset=["text", "category"])
    all_data = all_data[all_data["text"].str.strip() != ""]
    all_data = all_data.drop_duplicates(subset=["text", "task", "category"]).reset_index(drop=True)

    humanitarian = all_data[all_data["task"] == "humanitarian_category"].copy()
    informativeness = all_data[all_data["task"] == "informativeness"].copy()
    sentiment = all_data[all_data["task"] == "sentiment"].copy()

    humanitarian, humanitarian_labels = add_label_ids(humanitarian)
    informativeness, informativeness_labels = add_label_ids(informativeness)
    sentiment, sentiment_labels = add_label_ids(sentiment)

    all_with_labels = pd.concat([humanitarian, informativeness, sentiment], ignore_index=True)

    master_path = output_dir / "disaster_posts_master.csv.gz"
    humanitarian_path = output_dir / "disaster_humanitarian_categories.csv"
    informativeness_path = output_dir / "disaster_informativeness.csv"
    sentiment_path = output_dir / "typhoon_yolanda_sentiment.csv"
    summary_path = output_dir / "disaster_dataset_summary.json"

    all_with_labels.to_csv(master_path, index=False, compression="gzip")
    humanitarian.to_csv(humanitarian_path, index=False)
    informativeness.to_csv(informativeness_path, index=False)
    sentiment.to_csv(sentiment_path, index=False)

    summary = {
        "files": {
            "master": str(master_path),
            "humanitarian_categories": str(humanitarian_path),
            "informativeness": str(informativeness_path),
            "typhoon_yolanda_sentiment": str(sentiment_path),
        },
        "row_counts": {
            "master": int(len(all_with_labels)),
            "humanitarian_categories": int(len(humanitarian)),
            "informativeness": int(len(informativeness)),
            "typhoon_yolanda_sentiment": int(len(sentiment)),
        },
        "source_counts": all_with_labels["source_dataset"].value_counts().to_dict(),
        "humanitarian_label_mapping": humanitarian_labels,
        "informativeness_label_mapping": informativeness_labels,
        "sentiment_label_mapping": sentiment_labels,
        "urgency_mapping": URGENCY_BY_CATEGORY,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_languages(value: str) -> set[str] | None:
    if not value.strip() or value.strip().lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import disaster NLP datasets for RescueText PH.")
    parser.add_argument("--output-dir", default="dataset/processed", help="Directory for normalized CSV outputs.")
    parser.add_argument("--crisislex-dir", default="dataset/CrisisLexT26", help="Path to local CrisisLexT26 folder.")
    parser.add_argument(
        "--crisisbench-languages",
        default="en,tl,fil,ceb",
        help="Comma-separated CrisisBench language codes to keep, or 'all'. Default targets English and PH languages.",
    )
    parser.add_argument(
        "--max-humaid-per-event-split",
        type=int,
        default=0,
        help="Optional cap per HumAID event/split. 0 means no cap.",
    )
    parser.add_argument(
        "--max-crisisbench-per-split",
        type=int,
        default=0,
        help="Optional cap per CrisisBench split after language filtering. 0 means no cap.",
    )
    parser.add_argument("--skip-humaid", action="store_true")
    parser.add_argument("--skip-crisisbench", action="store_true")
    parser.add_argument("--skip-typhoon-yolanda", action="store_true")
    parser.add_argument("--skip-local-crisislex", action="store_true")
    args = parser.parse_args()

    frames = []
    max_humaid = args.max_humaid_per_event_split or None
    max_crisisbench = args.max_crisisbench_per_split or None
    languages = parse_languages(args.crisisbench_languages)

    if not args.skip_humaid:
        print("Loading HumAID-events from Hugging Face...")
        frames.append(load_humaid(max_per_event_split=max_humaid))

    if not args.skip_crisisbench:
        print("Loading CrisisBench humanitarian data from Hugging Face...")
        frames.append(load_crisisbench("humanitarian", languages, max_per_split=max_crisisbench))
        print("Loading CrisisBench informativeness data from Hugging Face...")
        frames.append(load_crisisbench("informativeness", languages, max_per_split=max_crisisbench))

    if not args.skip_typhoon_yolanda:
        print("Loading Typhoon Yolanda Filipino sentiment tweets...")
        frames.append(load_typhoon_yolanda())

    if not args.skip_local_crisislex:
        print("Loading local CrisisLexT26 files...")
        frames.append(load_local_crisislex(Path(args.crisislex_dir)))

    if not frames:
        raise SystemExit("No datasets selected.")

    print("Writing normalized datasets...")
    all_data = pd.concat(frames, ignore_index=True)
    summary = write_outputs(all_data, Path(args.output_dir))

    print(json.dumps(summary["row_counts"], indent=2))
    print(f"Done. Summary written to {Path(args.output_dir) / 'disaster_dataset_summary.json'}")


if __name__ == "__main__":
    main()
