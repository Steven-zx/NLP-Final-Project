"""
Dataset and label mapping checks for RescueText PH.
"""

import pandas as pd

from disaster_config import DATASET_PATH, FINAL_LABELS, simplify_category, urgency_for_label


def test_disaster_dataset_loads():
    df = pd.read_csv(DATASET_PATH, nrows=100)
    assert {"text", "category", "urgency", "source_dataset"}.issubset(df.columns)
    assert df["text"].notna().all()
    assert df["category"].notna().all()


def test_simplified_label_mapping_is_valid():
    raw_labels = [
        "requests_or_needs",
        "injured_or_dead_people",
        "displaced_people_and_evacuations",
        "infrastructure_and_utility_damage",
        "caution_and_advice",
        "rescue_volunteering_or_donation_effort",
        "other_relevant_information",
        "not_humanitarian",
    ]
    mapped = [simplify_category(label) for label in raw_labels]
    assert set(mapped).issubset(set(FINAL_LABELS))
    assert urgency_for_label("rescue_or_urgent_needs") == "critical"
    assert urgency_for_label("not_humanitarian") == "low"
