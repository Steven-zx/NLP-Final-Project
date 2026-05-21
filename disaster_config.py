"""
Shared RescueText PH labels, urgency mapping, and dataset helpers.
"""

from __future__ import annotations

from typing import Dict


APP_NAME = "RescueText PH"
DATASET_PATH = "dataset/processed/disaster_humanitarian_categories.csv"

FINAL_LABELS = [
    "rescue_or_urgent_needs",
    "medical_or_casualties",
    "evacuation_or_displacement",
    "infrastructure_damage",
    "warnings_or_advice",
    "donation_or_volunteering",
    "general_update",
    "not_humanitarian",
]

LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(FINAL_LABELS)}
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}

LABEL_DISPLAY_NAMES = {
    "rescue_or_urgent_needs": "Rescue or Urgent Needs",
    "medical_or_casualties": "Medical or Casualties",
    "evacuation_or_displacement": "Evacuation or Displacement",
    "infrastructure_damage": "Infrastructure Damage",
    "warnings_or_advice": "Warnings or Advice",
    "donation_or_volunteering": "Donation or Volunteering",
    "general_update": "General Update",
    "not_humanitarian": "Not Humanitarian",
}

SOURCE_TO_FINAL_LABEL = {
    "rescue_or_urgent_needs": "rescue_or_urgent_needs",
    "medical_or_casualties": "medical_or_casualties",
    "evacuation_or_displacement": "evacuation_or_displacement",
    "infrastructure_damage": "infrastructure_damage",
    "warnings_or_advice": "warnings_or_advice",
    "donation_or_volunteering": "donation_or_volunteering",
    "general_update": "general_update",
    "not_humanitarian": "not_humanitarian",
    "requests_or_needs": "rescue_or_urgent_needs",
    "missing_or_found_people": "rescue_or_urgent_needs",
    "injured_or_dead_people": "medical_or_casualties",
    "disease_related": "medical_or_casualties",
    "affected_individuals": "general_update",
    "displaced_people_and_evacuations": "evacuation_or_displacement",
    "infrastructure_and_utility_damage": "infrastructure_damage",
    "physical_landslide": "infrastructure_damage",
    "caution_and_advice": "warnings_or_advice",
    "response_efforts": "donation_or_volunteering",
    "terrorism_related": "general_update",
    "rescue_volunteering_or_donation_effort": "donation_or_volunteering",
    "other_relevant_information": "general_update",
    "personal_update": "general_update",
    "sympathy_and_support": "general_update",
    "not_humanitarian": "not_humanitarian",
}

URGENCY_BY_FINAL_LABEL = {
    "rescue_or_urgent_needs": "critical",
    "medical_or_casualties": "critical",
    "evacuation_or_displacement": "high",
    "infrastructure_damage": "high",
    "warnings_or_advice": "moderate",
    "donation_or_volunteering": "moderate",
    "general_update": "low",
    "not_humanitarian": "low",
}

URGENCY_DISPLAY_NAMES = {
    "critical": "Critical",
    "high": "High",
    "moderate": "Moderate",
    "low": "Low",
}


def simplify_category(category: object) -> str:
    """Map raw humanitarian labels to the final 8-class project taxonomy."""
    key = str(category or "").strip()
    return SOURCE_TO_FINAL_LABEL.get(key, "general_update")


def label_id(label: object) -> int:
    """Return the integer class ID for a final label."""
    return LABEL_TO_ID[simplify_category(label) if label not in LABEL_TO_ID else str(label)]


def urgency_for_label(label: object) -> str:
    """Return deterministic urgency for a final label."""
    final_label = simplify_category(label) if str(label) not in LABEL_TO_ID else str(label)
    return URGENCY_BY_FINAL_LABEL.get(final_label, "low")


def display_label(label: object) -> str:
    """Return human-readable label text."""
    final_label = simplify_category(label) if str(label) not in LABEL_TO_ID else str(label)
    return LABEL_DISPLAY_NAMES.get(final_label, final_label.replace("_", " ").title())
