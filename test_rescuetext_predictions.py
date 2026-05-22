"""
Smoke tests for TulongText PH disaster triage predictions.
"""

from app import ModelManager


def test_baseline_model_loads_and_predicts():
    manager = ModelManager()
    assert "baseline" in manager.models

    result = manager.predict(
        "Need rescue sa Brgy. San Isidro, baha na hanggang bubong. May stranded family.",
        "baseline",
    )

    assert result["category"]
    assert result["urgency"] in {"critical", "high", "moderate", "low"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["top_predictions"]
    assert result["preprocessing"]["tokens"]


def test_empty_input_is_rejected():
    manager = ModelManager()
    try:
        manager.predict("", "baseline")
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty input")
