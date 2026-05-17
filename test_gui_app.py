"""
Frontend file smoke tests for RescueText PH.
"""

from pathlib import Path


def test_frontend_contains_rescuetext_ui():
    html = Path("index.html").read_text(encoding="utf-8")
    js = Path("main.js").read_text(encoding="utf-8")

    assert "RescueText PH" in html
    assert "Predicted Category" in html
    assert "Top Predictions" in html
    assert "/api/predict" in js
    assert "top_predictions" in js
