"""
API route smoke tests for RescueText PH.
"""

import app as rescue_app


def test_predict_endpoint_with_baseline():
    rescue_app.model_manager = rescue_app.ModelManager()
    client = rescue_app.app.test_client()

    response = client.post(
        "/api/predict",
        json={
            "text": "Power lines are down along Mabini Street after the typhoon.",
            "model": "baseline",
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["category"]
    assert data["urgency"] in {"critical", "high", "moderate", "low"}


def test_predict_endpoint_rejects_empty_text():
    rescue_app.model_manager = rescue_app.ModelManager()
    client = rescue_app.app.test_client()

    response = client.post("/api/predict", json={"text": "", "model": "baseline"})
    assert response.status_code == 400
