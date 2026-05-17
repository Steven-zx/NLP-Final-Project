"""
RescueText PH Flask API.

Classifies disaster-related social media posts into humanitarian response
categories and derives an urgency level from the predicted category.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List

import joblib
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from disaster_config import (
    APP_NAME,
    FINAL_LABELS,
    ID_TO_LABEL,
    LABEL_DISPLAY_NAMES,
    URGENCY_BY_FINAL_LABEL,
    URGENCY_DISPLAY_NAMES,
    display_label,
    urgency_for_label,
)
from preprocessing import TextPreprocessor


FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

BASELINE_MODEL_PATH = "models/disaster_baseline.pkl"
TRANSFORMER_MODEL_PATH = "models/disaster_transformer"

DEFAULT_MODEL = "transformer"
LOG_FILE = "api_predictions.log"
MAX_LOG_SIZE = 10 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KEYWORD_BOOSTS = {
    "rescue_or_urgent_needs": [
        "rescue",
        "stranded",
        "trapped",
        "sos",
        "asap",
        "urgent",
        "bubong",
        "rooftop",
        "baha",
        "flood",
        "saklolo",
        "tulong",
        "help us",
    ],
    "medical_or_casualties": [
        "injured",
        "wounded",
        "dead",
        "casualty",
        "medical",
        "ambulance",
        "hospital",
        "gamot",
        "nasugatan",
        "patay",
    ],
    "evacuation_or_displacement": [
        "evacuation",
        "evacuate",
        "shelter",
        "center",
        "evacuee",
        "lumikas",
        "relocation",
    ],
    "infrastructure_damage": [
        "bridge",
        "road",
        "power line",
        "kuryente",
        "collapsed",
        "landslide",
        "damage",
        "sirang",
        "down",
    ],
    "warnings_or_advice": [
        "avoid",
        "warning",
        "advisory",
        "caution",
        "alert",
        "stay away",
        "mag ingat",
        "ingat",
    ],
    "donation_or_volunteering": [
        "donate",
        "donation",
        "volunteer",
        "relief goods",
        "food packs",
        "tubig",
        "water",
        "blankets",
    ],
    "not_humanitarian": [
        "selling",
        "sale",
        "discount",
        "promo",
        "buy now",
    ],
}

KEYWORD_BOOST_WEIGHTS = {
    "rescue_or_urgent_needs": 0.30,
    "medical_or_casualties": 0.34,
    "evacuation_or_displacement": 0.26,
    "infrastructure_damage": 0.20,
    "warnings_or_advice": 0.18,
    "donation_or_volunteering": 0.18,
    "not_humanitarian": 0.30,
}


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("RescueTextPH")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.config["JSON_AS_ASCII"] = False
    return app


class ModelManager:
    def __init__(self) -> None:
        self.preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True, stem=True)
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self._load_models()

    def _load_models(self) -> None:
        for name, loader in (
            ("baseline", self._load_baseline),
            ("transformer", self._load_transformer),
        ):
            try:
                loader()
            except Exception as exc:
                logger.error("Failed to load %s model: %s", name, exc)

    def _load_baseline(self) -> None:
        if not os.path.exists(BASELINE_MODEL_PATH):
            logger.warning("Baseline model not found: %s", BASELINE_MODEL_PATH)
            return

        artifact = joblib.load(BASELINE_MODEL_PATH)
        pipeline = artifact["pipeline"] if isinstance(artifact, dict) and "pipeline" in artifact else artifact
        self.models["baseline"] = pipeline
        self.model_info["baseline"] = {
            "name": "TF-IDF + Logistic Regression",
            "type": "traditional_ml",
            "description": "Classical baseline trained on disaster humanitarian categories",
            "status": "loaded",
            "inference_device": "cpu",
        }
        logger.info("Loaded baseline model")

    def _load_transformer(self) -> None:
        if not os.path.isdir(TRANSFORMER_MODEL_PATH):
            logger.warning("Transformer model not found: %s", TRANSFORMER_MODEL_PATH)
            return

        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_PATH)
        model.to(DEVICE)
        model.eval()

        self.models["transformer"] = (model, tokenizer)
        self.model_info["transformer"] = {
            "name": "DistilBERT Multilingual Fine-tuned",
            "type": "transformer",
            "description": "Transformer classifier fine-tuned locally for disaster response triage",
            "status": "loaded",
            "inference_device": "cuda" if DEVICE.type == "cuda" else "cpu",
        }
        logger.info("Loaded transformer model")

    def available_models(self) -> Dict[str, Dict[str, Any]]:
        return self.model_info

    def default_model(self) -> str:
        if DEFAULT_MODEL in self.models:
            return DEFAULT_MODEL
        if "baseline" in self.models:
            return "baseline"
        if self.models:
            return next(iter(self.models))
        return DEFAULT_MODEL

    def preprocessing_details(self, text: str) -> dict:
        processed = self.preprocessor.preprocess_text(text)
        return {
            "original_text": processed.original_text,
            "cleaned_text": processed.cleaned_text,
            "tokens": processed.stemmed_tokens,
            "token_count": processed.token_count,
        }

    def _format_prediction(
        self,
        text: str,
        model_name: str,
        probabilities: np.ndarray,
        inference_time_ms: float,
    ) -> Dict[str, Any]:
        top_indices = probabilities.argsort()[::-1][:3]
        top_predictions = [
            {
                "category": ID_TO_LABEL[int(idx)],
                "display_name": display_label(ID_TO_LABEL[int(idx)]),
                "confidence": float(probabilities[int(idx)]),
            }
            for idx in top_indices
        ]

        best = top_predictions[0]["category"]
        urgency = urgency_for_label(best)

        return {
            "category": best,
            "category_display": display_label(best),
            "urgency": urgency,
            "urgency_display": URGENCY_DISPLAY_NAMES.get(urgency, urgency.title()),
            "confidence": float(top_predictions[0]["confidence"]),
            "top_predictions": top_predictions,
            "preprocessing": self.preprocessing_details(text),
            "model": model_name,
            "inference_time_ms": round(inference_time_ms, 2),
        }

    def _apply_keyword_boosts(self, text: str, probabilities: np.ndarray) -> np.ndarray:
        lowered = text.lower()
        boosted = probabilities.astype(float).copy()
        applied = False

        for label, keywords in KEYWORD_BOOSTS.items():
            matches = sum(1 for keyword in keywords if keyword in lowered)
            if matches:
                weight = KEYWORD_BOOST_WEIGHTS.get(label, 0.18)
                boosted[FINAL_LABELS.index(label)] += min(0.75, weight * matches)
                applied = True

        if applied and boosted.sum() > 0:
            boosted = boosted / boosted.sum()
        return boosted

    def predict_baseline(self, text: str) -> Dict[str, Any]:
        start = time.time()
        processed = self.preprocessor.get_tokens_as_string(text)
        pipeline = self.models["baseline"]
        probabilities = pipeline.predict_proba([processed])[0]
        probabilities = self._apply_keyword_boosts(text, probabilities)
        return self._format_prediction(text, "baseline", probabilities, (time.time() - start) * 1000)

    def predict_transformer(self, text: str) -> Dict[str, Any]:
        start = time.time()
        model, tokenizer = self.models["transformer"]
        processed = self.preprocessor.get_tokens_as_string(text)
        encoding = tokenizer(
            processed,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        probabilities = self._apply_keyword_boosts(text, probabilities)
        return self._format_prediction(text, "transformer", probabilities, (time.time() - start) * 1000)

    def predict(self, text: str, model_name: str | None = None) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")

        selected = model_name or self.default_model()
        if selected not in self.models:
            raise ValueError(f"Model '{selected}' is not available. Available models: {list(self.models)}")

        if selected == "baseline":
            return self.predict_baseline(text)
        if selected == "transformer":
            return self.predict_transformer(text)
        raise ValueError(f"Unsupported model: {selected}")


app = create_app()
model_manager: ModelManager | None = None


def manager() -> ModelManager:
    if model_manager is None:
        raise RuntimeError("Models are not initialized")
    return model_manager


@app.errorhandler(400)
def bad_request(error):
    return jsonify({"success": False, "error": "Bad Request", "message": str(error)}), 400


@app.errorhandler(500)
def internal_error(error):
    logger.error("Internal server error: %s\n%s", error, traceback.format_exc())
    return jsonify({"success": False, "error": "Internal Server Error"}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    loaded = manager().available_models()
    return jsonify(
        {
            "success": True,
            "app": APP_NAME,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "device": str(DEVICE),
            "models_loaded": len(loaded),
            "models": list(loaded.keys()),
        }
    )


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify(
        {
            "success": True,
            "models": manager().available_models(),
            "default_model": manager().default_model(),
            "labels": [
                {
                    "category": label,
                    "display_name": LABEL_DISPLAY_NAMES[label],
                    "urgency": URGENCY_BY_FINAL_LABEL[label],
                }
                for label in FINAL_LABELS
            ],
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict_single():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        model_name = data.get("model") or manager().default_model()
        result = manager().predict(text, model_name)

        logger.info(
            "Prediction model=%s len=%s category=%s confidence=%.3f",
            result["model"],
            len(text),
            result["category"],
            result["confidence"],
        )
        return jsonify({"success": True, **result}), 200
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.error("Prediction failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"success": False, "error": "Prediction failed", "message": str(exc)}), 500


@app.route("/api/batch_predict", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json(silent=True) or {}
        texts = data.get("texts")
        model_name = data.get("model") or manager().default_model()
        if not isinstance(texts, list) or not texts:
            return jsonify({"success": False, "error": "texts must be a non-empty list"}), 400

        predictions = []
        for text in texts:
            try:
                predictions.append({"success": True, **manager().predict(text, model_name)})
            except Exception as exc:
                predictions.append({"success": False, "error": str(exc), "text": text})

        return jsonify({"success": True, "predictions": predictions, "total": len(predictions)}), 200
    except Exception as exc:
        logger.error("Batch prediction failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"success": False, "error": str(exc)}), 500


def run_server(host: str = FLASK_HOST, port: int = FLASK_PORT, debug: bool = FLASK_DEBUG) -> None:
    global model_manager

    print("\n" + "=" * 72)
    print(f"{APP_NAME.upper()} API")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print("Loading models...")

    model_manager = ModelManager()
    if not model_manager.models:
        print("No RescueText PH models found. Train a model before starting the API.")
        print("Example: python train_disaster_baseline.py --sample-size 40000")
        sys.exit(1)

    print(f"Loaded models: {', '.join(model_manager.models.keys())}")
    print(f"Starting server on http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"{APP_NAME} API server")
    parser.add_argument("--host", default=FLASK_HOST)
    parser.add_argument("--port", type=int, default=FLASK_PORT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug)
