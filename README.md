# RescueText PH

Filipino-English disaster relief post classification and urgency triage system for CCS 249.

RescueText PH accepts a disaster-related social media post, classifies it into a humanitarian response category, derives an urgency level, and displays confidence, top predictions, and preprocessing details.

## Project Fit

This project follows the CCS 249 final project requirements:

- NLP preprocessing and feature extraction
- Model training and development
- Baseline model comparison
- Model evaluation
- Flask backend API
- Frontend interface
- Software documentation
- Presentation/demo outline

No GPT, third-party classifier API, or external prediction service is used.

## Main Dataset

The main training file is:

```text
dataset/processed/disaster_humanitarian_categories.csv
```

It is generated from:

- CrisisLexT26
- QCRI/HumAID-events
- QCRI/CrisisBench-all-lang
- SEACrowd Typhoon Yolanda Tweets for local relevance support

To regenerate processed datasets:

```bash
python scripts/import_disaster_datasets.py
```

## Labels

The final 8-class taxonomy is:

- `rescue_or_urgent_needs`
- `medical_or_casualties`
- `evacuation_or_displacement`
- `infrastructure_damage`
- `warnings_or_advice`
- `donation_or_volunteering`
- `general_update`
- `not_humanitarian`

Urgency is derived from the predicted category:

- `critical`
- `high`
- `moderate`
- `low`

## Train Models

Train the baseline model:

```bash
python train_disaster_baseline.py --sample-size 40000
```

Train the transformer model:

```bash
python train_disaster_transformer.py --sample-size 24000 --epochs 1
```

The transformer model can be several hundred MB. It is ignored by `.gitignore` by default so GitHub will not reject the push. Use Git LFS only if the final transformer artifact must be committed.

Baseline artifacts are saved to:

```text
models/disaster_baseline.pkl
outputs/disaster_baseline_evaluation.txt
outputs/disaster_baseline_predictions.csv
```

Transformer artifacts are saved to:

```text
models/disaster_transformer/
outputs/disaster_transformer_evaluation.txt
outputs/disaster_transformer_predictions.csv
```

## Run the App

Start the backend:

```bash
python app.py
```

Open `index.html` in a browser.

Main API endpoints:

- `GET /api/health`
- `GET /api/models`
- `POST /api/predict`
- `POST /api/batch_predict`

Example request:

```json
{
  "text": "Need rescue sa Brgy. San Isidro, baha na hanggang bubong",
  "model": "baseline"
}
```

Example response:

```json
{
  "success": true,
  "category": "rescue_or_urgent_needs",
  "urgency": "critical",
  "confidence": 0.94,
  "top_predictions": [],
  "preprocessing": {
    "original_text": "...",
    "cleaned_text": "...",
    "tokens": []
  },
  "model": "baseline",
  "inference_time_ms": 25.0
}
```

## Evaluation

Current baseline result using a balanced 40,000-row sample:

- Accuracy: `0.7333`
- Macro Precision: `0.7336`
- Macro Recall: `0.7364`
- Macro F1-score: `0.7332`

See:

```text
outputs/disaster_baseline_evaluation.txt
```

## Documentation

- `SOFTWARE_DOCUMENTATION.md`
- `PRESENTATION_OUTLINE.md`
- `dataset/README.md`
