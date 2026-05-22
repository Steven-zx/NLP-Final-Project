# TulongText PH

Filipino-English disaster relief post classification and urgency triage system for CCS 249.

TulongText PH accepts a disaster-related social media post, classifies it into a humanitarian response category, derives an urgency level, and displays confidence, top predictions, and preprocessing details.
It also includes a secondary binary actionability classifier that estimates whether a post is actionable for disaster triage.

## Project Fit

This project follows the CCS 249 final project requirements:

- NLP preprocessing and feature extraction
- Model training and development
- Baseline model comparison
- Model evaluation
- Secondary actionability classification
- Flask backend API
- Frontend interface
- Software documentation
- Presentation/demo outline

No GPT, third-party classifier API, or external prediction service is used. The transformer option uses a pretrained DistilBERT checkpoint fine-tuned locally inside this project, which keeps prediction offline and reproducible.

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

To generate the optional expanded dataset without overwriting the stable original files:

```bash
python scripts/import_disaster_datasets.py --include-disaster-response-messages --include-crisismmd
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
python scripts/create_clean_training_subset.py
python train_disaster_baseline.py --dataset dataset/processed/clean_training_subset.csv --sample-size 80000
```

Train the baseline on the expanded dataset:

```bash
python train_disaster_baseline.py --dataset dataset/processed/disaster_humanitarian_categories_expanded.csv --sample-size 80000
```

Train the secondary actionability model:

```bash
python train_actionability.py --dataset dataset/processed/clean_training_subset.csv
```

Train the transformer model:

```bash
python train_disaster_transformer.py --sample-size 24000 --epochs 1
```

The script can run with `--sample-size 24000`, but the reported final transformer evaluation below used `--sample-size 12000 --epochs 1` because of local compute limits.

The transformer model can be several hundred MB. It is ignored by `.gitignore` by default so GitHub will not reject the push. Use Git LFS only if the final transformer artifact must be committed.

Baseline artifacts are saved to:

```text
models/disaster_baseline.pkl
outputs/disaster_baseline_evaluation.txt
outputs/disaster_baseline_predictions.csv
outputs/disaster_baseline_model_comparison.csv
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
  "actionability": {
    "label": "actionable",
    "display_name": "Actionable",
    "confidence": 0.91
  },
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

Current improved baseline result using a clean balanced 80,000-row sample:

- Selected model: `tfidf_word_bigram_calibrated_linearsvc`
- Accuracy: `0.7600`
- Macro Precision: `0.7654`
- Macro Recall: `0.7733`
- Macro F1-score: `0.7689`
- Weighted F1-score: `0.7576`

Secondary actionability result:

- Selected model: `actionability_word_char_sgd`
- Accuracy: `0.8683`
- Macro Precision: `0.8535`
- Macro Recall: `0.8690`
- Macro F1-score: `0.8596`
- Weighted F1-score: `0.8697`

Current transformer result using a balanced 12,000-row sample for 1 epoch:

- Accuracy: `0.6661`
- Macro Precision: `0.6600`
- Macro Recall: `0.6661`
- Macro F1-score: `0.6608`

See:

```text
outputs/disaster_baseline_evaluation.txt
outputs/disaster_transformer_evaluation.txt
DATASET_EXPANSION_REPORT.md
METRIC_IMPROVEMENT_REPORT.md
```

## Documentation

- `SOFTWARE_DOCUMENTATION.md`
- `PRESENTATION_OUTLINE.md`
- `dataset/README.md`
