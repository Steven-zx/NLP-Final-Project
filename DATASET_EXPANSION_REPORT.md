# TulongText PH Dataset Expansion Report

## What Was Added

The importer now supports optional expansion sources without overwriting the stable original dataset:

```bash
python scripts/import_disaster_datasets.py --include-disaster-response-messages --include-crisismmd
```

New expanded files:

- `dataset/processed/disaster_humanitarian_categories_expanded.csv`
- `dataset/processed/disaster_informativeness_expanded.csv`
- `dataset/processed/typhoon_yolanda_sentiment_expanded.csv`
- `dataset/processed/disaster_posts_master_expanded.csv.gz`
- `dataset/processed/disaster_dataset_summary_expanded.json`

## Expanded Dataset Checks

- Humanitarian category rows: 225,568
- Missing text values: 0
- Missing category values: 0
- Missing final label values after TulongText PH mapping: 0
- Invalid final label values: 0
- Exact normalized duplicate texts: 0

Added source rows after duplicate protection:

- `HFAbrar/disaster_response_messages`: 5,469
- `QCRI/CrisisMMD/humanitarian`: 38

CrisisMMD contributed few unique rows because CrisisBench already consolidates overlapping crisis datasets.

## Expanded Baseline Experiment

Command:

```bash
python train_disaster_baseline.py --dataset dataset/processed/disaster_humanitarian_categories_expanded.csv --sample-size 80000 --model-output models/disaster_baseline_expanded.pkl --output-dir outputs/expanded_baseline
```

Result:

- Selected model: `tfidf_word_bigram_logreg`
- Accuracy: 0.7197
- Macro F1-score: 0.7244

Current final baseline for comparison:

- Selected model: `tfidf_word_char_linearsvc`
- Accuracy: 0.7360
- Macro F1-score: 0.7459

## Decision

The expanded dataset is useful for future cleaning and label coverage experiments, but it should not replace the current final demo dataset yet because the first expanded 8-class baseline score is lower.

For the final project, use:

- Main final model: `models/disaster_baseline.pkl`
- Main final dataset: `dataset/processed/disaster_humanitarian_categories.csv`
- Expansion evidence: this report and the `_expanded` processed files

This is a strong result academically because it shows that the project tested dataset expansion honestly and chose the better-performing dataset instead of assuming more data always improves performance.
