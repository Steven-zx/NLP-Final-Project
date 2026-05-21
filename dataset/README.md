# Disaster Response Dataset Imports

This project uses `scripts/import_disaster_datasets.py` to normalize crisis and disaster-response datasets for the RescueText PH idea.

Run from the repository root:

```bash
python scripts/import_disaster_datasets.py
```

The script imports:

- Local `dataset/CrisisLexT26`
- Hugging Face `QCRI/HumAID-events`
- Hugging Face `QCRI/CrisisBench-all-lang`
- Filipino Typhoon Yolanda sentiment tweets from the original SEACrowd source URLs

Optional expansion imports are available for label-compatible datasets:

- Hugging Face `HFAbrar/disaster_response_messages`
- Hugging Face `QCRI/CrisisMMD`
- TREC-IS is documented as a future extension because its import and label mapping are more complex.

Generated files are written to `dataset/processed/`:

- `disaster_humanitarian_categories.csv`: main triage/category training file
- `disaster_informativeness.csv`: informative vs not-informative training file
- `typhoon_yolanda_sentiment.csv`: Filipino Typhoon Yolanda sentiment data
- `disaster_posts_master.csv.gz`: compressed combined dataset
- `disaster_dataset_summary.json`: row counts, label IDs, and urgency mapping

By default, CrisisBench is filtered to English and Philippine-related language codes:

```bash
python scripts/import_disaster_datasets.py --crisisbench-languages en,tl,fil,ceb
```

To import every CrisisBench language:

```bash
python scripts/import_disaster_datasets.py --crisisbench-languages all
```

## Expanded Dataset

To create an expanded dataset without overwriting the stable original files:

```bash
python scripts/import_disaster_datasets.py --include-disaster-response-messages --include-crisismmd
```

This writes separate `_expanded` files:

- `disaster_humanitarian_categories_expanded.csv`
- `disaster_informativeness_expanded.csv`
- `typhoon_yolanda_sentiment_expanded.csv`
- `disaster_posts_master_expanded.csv.gz`
- `disaster_dataset_summary_expanded.json`

The original `disaster_humanitarian_categories.csv` is kept unchanged for reproducibility.

## Expansion Label Mapping

Disaster Response Messages is mapped into the RescueText PH taxonomy using priority rules:

- `search_and_rescue`, `request`, `missing_people` -> rescue or urgent needs
- `medical_help`, `medical_products`, `hospitals`, `death` -> medical or casualties
- `shelter`, `refugees`, `aid_centers` -> evacuation or displacement
- `infrastructure_related`, `transport`, `buildings`, `electricity`, `other_infrastructure` -> infrastructure damage
- `weather_related`, `floods`, `storm`, `fire`, `earthquake`, `other_weather` -> warnings or advice
- `water`, `food`, `clothing`, `money`, `offer`, `other_aid` -> donation or volunteering
- `related=0` -> not humanitarian
- fallback related messages -> general update

CrisisMMD humanitarian labels are normalized through the same crisis-category mapping used for HumAID and CrisisBench. CrisisMMD damage labels are imported as an optional damage-focused source, but duplicate protection may remove most rows because CrisisBench already consolidates CrisisMMD-derived content.

Expanded import uses normalized-text duplicate removal so the same message is not repeated across sources.

## Current Expanded Import Check

Latest expanded import:

- Humanitarian category rows: 225,568
- Added source rows after duplicate protection:
  - `HFAbrar/disaster_response_messages`: 5,469
  - `QCRI/CrisisMMD/humanitarian`: 38
- Missing text/category/final label values: 0
- Exact normalized duplicate texts in the expanded humanitarian file: 0
