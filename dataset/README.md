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
