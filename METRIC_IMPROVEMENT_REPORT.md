# RescueText PH Metric Improvement Report

## What Changed

The final 8-class baseline was improved through cleaner label handling, stricter training data filtering, and a larger classical model comparison.

Main changes:

- Built `dataset/processed/clean_training_subset.csv`.
- Removed short low-information texts and plain retweets.
- Removed normalized duplicate texts.
- Remapped or dropped ambiguous source labels:
  - `affected_individuals` is no longer treated as automatically medical.
  - `response_efforts` is treated as response/donation-related when appropriate.
  - `rescue_volunteering_or_donation_effort` is split using rescue vs donation keywords.
  - `terrorism_related` is dropped from the clean subset.
- Added more baseline candidates:
  - Complement Naive Bayes
  - SGD logistic classifier
  - calibrated LinearSVC
  - word and character TF-IDF variants
- Added a secondary binary actionability classifier.

## Clean Dataset Summary

- Original category rows: 220,495
- Rows after quality filters: 164,408
- Rows after normalized deduplication: 164,075
- Duplicate normalized texts removed: 333
- Minimum token threshold: 4

## Final 8-Class Baseline Improvement

Previous final baseline:

- Selected model: `tfidf_word_char_linearsvc`
- Accuracy: 0.7360
- Macro F1: 0.7459

Improved final baseline:

- Selected model: `tfidf_word_bigram_calibrated_linearsvc`
- Accuracy: 0.7600
- Macro Precision: 0.7654
- Macro Recall: 0.7733
- Macro F1: 0.7689
- Weighted F1: 0.7576

Improvement:

- Accuracy: +0.0240
- Macro F1: +0.0230

## Secondary Actionability Model

The secondary task predicts whether a post is actionable for disaster triage.

- Actionable: rescue, medical, evacuation, infrastructure, warnings, donations
- Non-actionable: general update, not humanitarian

Result:

- Selected model: `actionability_word_char_sgd`
- Accuracy: 0.8683
- Macro Precision: 0.8535
- Macro Recall: 0.8690
- Macro F1: 0.8596
- Weighted F1: 0.8697

This did not honestly reach 90%, so the project should not claim 90%+. It is still a strong secondary result and useful for explaining that binary triage is easier than fine-grained 8-class categorization.

## Final Decision

Use the improved clean 8-class baseline as the main final app model:

- `models/disaster_baseline.pkl`
- `outputs/disaster_baseline_evaluation.txt`

Use the actionability model as an additional app output:

- `models/actionability_baseline.pkl`
- `outputs/actionability/actionability_evaluation.txt`

Do not replace the final model with the expanded dataset experiment, because the expanded baseline macro F1 was lower.
