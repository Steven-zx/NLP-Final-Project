# RescueText PH PPT Outline

Target main presentation length: 5-10 minutes  
Recommended main deck: 14 slides  
Backup slides: use only during Q&A

## Slide 1: Title

**Title:** RescueText PH  
**Subtitle:** Filipino-English Disaster Relief Post Classification and Urgency Triage System  
**Course:** CCS 249 Final Project  
**Group Members:** Add names here

**Speaker Notes:**  
Introduce RescueText PH as an NLP system that turns disaster-related posts into response categories and urgency levels.

## Slide 2: Problem and Motivation

During disasters, people post rescue requests, evacuation updates, medical concerns, damage reports, warnings, and donation offers on social media.

These posts are:

- Noisy and informal
- Written in English, Filipino, or Taglish
- Time-sensitive
- Mixed with irrelevant or promotional content
- Difficult to manually sort at scale

**Speaker Notes:**  
Emphasize that the project is not only detecting disaster posts. It organizes them into actionable response categories.

## Slide 3: Project Goal and Users

**Goal:**  
Classify disaster-related social media posts into humanitarian categories and assign an urgency level for faster triage.

**Target Users:**

- Local Government Units
- Disaster response teams
- Volunteers and donation coordinators
- Social media monitoring teams
- Crisis informatics researchers

**Why NLP is appropriate:**

- Input is natural language text
- Posts contain meaningful keywords, context, and intent
- Automated classification can prioritize large message volumes

**Speaker Notes:**  
Connect this directly to the rubric: clear objective, target users, NLP justification, and real-world value.

## Slide 4: System Overview

**System Flow:**

1. User enters a disaster-related post.
2. Flask backend preprocesses the text.
3. Model predicts the humanitarian category.
4. System maps the category to urgency.
5. Frontend displays category, urgency, confidence, top predictions, and preprocessing details.

**Components:**

- Frontend: HTML, CSS, JavaScript
- Backend: Flask API
- NLP pipeline: preprocessing + classification
- Models: tuned TF-IDF baseline and locally fine-tuned multilingual transformer

**Speaker Notes:**  
This shows the project is a working end-to-end NLP application, not just a notebook.

## Slide 5: Datasets

**Main training file:**  
`dataset/processed/disaster_humanitarian_categories.csv`

**Sources integrated:**

- CrisisLexT26
- QCRI/HumAID-events
- QCRI/CrisisBench-all-lang
- SEACrowd Typhoon Yolanda Tweets

**Dataset sizes after import:**

- Master dataset: 393,514 rows
- Humanitarian category dataset: 220,495 rows
- Informativeness dataset: 172,284 rows
- Typhoon Yolanda sentiment dataset: 735 rows

**Speaker Notes:**  
Explain that HumAID and CrisisBench provide category labels, CrisisLexT26 adds crisis tweet variety, and Typhoon Yolanda supports Philippine relevance but is not the main category-training source because it is sentiment-labeled.

## Slide 6: Labels and Urgency Mapping

**Final 8 categories:**

1. Rescue or Urgent Needs
2. Medical or Casualties
3. Evacuation or Displacement
4. Infrastructure Damage
5. Warnings or Advice
6. Donation or Volunteering
7. General Update
8. Not Humanitarian

**Urgency mapping:**

| Urgency | Categories |
|---|---|
| Critical | Rescue or Urgent Needs, Medical or Casualties |
| High | Evacuation or Displacement, Infrastructure Damage |
| Moderate | Warnings or Advice, Donation or Volunteering |
| Low | General Update, Not Humanitarian |

**Speaker Notes:**  
Clarify that the model predicts the category. Urgency is a transparent deterministic layer derived from that category.

## Slide 7: Text Preprocessing

The system applies:

- URL and mention removal
- Hashtag text preservation
- Lowercasing
- Tokenization
- English and Filipino stopword removal
- English lemmatization
- Lightweight Filipino stemming

**Example:**

Input:

> Need rescue sa Brgy. San Isidro, baha na hanggang bubong.

Output tokens:

> need, rescue, brg, san, isidro, hangg, bubo

**Speaker Notes:**  
Mention that preprocessing makes noisy social media text more usable for TF-IDF and transformer training.

## Slide 8: NLP Task

**Primary NLP task:**  
Multi-class text classification

**Input:**  
A disaster-related social media post

**Output:**  
One of 8 humanitarian categories

**Additional outputs:**

- Urgency level
- Confidence score
- Top predictions
- Preprocessing preview

**Speaker Notes:**  
This slide directly satisfies the requirement to identify and explain the NLP task.

## Slide 9: Baseline Development

**Baseline approach:**  
Tuned classical TF-IDF models with class balancing.

**Compared variants:**

- Word bigram TF-IDF + Logistic Regression
- Word trigram TF-IDF + Logistic Regression
- Word trigram TF-IDF + LinearSVC
- Word + character TF-IDF + LinearSVC
- ComplementNB, SGD, and calibrated LinearSVC variants

**Selected baseline:**  
Word bigram TF-IDF + calibrated LinearSVC

**Reason selected:**  
Highest validation macro F1.

**Speaker Notes:**  
This is stronger than a single baseline because it shows model development, comparison, and validation-based selection.

## Slide 10: Model Evaluation

**Baseline final test results, clean balanced 80,000-row sample:**

- Accuracy: 0.7600
- Macro Precision: 0.7654
- Macro Recall: 0.7733
- Macro F1-score: 0.7689
- Weighted F1-score: 0.7576

**Secondary actionability result:**

- Accuracy: 0.8683
- Macro Precision: 0.8535
- Macro Recall: 0.8690
- Macro F1-score: 0.8596

**Transformer final test results, balanced 12,000-row sample, 1 epoch:**

- Accuracy: 0.6661
- Macro Precision: 0.6600
- Macro Recall: 0.6661
- Macro F1-score: 0.6608

**Comparison:**

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Improved TF-IDF + calibrated LinearSVC | 0.7600 | 0.7654 | 0.7733 | 0.7689 |
| Binary actionability model | 0.8683 | 0.8535 | 0.8690 | 0.8596 |
| Multilingual DistilBERT | 0.6661 | 0.6600 | 0.6661 | 0.6608 |

**Speaker Notes:**  
Explain that macro F1 matters because categories are uneven. The baseline is currently stronger because the transformer was limited to a small local run.

## Slide 11: Transformer Notes and API Rule

**Transformer model:**  
`distilbert-base-multilingual-cased`

**Training setup:**

- Locally fine-tuned on the project dataset
- Final reported run: `--sample-size 12000 --epochs 1`
- Batch size: 8
- About 8,399 training rows
- About 1,050 training steps

**Important compliance note:**

- No GPT
- No third-party classifier API
- No external prediction service
- Pretrained DistilBERT is fine-tuned and served locally

**Speaker Notes:**  
Mention that the script can run with `--sample-size 24000`, but the reported result used 12,000 because of local compute limits.

## Slide 12: Error Analysis and Limitations

**Observed challenges:**

- Rescue and donation posts may both contain words like "help" or "support."
- General Update can overlap with many categories.
- Infrastructure and medical reports can overlap when injuries happen near damaged roads or buildings.
- Filipino and Taglish category-labeled data is smaller than English crisis data.

**Limitations:**

- Urgency is rule-derived from category.
- Location extraction is not yet implemented.
- Transformer training was limited by local compute.
- The system should support, not replace, human responders.

**Speaker Notes:**  
This slide is important for high marks because it shows critical analysis instead of only reporting accuracy.

## Slide 13: Web App and Live Demo

**Frontend features:**

- Text input
- Model selector
- Prediction result panel
- Category and urgency display
- Confidence and top predictions
- Actionability output
- Preprocessing preview
- Demo examples and category guide

**Backend endpoints:**

- `GET /api/health`
- `GET /api/models`
- `POST /api/predict`
- `POST /api/batch_predict`

**Demo examples:**

- `Need rescue sa Brgy. San Isidro, baha na hanggang bubong.`
- `Two injured residents near the collapsed bridge need medical assistance immediately.`
- `Evacuation center at City High School is open.`
- `Selling raincoats and flashlights at discounted prices today only.`

**Speaker Notes:**  
Use only 2-3 examples during the actual presentation to stay within time.

## Slide 14: Conclusion

RescueText PH demonstrates a complete NLP pipeline:

- Real-world disaster response problem
- Multiple crisis datasets
- Text preprocessing
- Multi-class classification
- Baseline comparison and transformer experiment
- Evaluation with precision, recall, F1, and confusion matrix
- Flask backend and browser frontend

**Closing line:**  
RescueText PH helps turn noisy disaster posts into categorized, urgency-aware information that can support faster human response.

## Backup Slide: Technology Stack

**Frontend:** HTML, CSS, JavaScript  
**Backend:** Python, Flask, Flask-CORS  
**NLP and ML:** pandas, numpy, scikit-learn, NLTK, PyTorch, Hugging Face Transformers, Hugging Face Datasets

## Backup Slide: Baseline Validation Comparison

| Baseline Variant | Validation Macro F1 |
|---|---:|
| Word bigram TF-IDF + calibrated LinearSVC | 0.7673 |
| Word bigram TF-IDF + SGD logistic classifier | 0.7656 |
| Word bigram TF-IDF + Logistic Regression | 0.7635 |
| Word trigram TF-IDF + Logistic Regression | 0.7624 |
| Word + character TF-IDF + LinearSVC | 0.7606 |

## Backup Slide: Epochs and Steps

- `sample-size 12000` means 12,000 total selected rows.
- The data is split into train, validation, and test.
- Around 8,399 rows are used for training.
- With batch size 8, one training step processes 8 rows.
- 8,399 divided by 8 is about 1,050 steps.
- 1 epoch means the model sees the whole training split once.

## Backup Slide: Files to Mention

- `dataset/processed/disaster_humanitarian_categories.csv`
- `scripts/import_disaster_datasets.py`
- `train_disaster_baseline.py`
- `train_disaster_transformer.py`
- `app.py`
- `index.html`
- `main.js`
- `SOFTWARE_DOCUMENTATION.md`
- `outputs/disaster_baseline_evaluation.txt`
- `outputs/disaster_transformer_evaluation.txt`
