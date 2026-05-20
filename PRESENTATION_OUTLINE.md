# RescueText PH Full PPT Outline

Target presentation length: 5-10 minutes  
Target Q&A length: 5-10 minutes

## Slide 1: Title Slide

**Title:** RescueText PH  
**Subtitle:** Filipino-English Disaster Relief Post Classification and Urgency Triage System  
**Course:** CCS 249 Final Project  
**Group Members:** Add names here

**Speaker Notes:**  
Introduce the system as an NLP application that helps classify disaster-related social media posts into response categories and urgency levels.

## Slide 2: Problem Statement

During disasters, people post urgent information on social media, such as rescue requests, evacuation updates, damage reports, medical concerns, and donation offers.

However, these posts are:

- Noisy and informal
- Written in English, Filipino, or Taglish
- Time-sensitive
- Mixed with irrelevant or promotional content
- Difficult to manually sort at scale

**Speaker Notes:**  
Emphasize that the problem is not only detecting disaster posts, but organizing them into actionable categories for faster response.

## Slide 3: Purpose and Target Users

**Purpose:**  
RescueText PH automatically classifies disaster-related posts and assigns urgency levels to support faster disaster information triage.

**Target Users:**

- Local Government Units
- Disaster response teams
- Volunteers and donation coordinators
- Social media monitoring teams
- Crisis informatics researchers

**Why NLP is appropriate:**

- Input data is natural language text
- Posts contain meaningful patterns, keywords, and context
- Classification can help prioritize large volumes of messages

**Speaker Notes:**  
Connect directly to the rubric: clear problem, objectives, target users, NLP justification, and real-world relevance.

## Slide 4: System Overview

**System Flow:**

1. User enters a disaster-related post.
2. Text is sent to the Flask backend.
3. Backend preprocesses the text.
4. Trained model predicts the humanitarian category.
5. System derives urgency from the predicted category.
6. Frontend displays category, urgency, confidence, top predictions, and preprocessing details.

**Components:**

- Frontend: HTML, CSS, JavaScript
- Backend: Flask API
- NLP Pipeline: preprocessing + classification
- Models: TF-IDF Logistic Regression baseline and multilingual transformer

**Speaker Notes:**  
Briefly explain that the frontend is for user interaction while the backend contains the NLP methods, satisfying the project requirement.

## Slide 5: Dataset Sources

**Main Dataset:**  
`dataset/processed/disaster_humanitarian_categories.csv`

**Sources integrated:**

- CrisisLexT26
- QCRI/HumAID-events
- QCRI/CrisisBench-all-lang
- SEACrowd Typhoon Yolanda Tweets

**Dataset Size After Import:**

- Master dataset: 393,514 rows
- Humanitarian category dataset: 220,495 rows
- Informativeness dataset: 172,284 rows
- Typhoon Yolanda sentiment dataset: 735 rows

**Speaker Notes:**  
Explain that HumAID and CrisisBench provide humanitarian labels, CrisisLexT26 adds crisis tweet variety, and Typhoon Yolanda supports local relevance even though it is sentiment-labeled.

## Slide 6: Final Classification Categories

The raw humanitarian labels were simplified into 8 final categories:

1. Rescue or Urgent Needs
2. Medical or Casualties
3. Evacuation or Displacement
4. Infrastructure Damage
5. Warnings or Advice
6. Donation or Volunteering
7. General Update
8. Not Humanitarian

**Speaker Notes:**  
Explain that simplifying labels helps the model learn clearer classes and makes the demo easier to understand.

## Slide 7: Urgency Mapping

Urgency is derived from predicted category:

**Critical**

- Rescue or Urgent Needs
- Medical or Casualties

**High**

- Evacuation or Displacement
- Infrastructure Damage

**Moderate**

- Warnings or Advice
- Donation or Volunteering

**Low**

- General Update
- Not Humanitarian

**Speaker Notes:**  
Clarify that urgency is deterministic and rule-based. The model predicts the category, then the system maps that category to urgency.

## Slide 8: Text Preprocessing Pipeline

The system applies a consistent preprocessing pipeline:

1. Remove URLs
2. Remove mentions
3. Preserve hashtag text
4. Normalize whitespace
5. Lowercase text
6. Tokenize words
7. Remove English and Filipino stopwords
8. Apply English lemmatization
9. Apply lightweight Filipino stemming

**Example:**

Input:

> Need rescue sa Brgy. San Isidro, baha na hanggang bubong.

Output tokens:

> need, rescue, brg, san, isidro, hangg, bubo

**Speaker Notes:**  
Mention that preprocessing handles noisy social media text and improves feature extraction for both baseline and transformer models.

## Slide 9: NLP Task

**Primary NLP Task:**  
Multi-class text classification

**Input:**  
A disaster-related social media post

**Output:**  
One of the 8 humanitarian categories

**Additional Output:**  
Urgency level, confidence score, top predictions, and preprocessing preview

**Speaker Notes:**  
This slide directly satisfies the requirement to identify and explain the NLP task.

## Slide 10: Baseline Model

**Model:** TF-IDF + Logistic Regression

**Why this baseline was selected:**

- Fast to train
- Easy to interpret
- Works well for text classification
- Provides a comparison point for advanced models
- Supports class balancing for uneven categories

**Training Setup:**

- Dataset: disaster humanitarian category dataset
- Sample size: 40,000 balanced rows
- Split: 70% train, 15% validation, 15% test
- Features: TF-IDF unigrams and bigrams
- Classifier: Logistic Regression with balanced class weights

**Speaker Notes:**  
Emphasize that the baseline is required for comparison and is stable enough for the live demo.

## Slide 11: Baseline Evaluation

**Baseline Results:**

- Accuracy: 0.7333
- Macro Precision: 0.7336
- Macro Recall: 0.7364
- Macro F1-score: 0.7332

**Strongest classes:**

- Evacuation or Displacement
- Medical or Casualties
- Infrastructure Damage

**Common challenge:**

- General Update can overlap with other categories because disaster updates often contain broad or vague language.

**Speaker Notes:**  
Discuss the confusion matrix and explain that macro F1 is important because the system has multiple categories.

## Slide 12: Transformer Model

**Model:** DistilBERT Multilingual for Sequence Classification

**Why this model was selected:**

- Supports multilingual text
- Better suited for English, Filipino, and Taglish
- Learns contextual meaning beyond keyword frequency
- Fine-tuned locally on the project dataset

**Training Setup:**

- Model: `distilbert-base-multilingual-cased`
- Sample size: 12,000 balanced rows
- Epochs: 1
- Batch size: 8
- Approximate training rows: 8,399
- Approximate training steps: 1,050
- Validation rows: 1,801
- Test rows: 1,800

**Speaker Notes:**  
Explain that 12,000 rows are split into train, validation, and test. The 1,050 steps are batches, not samples.

## Slide 13: Transformer Evaluation

**Fill this after training finishes:**

- Accuracy: `_____`
- Macro Precision: `_____`
- Macro Recall: `_____`
- Macro F1-score: `_____`

**Comparison Table:**

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.7333 | 0.7336 | 0.7364 | 0.7332 |
| Multilingual Transformer | TBD | TBD | TBD | TBD |

**Speaker Notes:**  
Once training is finished, copy results from `outputs/disaster_transformer_evaluation.txt`.

## Slide 14: Error Analysis and Limitations

**Observed/Expected Error Patterns:**

- Rescue and donation posts may both contain words like “help” or “support.”
- Infrastructure and medical posts can overlap when injuries happen near damaged roads or bridges.
- General Update is broad and may absorb vague posts.
- Filipino and Taglish category-labeled data is smaller than English data.

**System Limitations:**

- Urgency is rule-derived from category.
- Location extraction is not yet implemented.
- Transformer training is limited by local compute.
- The system should support, not replace, human responders.

**Speaker Notes:**  
This slide is important for the highest evaluation score because the rubric asks for insightful analysis of errors and limitations.

## Slide 15: Web Application Features

**Frontend Features:**

- Text input for disaster/social media posts
- Model selector
- Prediction result panel
- Category and urgency display
- Confidence score
- Top predictions
- Preprocessing preview
- Demo examples with expected categories
- Category guide

**Backend Features:**

- `/api/health`
- `/api/models`
- `/api/predict`
- `/api/batch_predict`

**Speaker Notes:**  
Show that the system is not just a model script. It is a usable frontend-backend NLP application.

## Slide 16: Live Demo

**Demo Steps:**

1. Open RescueText PH frontend.
2. Select model.
3. Use a prepared example or type a new post.
4. Click Analyze Post.
5. Explain predicted category, urgency, confidence, and preprocessing preview.

**Prepared Examples:**

- Rescue needed:
  > Need rescue sa Brgy. San Isidro, baha na hanggang bubong.

- Medical:
  > Two injured residents near the collapsed bridge need medical assistance immediately.

- Evacuation:
  > Evacuation center at City High School is open.

- Infrastructure:
  > Power lines are down along Mabini Street after the typhoon.

- Not humanitarian:
  > Selling raincoats and flashlights at discounted prices today only.

**Speaker Notes:**  
Keep the demo short. Use 2-3 examples if time is limited.

## Slide 17: Technology Stack

**Frontend:**

- HTML
- CSS
- JavaScript

**Backend:**

- Python
- Flask
- Flask-CORS

**NLP and ML:**

- pandas
- numpy
- scikit-learn
- NLTK
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

**Speaker Notes:**  
Mention that all libraries and modules are disclosed, as required by the project instructions.

## Slide 18: Future Enhancements

Planned improvements:

- Add location extraction for barangays, roads, schools, and cities.
- Add batch triage dashboard sorted by urgency.
- Improve Filipino and Taglish disaster-labeled data.
- Add explainability for model decisions.
- Add CSV export for responders.
- Train transformer longer with better compute resources.

**Speaker Notes:**  
These future enhancements show awareness of the project’s limitations and possible real-world extension.

## Slide 19: Conclusion

RescueText PH demonstrates a complete NLP pipeline:

- Real-world problem
- Relevant crisis datasets
- Text preprocessing
- Feature extraction
- Baseline and transformer model development
- Model evaluation
- Flask backend
- Frontend demo

**Closing Line:**  
RescueText PH helps turn noisy disaster posts into categorized, urgency-aware information that can support faster human response.

## Slide 20: Q&A

Possible questions to prepare for:

1. Why did you choose this project?
2. Why is NLP needed?
3. Why did you simplify the labels?
4. What is the difference between baseline and transformer?
5. Why is urgency rule-based?
6. How accurate is the system?
7. What are the limitations?
8. Can it handle Filipino or Taglish?
9. Why not use GPT or an API?
10. How can this be improved?

## Backup Slide: Why Not GPT?

The project requirements prohibit third-party language models, classifiers, GPTs, and external APIs.

Our system uses:

- Locally trained baseline classifier
- Locally fine-tuned transformer classifier
- No external prediction API

## Backup Slide: Epochs and Steps

In transformer training:

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
