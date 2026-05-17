# RescueText PH Presentation Outline

Target length: 5-10 minutes, followed by 5-10 minutes of Q&A.

## Slide 1: Title

**RescueText PH: Filipino-English Disaster Relief Post Classification and Urgency Triage**

Introduce the group and state that the system classifies noisy disaster-related posts into response categories.

## Slide 2: Problem Statement and Purpose

During disasters, people post urgent needs, warnings, damage reports, and donation offers on social media. These posts are noisy, multilingual, and time-sensitive.

Purpose: automatically categorize disaster posts and assign urgency so responders can prioritize information faster.

## Slide 3: Target Users and Real-World Relevance

Target users:

- LGUs
- Disaster responders
- Volunteers
- Social media monitoring teams
- Crisis informatics researchers

Real-world value: helps organize public disaster messages during typhoons, floods, earthquakes, and similar emergencies.

## Slide 4: Training Corpora

Datasets used:

- CrisisLexT26
- HumAID-events
- CrisisBench-all-lang
- Typhoon Yolanda Filipino tweets for local relevance discussion

Explain why these corpora are appropriate: they contain real crisis/disaster social media text with humanitarian and informativeness labels.

## Slide 5: Text Preprocessing

Pipeline:

- URL removal
- Mention removal
- Hashtag text preservation
- Lowercasing
- Tokenization
- English and Filipino stopword removal
- English lemmatization
- Lightweight Filipino stemming

Show one before-and-after example.

## Slide 6: NLP Task

Task: multi-class text classification.

Final labels:

- Rescue or Urgent Needs
- Medical or Casualties
- Evacuation or Displacement
- Infrastructure Damage
- Warnings or Advice
- Donation or Volunteering
- General Update
- Not Humanitarian

Urgency is derived from predicted category.

## Slide 7: Model Training and Development

Baseline model:

- TF-IDF features
- Logistic Regression
- Class balancing

Main model:

- Multilingual transformer fine-tuning script
- Locally trained classifier head
- No GPT or third-party classifier API

## Slide 8: Evaluation

Current baseline evaluation:

- Accuracy: 0.7333
- Macro Precision: 0.7336
- Macro Recall: 0.7364
- Macro F1-score: 0.7332

Discuss confusion matrix and expected error patterns.

## Slide 9: Demo

Demo flow:

1. Start Flask backend.
2. Open `index.html`.
3. Enter or choose a sample disaster post.
4. Show predicted category, urgency, confidence, top predictions, and preprocessing preview.

Prepared examples:

- Rescue needed
- Medical assistance
- Evacuation center
- Infrastructure damage
- Not humanitarian

## Slide 10: Limitations and Roadmap

Known limitations:

- Less Filipino/Taglish category-labeled data than English data
- Urgency is rule-derived
- No location extraction yet
- Transformer needs training time/GPU for best result

Roadmap:

- Add location extraction
- Improve Filipino/Taglish data
- Add responder dashboard and CSV export
- Add explainability features
