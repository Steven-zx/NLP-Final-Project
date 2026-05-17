"""
Baseline ML Model Training: TF-IDF + Logistic Regression
========================================================

This script trains a traditional machine learning baseline model for hate speech
detection using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
combined with Logistic Regression classification.

This baseline is required for:
1. Rubric comparison (evaluating advanced models against traditional ML)
2. Understanding the impact of neural network approaches
3. Fast inference capability for production use

Pipeline:
1. Load unified bilingual hate speech dataset
2. Split into train/validation/test (70/15/15)
3. Apply text preprocessing (using preprocessing.py)
4. Fit TF-IDF vectorizer on training data
5. Train Logistic Regression classifier
6. Evaluate on test set (Accuracy, Precision, Recall, F1-score)
7. Save pipeline (TF-IDF + LR) using joblib

Model Performance Target:
- Expected Accuracy: ~65-75% (baseline for rubric)
- Inference Speed: <100ms per sample (very fast)

Dependencies:
    - pandas, numpy
    - scikit-learn (TfidfVectorizer, LogisticRegression, metrics)
    - joblib (model serialization)
    - preprocessing (custom module)

Output Files:
    - baseline_model.pkl: Saved TF-IDF + LR pipeline (joblib format)
    - baseline_evaluation.txt: Metrics and confusion matrix
    - baseline_predictions.csv: Test set predictions with confidence
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
import joblib

# Custom preprocessing module
from preprocessing import TextPreprocessor, get_tokens_string


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# TF-IDF Configuration
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
TFIDF_MIN_DF = 2             # Minimum document frequency
TFIDF_MAX_DF = 0.8           # Maximum document frequency (80% of docs)

# Logistic Regression Configuration
LR_MAX_ITER = 1000
LR_C = 1.0                    # Inverse of regularization strength
LR_CLASS_WEIGHT = 'balanced'  # Handle class imbalance

# Data paths
DEFAULT_DATASET_PATH = 'unified_bilingual_hatespeech.csv'
OUTPUT_MODEL_PATH = 'baseline_model.pkl'
OUTPUT_EVAL_PATH = 'baseline_evaluation.txt'
OUTPUT_PREDICTIONS_PATH = 'baseline_predictions.csv'

# Random seed for reproducibility
RANDOM_SEED = 42


# ==============================================================================
# DATASET LOADING & PREPROCESSING
# ==============================================================================

def load_dataset(csv_path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """
    Load the unified bilingual hate speech dataset.
    
    Supports multiple dataset sources combined into a single CSV with
    'text' and 'label' columns (0 = non-hate, 1 = hate speech).
    
    Args:
        csv_path (str): Path to the dataset CSV file
        
    Returns:
        pd.DataFrame: Dataset with 'text' and 'label' columns
        
    Raises:
        FileNotFoundError: If dataset file not found
        ValueError: If required columns missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"Dataset must have 'text' and 'label' columns. Found: {df.columns.tolist()}")
    
    # Remove rows with missing text or label
    initial_count = len(df)
    df = df.dropna(subset=['text', 'label'])
    
    if len(df) < initial_count:
        print(f"  ⚠ Removed {initial_count - len(df)} rows with missing values")
    
    # Ensure label is binary (0 or 1)
    df['label'] = df['label'].astype(int)
    
    print(f"✓ Loaded {len(df)} samples")
    print(f"  - Hate speech: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")
    print(f"  - Non-hate: {len(df) - df['label'].sum()} ({(1-df['label'].mean())*100:.2f}%)")
    
    return df


def preprocess_texts(texts: List[str], processor: TextPreprocessor) -> List[str]:
    """
    Preprocess a batch of texts using the unified preprocessing pipeline.
    
    Args:
        texts (List[str]): List of raw text samples
        processor (TextPreprocessor): Configured preprocessor instance
        
    Returns:
        List[str]: List of preprocessed texts (tokens joined as strings)
    """
    print(f"Preprocessing {len(texts)} texts...")
    preprocessed = []
    
    for idx, text in enumerate(texts):
        if (idx + 1) % 10000 == 0:
            print(f"  [{idx + 1}/{len(texts)}]", end='\r')
        
        processed = processor.get_tokens_as_string(text)
        preprocessed.append(processed)
    
    print(f"✓ Preprocessing complete ({len(texts)} texts)")
    return preprocessed


def split_dataset(df: pd.DataFrame, test_size: float = 0.15, 
                  val_size: float = 0.15, random_state: int = RANDOM_SEED
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.
    
    Uses stratified splitting to maintain class distribution across all splits.
    
    Default split: 70% train, 15% validation, 15% test
    
    Args:
        df (pd.DataFrame): Full dataset
        test_size (float): Proportion for test set (default: 0.15)
        val_size (float): Proportion for validation set (default: 0.15)
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train, val, test) splits
    """
    # First split: separate test set (15%)
    train_val, test = split_stratified(df, test_size=test_size, random_state=random_state)
    
    # Second split: separate validation from train (15% of original = 15/(100-15) ≈ 17.65% of train_val)
    val_split_ratio = val_size / (1 - test_size)
    train, val = split_stratified(train_val, test_size=val_split_ratio, random_state=random_state + 1)
    
    print(f"Split dataset into:")
    print(f"  - Train: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
    print(f"  - Validation: {len(val)} samples ({len(val)/len(df)*100:.1f}%)")
    print(f"  - Test: {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
    
    return train, val, test


def split_stratified(df: pd.DataFrame, test_size: float = 0.15,
                     random_state: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train-test split (maintains class distribution).
    
    Args:
        df (pd.DataFrame): Dataset to split
        test_size (float): Proportion for test set
        random_state (int): Random seed
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train, test) splits
    """
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=df['label'].values
    )
    
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


# ==============================================================================
# MODEL TRAINING & EVALUATION
# ==============================================================================

def train_baseline_model(X_train: List[str], y_train: np.ndarray
                        ) -> Pipeline:
    """
    Train TF-IDF + Logistic Regression baseline model.
    
    Pipeline stages:
    1. TF-IDF Vectorization: Convert text to sparse feature matrix
    2. Logistic Regression: Binary classification with L2 regularization
    
    Args:
        X_train (List[str]): Training texts (preprocessed)
        y_train (np.ndarray): Training labels (0/1)
        
    Returns:
        Pipeline: Fitted sklearn Pipeline containing TF-IDF + LR
    """
    print("\nTraining baseline model (TF-IDF + Logistic Regression)...")
    print(f"  - TF-IDF: max_features={TFIDF_MAX_FEATURES}, ngrams={TFIDF_NGRAM_RANGE}")
    print(f"  - LR: C={LR_C}, class_weight={LR_CLASS_WEIGHT}, max_iter={LR_MAX_ITER}")
    
    start_time = time.time()
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            lowercase=True,
            strip_accents='unicode',
            stop_words=None  # Already handled in preprocessing
        )),
        ('lr', LogisticRegression(
            max_iter=LR_MAX_ITER,
            C=LR_C,
            class_weight=LR_CLASS_WEIGHT,
            solver='lbfgs',
            random_state=RANDOM_SEED,
            n_jobs=-1  # Use all CPU cores
        ))
    ])
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"✓ Model trained in {training_time:.2f}s")
    
    # Print feature statistics
    tfidf = pipeline.named_steps['tfidf']
    print(f"  - TF-IDF vocabulary size: {len(tfidf.get_feature_names_out())}")
    
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: List[str], y_test: np.ndarray,
                   thresholds: List[float] = None) -> Dict:
    """
    Evaluate baseline model on test set with multiple thresholds.
    
    Computes:
    - Raw predictions (probability scores)
    - Thresholded predictions (binary labels for different thresholds)
    - Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
    
    Args:
        pipeline (Pipeline): Trained sklearn Pipeline
        X_test (List[str]): Test texts
        y_test (np.ndarray): Test labels
        thresholds (List[float]): Thresholds to evaluate (default: [0.5])
        
    Returns:
        Dict: Evaluation results including predictions and metrics
    """
    if thresholds is None:
        thresholds = [0.5]
    
    print(f"\nEvaluating baseline model on {len(X_test)} test samples...")
    
    start_time = time.time()
    
    # Get probability predictions
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of hate speech (class 1)
    
    inference_time = time.time() - start_time
    avg_inference_time = (inference_time / len(X_test)) * 1000  # Convert to ms
    
    print(f"✓ Inference completed in {inference_time:.2f}s ({avg_inference_time:.2f}ms per sample)")
    
    # Evaluate at each threshold
    results = {
        'raw_scores': y_proba,
        'test_labels': y_test,
        'inference_time_ms': avg_inference_time,
        'thresholds': {}
    }
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results['thresholds'][threshold] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        print(f"\n  Threshold: {threshold:.2f}")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
    
    return results


def save_model(pipeline: Pipeline, output_path: str = OUTPUT_MODEL_PATH) -> None:
    """
    Save trained model pipeline using joblib.
    
    Args:
        pipeline (Pipeline): Trained pipeline to save
        output_path (str): Output file path
    """
    print(f"\nSaving model to: {output_path}")
    joblib.dump(pipeline, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Model saved ({file_size_mb:.2f}MB)")


def save_evaluation_report(results: Dict, pipeline: Pipeline,
                           output_path: str = OUTPUT_EVAL_PATH,
                           predictions_path: str = OUTPUT_PREDICTIONS_PATH,
                           X_test: List[str] = None, y_test: np.ndarray = None) -> None:
    """
    Save comprehensive evaluation report to file.
    
    Args:
        results (Dict): Evaluation results from evaluate_model()
        pipeline (Pipeline): Trained pipeline
        output_path (str): Report file path
        predictions_path (str): Predictions CSV path
        X_test (List[str]): Test texts (for saving predictions)
        y_test (np.ndarray): Test labels (for saving predictions)
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BASELINE MODEL EVALUATION REPORT\n")
        f.write("TF-IDF + Logistic Regression\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Set Size: {len(y_test)} samples\n")
        f.write(f"Average Inference Time: {results['inference_time_ms']:.2f}ms per sample\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("THRESHOLD-BASED EVALUATION\n")
        f.write("-" * 80 + "\n\n")
        
        for threshold, metrics in results['thresholds'].items():
            f.write(f"THRESHOLD: {threshold:.2f}\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
            
            cm = metrics['confusion_matrix']
            f.write(f"\n  Confusion Matrix:\n")
            f.write(f"    True Negatives:  {cm[0, 0]}\n")
            f.write(f"    False Positives: {cm[0, 1]}\n")
            f.write(f"    False Negatives: {cm[1, 0]}\n")
            f.write(f"    True Positives:  {cm[1, 1]}\n")
            
            f.write(f"\n  Class Distribution (Predicted):\n")
            f.write(f"    Non-Hate: {(metrics['predictions'] == 0).sum()} samples\n")
            f.write(f"    Hate:     {(metrics['predictions'] == 1).sum()} samples\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"TF-IDF Max Features: {TFIDF_MAX_FEATURES}\n")
        f.write(f"TF-IDF N-gram Range: {TFIDF_NGRAM_RANGE}\n")
        f.write(f"TF-IDF Min DF: {TFIDF_MIN_DF}\n")
        f.write(f"TF-IDF Max DF: {TFIDF_MAX_DF}\n")
        f.write(f"LR Regularization (C): {LR_C}\n")
        f.write(f"LR Class Weight: {LR_CLASS_WEIGHT}\n")
        f.write(f"LR Max Iterations: {LR_MAX_ITER}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
    
    print(f"✓ Evaluation report saved to: {output_path}")
    
    # Save predictions CSV
    if X_test is not None and y_test is not None:
        predictions_df = pd.DataFrame({
            'text': X_test,
            'true_label': y_test,
            'confidence_hate': results['raw_scores'],
            'predicted_label_05': results['thresholds'][0.5]['predictions'],
            'predicted_label_08': results['thresholds'].get(0.8, {}).get('predictions', None),
        })
        
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✓ Predictions saved to: {predictions_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main(dataset_path: str = DEFAULT_DATASET_PATH,
         model_output_path: str = OUTPUT_MODEL_PATH,
         eval_output_path: str = OUTPUT_EVAL_PATH,
         predictions_output_path: str = OUTPUT_PREDICTIONS_PATH) -> None:
    """
    Main execution pipeline for training baseline model.
    
    Args:
        dataset_path (str): Path to unified dataset CSV
        model_output_path (str): Output path for model pickle
        eval_output_path (str): Output path for evaluation report
        predictions_output_path (str): Output path for predictions CSV
    """
    print("\n" + "=" * 80)
    print("BASELINE MODEL TRAINING: TF-IDF + LOGISTIC REGRESSION")
    print("=" * 80 + "\n")
    
    try:
        # Step 1: Load dataset
        print("[Step 1/6] Loading dataset...")
        df = load_dataset(dataset_path)
        
        # Step 2: Initialize preprocessor
        print("\n[Step 2/6] Initializing text preprocessor...")
        processor = TextPreprocessor(remove_stopwords=True, lemmatize=True, stem=True)
        print("✓ Preprocessor initialized")
        
        # Step 3: Split dataset
        print("\n[Step 3/6] Splitting dataset (70/15/15)...")
        train_df, val_df, test_df = split_dataset(df)
        
        # Step 4: Preprocess texts
        print("\n[Step 4/6] Preprocessing texts...")
        X_train = preprocess_texts(train_df['text'].tolist(), processor)
        X_test = preprocess_texts(test_df['text'].tolist(), processor)
        X_val = preprocess_texts(val_df['text'].tolist(), processor)
        
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        # Step 5: Train model
        print("\n[Step 5/6] Training baseline model...")
        pipeline = train_baseline_model(X_train, y_train)
        
        # Step 6: Evaluate model
        print("\n[Step 6/6] Evaluating model...")
        # Test on multiple thresholds for comprehensive evaluation
        results = evaluate_model(pipeline, X_test, y_test,
                                thresholds=[0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Save outputs
        print("\n[Saving] Saving model and reports...")
        save_model(pipeline, model_output_path)
        save_evaluation_report(results, pipeline, eval_output_path,
                              predictions_output_path, X_test, y_test)
        
        print("\n" + "=" * 80)
        print("✓ BASELINE MODEL TRAINING COMPLETE")
        print("=" * 80)
        print(f"\nGenerated files:")
        print(f"  - Model: {model_output_path}")
        print(f"  - Report: {eval_output_path}")
        print(f"  - Predictions: {predictions_output_path}")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # Optional: Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Train baseline hate speech detection model (TF-IDF + LR)'
    )
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_PATH,
                        help=f'Path to dataset CSV (default: {DEFAULT_DATASET_PATH})')
    parser.add_argument('--model-output', type=str, default=OUTPUT_MODEL_PATH,
                        help=f'Output path for model (default: {OUTPUT_MODEL_PATH})')
    parser.add_argument('--eval-output', type=str, default=OUTPUT_EVAL_PATH,
                        help=f'Output path for evaluation report (default: {OUTPUT_EVAL_PATH})')
    
    args = parser.parse_args()
    
    main(dataset_path=args.dataset,
         model_output_path=args.model_output,
         eval_output_path=args.eval_output)
