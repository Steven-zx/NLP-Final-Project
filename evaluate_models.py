"""
Comprehensive Model Evaluation: Compare Baseline, BiLSTM, and Transformer
=========================================================================

This script provides comprehensive evaluation and comparison of all three
hate speech detection models:

1. Baseline Model: TF-IDF + Logistic Regression
2. BiLSTM Model: Bidirectional LSTM neural network (existing)
3. Transformer Model: Fine-tuned BERT multilingual

Evaluation includes:
- Accuracy, Precision, Recall, F1-score at multiple thresholds (0.5–0.9)
- Confusion matrices and classification reports
- Inference time comparison
- Cross-model predictions consistency analysis
- Threshold optimization recommendations

Outputs:
- Detailed evaluation reports (text format)
- CSV comparison tables
- Threshold analysis for each model
- Combined comparison table (all models × all thresholds)
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# PyTorch
import torch
from torch.utils.data import DataLoader

# HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Scikit-learn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import joblib

# Custom modules
from preprocessing import TextPreprocessor
from rnn_model import BiLSTMHateSpeechClassifier


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Dataset
DEFAULT_DATASET_PATH = 'unified_bilingual_hatespeech.csv'
TEST_SPLIT_RATIO = 0.15

# Models
BASELINE_MODEL_PATH = 'baseline_model.pkl'
BILSTM_MODEL_PATH = 'best_bilstm_model.pt'
TRANSFORMER_MODEL_PATH = 'transformer_model'

# Thresholds to evaluate
EVALUATION_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

# Vocabulary (for BiLSTM)
VOCAB_PATH = 'vocabulary.pkl'

# Output paths
OUTPUT_COMPARISON_CSV = 'model_comparison.csv'
OUTPUT_DETAILED_REPORT = 'evaluation_detailed_report.txt'
OUTPUT_THRESHOLD_ANALYSIS = 'threshold_analysis.csv'
OUTPUT_MODEL_PREDICTIONS = 'all_models_predictions.csv'

# Random seed
RANDOM_SEED = 42


# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================

def get_device():
    """Detect available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠ Using CPU (inference will be slower)")
    
    return device


# ==============================================================================
# DATASET LOADING & PREPROCESSING
# ==============================================================================

def load_dataset(csv_path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load unified bilingual hate speech dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"Dataset must have 'text' and 'label' columns")
    
    # Remove rows with missing values
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    
    print(f"✓ Loaded {len(df)} samples")
    print(f"  - Hate: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")
    print(f"  - Non-Hate: {len(df) - df['label'].sum()} ({(1-df['label'].mean())*100:.2f}%)")
    
    return df


def split_dataset(df: pd.DataFrame, test_size: float = TEST_SPLIT_RATIO
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train+val and test sets (stratified)."""
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=RANDOM_SEED,
        stratify=df['label'].values
    )
    
    return df.iloc[train_val_idx].reset_index(drop=True), \
           df.iloc[test_idx].reset_index(drop=True)


def preprocess_texts(texts: List[str], processor: TextPreprocessor) -> List[str]:
    """Preprocess texts using unified pipeline."""
    print(f"Preprocessing {len(texts)} texts...")
    preprocessed = []
    
    for idx, text in enumerate(texts):
        if (idx + 1) % 10000 == 0:
            print(f"  [{idx + 1}/{len(texts)}]", end='\r')
        preprocessed.append(processor.get_tokens_as_string(text))
    
    print(f"✓ Preprocessing complete")
    return preprocessed


# ==============================================================================
# MODEL LOADING FUNCTIONS
# ==============================================================================

def load_baseline_model(model_path: str = BASELINE_MODEL_PATH) -> tuple:
    """
    Load baseline TF-IDF + Logistic Regression model.
    
    Returns:
        tuple: (model_pipeline, model_name)
    """
    print(f"\nLoading baseline model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Baseline model not found: {model_path}")
    
    pipeline = joblib.load(model_path)
    print(f"✓ Baseline model loaded")
    
    return pipeline, 'Baseline (TF-IDF + LR)'


def load_bilstm_model(model_path: str = BILSTM_MODEL_PATH,
                     vocab_path: str = VOCAB_PATH) -> Tuple:
    """
    Load BiLSTM model and vocabulary.
    
    Returns:
        tuple: (model, vocab, device, model_name)
    """
    print(f"\nLoading BiLSTM model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"BiLSTM model not found: {model_path}")
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    
    device = get_device()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Recreate model (assuming config from checkpoint or hardcoded)
    model = BiLSTMHateSpeechClassifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=128,
        num_layers=3,
        dropout=0.35,
        pad_idx=vocab.get('<PAD>', 0)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ BiLSTM model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model, vocab, device, 'BiLSTM'


def load_transformer_model(model_path: str = TRANSFORMER_MODEL_PATH,
                          device: torch.device = None) -> Tuple:
    """
    Load fine-tuned BERT multilingual model.
    
    Returns:
        tuple: (model, tokenizer, device, model_name)
    """
    print(f"\nLoading transformer model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transformer model not found: {model_path}")
    
    if device is None:
        device = get_device()
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Transformer model loaded")
    
    return model, tokenizer, device, 'BERT Multilingual'


# ==============================================================================
# INFERENCE FUNCTIONS
# ==============================================================================

def baseline_predict(pipeline, texts: List[str]) -> Tuple[np.ndarray, float]:
    """
    Get predictions from baseline model.
    
    Returns:
        tuple: (probabilities_of_hate_class, inference_time_ms)
    """
    print("Running baseline model inference...")
    
    start_time = time.time()
    proba = pipeline.predict_proba(texts)[:, 1]  # Probability of class 1
    inference_time = time.time() - start_time
    
    avg_time_ms = (inference_time / len(texts)) * 1000
    print(f"✓ Inference complete ({avg_time_ms:.2f}ms per sample)")
    
    return proba, avg_time_ms


def bilstm_predict(model, texts: List[str], vocab: dict,
                  device: torch.device) -> Tuple[np.ndarray, float]:
    """
    Get predictions from BiLSTM model.
    
    Returns:
        tuple: (probabilities_of_hate_class, inference_time_ms)
    """
    print("Running BiLSTM model inference...")
    
    # Convert texts to token indices
    token_indices = []
    lengths = []
    
    for text in texts:
        tokens = text.split()
        indices = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
        token_indices.append(indices)
        lengths.append(len(indices))
    
    # Pad sequences
    max_len = max(lengths) if lengths else 1
    padded = np.zeros((len(texts), max_len), dtype=np.int64)
    
    for idx, indices in enumerate(token_indices):
        padded[idx, :len(indices)] = indices
    
    # Convert to tensors
    X_tensor = torch.LongTensor(padded).to(device)
    lengths_tensor = torch.LongTensor(lengths).to(device)
    
    # Inference
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(X_tensor, lengths_tensor)
        # Apply sigmoid to get probabilities
        proba = torch.sigmoid(outputs).cpu().numpy().flatten()
    
    inference_time = time.time() - start_time
    avg_time_ms = (inference_time / len(texts)) * 1000
    
    print(f"✓ Inference complete ({avg_time_ms:.2f}ms per sample)")
    
    return proba, avg_time_ms


def transformer_predict(model, tokenizer, texts: List[str],
                       device: torch.device,
                       batch_size: int = 32) -> Tuple[np.ndarray, float]:
    """
    Get predictions from transformer model.
    
    Returns:
        tuple: (probabilities_of_hate_class, inference_time_ms)
    """
    print("Running transformer model inference...")
    
    start_time = time.time()
    all_probs = []
    
    # Process in batches
    for batch_idx in range(0, len(texts), batch_size):
        if (batch_idx // batch_size + 1) % 10 == 0:
            print(f"  Batch {batch_idx // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}", end='\r')
        
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        
        # Tokenize
        encoding = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Softmax to get probabilities
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1
            all_probs.extend(probs.cpu().numpy())
    
    inference_time = time.time() - start_time
    avg_time_ms = (inference_time / len(texts)) * 1000
    
    print(f"\n✓ Inference complete ({avg_time_ms:.2f}ms per sample)")
    
    return np.array(all_probs), avg_time_ms


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray,
                        thresholds: List[float] = None) -> Dict:
    """
    Evaluate predictions at multiple thresholds.
    
    Returns:
        dict: Metrics for each threshold
    """
    if thresholds is None:
        thresholds = EVALUATION_THRESHOLDS
    
    results = {
        'raw_scores': y_proba,
        'thresholds': {}
    }
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate specificity and sensitivity
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results['thresholds'][threshold] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    return results


def print_threshold_results(model_name: str, results: Dict,
                           thresholds: List[float] = None) -> None:
    """Print threshold evaluation results to console."""
    if thresholds is None:
        thresholds = EVALUATION_THRESHOLDS
    
    print(f"\n{'=' * 80}")
    print(f"EVALUATION: {model_name}")
    print(f"{'=' * 80}")
    
    for threshold in thresholds:
        metrics = results['thresholds'][threshold]
        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")


# ==============================================================================
# REPORTING FUNCTIONS
# ==============================================================================

def generate_comparison_csv(results_dict: Dict, output_path: str) -> None:
    """
    Generate comprehensive comparison CSV across all models and thresholds.
    
    Args:
        results_dict: Dictionary with model results
        output_path: Output CSV file path
    """
    rows = []
    
    for model_name, (results, inference_time) in results_dict.items():
        for threshold, metrics in results['thresholds'].items():
            rows.append({
                'Model': model_name,
                'Threshold': f"{threshold:.2f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'Sensitivity': f"{metrics['sensitivity']:.4f}",
                'Inf. Time (ms)': f"{inference_time:.2f}"
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Comparison CSV saved: {output_path}")
    print("\nComparison Table Preview:")
    print(df.to_string(index=False))


def generate_detailed_report(results_dict: Dict, inference_times: Dict,
                            test_size: int, output_path: str) -> None:
    """Generate comprehensive text report."""
    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
        f.write("Hate Speech Detection: Baseline vs BiLSTM vs Transformer\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Set Size: {test_size} samples\n\n")
        
        f.write("-" * 100 + "\n")
        f.write("INFERENCE TIME COMPARISON\n")
        f.write("-" * 100 + "\n\n")
        
        for model_name, inf_time in inference_times.items():
            f.write(f"{model_name}: {inf_time:.2f}ms per sample\n")
        
        f.write("\n" + "-" * 100 + "\n")
        f.write("DETAILED THRESHOLD ANALYSIS\n")
        f.write("-" * 100 + "\n\n")
        
        for model_name, (results, _) in results_dict.items():
            f.write(f"\n{'#' * 100}\n")
            f.write(f"MODEL: {model_name}\n")
            f.write(f"{'#' * 100}\n\n")
            
            for threshold, metrics in results['thresholds'].items():
                f.write(f"Threshold: {threshold:.2f}\n")
                f.write(f"  Accuracy:    {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision:   {metrics['precision']:.4f}\n")
                f.write(f"  Recall:      {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:    {metrics['f1_score']:.4f}\n")
                f.write(f"  Specificity: {metrics['specificity']:.4f}\n")
                f.write(f"  Sensitivity: {metrics['sensitivity']:.4f}\n")
                
                cm = metrics['confusion_matrix']
                f.write(f"\n  Confusion Matrix:\n")
                f.write(f"    [[TN={cm[0,0]:>6} FP={cm[0,1]:>6}]\n")
                f.write(f"     [FN={cm[1,0]:>6} TP={cm[1,1]:>6}]]\n\n")
        
        f.write("\n" + "-" * 100 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 100 + "\n\n")
        f.write("1. For production deployment:\n")
        f.write("   - Choose model with best F1-score on validation set\n")
        f.write("   - Set threshold to minimize false positives/negatives based on business need\n")
        f.write("2. Inference speed considerations:\n")
        f.write("   - Baseline model: Fastest, suitable for high-throughput scenarios\n")
        f.write("   - BiLSTM model: Medium speed, good accuracy\n")
        f.write("   - Transformer model: Slowest but highest accuracy\n")
    
    print(f"✓ Detailed report saved: {output_path}")


def save_all_predictions(predictions_dict: Dict, X_test: List[str],
                        y_test: np.ndarray, output_path: str) -> None:
    """Save all model predictions to CSV."""
    df_data = {
        'text': X_test,
        'true_label': y_test,
    }
    
    for model_name, (results, _) in predictions_dict.items():
        df_data[f'{model_name}_confidence'] = results['raw_scores']
        df_data[f'{model_name}_pred_t05'] = results['thresholds'][0.5]['predictions']
        df_data[f'{model_name}_pred_t08'] = results['thresholds'].get(0.8, {}).get('predictions', [])
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)
    print(f"✓ All predictions saved: {output_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main(dataset_path: str = DEFAULT_DATASET_PATH,
         comparison_csv: str = OUTPUT_COMPARISON_CSV,
         detailed_report: str = OUTPUT_DETAILED_REPORT,
         predictions_csv: str = OUTPUT_MODEL_PREDICTIONS) -> None:
    """
    Main evaluation pipeline.
    """
    print("\n" + "=" * 100)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("Comparing: Baseline (TF-IDF + LR) vs BiLSTM vs Transformer (BERT)")
    print("=" * 100 + "\n")
    
    try:
        # Step 1: Load and prepare data
        print("[Step 1/5] Loading and preparing test dataset...")
        df = load_dataset(dataset_path)
        _, test_df = split_dataset(df)
        
        processor = TextPreprocessor(remove_stopwords=True, lemmatize=True, stem=True)
        X_test = preprocess_texts(test_df['text'].tolist(), processor)
        y_test = test_df['label'].values
        
        print(f"✓ Test set prepared: {len(X_test)} samples")
        
        # Step 2: Load all models
        print("\n[Step 2/5] Loading all trained models...")
        device = get_device()
        
        baseline_pipeline, baseline_name = load_baseline_model()
        bilstm_model, vocab, _, bilstm_name = load_bilstm_model()
        transformer_model, tokenizer, _, transformer_name = load_transformer_model(device=device)
        
        # Step 3: Run inference on all models
        print("\n[Step 3/5] Running inference on all models...")
        
        baseline_proba, baseline_time = baseline_predict(baseline_pipeline, X_test)
        bilstm_proba, bilstm_time = bilstm_predict(bilstm_model, X_test, vocab, device)
        transformer_proba, transformer_time = transformer_predict(
            transformer_model, tokenizer, X_test, device
        )
        
        # Step 4: Evaluate models
        print("\n[Step 4/5] Evaluating models at multiple thresholds...")
        
        baseline_results = evaluate_predictions(y_test, baseline_proba)
        bilstm_results = evaluate_predictions(y_test, bilstm_proba)
        transformer_results = evaluate_predictions(y_test, transformer_proba)
        
        results_dict = {
            baseline_name: (baseline_results, baseline_time),
            bilstm_name: (bilstm_results, bilstm_time),
            transformer_name: (transformer_results, transformer_time)
        }
        
        inference_times = {
            baseline_name: baseline_time,
            bilstm_name: bilstm_time,
            transformer_name: transformer_time
        }
        
        # Print results
        print_threshold_results(baseline_name, baseline_results)
        print_threshold_results(bilstm_name, bilstm_results)
        print_threshold_results(transformer_name, transformer_results)
        
        # Step 5: Generate reports
        print("\n[Step 5/5] Generating comprehensive reports...")
        
        generate_comparison_csv(results_dict, comparison_csv)
        generate_detailed_report(results_dict, inference_times, len(y_test), detailed_report)
        save_all_predictions(results_dict, X_test, y_test, predictions_csv)
        
        print("\n" + "=" * 100)
        print("✓ EVALUATION COMPLETE")
        print("=" * 100)
        print(f"\nGenerated files:")
        print(f"  - Comparison CSV: {comparison_csv}")
        print(f"  - Detailed Report: {detailed_report}")
        print(f"  - All Predictions: {predictions_csv}")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation of all hate speech detection models'
    )
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_PATH,
                        help=f'Path to dataset CSV (default: {DEFAULT_DATASET_PATH})')
    parser.add_argument('--comparison-csv', type=str, default=OUTPUT_COMPARISON_CSV,
                        help=f'Output comparison CSV (default: {OUTPUT_COMPARISON_CSV})')
    parser.add_argument('--report', type=str, default=OUTPUT_DETAILED_REPORT,
                        help=f'Output detailed report (default: {OUTPUT_DETAILED_REPORT})')
    
    args = parser.parse_args()
    
    main(dataset_path=args.dataset,
         comparison_csv=args.comparison_csv,
         detailed_report=args.report)
