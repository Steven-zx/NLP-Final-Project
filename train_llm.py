"""
Large Language Model Fine-Tuning: BERT Multilingual for Hate Speech Detection
==============================================================================

This script fine-tunes a pretrained BERT multilingual model on the unified
bilingual hate speech dataset for binary classification (hate vs. non-hate).

Selected Model: bert-base-multilingual-cased
- True multilingual support (110+ languages)
- 110M parameters (manageable GPU memory)
- Well-suited for English + Filipino mixed text
- Strong baseline for transformer-based approaches

Pipeline:
1. Load unified bilingual hate speech dataset
2. Split into train/validation/test (70/15/15)
3. Apply text preprocessing (using preprocessing.py)
4. Tokenize using AutoTokenizer (BERT wordpiece tokenizer)
5. Create PyTorch datasets with attention masks and token type IDs
6. Fine-tune using HuggingFace Trainer API
7. Evaluate on test set (Accuracy, Precision, Recall, F1-score)
8. Save fine-tuned model and tokenizer

Model Performance Target:
- Expected Accuracy: ~72-78% (improvement over baseline)
- Inference Speed: ~200-500ms per sample (CUDA-accelerated)

Training Configuration:
- Batch Size: 32 (adjust for GPU memory constraints)
- Learning Rate: 2e-5 (standard for fine-tuning)
- Epochs: 3-5 (tune for convergence without overfitting)
- Max Sequence Length: 128 (consistent with BiLSTM baseline)
- Mixed Precision: FP16 if CUDA available (2-3x speedup, minimal accuracy loss)

Dependencies:
    - torch, transformers, datasets
    - pandas, numpy
    - preprocessing (custom module)
    - scipy (for metrics computation)

Output Files:
    - transformer_model/: Model directory with config, weights, tokenizer
    - llm_evaluation.txt: Metrics and confusion matrix
    - llm_predictions.csv: Test set predictions with confidence
    - training_logs.txt: Training progress and loss curves
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Optional

# PyTorch
import torch
from torch.utils.data import Dataset

# HuggingFace Transformers
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from transformers.utils import logging as hf_logging

# Scikit-learn (for metrics)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# Custom preprocessing
from preprocessing import TextPreprocessor

# Suppress HuggingFace warnings
hf_logging.set_verbosity_error()


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model Configuration
# Use a smaller model and conservative settings for local/CPU runs.
# If you have a GPU and want full fine-tuning, change these back to
# 'bert-base-multilingual-cased', larger batch sizes, and more epochs.
MODEL_NAME = 'distilbert-base-multilingual-cased'
MAX_SEQ_LENGTH = 128  # Consistent with BiLSTM baseline

# Training Configuration (reduced for CPU / quick run)
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 1
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1

# Mixed Precision Training
# Disable FP16 for CPU runs
USE_FP16 = False

# Data Configuration
DEFAULT_DATASET_PATH = 'unified_bilingual_hatespeech.csv'
OUTPUT_MODEL_DIR = 'transformer_model'
OUTPUT_EVAL_PATH = 'llm_evaluation.txt'
OUTPUT_PREDICTIONS_PATH = 'llm_predictions.csv'
OUTPUT_LOGS_PATH = 'training_logs.txt'

# Random seed for reproducibility
RANDOM_SEED = 42

# Data sampling for quick local runs (1.0 = use full dataset)
DATA_PCT = 0.02  # set to small fraction for quick CPU runs; increase if you have GPU


# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================

def get_device():
    """
    Detect available device (GPU with CUDA or CPU).
    
    Returns:
        torch.device: Device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    else:
        device = torch.device('cpu')
        print("WARNING: Using CPU (training will be slow; consider using GPU)")
    
    return device


# ==============================================================================
# DATASET LOADING & PREPROCESSING
# ==============================================================================

def load_dataset(csv_path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """
    Load the unified bilingual hate speech dataset.
    
    Args:
        csv_path (str): Path to dataset CSV
        
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
        print(f"  WARNING: Removed {initial_count - len(df)} rows with missing values")
    
    # Ensure label is binary (0 or 1)
    df['label'] = df['label'].astype(int)
    
    print(f"Loaded {len(df)} samples")
    print(f"  - Hate speech: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")
    print(f"  - Non-hate: {len(df) - df['label'].sum()} ({(1-df['label'].mean())*100:.2f}%)")
    
    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.15,
                  val_size: float = 0.15, random_state: int = RANDOM_SEED
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets (stratified).
    
    Args:
        df (pd.DataFrame): Full dataset
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set
        random_state (int): Random seed
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train, val, test)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: test set
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=df['label'].values
    )
    
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    
    # Second split: validation set
    val_split_ratio = val_size / (1 - test_size)
    val_split_indices = np.arange(len(train_val))
    train_idx, val_idx = train_test_split(
        val_split_indices, test_size=val_split_ratio,
        random_state=random_state + 1,
        stratify=train_val['label'].values
    )
    
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    
    print(f"Split dataset into:")
    print(f"  - Train: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
    print(f"  - Validation: {len(val)} samples ({len(val)/len(df)*100:.1f}%)")
    print(f"  - Test: {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
    
    return train, val, test


def preprocess_texts(texts: List[str], processor: TextPreprocessor) -> List[str]:
    """
    Preprocess texts using the unified pipeline.
    
    Args:
        texts (List[str]): List of raw texts
        processor (TextPreprocessor): Preprocessor instance
        
    Returns:
        List[str]: List of preprocessed texts
    """
    print(f"Preprocessing {len(texts)} texts...")
    preprocessed = []
    
    for idx, text in enumerate(texts):
        if (idx + 1) % 10000 == 0:
            print(f"  [{idx + 1}/{len(texts)}]", end='\r')
        
        processed = processor.get_tokens_as_string(text)
        preprocessed.append(processed)
    
    print(f"Preprocessing complete")
    return preprocessed


# ==============================================================================
# PYTORCH DATASET
# ==============================================================================

class HateSpeechDataset(Dataset):
    """
    PyTorch Dataset for hate speech classification.
    
    Handles tokenization and encoding of texts using BERT tokenizer.
    """
    
    def __init__(self, texts: List[str], labels: np.ndarray,
                 tokenizer, max_length: int = MAX_SEQ_LENGTH):
        """
        Initialize dataset.
        
        Args:
            texts (List[str]): List of preprocessed texts
            labels (np.ndarray): Binary labels (0/1)
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns tokenized text with input IDs, attention mask, and token type IDs.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze() if 'token_type_ids' in encoding else torch.zeros(self.max_length),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ==============================================================================
# MODEL TRAINING & EVALUATION
# ==============================================================================

def train_transformer_model(X_train: List[str], y_train: np.ndarray,
                           X_val: List[str], y_val: np.ndarray,
                           device: torch.device) -> Tuple:
    """
    Fine-tune BERT multilingual model on training data.
    
    Uses HuggingFace Trainer API for simplified training loop with:
    - Automatic mixed precision (FP16) if available
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    
    Args:
        X_train (List[str]): Training texts (preprocessed)
        y_train (np.ndarray): Training labels
        X_val (List[str]): Validation texts
        y_val (np.ndarray): Validation labels
        device (torch.device): Device to use
        
    Returns:
        Tuple: (model, tokenizer, trainer)
    """
    print(f"\nLoading pretrained model and tokenizer: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2  # Binary classification
    )
    
    print(f"Model loaded")
    print(f"  - Total parameters: {model.num_parameters():,}")
    
    # Create datasets
    print(f"\nCreating PyTorch datasets...")
    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = HateSpeechDataset(X_val, y_val, tokenizer, MAX_SEQ_LENGTH)
    print(f"Datasets created")
    
    # Define metrics computation function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0)
        }
    
    # Training configuration
    # Use a simplified TrainingArguments configuration to maintain
    # compatibility with different transformers versions.
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=100,
        eval_strategy='epoch',
        metric_for_best_model='f1',
        # Older transformers may not accept evaluation/save strategy kwargs,
        # so keep defaults and rely on Trainer methods for evaluation.
        fp16=(USE_FP16 and torch.cuda.is_available()),
        seed=RANDOM_SEED,
        dataloader_num_workers=0,
        report_to='none'
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        # No early-stopping callback for compatibility on local CPU runs
    )
    
    # Train
    print(f"\nStarting fine-tuning training...")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - FP16: {training_args.fp16}")
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"Training complete ({training_time/60:.1f} minutes)")
    
    return model, tokenizer, trainer


def evaluate_model(model, tokenizer, X_test: List[str], y_test: np.ndarray,
                   device: torch.device, thresholds: List[float] = None) -> Dict:
    """
    Evaluate fine-tuned model on test set with multiple thresholds.
    
    Args:
        model: Fine-tuned transformer model
        tokenizer: Tokenizer used during training
        X_test (List[str]): Test texts
        y_test (np.ndarray): Test labels
        device (torch.device): Device to use
        thresholds (List[float]): Thresholds to evaluate
        
    Returns:
        Dict: Evaluation results
    """
    if thresholds is None:
        thresholds = [0.5]
    
    print(f"\nEvaluating model on {len(X_test)} test samples...")
    
    # Create test dataset
    test_dataset = HateSpeechDataset(X_test, y_test, tokenizer, MAX_SEQ_LENGTH)
    
    # Setup model for inference
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_logits = []
    
    start_time = time.time()
    
    # Inference loop
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            if (idx + 1) % 5000 == 0:
                print(f"  [{idx + 1}/{len(test_dataset)}]", end='\r')
            
            sample = test_dataset[idx]
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()[0]
            
            # Softmax to get probabilities
            probs = softmax(logits)
            all_logits.append(logits)
            all_predictions.append(probs[1])  # Probability of class 1 (hate speech)
    
    inference_time = time.time() - start_time
    avg_inference_time = (inference_time / len(X_test)) * 1000
    
    print(f"\nInference complete in {inference_time:.2f}s ({avg_inference_time:.2f}ms per sample)")
    
    y_proba = np.array(all_predictions)
    
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


def softmax(x):
    """Compute softmax values for each set of scores."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def save_model(model, tokenizer, output_dir: str = OUTPUT_MODEL_DIR) -> None:
    """
    Save fine-tuned model and tokenizer.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        output_dir (str): Output directory
    """
    print(f"\nSaving model and tokenizer to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved")


def save_evaluation_report(results: Dict, output_path: str = OUTPUT_EVAL_PATH,
                          predictions_path: str = OUTPUT_PREDICTIONS_PATH,
                          X_test: List[str] = None, y_test: np.ndarray = None) -> None:
    """
    Save comprehensive evaluation report.
    
    Args:
        results (Dict): Evaluation results
        output_path (str): Report file path
        predictions_path (str): Predictions CSV path
        X_test (List[str]): Test texts
        y_test (np.ndarray): Test labels
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRANSFORMER MODEL EVALUATION REPORT\n")
        f.write(f"Model: {MODEL_NAME}\n")
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
        f.write(f"Pretrained Model: {MODEL_NAME}\n")
        f.write(f"Max Sequence Length: {MAX_SEQ_LENGTH}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Warmup Ratio: {WARMUP_RATIO}\n")
        f.write(f"Weight Decay: {WEIGHT_DECAY}\n")
        f.write(f"FP16 Training: {USE_FP16}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
    
    print(f"Evaluation report saved to: {output_path}")
    
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
        print(f"Predictions saved to: {predictions_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main(dataset_path: str = DEFAULT_DATASET_PATH,
         model_output_dir: str = OUTPUT_MODEL_DIR,
         eval_output_path: str = OUTPUT_EVAL_PATH,
         predictions_output_path: str = OUTPUT_PREDICTIONS_PATH) -> None:
    """
    Main execution pipeline for fine-tuning LLM.
    
    Args:
        dataset_path (str): Path to dataset CSV
        model_output_dir (str): Output directory for model
        eval_output_path (str): Output path for evaluation report
        predictions_output_path (str): Output path for predictions
    """
    print("\n" + "=" * 80)
    print("TRANSFORMER FINE-TUNING: BERT MULTILINGUAL")
    print("=" * 80 + "\n")
    
    try:
        # Step 0: Setup device
        print("[Step 0/7] Setting up device...")
        device = get_device()
        
        # Step 1: Load dataset
        print("\n[Step 1/7] Loading dataset...")
        df = load_dataset(dataset_path)

        # Optionally sample a subset for quick local runs
        if DATA_PCT is not None and 0.0 < DATA_PCT < 1.0:
            sample_n = int(len(df) * DATA_PCT)
            print(f"Sampling {DATA_PCT*100:.2f}% of data -> {sample_n} samples for quick run")
            df = df.sample(n=max(1, sample_n), random_state=RANDOM_SEED).reset_index(drop=True)
        
        # Step 2: Initialize preprocessor
        print("\n[Step 2/7] Initializing text preprocessor...")
        processor = TextPreprocessor(remove_stopwords=True, lemmatize=True, stem=True)
        print("Preprocessor initialized")
        
        # Step 3: Split dataset
        print("\n[Step 3/7] Splitting dataset (70/15/15)...")
        train_df, val_df, test_df = split_dataset(df)
        
        # Step 4: Preprocess texts
        print("\n[Step 4/7] Preprocessing texts...")
        X_train = preprocess_texts(train_df['text'].tolist(), processor)
        X_val = preprocess_texts(val_df['text'].tolist(), processor)
        X_test = preprocess_texts(test_df['text'].tolist(), processor)
        
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values
        
        # Step 5: Train model
        print("\n[Step 5/7] Fine-tuning transformer model...")
        model, tokenizer, trainer = train_transformer_model(
            X_train, y_train, X_val, y_val, device
        )
        
        # Step 6: Evaluate model
        print("\n[Step 6/7] Evaluating model...")
        results = evaluate_model(model, tokenizer, X_test, y_test, device,
                                thresholds=[0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Step 7: Save outputs
        print("\n[Step 7/7] Saving model and reports...")
        save_model(model, tokenizer, model_output_dir)
        save_evaluation_report(results, eval_output_path, predictions_output_path,
                              X_test, y_test)
        
        print("\n" + "=" * 80)
        print("TRANSFORMER FINE-TUNING COMPLETE")
        print("=" * 80)
        print(f"\nGenerated files:")
        print(f"  - Model: {model_output_dir}/")
        print(f"  - Report: {eval_output_path}")
        print(f"  - Predictions: {predictions_output_path}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-tune BERT multilingual for hate speech detection'
    )
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_PATH,
                        help=f'Path to dataset CSV (default: {DEFAULT_DATASET_PATH})')
    parser.add_argument('--model-output', type=str, default=OUTPUT_MODEL_DIR,
                        help=f'Output directory for model (default: {OUTPUT_MODEL_DIR})')
    parser.add_argument('--eval-output', type=str, default=OUTPUT_EVAL_PATH,
                        help=f'Output path for evaluation report (default: {OUTPUT_EVAL_PATH})')
    
    args = parser.parse_args()
    
    main(dataset_path=args.dataset,
         model_output_dir=args.model_output,
         eval_output_path=args.eval_output)
