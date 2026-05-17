"""
Training Script for BiLSTM Hate Speech Detection
Train on bilingual Filipino-English dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import time
import os
from tqdm import tqdm

from rnn_model import BiLSTMHateSpeechClassifier, SimpleBiLSTM, count_parameters
from text_preprocessing import TextPreprocessor, Vocabulary, pad_sequences
from load_unified_dataset import UnifiedDatasetLoader


class HateSpeechDataset(Dataset):
    """PyTorch Dataset for hate speech detection"""
    
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to sequence
        sequence = self.vocab.text_to_sequence(text, max_length=self.max_length)
        
        # Ensure sequence is not empty (add UNK token if needed)
        if len(sequence) == 0:
            sequence = [self.vocab.word2idx[self.vocab.UNK_TOKEN]]
        
        length = min(len(sequence), self.max_length)
        
        return sequence, length, label


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    sequences, lengths, labels = zip(*batch)
    
    # Pad sequences
    padded_seqs, lengths = pad_sequences(sequences, padding_value=0)
    
    # Convert to tensors
    padded_seqs = torch.LongTensor(padded_seqs)
    lengths = torch.LongTensor(lengths)
    labels = torch.FloatTensor(labels)
    
    return padded_seqs, lengths, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch_idx, (texts, lengths, labels) in enumerate(progress_bar):
        texts = texts.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(texts, lengths).squeeze(1)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for texts, lengths, labels in tqdm(dataloader, desc='Evaluating'):
            texts = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(texts, lengths).squeeze(1)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1, all_labels, all_preds


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, save_path='best_model.pt'):
    """Full training loop"""
    
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, save_path)
            print(f"✓ Best model saved! (Val Acc: {val_acc:.4f})")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    return train_losses, val_losses, train_accs, val_accs


def test_model(model, test_loader, criterion, device):
    """Test the model and print detailed metrics"""
    print("\n" + "=" * 70)
    print("TESTING MODEL")
    print("=" * 70)
    
    test_loss, test_acc, test_precision, test_recall, test_f1, labels, preds = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Non-Hate', 'Hate']))
    
    return test_loss, test_acc, test_f1


def main():
    """Main training pipeline"""
    
    # Hyperparameters
    BATCH_SIZE = 64
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    MAX_VOCAB_SIZE = 20000
    MIN_FREQ = 2
    MAX_SEQ_LENGTH = 100
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    
    loader = UnifiedDatasetLoader()
    loader.load_all_datasets()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.get_train_val_test_split()
    
    # Preprocess texts
    print("\n" + "=" * 70)
    print("PREPROCESSING TEXTS")
    print("=" * 70)
    
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=False,
        remove_punctuation=False
    )
    
    X_train_processed = preprocessor.preprocess_batch(X_train)
    X_val_processed = preprocessor.preprocess_batch(X_val)
    X_test_processed = preprocessor.preprocess_batch(X_test)
    
    # Filter out empty texts (length 0 after preprocessing)
    def filter_empty_texts(texts, labels):
        """Remove empty texts and their corresponding labels"""
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(texts, labels):
            if text.strip():  # Only keep non-empty texts
                filtered_texts.append(text)
                filtered_labels.append(label)
        return filtered_texts, filtered_labels
    
    original_train = len(X_train_processed)
    X_train_processed, y_train = filter_empty_texts(X_train_processed, y_train)
    X_val_processed, y_val = filter_empty_texts(X_val_processed, y_val)
    X_test_processed, y_test = filter_empty_texts(X_test_processed, y_test)
    
    print(f"✓ Preprocessed {len(X_train_processed)} training samples (removed {original_train - len(X_train_processed)} empty)")
    print(f"✓ Preprocessed {len(X_val_processed)} validation samples")
    print(f"✓ Preprocessed {len(X_test_processed)} test samples")
    
    # Build vocabulary
    print("\n" + "=" * 70)
    print("BUILDING VOCABULARY")
    print("=" * 70)
    
    vocab = Vocabulary(max_vocab_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQ)
    vocab.build_vocab(X_train_processed)
    vocab.save('vocabulary.pkl')
    
    # Create datasets
    train_dataset = HateSpeechDataset(X_train_processed, y_train, vocab, MAX_SEQ_LENGTH)
    val_dataset = HateSpeechDataset(X_val_processed, y_val, vocab, MAX_SEQ_LENGTH)
    test_dataset = HateSpeechDataset(X_test_processed, y_test, vocab, MAX_SEQ_LENGTH)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"✓ Created dataloaders (batch_size={BATCH_SIZE})")
    
    # Initialize model
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    
    model = BiLSTMHateSpeechClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=vocab.word2idx[vocab.PAD_TOKEN]
    ).to(device)
    
    print(f"Model: BiLSTMHateSpeechClassifier")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"Loss: BCEWithLogitsLoss")
    
    # Train
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=NUM_EPOCHS, device=device, save_path='best_bilstm_model.pt'
    )
    
    # Load best model and test
    print("\n" + "=" * 70)
    print("LOADING BEST MODEL FOR TESTING")
    print("=" * 70)
    
    checkpoint = torch.load('best_bilstm_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']+1}")
    
    # Test
    test_loss, test_acc, test_f1 = test_model(model, test_loader, criterion, device)
    
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test F1-Score: {test_f1:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
