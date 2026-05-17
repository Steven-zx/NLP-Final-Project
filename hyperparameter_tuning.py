"""
Hyperparameter Tuning for BiLSTM Hate Speech Classifier
Trains 3 different configurations and compares their performance
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import time
from datetime import datetime
import json

from rnn_model import BiLSTMHateSpeechClassifier
from text_preprocessing import TextPreprocessor, Vocabulary, pad_sequences


class HateSpeechDataset(Dataset):
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
        
        sequence = self.vocab.text_to_sequence(text, self.max_length)
        
        if len(sequence) == 0:
            sequence = [self.vocab.word2idx[self.vocab.UNK_TOKEN]]
        
        return torch.LongTensor(sequence), len(sequence), torch.FloatTensor([label])


def collate_fn(batch):
    sequences, lengths, labels = zip(*batch)
    
    max_len = max(lengths)
    padded_sequences = []
    
    for seq in sequences:
        if len(seq) < max_len:
            padded = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])
        else:
            padded = seq
        padded_sequences.append(padded)
    
    return torch.stack(padded_sequences), torch.LongTensor(lengths), torch.stack(labels)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for texts, lengths, labels in tqdm(train_loader, desc="Training"):
        texts = texts.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        predictions = torch.sigmoid(outputs) > 0.5
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for texts, lengths, labels in tqdm(data_loader, desc="Evaluating"):
            texts = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            predictions = torch.sigmoid(outputs) > 0.5
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    return avg_loss, accuracy, precision, recall, f1


def train_model_with_config(config, train_dataset, val_dataset, vocab, device):
    """Train a model with specific hyperparameter configuration"""
    
    print(f"\n{'='*80}")
    print(f"Training Configuration: {config['name']}")
    print(f"{'='*80}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Hidden Units: {config['hidden_dim']}")
    print(f"Dropout: {config['dropout']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"LSTM Layers: {config['num_layers']}")
    print(f"{'='*80}\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = BiLSTMHateSpeechClassifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        pad_idx=vocab.word2idx[vocab.PAD_TOKEN]
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    else:  # RMSprop
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'])
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    best_val_acc = 0
    start_time = time.time()
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'config': config
            }, f"{config['model_name']}.pt")
            print(f"✓ Best model saved with Val Acc: {val_acc:.4f}")
    
    training_time = time.time() - start_time
    
    # Final results
    results = {
        'config_name': config['name'],
        'model_file': f"{config['model_name']}.pt",
        'best_val_acc': best_val_acc,
        'final_val_loss': history['val_loss'][-1],
        'final_val_f1': history['val_f1'][-1],
        'final_val_precision': history['val_precision'][-1],
        'final_val_recall': history['val_recall'][-1],
        'training_time_seconds': training_time,
        'history': history,
        'hyperparameters': config
    }
    
    return results


def main():
    print("\n" + "="*80)
    print("BiLSTM HYPERPARAMETER TUNING")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load vocabulary
    print("Loading vocabulary...")
    vocab = Vocabulary.load('vocabulary.pkl')
    print(f"✓ Vocabulary loaded: {len(vocab)} words\n")
    
    # Load and prepare data
    print("Loading datasets...")
    train_df = pd.read_csv('hatespeech/train.csv')
    val_df = pd.read_csv('hatespeech/valid.csv')
    
    print(f"✓ Training samples: {len(train_df)}")
    print(f"✓ Validation samples: {len(val_df)}\n")
    
    # Create datasets
    train_dataset = HateSpeechDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        vocab
    )
    
    val_dataset = HateSpeechDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        vocab
    )
    
    # Define 5 different hyperparameter configurations
    configurations = [
        {
            'name': 'Config 1: Baseline (10 epochs)',
            'model_name': 'model_config1_baseline',
            'learning_rate': 0.001,
            'batch_size': 64,
            'hidden_dim': 128,
            'dropout': 0.3,
            'epochs': 10,
            'optimizer': 'Adam',
            'num_layers': 2
        },
        {
            'name': 'Config 2: Extended Training (15 epochs)',
            'model_name': 'model_config2_extended',
            'learning_rate': 0.0005,
            'batch_size': 32,
            'hidden_dim': 256,
            'dropout': 0.4,
            'epochs': 15,
            'optimizer': 'Adam',
            'num_layers': 2
        },
        {
            'name': 'Config 3: Fast Training (20 epochs, smaller model)',
            'model_name': 'model_config3_fast',
            'learning_rate': 0.01,
            'batch_size': 128,
            'hidden_dim': 64,
            'dropout': 0.2,
            'epochs': 20,
            'optimizer': 'RMSprop',
            'num_layers': 2
        },
        {
            'name': 'Config 4: Deep Model (3 LSTM layers, 12 epochs)',
            'model_name': 'model_config4_deep',
            'learning_rate': 0.0008,
            'batch_size': 64,
            'hidden_dim': 128,
            'dropout': 0.35,
            'epochs': 12,
            'optimizer': 'Adam',
            'num_layers': 3
        },
        {
            'name': 'Config 5: High Dropout Regularization (10 epochs)',
            'model_name': 'model_config5_regularized',
            'learning_rate': 0.001,
            'batch_size': 64,
            'hidden_dim': 128,
            'dropout': 0.5,
            'epochs': 10,
            'optimizer': 'SGD',
            'num_layers': 2
        }
    ]
    
    # Train all configurations
    all_results = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n{'#'*80}")
        print(f"# Training Model {i}/3")
        print(f"{'#'*80}\n")
        
        results = train_model_with_config(config, train_dataset, val_dataset, vocab, device)
        all_results.append(results)
        
        # Short break between trainings
        if i < len(configurations):
            print("\n" + "-"*80)
            print("Taking a short break before next configuration...")
            print("-"*80 + "\n")
            time.sleep(2)
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print("="*80 + "\n")
    
    # Create comparison table
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Configuration': result['config_name'],
            'Learning Rate': result['hyperparameters']['learning_rate'],
            'Batch Size': result['hyperparameters']['batch_size'],
            'Hidden Dim': result['hyperparameters']['hidden_dim'],
            'Dropout': result['hyperparameters']['dropout'],
            'Epochs': result['hyperparameters']['epochs'],
            'LSTM Layers': result['hyperparameters']['num_layers'],
            'Optimizer': result['hyperparameters']['optimizer'],
            'Best Val Acc': f"{result['best_val_acc']:.4f}",
            'Final F1': f"{result['final_val_f1']:.4f}",
            'Final Precision': f"{result['final_val_precision']:.4f}",
            'Final Recall': f"{result['final_val_recall']:.4f}",
            'Training Time (min)': f"{result['training_time_seconds']/60:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print(comparison_df.to_string(index=False))
    print("\n")
    
    # Find best configuration
    best_config = max(all_results, key=lambda x: x['best_val_acc'])
    print(f"🏆 BEST CONFIGURATION: {best_config['config_name']}")
    print(f"   Best Validation Accuracy: {best_config['best_val_acc']:.4f}")
    print(f"   F1-Score: {best_config['final_val_f1']:.4f}")
    print(f"   Model saved as: {best_config['model_file']}")
    print("\n")
    
    # Save comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    comparison_df.to_csv(f'hyperparameter_comparison_{timestamp}.csv', index=False)
    print(f"✓ Comparison table saved: hyperparameter_comparison_{timestamp}.csv")
    
    # Save detailed results as JSON
    with open(f'detailed_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"✓ Detailed results saved: detailed_results_{timestamp}.json")
    
    # Create summary text report
    with open(f'tuning_summary_{timestamp}.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("HYPERPARAMETER TUNING SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total Configurations Tested: {len(configurations)}\n\n")
        
        f.write("CONFIGURATIONS TESTED:\n")
        f.write("-"*80 + "\n\n")
        
        for i, result in enumerate(all_results, 1):
            config = result['hyperparameters']
            f.write(f"{i}. {result['config_name']}\n")
            f.write(f"   Learning Rate: {config['learning_rate']}\n")
            f.write(f"   Batch Size: {config['batch_size']}\n")
            f.write(f"   Hidden Units: {config['hidden_dim']}\n")
            f.write(f"   Dropout: {config['dropout']}\n")
            f.write(f"   Epochs: {config['epochs']}\n")
            f.write(f"   LSTM Layers: {config['num_layers']}\n")
            f.write(f"   Optimizer: {config['optimizer']}\n")
            f.write(f"   \n")
            f.write(f"   Results:\n")
            f.write(f"   - Best Validation Accuracy: {result['best_val_acc']:.4f} ({result['best_val_acc']*100:.2f}%)\n")
            f.write(f"   - F1-Score: {result['final_val_f1']:.4f}\n")
            f.write(f"   - Precision: {result['final_val_precision']:.4f}\n")
            f.write(f"   - Recall: {result['final_val_recall']:.4f}\n")
            f.write(f"   - Training Time: {result['training_time_seconds']/60:.2f} minutes\n")
            f.write(f"   - Model File: {result['model_file']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("BEST CONFIGURATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Configuration: {best_config['config_name']}\n")
        f.write(f"Validation Accuracy: {best_config['best_val_acc']:.4f} ({best_config['best_val_acc']*100:.2f}%)\n")
        f.write(f"F1-Score: {best_config['final_val_f1']:.4f}\n")
        f.write(f"Model File: {best_config['model_file']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        # Performance analysis
        accuracies = [r['best_val_acc'] for r in all_results]
        if max(accuracies) - min(accuracies) < 0.02:
            f.write("- All configurations performed similarly (accuracy difference < 2%).\n")
            f.write("- Consider using the fastest training configuration for efficiency.\n")
        else:
            f.write(f"- Significant performance difference detected ({(max(accuracies) - min(accuracies))*100:.2f}% accuracy gap).\n")
            f.write(f"- Recommend using {best_config['config_name']} for best performance.\n")
        
        f.write("\n")
    
    print(f"✓ Summary report saved: tuning_summary_{timestamp}.txt")
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETED!")
    print("="*80 + "\n")
    
    print("Generated files:")
    print(f"  1. model_config1_baseline.pt")
    print(f"  2. model_config2_extended.pt")
    print(f"  3. model_config3_fast.pt")
    print(f"  4. model_config4_deep.pt")
    print(f"  5. model_config5_regularized.pt")
    print(f"  6. hyperparameter_comparison_{timestamp}.csv")
    print(f"  7. detailed_results_{timestamp}.json")
    print(f"  8. tuning_summary_{timestamp}.txt")
    print("\n")


if __name__ == "__main__":
    main()
