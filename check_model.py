import torch

# Load the model state dict
state_dict = torch.load('best_bilstm_model.pt', map_location='cpu', weights_only=False)

print("="*60)
print("MODEL FILE CONTENTS")
print("="*60)

print("\nModel state_dict keys:")
for key in state_dict.keys():
    print(f"  - {key}")

print("\n" + "="*60)
print("CHECKPOINT INFORMATION")
print("="*60)

for key, value in state_dict.items():
    if isinstance(value, dict):
        print(f"\n{key}: (dictionary with {len(value)} items)")
    elif hasattr(value, 'shape'):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {value}")

# Extract actual model weights
if 'model_state_dict' in state_dict:
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE DETAILS")
    print("="*60)
    
    model_weights = state_dict['model_state_dict']
    for key, tensor in model_weights.items():
        print(f"{key:.<45} {str(tensor.shape)}")
else:
    model_weights = state_dict

# Analyze the architecture
print("\n" + "="*60)
print("ARCHITECTURE ANALYSIS")
print("="*60)

# Use the actual model weights
if 'model_state_dict' in state_dict:
    weights = state_dict['model_state_dict']
else:
    weights = state_dict

# Count LSTM layers
lstm_layers = sum(1 for key in weights.keys() if 'lstm.weight_ih_l' in key)
print(f"Number of LSTM layers: {lstm_layers}")

# Get dimensions
if 'embedding.weight' in weights:
    vocab_size, embed_dim = weights['embedding.weight'].shape
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Embedding dimension: {embed_dim}")

if 'lstm.weight_ih_l0' in weights:
    hidden_dim = weights['lstm.weight_ih_l0'].shape[0] // 4  # LSTM has 4 gates
    print(f"Hidden dimension: {hidden_dim}")

if 'fc.weight' in weights:
    fc_out, fc_in = weights['fc.weight'].shape
    print(f"FC layer: {fc_in} → {fc_out}")

print("\n" + "="*60)
print("CONFIGURATION MATCH")
print("="*60)

# Compare with known configs
configs = {
    'Config 1 (Baseline)': {'hidden': 128, 'layers': 2, 'dropout': 0.3},
    'Config 2 (Extended)': {'hidden': 256, 'layers': 2, 'dropout': 0.4},
    'Config 3 (Fast)': {'hidden': 64, 'layers': 2, 'dropout': 0.2},
    'Config 4 (Deep)': {'hidden': 128, 'layers': 3, 'dropout': 0.35},
    'Config 5 (Regularized)': {'hidden': 128, 'layers': 2, 'dropout': 0.5}
}

# Use the actual model weights
if 'model_state_dict' in state_dict:
    weights = state_dict['model_state_dict']
else:
    weights = state_dict

if 'lstm.weight_ih_l0' in weights:
    model_hidden = weights['lstm.weight_ih_l0'].shape[0] // 4
    model_layers = sum(1 for key in weights.keys() if 'lstm.weight_ih_l' in key)
    
    print(f"Model has: {model_layers} LSTM layers, {model_hidden} hidden dim")
    print("\nPotential match:")
    for name, cfg in configs.items():
        if cfg['hidden'] == model_hidden and cfg['layers'] == model_layers:
            print(f"  ✓ {name}")
            print(f"    (Hidden: {cfg['hidden']}, Layers: {cfg['layers']}, Dropout: {cfg['dropout']})")

# Show checkpoint metadata
if 'epoch' in state_dict and 'val_acc' in state_dict and 'val_f1' in state_dict:
    print("\n" + "="*60)
    print("CHECKPOINT METADATA")
    print("="*60)
    print(f"Epoch: {state_dict['epoch']}")
    print(f"Validation Accuracy: {state_dict['val_acc']:.4f}")
    print(f"Validation F1 Score: {state_dict['val_f1']:.4f}")
