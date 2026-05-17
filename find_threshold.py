"""
Test model with various texts and find optimal threshold
"""

import torch
from rnn_model import BiLSTMHateSpeechClassifier
from text_preprocessing import TextPreprocessor, Vocabulary, pad_sequences

# Load model
device = torch.device('cpu')
vocab = Vocabulary.load('vocabulary.pkl')

preprocessor = TextPreprocessor(
    lowercase=True,
    remove_urls=True,
    remove_mentions=True,
    remove_hashtags=False,
    remove_punctuation=False
)

model = BiLSTMHateSpeechClassifier(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3,
    pad_idx=vocab.word2idx[vocab.PAD_TOKEN]
).to(device)

checkpoint = torch.load('best_bilstm_model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Testing with various thresholds:")
print("="*80)

# Test cases - clearly non-hate
non_hate_texts = [
    "you are the best thing in the world",
    "thank you so much for your help",
    "i love this product",
    "have a great day",
    "congratulations on your success",
    "you're amazing and talented",
    "this is wonderful news",
    "i appreciate your kindness"
]

# Test cases - clearly hate
hate_texts = [
    "you're a stupid idiot",
    "i hate you so much",
    "kill yourself",
    "you're worthless trash",
    "tang ina mo gago ka",
    "go die in a hole"
]

def predict_with_threshold(text, threshold=0.5):
    processed = preprocessor.preprocess(text)
    sequence = vocab.text_to_sequence(processed, max_length=100)
    
    if len(sequence) == 0:
        sequence = [vocab.word2idx[vocab.UNK_TOKEN]]
    
    padded_seq, length = pad_sequences([sequence], max_length=100)
    text_tensor = torch.LongTensor(padded_seq).to(device)
    length_tensor = torch.LongTensor(length).to(device)
    
    with torch.no_grad():
        output = model(text_tensor, length_tensor)
        probability = torch.sigmoid(output).item()
    
    return probability

# Test with different thresholds
thresholds = [0.5, 0.6, 0.7, 0.75, 0.8]

for threshold in thresholds:
    print(f"\n{'='*80}")
    print(f"THRESHOLD: {threshold}")
    print(f"{'='*80}")
    
    # Test non-hate
    print("\nNON-HATE TEXTS (should be below threshold):")
    print("-"*80)
    non_hate_errors = 0
    for text in non_hate_texts:
        prob = predict_with_threshold(text, threshold)
        is_hate = prob >= threshold
        status = "❌ WRONG" if is_hate else "✅ CORRECT"
        if is_hate:
            non_hate_errors += 1
        print(f"{status} | Prob: {prob:.3f} | {text}")
    
    # Test hate
    print("\nHATE SPEECH TEXTS (should be above threshold):")
    print("-"*80)
    hate_errors = 0
    for text in hate_texts:
        prob = predict_with_threshold(text, threshold)
        is_hate = prob >= threshold
        status = "✅ CORRECT" if is_hate else "❌ WRONG"
        if not is_hate:
            hate_errors += 1
        print(f"{status} | Prob: {prob:.3f} | {text}")
    
    total_errors = non_hate_errors + hate_errors
    accuracy = (len(non_hate_texts) + len(hate_texts) - total_errors) / (len(non_hate_texts) + len(hate_texts)) * 100
    
    print(f"\nRESULTS: {non_hate_errors} false positives, {hate_errors} false negatives")
    print(f"Test Accuracy: {accuracy:.1f}%")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("Choose the threshold with best balance (fewest total errors)")
