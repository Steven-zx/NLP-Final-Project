"""
Complete Hyperparameter Tuning Comparison - All 5 Configurations
"""

import pandas as pd
from datetime import datetime

# All 5 configurations results
all_results = {
    'Configuration': [
        'Config 1: Baseline (10 epochs)',
        'Config 2: Extended (15 epochs)',
        'Config 3: Fast (20 epochs)',
        'Config 4: Deep Model (3 layers, 12 epochs)',
        'Config 5: Regularized (High dropout, 10 epochs)'
    ],
    'Learning Rate': [0.001, 0.0005, 0.01, 0.0008, 0.001],
    'Batch Size': [64, 32, 128, 64, 64],
    'Hidden Dim': [128, 256, 64, 128, 128],
    'Dropout': [0.3, 0.4, 0.2, 0.35, 0.5],
    'Epochs': [10, 15, 20, 12, 10],
    'LSTM Layers': [2, 2, 2, 3, 2],
    'Optimizer': ['Adam', 'Adam', 'RMSprop', 'Adam', 'SGD'],
    'Best Val Acc': [0.7042, 0.6949, 0.7039, 0.7053, 0.5432],
    'Final F1': [0.6426, 0.6479, 0.6757, 0.6650, 0.0000],
    'Final Precision': [0.6863, 0.6656, 0.6572, 0.6605, 0.0000],
    'Final Recall': [0.6042, 0.6311, 0.6953, 0.6694, 0.0000],
    'Training Time (min)': [4.46, 17.93, 4.22, 7.27, 3.98]
}

df = pd.DataFrame(all_results)

# Create summary report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("="*100)
print("COMPLETE HYPERPARAMETER TUNING COMPARISON - ALL 5 CONFIGURATIONS")
print("="*100)
print()

# Display comparison table
print(df.to_string(index=False))
print()

# Analysis
print("="*100)
print("ANALYSIS & KEY FINDINGS")
print("="*100)
print()

best_acc_idx = df['Best Val Acc'].idxmax()
best_f1_idx = df['Final F1'].idxmax()
fastest_idx = df['Training Time (min)'].idxmin()

print(f"🏆 BEST ACCURACY: {df.loc[best_acc_idx, 'Configuration']}")
print(f"   Accuracy: {df.loc[best_acc_idx, 'Best Val Acc']:.4f} ({df.loc[best_acc_idx, 'Best Val Acc']*100:.2f}%)")
print(f"   F1-Score: {df.loc[best_acc_idx, 'Final F1']:.4f}")
print(f"   Training Time: {df.loc[best_acc_idx, 'Training Time (min)']:.2f} minutes")
print()

print(f"🎯 BEST F1-SCORE: {df.loc[best_f1_idx, 'Configuration']}")
print(f"   Accuracy: {df.loc[best_f1_idx, 'Best Val Acc']:.4f} ({df.loc[best_f1_idx, 'Best Val Acc']*100:.2f}%)")
print(f"   F1-Score: {df.loc[best_f1_idx, 'Final F1']:.4f}")
print(f"   Training Time: {df.loc[best_f1_idx, 'Training Time (min)']:.2f} minutes")
print()

print(f"⚡ FASTEST TRAINING: {df.loc[fastest_idx, 'Configuration']}")
print(f"   Accuracy: {df.loc[fastest_idx, 'Best Val Acc']:.4f} ({df.loc[fastest_idx, 'Best Val Acc']*100:.2f}%)")
print(f"   F1-Score: {df.loc[fastest_idx, 'Final F1']:.4f}")
print(f"   Training Time: {df.loc[fastest_idx, 'Training Time (min)']:.2f} minutes")
print()

print("="*100)
print("DETAILED INSIGHTS")
print("="*100)
print()

# Top performers (excluding failed config 5)
valid_configs = df[df['Final F1'] > 0.0]

print("✅ TOP 4 PERFORMING CONFIGURATIONS (excluding failed Config 5):")
print()
for idx in valid_configs.nlargest(4, 'Best Val Acc').index:
    print(f"{idx+1}. {df.loc[idx, 'Configuration']}")
    print(f"   Accuracy: {df.loc[idx, 'Best Val Acc']:.4f} | F1: {df.loc[idx, 'Final F1']:.4f} | Time: {df.loc[idx, 'Training Time (min)']:.1f}min")
    print()

print("❌ FAILED CONFIGURATION:")
print()
print("Config 5: High Dropout Regularization (50% dropout + SGD)")
print("   Issue: Model failed to learn - stuck at 54.3% (random guessing)")
print("   Cause: Dropout too high (0.5) + SGD optimizer too slow for this task")
print("   Lesson: High dropout (>0.4) can prevent learning, especially with SGD")
print()

print("="*100)
print("HYPERPARAMETER IMPACT ANALYSIS")
print("="*100)
print()

print("1. NUMBER OF LSTM LAYERS:")
print("   • 2 layers: 70.42% (Config 1), 70.39% (Config 3)")
print("   • 3 layers: 70.53% (Config 4)")
print("   → Deeper model (3 layers) gives SLIGHTLY better accuracy (+0.11%)")
print("   → But takes 63% longer to train (7.3 min vs 4.5 min)")
print()

print("2. DROPOUT RATE:")
print("   • 0.2: 70.39% accuracy, 67.57% F1 (Config 3)")
print("   • 0.3: 70.42% accuracy, 64.26% F1 (Config 1)")
print("   • 0.35: 70.53% accuracy, 66.50% F1 (Config 4)")
print("   • 0.4: 69.49% accuracy, 64.79% F1 (Config 2)")
print("   • 0.5: FAILED - 54.32% (Config 5)")
print("   → Optimal range: 0.2 - 0.35")
print("   → 0.5 is TOO HIGH and prevents learning")
print()

print("3. OPTIMIZER:")
print("   • Adam: 70.42% (Config 1), 69.49% (Config 2), 70.53% (Config 4)")
print("   • RMSprop: 70.39% (Config 3)")
print("   • SGD: FAILED - 54.32% (Config 5)")
print("   → Adam and RMSprop work well")
print("   → SGD needs careful tuning (momentum, learning rate scheduling)")
print()

print("4. TRAINING DURATION:")
print("   • 10 epochs: 70.42% (sufficient)")
print("   • 15 epochs: 69.49% (no improvement)")
print("   • 20 epochs: 70.39% (no improvement)")
print("   • 12 epochs: 70.53% (slight improvement with deeper model)")
print("   → 10-12 epochs is optimal - more doesn't help")
print()

print("="*100)
print("FINAL RECOMMENDATIONS")
print("="*100)
print()

print("🎯 FOR PRODUCTION USE:")
print("   Recommendation: Config 4 (Deep Model)")
print("   Reason: Highest accuracy (70.53%) with good F1-score (66.50%)")
print("   Trade-off: 63% longer training time is acceptable for best performance")
print()

print("⚡ FOR FAST PROTOTYPING:")
print("   Recommendation: Config 3 (Fast Training)")
print("   Reason: Best F1-score (67.57%), fastest training (4.2 min)")
print("   Trade-off: Only 0.14% less accuracy than best model")
print()

print("📊 FOR BALANCED PERFORMANCE:")
print("   Recommendation: Config 1 (Baseline)")
print("   Reason: Good accuracy (70.42%), reasonable training time (4.5 min)")
print("   Trade-off: Standard 2-layer model with proven configuration")
print()

print("="*100)
print("LESSONS LEARNED")
print("="*100)
print()

print("✅ What Works:")
print("   • Adam optimizer with learning rate 0.0008-0.001")
print("   • Dropout between 0.2-0.35")
print("   • 10-12 epochs sufficient")
print("   • Deeper models (3 layers) give marginal improvement")
print()

print("❌ What Doesn't Work:")
print("   • High dropout (0.5) - prevents learning")
print("   • SGD optimizer without proper tuning")
print("   • Training beyond 12-15 epochs - no benefit")
print()

print("="*100)
print()

# Save to CSV
df.to_csv(f'complete_comparison_5configs_{timestamp}.csv', index=False)
print(f"✓ Saved comparison table: complete_comparison_5configs_{timestamp}.csv")

# Save detailed report
with open(f'complete_analysis_{timestamp}.txt', 'w') as f:
    f.write("="*100 + "\n")
    f.write("COMPLETE HYPERPARAMETER TUNING ANALYSIS - ALL 5 CONFIGURATIONS\n")
    f.write("="*100 + "\n\n")
    f.write(df.to_string(index=False))
    f.write("\n\n")
    
    f.write("BEST CONFIGURATION FOR PRODUCTION:\n")
    f.write(f"Config 4: Deep Model (3 LSTM layers, 12 epochs)\n")
    f.write(f"Accuracy: 70.53%\n")
    f.write(f"F1-Score: 66.50%\n\n")
    
    f.write("KEY FINDING:\n")
    f.write("High dropout (0.5) combined with SGD optimizer causes training failure.\n")
    f.write("Optimal dropout range is 0.2-0.35 with Adam optimizer.\n")

print(f"✓ Saved detailed analysis: complete_analysis_{timestamp}.txt")
print()
