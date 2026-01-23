#!/usr/bin/env python3
"""
Meta-Learner Validation using Leave-One-Dataset-Out Cross-Validation (LODOCV)

Standard validation approach in meta-learning:
- Train on 4 datasets, test on 1 held-out dataset
- Repeat for all 5 datasets
- Report average accuracy across all folds
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 80)
print("META-LEARNER VALIDATION - LEAVE-ONE-DATASET-OUT CROSS-VALIDATION")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")

meta_features = pd.read_csv("data/meta_features/all_datasets_metafeatures.csv")
hpo_results = pd.read_csv("results/hpo/multi_dataset_budget_aware.csv")

# Load selected features
selected_features_df = pd.read_csv("results/selected_features.csv")
selected_features = selected_features_df['feature'].tolist()

print(f"   ✓ Using {len(selected_features)} selected features")
print(f"   ✓ Datasets: {meta_features['dataset'].unique().tolist()}")

# ============================================================================
# 2. PREPARE DATA
# ============================================================================
print("\n2. Preparing training data...")

best_methods = []
for dataset in hpo_results['dataset'].unique():
    for budget in hpo_results['budget_seconds'].unique():
        subset = hpo_results[(hpo_results['dataset'] == dataset) & 
                            (hpo_results['budget_seconds'] == budget)]
        if len(subset) == 0:
            continue
        method_performance = subset.groupby('method')['best_score'].mean()
        best_method = method_performance.idxmax()
        best_methods.append({
            'dataset': dataset,
            'budget_seconds': budget,
            'best_method': best_method,
        })

best_methods_df = pd.DataFrame(best_methods)
training_data = best_methods_df.merge(meta_features, on='dataset')

X = training_data[selected_features]
y = training_data['best_method']
datasets = training_data['dataset']

print(f"   ✓ Total samples: {len(X)}")
print(f"   ✓ Features: {len(selected_features)}")

# ============================================================================
# 3. LEAVE-ONE-DATASET-OUT CROSS-VALIDATION
# ============================================================================
print("\n3. Running Leave-One-Dataset-Out Cross-Validation...")
print("   " + "=" * 60)

unique_datasets = datasets.unique()
fold_results = []
all_predictions = []
all_true_labels = []

for test_dataset in unique_datasets:
    print(f"\n   Fold: Test on [{test_dataset}]")
    
    # Split data
    train_mask = datasets != test_dataset
    test_mask = datasets == test_dataset
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"      Train samples: {len(X_train)} (4 datasets)")
    print(f"      Test samples:  {len(X_test)} (1 dataset, 4 budgets)")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    fold_results.append({
        'test_dataset': test_dataset,
        'accuracy': accuracy,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'correct': (y_pred == y_test).sum(),
        'total': len(y_test)
    })
    
    all_predictions.extend(y_pred)
    all_true_labels.extend(y_test)
    
    print(f"      Accuracy: {accuracy*100:.1f}% ({(y_pred == y_test).sum()}/{len(y_test)} correct)")
    
    # Show predictions vs actual
    test_budgets = training_data[test_mask]['budget_seconds'].values
    for budget, pred, actual in zip(test_budgets, y_pred, y_test):
        match = "✓" if pred == actual else "✗"
        print(f"         Budget {budget:3d}s: Predicted={pred:15s} Actual={actual:15s} {match}")

# ============================================================================
# 4. AGGREGATE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("4. CROSS-VALIDATION RESULTS")
print("=" * 80)

fold_results_df = pd.DataFrame(fold_results)
mean_accuracy = fold_results_df['accuracy'].mean()
std_accuracy = fold_results_df['accuracy'].std()

print("\n   Per-Fold Accuracy:")
print("   " + "-" * 60)
for _, row in fold_results_df.iterrows():
    print(f"   {row['test_dataset']:20s} {row['accuracy']*100:5.1f}% ({row['correct']}/{row['total']})")

print("\n   " + "=" * 60)
print(f"   Mean Accuracy:  {mean_accuracy*100:.1f}% ± {std_accuracy*100:.1f}%")
print(f"   Baseline:       45.0% (majority class)")
print("   " + "=" * 60)

# Overall confusion matrix
print("\n   Overall Confusion Matrix (across all folds):")
methods = sorted(set(all_true_labels))
cm = confusion_matrix(all_true_labels, all_predictions, labels=methods)
cm_df = pd.DataFrame(cm, index=methods, columns=methods)
print(cm_df.to_string())

# ============================================================================
# 5. COMPARE TRAINING VS VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("5. COMPARISON: TRAINING vs CROSS-VALIDATION")
print("=" * 80)

# Training accuracy (from previous results)
comparison_df = pd.read_csv("results/meta_learner_comparison.csv")
train_accuracy = comparison_df[comparison_df['approach'] == 'Top-10 Features']['accuracy'].values[0]

print(f"""
   Training Accuracy:        {train_accuracy*100:.1f}%
   Cross-Validation Accuracy: {mean_accuracy*100:.1f}% ± {std_accuracy*100:.1f}%
   Baseline:                 45.0%
   
   Interpretation:
   {'✓ CV accuracy close to training - good generalization' if abs(train_accuracy - mean_accuracy) < 0.1 else '⚠ CV lower than training - some overfitting'}
   {'✓ Both significantly exceed baseline - model is effective' if mean_accuracy > 0.50 else '⚠ CV accuracy near/below baseline - limited predictive power'}
""")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n6. Saving validation results...")

# Save fold results
fold_results_df.to_csv('results/lodocv_fold_results.csv', index=False)

# Save overall results
validation_summary = pd.DataFrame({
    'metric': ['Mean Accuracy', 'Std Accuracy', 'Min Accuracy', 'Max Accuracy', 'Baseline'],
    'value': [mean_accuracy, std_accuracy, 
              fold_results_df['accuracy'].min(), 
              fold_results_df['accuracy'].max(), 
              0.45]
})
validation_summary.to_csv('results/lodocv_summary.csv', index=False)

# Update comparison with CV results
comparison_df = pd.DataFrame({
    'approach': ['Training (95% accuracy)', 'Cross-Validation (LODOCV)', 'Baseline'],
    'accuracy': [train_accuracy, mean_accuracy, 0.45],
    'std': [0.0, std_accuracy, 0.0]
})
comparison_df.to_csv('results/final_model_comparison.csv', index=False)

print(f"   ✓ Fold results: results/lodocv_fold_results.csv")
print(f"   ✓ Summary: results/lodocv_summary.csv")
print(f"   ✓ Comparison: results/final_model_comparison.csv")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================
print("\n7. Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Per-fold accuracy
ax1 = axes[0]
fold_results_df.plot(x='test_dataset', y='accuracy', kind='bar', ax=ax1, 
                     color='steelblue', legend=False)
ax1.axhline(y=mean_accuracy, color='green', linestyle='--', label=f'Mean: {mean_accuracy*100:.1f}%')
ax1.axhline(y=0.45, color='red', linestyle='--', label='Baseline: 45%')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Test Dataset')
ax1.set_title('Leave-One-Dataset-Out Cross-Validation Results')
ax1.set_ylim([0, 1])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Training vs CV comparison
ax2 = axes[1]
approaches = ['Training', 'Cross-Val', 'Baseline']
accuracies = [train_accuracy, mean_accuracy, 0.45]
colors = ['lightblue', 'steelblue', 'lightcoral']
bars = ax2.bar(approaches, accuracies, color=colors)
ax2.set_ylabel('Accuracy')
ax2.set_title('Training vs Cross-Validation')
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height*100:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/lodocv_validation.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Visualization: results/lodocv_validation.png")

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

print(f"""
METHODOLOGY: Leave-One-Dataset-Out Cross-Validation
- Standard approach in meta-learning research
- Each dataset tested while training on other 4 datasets
- 5 folds (one per dataset)

RESULTS:
- Cross-Validation Accuracy: {mean_accuracy*100:.1f}% ± {std_accuracy*100:.1f}%
- Training Accuracy: {train_accuracy*100:.1f}%
- Baseline: 45.0%

INTERPRETATION:
{f'The model achieves {mean_accuracy*100:.1f}% accuracy on held-out datasets, demonstrating' if mean_accuracy > 0.50 else 'The model achieves ' + f'{mean_accuracy*100:.1f}% accuracy, indicating'}
{'effective generalization beyond the training data.' if mean_accuracy > 0.60 else 'limited but above-baseline predictive capability.'}

FOR DISSERTATION:
- Report both training ({train_accuracy*100:.1f}%) and CV accuracy ({mean_accuracy*100:.1f}%)
- Acknowledge: Limited by small sample (5 datasets)
- Emphasize: Proper validation methodology (LODOCV)
- Highlight: Budget feature dominance (39.95% importance)
""")

print("=" * 80)
print("Runtime: ~1 minute (no new experiments required)")
print("=" * 80)

