#!/usr/bin/env python3
"""
Meta-Learning Budget-Aware HPO
Train a meta-learner to predict optimal HPO methods based on dataset characteristics
and computational budgets.

Uses Leave-One-Dataset-Out Cross-Validation for robust evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("META-LEARNING BUDGET-AWARE HPO")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")

# Load meta-features
meta_features_path = Path("data/meta_features/all_datasets_metafeatures.csv")
if not meta_features_path.exists():
    print(f"   ERROR: {meta_features_path} not found!")
    print("   Run: python extract_meta_features.py")
    exit(1)

meta_features = pd.read_csv(meta_features_path)
print(f"   ✓ Meta-features loaded: {meta_features.shape}")
print(f"   Datasets: {meta_features['dataset'].unique().tolist()}")

# Load HPO results
hpo_results_path = Path("results/hpo/multi_dataset_budget_aware.csv")
if not hpo_results_path.exists():
    print(f"   ERROR: {hpo_results_path} not found!")
    exit(1)

hpo_results = pd.read_csv(hpo_results_path)
print(f"   ✓ HPO results loaded: {hpo_results.shape}")
print(f"   Budget levels: {sorted(hpo_results['budget_seconds'].unique())}")
print(f"   Methods: {hpo_results['method'].unique().tolist()}")

# ============================================================================
# 2. PREPARE TRAINING DATA
# ============================================================================
print("\n2. Preparing training data...")

# For each (dataset, budget) combination, find the best method
best_methods = []

for dataset in hpo_results['dataset'].unique():
    for budget in hpo_results['budget_seconds'].unique():
        subset = hpo_results[(hpo_results['dataset'] == dataset) & 
                            (hpo_results['budget_seconds'] == budget)]
        
        if len(subset) == 0:
            continue
            
        # Calculate mean performance per method
        method_performance = subset.groupby('method')['best_score'].mean()
        best_method = method_performance.idxmax()
        best_score = method_performance.max()
        
        best_methods.append({
            'dataset': dataset,
            'budget_seconds': budget,
            'best_method': best_method,
            'best_score': best_score,
            'n_runs': len(subset)
        })

best_methods_df = pd.DataFrame(best_methods)
print(f"   ✓ Best methods identified: {len(best_methods_df)} combinations")

# Merge with meta-features
training_data = best_methods_df.merge(meta_features, on='dataset')
print(f"   ✓ Training data prepared: {training_data.shape}")

# Select features for meta-learning
feature_cols = [col for col in training_data.columns 
                if col not in ['dataset', 'best_method', 'best_score', 'n_runs']]
print(f"   Features used: {len(feature_cols)}")
print(f"   {feature_cols}")

X = training_data[feature_cols].values
y = training_data['best_method'].values
datasets = training_data['dataset'].values

print(f"   X shape: {X.shape}, y shape: {y.shape}")
print(f"   Target distribution: {pd.Series(y).value_counts().to_dict()}")

# ============================================================================
# 3. LEAVE-ONE-DATASET-OUT CROSS-VALIDATION
# ============================================================================
print("\n3. Leave-One-Dataset-Out Cross-Validation...")

unique_datasets = np.unique(datasets)
n_datasets = len(unique_datasets)
print(f"   Datasets for LODO-CV: {n_datasets}")

lodo_results = []
lodo_predictions = []
feature_importances_all = []

for i, test_dataset in enumerate(unique_datasets):
    print(f"\n   Fold {i+1}/{n_datasets}: Testing on {test_dataset}")
    
    # Split data
    train_mask = datasets != test_dataset
    test_mask = datasets == test_dataset
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"      Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest meta-learner
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"      Accuracy: {accuracy:.4f}")
    
    # Store results
    for j in range(len(y_test)):
        lodo_predictions.append({
            'test_dataset': test_dataset,
            'budget_seconds': training_data[test_mask].iloc[j]['budget_seconds'],
            'true_method': y_test[j],
            'predicted_method': y_pred[j],
            'correct': y_test[j] == y_pred[j]
        })
    
    lodo_results.append({
        'test_dataset': test_dataset,
        'accuracy': accuracy,
        'n_test': len(y_test)
    })
    
    # Store feature importances
    feature_importances_all.append(rf.feature_importances_)

lodo_results_df = pd.DataFrame(lodo_results)
lodo_predictions_df = pd.DataFrame(lodo_predictions)

print("\n" + "=" * 80)
print("LODO-CV Results:")
print(lodo_results_df.to_string(index=False))
print(f"\nOverall Accuracy: {lodo_results_df['accuracy'].mean():.4f} "
      f"(±{lodo_results_df['accuracy'].std():.4f})")

# ============================================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n4. Feature Importance Analysis...")

# Average feature importances across folds
avg_importances = np.mean(feature_importances_all, axis=0)
std_importances = np.std(feature_importances_all, axis=0)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': avg_importances,
    'std': std_importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(importance_df.head(10).to_string(index=False))

# Save feature importances
importance_df.to_csv('results/meta_learning_feature_importance.csv', index=False)
print(f"   ✓ Saved: results/meta_learning_feature_importance.csv")

# ============================================================================
# 5. BASELINE COMPARISONS
# ============================================================================
print("\n5. Baseline Comparisons...")

# Oracle: Always pick the best method (upper bound)
oracle_correct = lodo_predictions_df['correct'].sum()
oracle_accuracy = oracle_correct / len(lodo_predictions_df)

# Always-Random-Search baseline
always_rs_correct = (lodo_predictions_df['true_method'] == 'random_search').sum()
always_rs_accuracy = always_rs_correct / len(lodo_predictions_df)

# Always-SHA baseline
always_sha_correct = (lodo_predictions_df['true_method'] == 'sha').sum()
always_sha_accuracy = always_sha_correct / len(lodo_predictions_df)

# Always-Hyperband baseline
always_hb_correct = (lodo_predictions_df['true_method'] == 'hyperband').sum()
always_hb_accuracy = always_hb_correct / len(lodo_predictions_df)

meta_learner_accuracy = lodo_results_df['accuracy'].mean()

baselines = pd.DataFrame({
    'Strategy': ['Oracle (Upper Bound)', 'Meta-Learner (Ours)', 
                 'Always-Random-Search', 'Always-SHA', 'Always-Hyperband'],
    'Accuracy': [oracle_accuracy, meta_learner_accuracy, 
                 always_rs_accuracy, always_sha_accuracy, always_hb_accuracy],
    'Improvement_over_best_baseline': [
        np.nan,
        meta_learner_accuracy - max(always_rs_accuracy, always_sha_accuracy, always_hb_accuracy),
        np.nan,
        np.nan,
        np.nan
    ]
})

print("\n" + "=" * 80)
print("BASELINE COMPARISONS")
print("=" * 80)
print(baselines.to_string(index=False))

# Statistical test: Meta-learner vs Always-Hyperband
meta_correct = lodo_predictions_df['correct'].astype(int).values
always_hb_correct = (lodo_predictions_df['true_method'] == 'hyperband').astype(int).values

# Paired t-test
t_stat, p_value = stats.ttest_rel(meta_correct, always_hb_correct)
print(f"\nPaired t-test (Meta-Learner vs Always-Hyperband):")
print(f"   t-statistic: {t_stat:.4f}")
print(f"   p-value: {p_value:.4f}")
print(f"   Significant at α=0.05: {'YES' if p_value < 0.05 else 'NO'}")

# ============================================================================
# 6. PERFORMANCE IMPROVEMENT ANALYSIS
# ============================================================================
print("\n6. Performance Improvement Analysis...")

# Merge predictions with actual scores
performance_analysis = lodo_predictions_df.merge(
    training_data[['dataset', 'budget_seconds', 'best_method', 'best_score']],
    left_on=['test_dataset', 'budget_seconds', 'predicted_method'],
    right_on=['dataset', 'budget_seconds', 'best_method'],
    how='left'
)

# Get oracle (true best) scores
oracle_scores = training_data[['dataset', 'budget_seconds', 'best_method', 'best_score']].copy()
oracle_scores = oracle_scores.rename(columns={'best_score': 'oracle_score'})

performance_analysis = performance_analysis.merge(
    oracle_scores[['dataset', 'budget_seconds', 'oracle_score']],
    left_on=['test_dataset', 'budget_seconds'],
    right_on=['dataset', 'budget_seconds'],
    how='left'
)

# Calculate prediction accuracy vs Oracle
performance_analysis['score_ratio'] = (performance_analysis['best_score'] / 
                                       performance_analysis['oracle_score'])

mean_score_ratio = performance_analysis['score_ratio'].mean()
print(f"\nMeta-Learner achieves {mean_score_ratio:.2%} of Oracle performance")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n7. Saving results...")

# Save LODO results
lodo_results_df.to_csv('results/meta_learning_lodo_results.csv', index=False)
print(f"   ✓ Saved: results/meta_learning_lodo_results.csv")

# Save predictions
lodo_predictions_df.to_csv('results/meta_learning_predictions.csv', index=False)
print(f"   ✓ Saved: results/meta_learning_predictions.csv")

# Save baselines
baselines.to_csv('results/meta_learning_baselines.csv', index=False)
print(f"   ✓ Saved: results/meta_learning_baselines.csv")

# Save summary JSON
summary = {
    'n_datasets': n_datasets,
    'n_training_samples': len(training_data),
    'n_features': len(feature_cols),
    'lodo_cv_accuracy': float(meta_learner_accuracy),
    'lodo_cv_accuracy_std': float(lodo_results_df['accuracy'].std()),
    'oracle_accuracy': float(oracle_accuracy),
    'always_random_search_accuracy': float(always_rs_accuracy),
    'always_sha_accuracy': float(always_sha_accuracy),
    'always_hyperband_accuracy': float(always_hb_accuracy),
    'improvement_over_best_baseline': float(meta_learner_accuracy - max(always_rs_accuracy, always_sha_accuracy, always_hb_accuracy)),
    'score_ratio_vs_oracle': float(mean_score_ratio),
    'paired_ttest_vs_hyperband': {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_at_0.05': p_value < 0.05
    },
    'top_features': importance_df.head(5)[['feature', 'importance']].to_dict('records')
}

with open('results/meta_learning_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"   ✓ Saved: results/meta_learning_summary.json")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n8. Creating visualizations...")

# Create figures directory
Path('figures/meta_learning').mkdir(parents=True, exist_ok=True)

# 8.1 Feature Importance Plot
plt.figure(figsize=(10, 6))
top_n = 10
top_features = importance_df.head(top_n)
plt.barh(range(top_n), top_features['importance'].values)
plt.yticks(range(top_n), top_features['feature'].values)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 10 Meta-Features by Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/meta_learning/feature_importance.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: figures/meta_learning/feature_importance.png")
plt.close()

# 8.2 LODO-CV Accuracy by Dataset
plt.figure(figsize=(10, 6))
datasets_sorted = lodo_results_df.sort_values('accuracy')
plt.barh(range(len(datasets_sorted)), datasets_sorted['accuracy'].values)
plt.yticks(range(len(datasets_sorted)), datasets_sorted['test_dataset'].values)
plt.xlabel('Accuracy', fontsize=12)
plt.ylabel('Test Dataset', fontsize=12)
plt.title('LODO-CV Accuracy by Test Dataset', fontsize=14, fontweight='bold')
plt.axvline(meta_learner_accuracy, color='red', linestyle='--', 
            label=f'Overall Mean: {meta_learner_accuracy:.3f}')
plt.legend()
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('figures/meta_learning/lodo_accuracy_by_dataset.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: figures/meta_learning/lodo_accuracy_by_dataset.png")
plt.close()

# 8.3 Baseline Comparison
plt.figure(figsize=(10, 6))
baseline_plot = baselines.sort_values('Accuracy')
colors = ['lightcoral' if 'baseline' in str(s).lower() or 'Always' in str(s) 
          else 'lightgreen' if 'Meta' in str(s) 
          else 'lightblue' 
          for s in baseline_plot['Strategy']]
plt.barh(range(len(baseline_plot)), baseline_plot['Accuracy'].values, color=colors)
plt.yticks(range(len(baseline_plot)), baseline_plot['Strategy'].values)
plt.xlabel('Accuracy', fontsize=12)
plt.ylabel('Strategy', fontsize=12)
plt.title('Meta-Learner vs Baseline Strategies', fontsize=14, fontweight='bold')
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('figures/meta_learning/baseline_comparison.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: figures/meta_learning/baseline_comparison.png")
plt.close()

# 8.4 Confusion Matrix (aggregated across all folds)
methods = sorted(lodo_predictions_df['true_method'].unique())
cm = confusion_matrix(lodo_predictions_df['true_method'], 
                      lodo_predictions_df['predicted_method'],
                      labels=methods)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=methods, yticklabels=methods)
plt.xlabel('Predicted Method', fontsize=12)
plt.ylabel('True Method', fontsize=12)
plt.title('Confusion Matrix (All LODO Folds)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/meta_learning/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: figures/meta_learning/confusion_matrix.png")
plt.close()

print("\n" + "=" * 80)
print("META-LEARNING TRAINING COMPLETE!")
print("=" * 80)
print(f"\nKey Results:")
print(f"  • LODO-CV Accuracy: {meta_learner_accuracy:.2%}")
print(f"  • Improvement over best baseline: {(meta_learner_accuracy - max(always_rs_accuracy, always_sha_accuracy, always_hb_accuracy)):.2%}")
print(f"  • Performance vs Oracle: {mean_score_ratio:.2%}")
print(f"  • Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.2%})")
print(f"\nFiles saved to:")
print(f"  • results/meta_learning_*.csv")
print(f"  • results/meta_learning_summary.json")
print(f"  • figures/meta_learning/*.png")
print("=" * 80)
