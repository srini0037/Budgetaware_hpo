"""
Build Meta-Learning Dataset - 40 Samples
=========================================

Combines meta-features with experimental results to create
the final meta-learning dataset for training.

Input:
  - Meta-features: data/meta_features/all_datasets_metafeatures.csv
  - Experiments: results/hpo/ALL_EXPERIMENTS_1200_FINAL.csv

Output:
  - Meta-learning dataset: 40 samples (10 datasets × 4 budgets)
  - Each sample: meta-features + budget_seconds + best_method (target)

Author: Srinivas
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("BUILDING META-LEARNING DATASET - 40 SAMPLES")
print("=" * 80)

# Paths
project_root = Path("~/Budgetaware_hpo").expanduser()
meta_features_file = project_root / "data/meta_features/all_datasets_metafeatures.csv"
experiments_file = project_root / "results/hpo/ALL_EXPERIMENTS_1200_FINAL.csv"
output_dir = project_root / "results/meta_learning"
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("\n📊 Loading data...")
print(f"  Meta-features: {meta_features_file}")
print(f"  Experiments: {experiments_file}")

meta_df = pd.read_csv(meta_features_file)
exp_df = pd.read_csv(experiments_file)

print(f"\n✓ Loaded meta-features: {len(meta_df)} datasets × {len(meta_df.columns)-1} features")
print(f"✓ Loaded experiments: {len(exp_df)} experiments")

# Check datasets match
meta_datasets = set(meta_df['dataset'].values)
exp_datasets = set(exp_df['dataset'].unique())

print(f"\nDatasets in meta-features: {sorted(meta_datasets)}")
print(f"Datasets in experiments: {sorted(exp_datasets)}")

common_datasets = meta_datasets & exp_datasets
if len(common_datasets) < 10:
    print(f"\n⚠️  WARNING: Only {len(common_datasets)} datasets in common!")
    print(f"   Missing from meta-features: {exp_datasets - meta_datasets}")
    print(f"   Missing from experiments: {meta_datasets - exp_datasets}")
else:
    print(f"\n✓ All {len(common_datasets)} datasets present in both files")

# ============================================================================
# BUILD META-LEARNING DATASET
# ============================================================================

print("\n" + "=" * 80)
print("BUILDING META-LEARNING DATASET")
print("=" * 80)

budget_mapping = {
    'very_low': 60,
    'low': 120,
    'medium': 300,
    'high': 600
}

meta_learning_samples = []

for dataset in sorted(common_datasets):
    print(f"\n  Processing {dataset}...")
    
    # Get meta-features for this dataset
    dataset_meta = meta_df[meta_df['dataset'] == dataset].iloc[0].to_dict()
    
    for budget_level in ['very_low', 'low', 'medium', 'high']:
        # Get all experiments for this dataset-budget combination
        budget_exps = exp_df[
            (exp_df['dataset'] == dataset) & 
            (exp_df['budget_level'] == budget_level)
        ]
        
        if len(budget_exps) == 0:
            print(f"    ⚠️  No experiments for {budget_level}")
            continue
        
        # Calculate mean performance for each method
        method_performance = budget_exps.groupby('method')['best_score'].agg(['mean', 'std', 'count'])
        
        # Determine best method
        best_method = method_performance['mean'].idxmax()
        best_score = method_performance['mean'].max()
        
        # Get performance of all three methods
        rs_score = method_performance.loc['random_search', 'mean'] if 'random_search' in method_performance.index else np.nan
        sha_score = method_performance.loc['sha', 'mean'] if 'sha' in method_performance.index else np.nan
        hb_score = method_performance.loc['hyperband', 'mean'] if 'hyperband' in method_performance.index else np.nan
        
        # Create meta-learning sample
        sample = dataset_meta.copy()
        sample['budget_level'] = budget_level
        sample['budget_seconds'] = budget_mapping[budget_level]
        
        # Add method performance (useful for analysis)
        sample['random_search_score'] = rs_score
        sample['sha_score'] = sha_score
        sample['hyperband_score'] = hb_score
        
        # Target variable
        sample['best_method'] = best_method
        sample['best_score'] = best_score
        
        meta_learning_samples.append(sample)
        
        print(f"    {budget_level:10s} ({budget_mapping[budget_level]:3d}s): {best_method:15s} (F1={best_score:.4f})")

# Convert to DataFrame
ml_df = pd.DataFrame(meta_learning_samples)

print("\n" + "=" * 80)
print("META-LEARNING DATASET SUMMARY")
print("=" * 80)

print(f"\n✓ Created {len(ml_df)} meta-learning samples")
print(f"  ({len(common_datasets)} datasets × 4 budget levels)")

# Dataset distribution
print("\n📊 Samples per dataset:")
for ds in sorted(ml_df['dataset'].unique()):
    count = len(ml_df[ml_df['dataset'] == ds])
    print(f"  {ds:20s}: {count} samples")

# Budget distribution
print("\n📊 Samples per budget level:")
for budget in ['very_low', 'low', 'medium', 'high']:
    count = len(ml_df[ml_df['budget_level'] == budget])
    print(f"  {budget:10s}: {count} samples")

# Target variable distribution
print("\n📊 Best method distribution (target variable):")
method_counts = ml_df['best_method'].value_counts()
for method, count in method_counts.items():
    pct = 100 * count / len(ml_df)
    print(f"  {method:15s}: {count:2d} samples ({pct:.1f}%)")

# Feature count
n_features = len([col for col in ml_df.columns if col not in ['dataset', 'budget_level', 'best_method', 'best_score', 'random_search_score', 'sha_score', 'hyperband_score']])
print(f"\n📊 Features per sample: {n_features}")
print(f"   (including meta-features + budget_seconds)")

# Sample-to-feature ratio
ratio = len(ml_df) / n_features
print(f"\n📊 Sample-to-feature ratio: {len(ml_df)}:{n_features} = {ratio:.2f}:1")
if ratio < 2:
    print("   ⚠️  Low ratio - feature selection recommended!")
elif ratio < 5:
    print("   ⚠️  Moderate ratio - feature selection may help")
else:
    print("   ✓ Good ratio for machine learning")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING OUTPUTS")
print("=" * 80)

# Save full meta-learning dataset
output_file = output_dir / "meta_learning_dataset_40_samples.csv"
ml_df.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")
print(f"  Columns: {len(ml_df.columns)}")
print(f"  Rows: {len(ml_df)}")

# Create feature-only version (for training)
feature_cols = [col for col in ml_df.columns if col not in ['dataset', 'budget_level', 'best_method', 'best_score', 'random_search_score', 'sha_score', 'hyperband_score']]
X_df = ml_df[feature_cols + ['best_method']]  # Features + target

features_file = output_dir / "meta_learning_features_and_target.csv"
X_df.to_csv(features_file, index=False)
print(f"\n✓ Saved feature matrix: {features_file}")
print(f"  Features: {len(feature_cols)}")
print(f"  Target: best_method")

# Save summary statistics
summary_file = output_dir / "meta_learning_summary.txt"
with open(summary_file, 'w') as f:
    f.write("META-LEARNING DATASET SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total samples: {len(ml_df)}\n")
    f.write(f"Datasets: {len(common_datasets)}\n")
    f.write(f"Budget levels: 4\n")
    f.write(f"Features: {len(feature_cols)}\n")
    f.write(f"Sample-to-feature ratio: {ratio:.2f}:1\n\n")
    
    f.write("TARGET DISTRIBUTION:\n")
    for method, count in method_counts.items():
        pct = 100 * count / len(ml_df)
        f.write(f"  {method}: {count} ({pct:.1f}%)\n")
    
    f.write(f"\nFEATURE LIST:\n")
    for i, feat in enumerate(feature_cols, 1):
        f.write(f"  {i:2d}. {feat}\n")

print(f"\n✓ Saved summary: {summary_file}")

# Display sample
print("\n" + "=" * 80)
print("SAMPLE DATA (first 3 rows)")
print("=" * 80)

display_cols = ['dataset', 'budget_level', 'budget_seconds', 'n_instances', 'n_features', 
                'n_classes', 'best_method', 'best_score']
available_cols = [col for col in display_cols if col in ml_df.columns]
print(ml_df[available_cols].head(3).to_string(index=False))

print("\n" + "=" * 80)
print("META-LEARNING DATASET READY!")
print("=" * 80)

print(f"\n📁 Output files:")
print(f"  1. {output_file.name}")
print(f"     - Complete dataset with all columns")
print(f"  2. {features_file.name}")
print(f"     - Features + target only (ready for ML)")
print(f"  3. {summary_file.name}")
print(f"     - Summary statistics")

print(f"\n🎯 Next steps:")
print(f"  1. ✅ Experiments complete (1,200)")
print(f"  2. ✅ Meta-features extracted (10 datasets)")
print(f"  3. ✅ Meta-learning dataset created (40 samples)")
print(f"  4. → Feature selection/ranking")
print(f"  5. → Train meta-learner with LODOCV")
print(f"  6. → Evaluate and compare with Phase 1")

print("\n" + "=" * 80)
