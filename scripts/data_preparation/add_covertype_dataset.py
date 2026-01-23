"""
Add Covertype Dataset - Run on Your Computer
=============================================

This script downloads Covertype, extracts meta-features, and prepares it
for budget-aware experiments.

Run this script on YOUR computer (not Claude's) where network access is available.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis, entropy
import time
import os

print("=" * 80)
print("ADDING COVERTYPE DATASET")
print("=" * 80)

# ============================================================================
# 1. Download Covertype Dataset
# ============================================================================
print("\n1. Downloading Covertype dataset from sklearn...")
print("   (This may take a few minutes on first run)")

try:
    covtype = fetch_covtype()
    X = covtype.data
    y = covtype.target
    
    print(f"   ✓ Downloaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Classes: {np.unique(y)}")
except Exception as e:
    print(f"   ✗ Error downloading: {e}")
    print("\n   Alternative: Download manually from:")
    print("   https://archive.ics.uci.edu/ml/datasets/covertype")
    exit(1)

# ============================================================================
# 2. Create Train/Val/Test Splits
# ============================================================================
print("\n2. Creating train/val/test splits (70/15/15)...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"   Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"   ✓ Features standardized")

# ============================================================================
# 3. Extract Meta-Features
# ============================================================================
print("\n3. Extracting 34 meta-features...")

meta_features = {}

# Simple features
meta_features['dataset'] = 'covertype'
meta_features['n_instances'] = X.shape[0]
meta_features['n_features'] = X.shape[1]
meta_features['n_classes'] = len(np.unique(y))
meta_features['dimensionality'] = X.shape[1] / X.shape[0]
meta_features['log_n_instances'] = np.log10(X.shape[0])
meta_features['log_n_features'] = np.log10(X.shape[1])
meta_features['log_dimensionality'] = np.log10(meta_features['dimensionality'])

# Class distribution
class_counts = pd.Series(y).value_counts()
class_probs = class_counts / len(y)

meta_features['class_prob_min'] = class_probs.min()
meta_features['class_prob_max'] = class_probs.max()
meta_features['class_prob_mean'] = class_probs.mean()
meta_features['class_prob_std'] = class_probs.std()
meta_features['class_imbalance_ratio'] = class_probs.max() / class_probs.min()

# Statistical features (sample for speed)
X_sample = X_train[:10000] if len(X_train) > 10000 else X_train

skewness_values = skew(X_sample, axis=0)
kurtosis_values = kurtosis(X_sample, axis=0)

meta_features['skewness_min'] = np.min(skewness_values)
meta_features['skewness_max'] = np.max(skewness_values)
meta_features['skewness_mean'] = np.mean(skewness_values)
meta_features['skewness_std'] = np.std(skewness_values)

meta_features['kurtosis_min'] = np.min(kurtosis_values)
meta_features['kurtosis_max'] = np.max(kurtosis_values)
meta_features['kurtosis_mean'] = np.mean(kurtosis_values)
meta_features['kurtosis_std'] = np.std(kurtosis_values)

# Correlation features
corr_matrix = np.corrcoef(X_sample.T)
mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
correlations = corr_matrix[mask]

meta_features['correlation_min'] = np.min(correlations)
meta_features['correlation_max'] = np.max(correlations)
meta_features['correlation_mean'] = np.mean(correlations)
meta_features['correlation_std'] = np.std(correlations)

# Information-theoretic
class_entropy = entropy(class_probs)
max_entropy = np.log2(meta_features['n_classes'])

meta_features['class_entropy'] = class_entropy
meta_features['normalized_class_entropy'] = class_entropy / max_entropy if max_entropy > 0 else 0

# PCA features
print("   Computing PCA...")
pca_full = PCA()
pca_full.fit(X_sample)

cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1

meta_features['pca_first_pc_variance'] = pca_full.explained_variance_ratio_[0]
meta_features['pca_95_percent_dims'] = n_components_95
meta_features['pca_95_percent_ratio'] = n_components_95 / X.shape[1]

# Timing features (NOVEL contribution!)
print("   Computing timing features (tree, NN, NB)...")

sample_size = min(5000, len(X_train))

# Decision tree
start = time.time()
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train_scaled[:sample_size], y_train[:sample_size])
meta_features['tree_time'] = time.time() - start
meta_features['tree_depth'] = tree.get_depth()
meta_features['tree_n_leaves'] = tree.get_n_leaves()

# Neural network  
start = time.time()
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, random_state=42)
nn.fit(X_train_scaled[:sample_size], y_train[:sample_size])
meta_features['nn_time'] = time.time() - start

# Naive Bayes
start = time.time()
nb = GaussianNB()
nb.fit(X_train_scaled[:sample_size], y_train[:sample_size])
meta_features['nb_time'] = time.time() - start

print(f"   ✓ Extracted {len(meta_features)-1} meta-features")

# ============================================================================
# 4. Save Meta-Features
# ============================================================================
print("\n4. Saving meta-features...")

meta_df = pd.DataFrame([meta_features])

# Save individual file
os.makedirs('data/meta_features', exist_ok=True)
meta_df.to_csv('data/meta_features/covertype_meta_features.csv', index=False)
print("   ✓ Saved: data/meta_features/covertype_meta_features.csv")

# Update all_datasets file
all_meta_path = 'data/meta_features/all_datasets_metafeatures.csv'
if os.path.exists(all_meta_path):
    existing = pd.read_csv(all_meta_path)
    existing = existing[existing['dataset'] != 'covertype']  # Remove if exists
    updated = pd.concat([existing, meta_df], ignore_index=True)
    updated.to_csv(all_meta_path, index=False)
    print(f"   ✓ Updated: {all_meta_path} (now {len(updated)} datasets)")
else:
    meta_df.to_csv(all_meta_path, index=False)
    print(f"   ✓ Created: {all_meta_path}")

# ============================================================================
# 5. Save Processed Data
# ============================================================================
print("\n5. Saving processed train/val/test splits...")

os.makedirs('data/processed/covertype', exist_ok=True)

np.save('data/processed/covertype/X_train.npy', X_train)
np.save('data/processed/covertype/X_val.npy', X_val)
np.save('data/processed/covertype/X_test.npy', X_test)
np.save('data/processed/covertype/y_train.npy', y_train)
np.save('data/processed/covertype/y_val.npy', y_val)
np.save('data/processed/covertype/y_test.npy', y_test)

print("   ✓ Saved to: data/processed/covertype/")
print(f"   Files: X_train.npy, X_val.npy, X_test.npy, y_train.npy, y_val.npy, y_test.npy")

# ============================================================================
# 6. Display Summary
# ============================================================================
print("\n" + "=" * 80)
print("COVERTYPE DATASET READY!")
print("=" * 80)

print("\nDataset Summary:")
print(f"  Instances:  {meta_features['n_instances']:,}")
print(f"  Features:   {meta_features['n_features']}")
print(f"  Classes:    {meta_features['n_classes']}")
print(f"  Imbalance:  {meta_features['class_imbalance_ratio']:.2f}:1")
print(f"  PCA 95%:    {meta_features['pca_95_percent_dims']}/{X.shape[1]} features ({meta_features['pca_95_percent_ratio']*100:.1f}%)")

print("\nNext Steps:")
print("1. Run budget-aware experiments: python run_covertype_budget_experiments.py")
print("2. Re-run meta-learning with 5 datasets")
print("=" * 80)
