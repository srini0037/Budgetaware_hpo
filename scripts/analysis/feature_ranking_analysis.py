"""
Feature Ranking and Selection for Meta-Learning (FIXED v2)
===========================================================

Properly handles NaN/Inf values and column removal

Author: Srinivas  
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE RANKING AND ANALYSIS")
print("=" * 80)

# Paths
project_root = Path("~/Budgetaware_hpo").expanduser()
data_file = project_root / "results/meta_learning/meta_learning_features_and_target.csv"
output_dir = project_root / "results/meta_learning/feature_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("\n📊 Loading meta-learning dataset...")
df = pd.read_csv(data_file)
print(f"✓ Loaded: {len(df)} samples × {len(df.columns)} columns")

# Separate features and target
target_col = 'best_method'
feature_cols = [col for col in df.columns if col != target_col]

X_raw = df[feature_cols].copy()
y = df[target_col]

print(f"\n  Features: {len(feature_cols)}")
print(f"  Target: {target_col}")

# ============================================================================
# DATA CLEANING
# ============================================================================

print("\n" + "=" * 80)
print("DATA CLEANING")
print("=" * 80)

print("\nChecking for problematic values...")

# Check for inf/nan
n_inf = np.isinf(X_raw.values).sum()
n_nan = np.isnan(X_raw.values).sum()

print(f"  Infinity values: {n_inf}")
print(f"  NaN values: {n_nan}")

# Replace inf with nan
X_clean = X_raw.replace([np.inf, -np.inf], np.nan)

# Check missing values per feature
missing_info = []
for col in X_clean.columns:
    n_missing = X_clean[col].isna().sum()
    if n_missing > 0:
        missing_info.append({
            'feature': col,
            'n_missing': n_missing,
            'pct_missing': 100 * n_missing / len(X_clean)
        })

if missing_info:
    print(f"\n⚠️  Found {len(missing_info)} features with missing values:")
    missing_df = pd.DataFrame(missing_info).sort_values('n_missing', ascending=False)
    print(missing_df.head(15).to_string(index=False))
    
    missing_file = output_dir / "features_with_missing_values.csv"
    missing_df.to_csv(missing_file, index=False)
    print(f"\n✓ Saved to: {missing_file.name}")

# Remove features that are 100% NaN
all_nan_features = [col for col in X_clean.columns if X_clean[col].isna().all()]
if all_nan_features:
    print(f"\n⚠️  Removing {len(all_nan_features)} features with 100% missing values:")
    for feat in all_nan_features:
        print(f"    - {feat}")
    X_clean = X_clean.drop(columns=all_nan_features)

# Now impute remaining missing values with median
print(f"\nImputing remaining missing values with median...")
for col in X_clean.columns:
    if X_clean[col].isna().any():
        median_val = X_clean[col].median()
        if np.isnan(median_val):  # If median is also NaN, use 0
            median_val = 0
        X_clean[col].fillna(median_val, inplace=True)

X = X_clean.copy()

print(f"✓ Data cleaned: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  Removed: {len(all_nan_features)} all-NaN features")
print(f"  Kept: {X.shape[1]} usable features")

# Verify no more inf/nan
assert not np.isinf(X.values).any(), "Still have infinity!"
assert not np.isnan(X.values).any(), "Still have NaN!"
print("✓ Clean data verified")

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n  Classes:")
for i, class_name in enumerate(le.classes_):
    count = (y == class_name).sum()
    print(f"    {i}: {class_name:15s} ({count} samples)")

# ============================================================================
# 1. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("1. CORRELATION ANALYSIS")
print("=" * 80)

print("\nComputing correlations...")
corr_matrix = X.corr().abs()

high_corr_pairs = []
threshold = 0.95

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > threshold:
            high_corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

print(f"\n✓ Found {len(high_corr_pairs)} highly correlated pairs (r > {threshold})")

if high_corr_pairs:
    corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
    print("\nTop 10:")
    print(corr_df.head(10).to_string(index=False))
    
    corr_file = output_dir / "highly_correlated_features.csv"
    corr_df.to_csv(corr_file, index=False)

# ============================================================================
# 2. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("2. RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y_encoded)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

importance_df = pd.DataFrame({
    'rank': range(1, len(X.columns) + 1),
    'feature': X.columns[indices],
    'importance': importances[indices],
    'importance_pct': 100 * importances[indices]
})

print(f"\n✓ Trained (accuracy: {rf.score(X, y_encoded):.2%})")
print("\n📊 Top 30 Features:")
print(importance_df.head(30).to_string(index=False))

rf_file = output_dir / "random_forest_importance.csv"
importance_df.to_csv(rf_file, index=False)
print(f"\n✓ Saved to: {rf_file.name}")

# ============================================================================
# 3. MUTUAL INFORMATION
# ============================================================================

print("\n" + "=" * 80)
print("3. MUTUAL INFORMATION")
print("=" * 80)

print("\nComputing MI scores...")
mi_scores = mutual_info_classif(X, y_encoded, random_state=42)

mi_df = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\n📊 Top 20:")
print(mi_df.head(20).to_string(index=False))

mi_file = output_dir / "mutual_information.csv"
mi_df.to_csv(mi_file, index=False)

# ============================================================================
# 4. ANOVA F-SCORES
# ============================================================================

print("\n" + "=" * 80)
print("4. ANOVA F-SCORES")
print("=" * 80)

print("\nComputing F-scores...")
f_scores, p_values = f_classif(X, y_encoded)

f_df = pd.DataFrame({
    'feature': X.columns,
    'f_score': f_scores,
    'p_value': p_values,
    'significant': p_values < 0.05
}).sort_values('f_score', ascending=False)

print("\n📊 Top 20:")
print(f_df.head(20).to_string(index=False))

n_sig = f_df['significant'].sum()
print(f"\n✓ Significant (p<0.05): {n_sig}/{len(f_df)}")

f_file = output_dir / "anova_fscores.csv"
f_df.to_csv(f_file, index=False)

# ============================================================================
# 5. CONSENSUS RANKING
# ============================================================================

print("\n" + "=" * 80)
print("5. CONSENSUS RANKING")
print("=" * 80)

# Normalize scores
def normalize(scores):
    return (scores - scores.min()) / (scores.max() - scores.min())

consensus = pd.DataFrame({'feature': X.columns})

consensus = consensus.merge(
    importance_df[['feature', 'importance_pct']],
    on='feature'
)
consensus = consensus.merge(
    mi_df[['feature', 'mi_score']],
    on='feature'
)
consensus = consensus.merge(
    f_df[['feature', 'f_score']],
    on='feature'
)

# Normalize and combine
consensus['rf_norm'] = normalize(consensus['importance_pct'])
consensus['mi_norm'] = normalize(consensus['mi_score'])
consensus['f_norm'] = normalize(consensus['f_score'])

consensus['consensus_score'] = (
    0.4 * consensus['rf_norm'] +
    0.3 * consensus['mi_norm'] +
    0.3 * consensus['f_norm']
)

consensus = consensus.sort_values('consensus_score', ascending=False)
consensus['rank'] = range(1, len(consensus) + 1)

print("\n📊 Top 30 by Consensus:")
cols = ['rank', 'feature', 'consensus_score', 'importance_pct', 'mi_score', 'f_score']
print(consensus[cols].head(30).to_string(index=False))

consensus_file = output_dir / "consensus_ranking.csv"
consensus.to_csv(consensus_file, index=False)
print(f"\n✓ Saved to: {consensus_file.name}")

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("6. CREATING VISUALIZATION")
print("=" * 80)

fig, ax = plt.subplots(figsize=(10, 8))
top_20 = consensus.head(20)

bars = ax.barh(range(len(top_20)), top_20['consensus_score'])

# Highlight budget_seconds
if 'budget_seconds' in top_20['feature'].values:
    idx = list(top_20['feature']).index('budget_seconds')
    bars[len(top_20)-1-idx].set_color('red')

ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'])
ax.invert_yaxis()
ax.set_xlabel('Consensus Score', fontweight='bold')
ax.set_title('Top 20 Features', fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plot_file = output_dir / 'top20_consensus.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved plot")

# ============================================================================
# 7. RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("7. RECOMMENDATIONS")
print("=" * 80)

cumsum = np.cumsum(importance_df['importance_pct'])
n_80 = np.searchsorted(cumsum, 80) + 1
n_90 = np.searchsorted(cumsum, 90) + 1

print(f"\n📊 Cumulative variance:")
print(f"  {n_80} features → 80%")
print(f"  {n_90} features → 90%")

for n in [10, 15, 20]:
    ratio = 40 / n
    print(f"\n{'='*60}")
    print(f"TOP {n} FEATURES (ratio {ratio:.1f}:1)")
    print(f"{'='*60}")
    selected = consensus.head(n)
    for i, row in selected.iterrows():
        marker = "🌟" if row['feature'] == 'budget_seconds' else "  "
        print(f"  {row['rank']:2d}. {marker} {row['feature']}")
    
    # Save
    out_file = output_dir / f"top{n}_features.txt"
    with open(out_file, 'w') as f:
        for _, row in selected.iterrows():
            f.write(f"{row['feature']}\n")
    print(f"\n✓ Saved: {out_file.name}")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\nOutputs: {output_dir}/")
print(f"\n💡 Recommendation: Use top 15 features")
print(f"   Next: Train meta-learner with selected features")