"""
Merge All 1,200 Budget-Aware HPO Experiments
============================================

Combines results from:
1. multi_dataset_budget_aware.csv (First 5 datasets)
2. new_5_datasets_results.csv (Second 5 datasets)

Output: Single master file with all 1,200 experiments
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("MERGING ALL BUDGET-AWARE HPO RESULTS")
print("=" * 80)

# File paths
results_dir = Path("~/Budgetaware_hpo/results/hpo").expanduser()
file1 = results_dir / "multi_dataset_budget_aware.csv"
file2 = results_dir / "new_5_datasets_results.csv"

# Check files exist
if not file1.exists():
    print(f"❌ ERROR: File not found: {file1}")
    exit(1)

if not file2.exists():
    print(f"❌ ERROR: File not found: {file2}")
    exit(1)

print(f"\n✓ Found file 1: {file1.name}")
print(f"✓ Found file 2: {file2.name}")

# Load the files
print("\nLoading files...")
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print(f"  File 1: {len(df1)} experiments")
print(f"  File 2: {len(df2)} experiments")

# Show what datasets are in each file
print(f"\n  File 1 datasets: {sorted(df1['dataset'].unique())}")
print(f"  File 2 datasets: {sorted(df2['dataset'].unique())}")

# Check columns
print("\nColumns in File 1:")
print(f"  {list(df1.columns)}")
print("\nColumns in File 2:")
print(f"  {list(df2.columns)}")

# Ensure columns match
if list(df1.columns) != list(df2.columns):
    print("\n⚠️  WARNING: Column order differs, aligning...")
    # Use the same column order
    common_cols = [col for col in df1.columns if col in df2.columns]
    df1 = df1[common_cols]
    df2 = df2[common_cols]
    print(f"  Using {len(common_cols)} common columns")

# Merge
print("\nMerging datasets...")
df_merged = pd.concat([df1, df2], ignore_index=True)

print(f"✓ Merged: {len(df_merged)} total experiments")

# Verify the merge
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

# Check datasets
datasets = df_merged['dataset'].unique()
print(f"\nDatasets found ({len(datasets)}):")
for ds in sorted(datasets):
    count = len(df_merged[df_merged['dataset'] == ds])
    expected = 120  # 4 budgets × 3 methods × 10 runs
    status = "✓" if count == expected else "⚠️"
    print(f"  {status} {ds:20s}: {count:4d} experiments (expected: {expected})")

# Check totals
expected_total = 1200
print(f"\n{'='*60}")
print(f"Expected total (10 datasets × 120): {expected_total}")
print(f"Actual total: {len(df_merged)}")

if len(df_merged) == expected_total:
    print("✅ PERFECT! All 1,200 experiments present!")
elif len(df_merged) < expected_total:
    missing = expected_total - len(df_merged)
    print(f"⚠️  WARNING: Missing {missing} experiments")
else:
    extra = len(df_merged) - expected_total
    print(f"⚠️  WARNING: Extra {extra} experiments (checking for duplicates...)")

# Check for duplicates
print("\nChecking for duplicates...")
duplicates = df_merged.duplicated(subset=['dataset', 'budget_level', 'method', 'run'], keep=False)
if duplicates.any():
    n_dups = duplicates.sum()
    print(f"⚠️  Found {n_dups} duplicate rows!")
    print("\nDuplicate details:")
    print(df_merged[duplicates][['dataset', 'budget_level', 'method', 'run']].sort_values(['dataset', 'budget_level', 'method', 'run']))
    print("\nRemoving duplicates (keeping first occurrence)...")
    df_merged = df_merged.drop_duplicates(subset=['dataset', 'budget_level', 'method', 'run'], keep='first')
    print(f"✓ After removal: {len(df_merged)} experiments")
else:
    print("✓ No duplicates found!")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nBudget levels:")
for budget in ['very_low', 'low', 'medium', 'high']:
    if budget in df_merged['budget_level'].values:
        count = len(df_merged[df_merged['budget_level'] == budget])
        expected = 10 * 3 * 10  # 10 datasets × 3 methods × 10 runs
        print(f"  {budget:10s}: {count:4d} experiments (expected: {expected})")

print("\nMethods:")
for method in ['random_search', 'sha', 'hyperband']:
    if method in df_merged['method'].values:
        count = len(df_merged[df_merged['method'] == method])
        expected = 10 * 4 * 10  # 10 datasets × 4 budgets × 10 runs
        print(f"  {method:15s}: {count:4d} experiments (expected: {expected})")

print("\nScore statistics:")
print(f"  Mean F1 score: {df_merged['best_score'].mean():.4f}")
print(f"  Std F1 score:  {df_merged['best_score'].std():.4f}")
print(f"  Min F1 score:  {df_merged['best_score'].min():.4f}")
print(f"  Max F1 score:  {df_merged['best_score'].max():.4f}")

print("\nTime statistics:")
print(f"  Mean time: {df_merged['time_used'].mean():.2f} seconds")
print(f"  Total time: {df_merged['time_used'].sum() / 3600:.2f} hours")

# Save merged file
output_file = results_dir / "ALL_EXPERIMENTS_1200_FINAL.csv"
print(f"\n{'='*80}")
print("SAVING MERGED RESULTS")
print(f"{'='*80}")
print(f"\nSaving to: {output_file}")
df_merged.to_csv(output_file, index=False)
print(f"✓ Saved successfully!")

# Also create a backup with timestamp
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_file = results_dir / f"ALL_EXPERIMENTS_1200_BACKUP_{timestamp}.csv"
df_merged.to_csv(backup_file, index=False)
print(f"✓ Backup saved: {backup_file.name}")

print("\n" + "=" * 80)
print("MERGE COMPLETE!")
print("=" * 80)
print(f"\n✅ Master file created: ALL_EXPERIMENTS_1200_FINAL.csv")
print(f"   Location: {output_file}")
print(f"   Total experiments: {len(df_merged)}")
print(f"   Datasets: {len(datasets)}")

print("\n📊 Ready for next steps:")
print("  1. ✅ Merge complete")
print("  2. → Statistical analysis")
print("  3. → Meta-feature extraction")
print("  4. → Meta-learning model training")
