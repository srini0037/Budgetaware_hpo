"""
Combine All Meta-Features - 10 Datasets
========================================
Combines all individual meta-feature CSV files into one master file
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("COMBINING ALL META-FEATURES - 10 DATASETS")
print("=" * 80)

# Find all individual meta-feature files
meta_dir = Path('data/meta_features')

# All 10 datasets
datasets = [
    'adult', 'fashion_mnist', 'mnist', 'letter', 'covertype',
    'bank', 'shuttle', 'creditcard', 'pendigits', 'satimage'
]

print(f"\nLooking for {len(datasets)} dataset files...")

found_files = []
missing_files = []

for dataset in datasets:
    filename = meta_dir / f'{dataset}_meta_features.csv'
    if filename.exists():
        file_size = filename.stat().st_size
        found_files.append(filename)
        print(f"  ✓ {dataset:20s} ({file_size:,} bytes)")
    else:
        missing_files.append(dataset)
        print(f"  ✗ {dataset:20s} (NOT FOUND)")

if missing_files:
    print(f"\n⚠️  WARNING: Missing {len(missing_files)} datasets: {missing_files}")
    print("   Will combine only the available datasets")

# Load all found files
print(f"\nLoading {len(found_files)} files...")
dfs = []

for f in found_files:
    df = pd.read_csv(f)
    dfs.append(df)
    dataset_name = df['dataset'].values[0] if 'dataset' in df.columns else f.stem.replace('_meta_features', '')
    n_features = len(df.columns) - 1  # Exclude 'dataset' column
    print(f"  ✓ {dataset_name:20s}: {n_features} features")

# Concatenate all
print("\nCombining datasets...")
combined = pd.concat(dfs, ignore_index=True)

print(f"✓ Combined: {len(combined)} datasets")
print(f"  Datasets: {sorted(combined['dataset'].unique())}")
print(f"  Features per dataset: {len(combined.columns) - 1}")

# Check feature consistency
feature_counts = []
for df in dfs:
    feature_counts.append(len(df.columns))

if len(set(feature_counts)) > 1:
    print(f"\n⚠️  WARNING: Inconsistent feature counts!")
    for i, (f, count) in enumerate(zip(found_files, feature_counts)):
        dataset = f.stem.replace('_meta_features', '')
        print(f"    {dataset:20s}: {count} columns")
    
    print("\n  This is okay if some datasets were extracted at different times")
    print("  The combined file will have all unique features (NaN for missing)")

# Save combined file
output_file = meta_dir / 'all_datasets_metafeatures.csv'
combined.to_csv(output_file, index=False)

print(f"\n✓ Saved to: {output_file}")
print(f"  Size: {output_file.stat().st_size:,} bytes")

# Display summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n✓ Successfully combined {len(combined)} datasets")
print(f"✓ Total features: {len(combined.columns) - 1}")
print(f"✓ Output: {output_file}")

if missing_files:
    print(f"\n⚠️  Missing datasets (not included): {missing_files}")

print("\n" + "=" * 80)
print("READY FOR META-LEARNING DATASET CREATION!")
print("=" * 80)

print("\nNext step:")
print("  Run: python build_meta_learning_dataset.py")
