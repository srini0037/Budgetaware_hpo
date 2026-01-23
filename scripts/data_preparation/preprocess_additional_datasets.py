"""
Preprocess 5 Additional Datasets
Follows same preprocessing pipeline as existing datasets:
1. Train/val/test split (60/20/20)
2. Standardize features (fit on train, transform val/test)
3. Subsample training if > 50K samples
4. Save to data/processed/{dataset_name}/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def preprocess_dataset(dataset_name, max_train_samples=50000):
    """
    Preprocess a single dataset - SAME AS EXISTING PREPROCESSING
    
    Steps:
    1. Load raw data
    2. Split into train/val/test (60/20/20)
    3. Standardize features
    4. Subsample training set to max_train_samples for budget constraints
    5. Save processed data
    """
    print(f"\n{'='*60}")
    print(f"Preprocessing: {dataset_name}")
    print(f"{'='*60}")
    
    # Load raw data
    print("  Loading raw data...")
    X = np.load(f'data/raw/{dataset_name}_X.npy', allow_pickle=True).astype(np.float32)
    y = np.load(f'data/raw/{dataset_name}_y.npy', allow_pickle=True).astype(np.int32)
    
    print(f"  Original shape: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Classes: {len(np.unique(y))}")
    
    # Step 1: Train/test split (80/20)
    print("  Splitting train/test...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 2: Train/val split (75/25 of remaining = 60/20 overall)
    print("  Splitting train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Step 3: Subsample training set if too large
    if X_train.shape[0] > max_train_samples:
        print(f"  Subsampling training set to {max_train_samples} samples...")
        indices = np.random.RandomState(42).choice(
            X_train.shape[0], max_train_samples, replace=False
        )
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"  New train size: {X_train.shape[0]} samples")
    
    # Step 4: Standardize features
    print("  Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 5: Save processed data
    print("  Saving processed data...")
    save_dir = Path(f'data/processed/{dataset_name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir / 'X_train.npy', X_train_scaled.astype(np.float32))
    np.save(save_dir / 'y_train.npy', y_train.astype(np.int32))
    np.save(save_dir / 'X_val.npy', X_val_scaled.astype(np.float32))
    np.save(save_dir / 'y_val.npy', y_val.astype(np.int32))
    np.save(save_dir / 'X_test.npy', X_test_scaled.astype(np.float32))
    np.save(save_dir / 'y_test.npy', y_test.astype(np.int32))
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'train_samples': X_train_scaled.shape[0],
        'val_samples': X_val_scaled.shape[0],
        'test_samples': X_test_scaled.shape[0],
        'n_features': X_train_scaled.shape[1],
        'n_classes': len(np.unique(y_train)),
        'subsampled': X_train.shape[0] < len(y_temp) * 0.75
    }
    
    pd.DataFrame([metadata]).to_csv(save_dir / 'metadata.csv', index=False)
    
    print(f"  ✓ Saved to {save_dir}")
    
    return metadata

def verify_preprocessing():
    """Verify all datasets (existing + new) were preprocessed correctly"""
    print("\n" + "="*60)
    print("VERIFICATION: Checking All 10 Datasets")
    print("="*60)
    
    all_datasets = [
        # Existing 5
        'adult', 'fashion_mnist', 'mnist', 'letter', 'covertype',
        # New 5
        'bank', 'shuttle', 'creditcard', 'pendigits', 'satimage'
    ]
    
    existing = []
    new_processed = []
    missing = []
    
    print("\nExisting datasets (already preprocessed):")
    for dataset in ['adult', 'fashion_mnist', 'mnist', 'letter', 'covertype']:
        metadata_path = Path(f'data/processed/{dataset}/metadata.csv')
        if metadata_path.exists():
            meta = pd.read_csv(metadata_path).iloc[0]
            print(f"  ✓ {dataset}: {meta['train_samples']} train, "
                  f"{meta['val_samples']} val, {meta['test_samples']} test, "
                  f"{meta['n_features']} features, {meta['n_classes']} classes")
            existing.append(dataset)
        else:
            print(f"  ✗ {dataset}: Not found")
            missing.append(dataset)
    
    print("\nNew datasets (just preprocessed):")
    for dataset in ['bank', 'shuttle', 'creditcard', 'pendigits', 'satimage']:
        metadata_path = Path(f'data/processed/{dataset}/metadata.csv')
        if metadata_path.exists():
            meta = pd.read_csv(metadata_path).iloc[0]
            print(f"  ✓ {dataset}: {meta['train_samples']} train, "
                  f"{meta['val_samples']} val, {meta['test_samples']} test, "
                  f"{meta['n_features']} features, {meta['n_classes']} classes")
            new_processed.append(dataset)
        else:
            print(f"  ✗ {dataset}: Not found")
            missing.append(dataset)
    
    print("\n" + "="*60)
    print(f"Total preprocessed: {len(existing) + len(new_processed)}/10 datasets")
    if missing:
        print(f"Missing: {', '.join(missing)}")
    print("="*60)
    
    return existing, new_processed, missing

def update_preprocessing_summary():
    """Update preprocessing_summary.csv with all 10 datasets"""
    print("\n" + "="*60)
    print("Updating preprocessing_summary.csv")
    print("="*60)
    
    all_datasets = [
        'adult', 'fashion_mnist', 'mnist', 'letter', 'covertype',
        'bank', 'shuttle', 'creditcard', 'pendigits', 'satimage'
    ]
    
    summary = []
    for dataset in all_datasets:
        metadata_path = Path(f'data/processed/{dataset}/metadata.csv')
        if metadata_path.exists():
            meta = pd.read_csv(metadata_path).iloc[0]
            summary.append(meta)
    
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('data/processed/preprocessing_summary.csv', index=False)
        print(f"  ✓ Updated with {len(summary)} datasets")
        print(f"  ✓ Saved to data/processed/preprocessing_summary.csv")
        return summary_df
    else:
        print("  ✗ No datasets found")
        return None

def main():
    """Main execution"""
    print("="*60)
    print("PREPROCESSING 5 ADDITIONAL DATASETS")
    print("="*60)
    print("\nDatasets to preprocess:")
    print("  1. bank - Bank Marketing")
    print("  2. shuttle - NASA Shuttle")
    print("  3. creditcard - Credit Card Fraud")
    print("  4. pendigits - Handwritten Digits")
    print("  5. satimage - Satellite Imagery")
    
    # Preprocess each dataset
    new_datasets = ['bank', 'shuttle', 'creditcard', 'pendigits', 'satimage']
    metadata_list = []
    
    for i, dataset in enumerate(new_datasets, 1):
        print(f"\n[{i}/5] Processing {dataset}...")
        try:
            metadata = preprocess_dataset(dataset, max_train_samples=50000)
            metadata_list.append(metadata)
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Verify all datasets
    existing, new_processed, missing = verify_preprocessing()
    
    # Update summary
    summary_df = update_preprocessing_summary()
    
    # Final summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Successfully preprocessed: {len(new_processed)}/5 new datasets")
    print(f"Total datasets ready: {len(existing) + len(new_processed)}/10")
    
    if len(existing) + len(new_processed) >= 10:
        print("\n✓ ALL 10 DATASETS PREPROCESSED!")
        print("\nDataset Summary:")
        if summary_df is not None:
            print(summary_df.to_string(index=False))
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Extract meta-features for 5 new datasets")
        print("2. Update experiment runner to include new datasets")
        print("3. Run budget experiments on all 10 datasets")
        print("4. Re-train meta-learner with 40 samples")
        print("\nExpected timeline:")
        print("  - Meta-feature extraction: ~10 minutes")
        print("  - Budget experiments: ~18-20 hours (overnight)")
        print("  - Meta-learner training: ~2 minutes")
    else:
        print(f"\n⚠ Only {len(existing) + len(new_processed)} datasets ready")
        if missing:
            print(f"Missing: {', '.join(missing)}")
    
    print("="*60)

if __name__ == '__main__':
    main()