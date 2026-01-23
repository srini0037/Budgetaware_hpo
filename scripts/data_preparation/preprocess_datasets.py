"""
Dataset Preprocessing Pipeline
Prepares all datasets for experiments with consistent train/val/test splits
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
    Preprocess a single dataset
    
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
    """Verify all datasets were preprocessed correctly"""
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    
    datasets = ['adult', 'fashion_mnist', 'mnist', 'letter']
    
    success = []
    failed = []
    
    for dataset_name in datasets:
        data_dir = Path(f'data/processed/{dataset_name}')
        
        if not data_dir.exists():
            failed.append(dataset_name)
            continue
        
        try:
            # Try loading all files
            X_train = np.load(data_dir / 'X_train.npy')
            y_train = np.load(data_dir / 'y_train.npy')
            X_val = np.load(data_dir / 'X_val.npy')
            y_val = np.load(data_dir / 'y_val.npy')
            X_test = np.load(data_dir / 'X_test.npy')
            y_test = np.load(data_dir / 'y_test.npy')
            
            # Verify shapes
            assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature dimension mismatch"
            assert len(y_train.shape) == 1, "y_train should be 1D"
            
            print(f"✓ {dataset_name}:")
            print(f"    Train: {X_train.shape}")
            print(f"    Val: {X_val.shape}")
            print(f"    Test: {X_test.shape}")
            
            success.append(dataset_name)
            
        except Exception as e:
            print(f"✗ {dataset_name}: {e}")
            failed.append(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"Successfully preprocessed: {len(success)}/{len(datasets)} datasets")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"{'='*60}")
    
    return success, failed

def main():
    """Main execution"""
    print("="*60)
    print("DATASET PREPROCESSING PIPELINE")
    print("="*60)
    
    # Datasets to preprocess
    datasets = ['adult', 'fashion_mnist', 'mnist', 'letter']
    
    # Process each dataset
    all_metadata = []
    
    for dataset_name in datasets:
        try:
            metadata = preprocess_dataset(dataset_name, max_train_samples=50000)
            all_metadata.append(metadata)
        except Exception as e:
            print(f"\n✗ Error preprocessing {dataset_name}: {e}")
    
    # Save summary
    if all_metadata:
        df_summary = pd.DataFrame(all_metadata)
        df_summary.to_csv('data/processed/preprocessing_summary.csv', index=False)
        print(f"\n✓ Saved summary to data/processed/preprocessing_summary.csv")
    
    # Verify all datasets
    success, failed = verify_preprocessing()
    
    # Final summary
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    
    if len(success) >= 4:
        print(f"✓ All datasets ready for experiments!")
        print(f"\nProcessed datasets:")
        for ds in success:
            print(f"  - {ds}")
        print(f"\nNext step: Run baseline experiments")
    else:
        print(f"✗ Some datasets failed preprocessing")
        print(f"  Please check error messages above")
    
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
