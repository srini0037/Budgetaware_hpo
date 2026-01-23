"""
Download 5 Additional Datasets for Enhanced Meta-Learning
Adds: bank, shuttle, creditcard, pendigits, satimage
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def download_bank():
    """Download Bank Marketing dataset"""
    print("\n=== Downloading Bank Marketing dataset ===")
    try:
        bank = fetch_openml('bank-marketing', version=1, parser='auto', as_frame=True)
        X = bank.data
        y = bank.target
        
        # Convert to numeric
        from sklearn.preprocessing import LabelEncoder
        
        # Encode categorical features
        for col in X.select_dtypes(include=['category', 'object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Convert y to numeric
        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(y)
        
        # Convert to arrays
        X_array = X.values.astype(np.float32)
        y_array = y_encoded.astype(np.int32)
        
        # Subsample if too large (keep 50K samples)
        if len(X_array) > 50000:
            print(f"  Subsampling from {len(X_array)} to 50000 samples...")
            X_array, _, y_array, _ = train_test_split(
                X_array, y_array, 
                train_size=50000,
                stratify=y_array,
                random_state=42
            )
        
        # Save
        np.save('data/raw/bank_X.npy', X_array)
        np.save('data/raw/bank_y.npy', y_array)
        
        print(f"✓ Bank: {X_array.shape[0]} samples, {X_array.shape[1]} features, {len(np.unique(y_array))} classes")
        return True
    except Exception as e:
        print(f"✗ Error downloading Bank: {e}")
        return False

def download_shuttle():
    """Download Shuttle dataset"""
    print("\n=== Downloading Shuttle dataset ===")
    try:
        shuttle = fetch_openml('shuttle', version=1, parser='auto', as_frame=True)
        X = shuttle.data
        y = shuttle.target
        
        # Encode target
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Convert to arrays
        X_array = X.values.astype(np.float32)
        y_array = y_encoded.astype(np.int32)
        
        # Subsample if too large
        if len(X_array) > 50000:
            print(f"  Subsampling from {len(X_array)} to 50000 samples...")
            X_array, _, y_array, _ = train_test_split(
                X_array, y_array, 
                train_size=50000,
                stratify=y_array,
                random_state=42
            )
        
        # Save
        np.save('data/raw/shuttle_X.npy', X_array)
        np.save('data/raw/shuttle_y.npy', y_array)
        
        print(f"✓ Shuttle: {X_array.shape[0]} samples, {X_array.shape[1]} features, {len(np.unique(y_array))} classes")
        return True
    except Exception as e:
        print(f"✗ Error downloading Shuttle: {e}")
        return False

def download_creditcard():
    """Download Creditcard (fraud detection) dataset"""
    print("\n=== Downloading Creditcard dataset ===")
    try:
        # Note: This dataset might be large, so we'll subsample
        creditcard = fetch_openml('creditcard', version=1, parser='auto', as_frame=True)
        X = creditcard.data
        y = creditcard.target
        
        # Encode target if needed
        from sklearn.preprocessing import LabelEncoder
        if y.dtype == object or isinstance(y.iloc[0], str):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y.values
        
        # Convert to arrays
        X_array = X.values.astype(np.float32)
        y_array = y_encoded.astype(np.int32)
        
        # Handle missing values if any
        if np.any(np.isnan(X_array)):
            print("  Imputing missing values...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_array = imputer.fit_transform(X_array)
        
        # Subsample if too large
        if len(X_array) > 50000:
            print(f"  Subsampling from {len(X_array)} to 50000 samples...")
            # Stratified sampling for imbalanced dataset
            try:
                X_array, _, y_array, _ = train_test_split(
                    X_array, y_array, 
                    train_size=50000,
                    stratify=y_array,
                    random_state=42
                )
            except:
                # If stratification fails (too imbalanced), do random sampling
                indices = np.random.RandomState(42).choice(len(X_array), 50000, replace=False)
                X_array = X_array[indices]
                y_array = y_array[indices]
        
        # Save
        np.save('data/raw/creditcard_X.npy', X_array)
        np.save('data/raw/creditcard_y.npy', y_array)
        
        print(f"✓ Creditcard: {X_array.shape[0]} samples, {X_array.shape[1]} features, {len(np.unique(y_array))} classes")
        print(f"  Note: Class imbalance ratio: {np.bincount(y_array).max() / np.bincount(y_array).min():.2f}")
        return True
    except Exception as e:
        print(f"✗ Error downloading Creditcard: {e}")
        return False

def download_pendigits():
    """Download Pendigits dataset"""
    print("\n=== Downloading Pendigits dataset ===")
    try:
        pendigits = fetch_openml('pendigits', version=1, parser='auto', as_frame=True)
        X = pendigits.data
        y = pendigits.target
        
        # Encode target
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Convert to arrays
        X_array = X.values.astype(np.float32)
        y_array = y_encoded.astype(np.int32)
        
        # Save
        np.save('data/raw/pendigits_X.npy', X_array)
        np.save('data/raw/pendigits_y.npy', y_array)
        
        print(f"✓ Pendigits: {X_array.shape[0]} samples, {X_array.shape[1]} features, {len(np.unique(y_array))} classes")
        return True
    except Exception as e:
        print(f"✗ Error downloading Pendigits: {e}")
        return False

def download_satimage():
    """Download Satimage dataset"""
    print("\n=== Downloading Satimage dataset ===")
    try:
        satimage = fetch_openml('satimage', version=1, parser='auto', as_frame=True)
        X = satimage.data
        y = satimage.target
        
        # Encode target
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Convert to arrays
        X_array = X.values.astype(np.float32)
        y_array = y_encoded.astype(np.int32)
        
        # Save
        np.save('data/raw/satimage_X.npy', X_array)
        np.save('data/raw/satimage_y.npy', y_array)
        
        print(f"✓ Satimage: {X_array.shape[0]} samples, {X_array.shape[1]} features, {len(np.unique(y_array))} classes")
        return True
    except Exception as e:
        print(f"✗ Error downloading Satimage: {e}")
        return False

def verify_downloads():
    """Verify all 5 new datasets were downloaded successfully"""
    print("\n=== Verification ===")
    
    datasets = {
        'bank': ('data/raw/bank_X.npy', 'data/raw/bank_y.npy'),
        'shuttle': ('data/raw/shuttle_X.npy', 'data/raw/shuttle_y.npy'),
        'creditcard': ('data/raw/creditcard_X.npy', 'data/raw/creditcard_y.npy'),
        'pendigits': ('data/raw/pendigits_X.npy', 'data/raw/pendigits_y.npy'),
        'satimage': ('data/raw/satimage_X.npy', 'data/raw/satimage_y.npy'),
    }
    
    success = []
    failed = []
    
    for name, (x_path, y_path) in datasets.items():
        if Path(x_path).exists() and Path(y_path).exists():
            try:
                X = np.load(x_path, allow_pickle=True)
                y = np.load(y_path, allow_pickle=True)
                print(f"✓ {name}: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
                success.append(name)
            except Exception as e:
                print(f"✗ {name}: Error loading - {e}")
                failed.append(name)
        else:
            print(f"✗ {name}: Files not found")
            failed.append(name)
    
    print(f"\n{'='*60}")
    print(f"Downloaded successfully: {len(success)}/5 datasets")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"{'='*60}")
    
    return success, failed

def verify_all_10_datasets():
    """Verify all 10 datasets (5 existing + 5 new)"""
    print("\n=== Verifying ALL 10 Datasets ===")
    
    all_datasets = {
        # Existing 5
        'adult': ('data/raw/adult_X.npy', 'data/raw/adult_y.npy'),
        'fashion_mnist': ('data/raw/fashion_mnist_X.npy', 'data/raw/fashion_mnist_y.npy'),
        'mnist': ('data/raw/mnist_X.npy', 'data/raw/mnist_y.npy'),
        'letter': ('data/raw/letter_X.npy', 'data/raw/letter_y.npy'),
        'covertype': ('data/raw/covertype_X.npy', 'data/raw/covertype_y.npy'),
        # New 5
        'bank': ('data/raw/bank_X.npy', 'data/raw/bank_y.npy'),
        'shuttle': ('data/raw/shuttle_X.npy', 'data/raw/shuttle_y.npy'),
        'creditcard': ('data/raw/creditcard_X.npy', 'data/raw/creditcard_y.npy'),
        'pendigits': ('data/raw/pendigits_X.npy', 'data/raw/pendigits_y.npy'),
        'satimage': ('data/raw/satimage_X.npy', 'data/raw/satimage_y.npy'),
    }
    
    available = []
    missing = []
    
    print("\nExisting datasets (should already be there):")
    for name in ['adult', 'fashion_mnist', 'mnist', 'letter', 'covertype']:
        x_path, y_path = all_datasets[name]
        if Path(x_path).exists() and Path(y_path).exists():
            try:
                X = np.load(x_path, allow_pickle=True)
                y = np.load(y_path, allow_pickle=True)
                print(f"  ✓ {name}: {X.shape[0]} samples, {X.shape[1]} features")
                available.append(name)
            except:
                print(f"  ✗ {name}: Error loading")
                missing.append(name)
        else:
            print(f"  ✗ {name}: Not found")
            missing.append(name)
    
    print("\nNew datasets (just downloaded):")
    for name in ['bank', 'shuttle', 'creditcard', 'pendigits', 'satimage']:
        x_path, y_path = all_datasets[name]
        if Path(x_path).exists() and Path(y_path).exists():
            try:
                X = np.load(x_path, allow_pickle=True)
                y = np.load(y_path, allow_pickle=True)
                print(f"  ✓ {name}: {X.shape[0]} samples, {X.shape[1]} features")
                available.append(name)
            except:
                print(f"  ✗ {name}: Error loading")
                missing.append(name)
        else:
            print(f"  ✗ {name}: Not found")
            missing.append(name)
    
    print(f"\n{'='*60}")
    print(f"TOTAL AVAILABLE: {len(available)}/10 datasets")
    if missing:
        print(f"Missing: {', '.join(missing)}")
    print(f"{'='*60}")
    
    return available, missing

def main():
    """Main execution"""
    print("="*60)
    print("DOWNLOADING 5 ADDITIONAL DATASETS")
    print("="*60)
    print("\nAdding diversity to existing 5 datasets:")
    print("  Existing: adult, fashion_mnist, mnist, letter, covertype")
    print("  Adding:   bank, shuttle, creditcard, pendigits, satimage")
    
    # Download datasets
    print("\n[1/5] Bank Marketing dataset...")
    bank_ok = download_bank()
    
    print("\n[2/5] Shuttle dataset...")
    shuttle_ok = download_shuttle()
    
    print("\n[3/5] Creditcard dataset...")
    creditcard_ok = download_creditcard()
    
    print("\n[4/5] Pendigits dataset...")
    pendigits_ok = download_pendigits()
    
    print("\n[5/5] Satimage dataset...")
    satimage_ok = download_satimage()
    
    # Verify new downloads
    success, failed = verify_downloads()
    
    # Verify all 10 datasets
    all_available, all_missing = verify_all_10_datasets()
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"New datasets downloaded: {len(success)}/5")
    print(f"Total datasets available: {len(all_available)}/10")
    
    if len(all_available) >= 10:
        print("\n✓ ALL 10 DATASETS READY!")
        print("\nNext steps:")
        print("  1. Update extract_meta_features.py to include new datasets")
        print("  2. Extract meta-features for all 10 datasets")
        print("  3. Update experiment scripts to run on all 10 datasets")
        print("  4. Run budget experiments")
        print("  5. Train meta-learner with 40 samples (10 datasets × 4 budgets)")
        print("\nExpected improvement:")
        print("  - Current CV accuracy: 55% ± 27%")
        print("  - Expected with 10 datasets: 70-75% ± 15-20%")
    elif len(all_available) >= 8:
        print(f"\n✓ Have {len(all_available)} datasets - Sufficient for improved results!")
        print(f"Missing: {', '.join(all_missing)}")
    else:
        print(f"\n⚠ Only {len(all_available)} datasets available")
        print(f"Missing: {', '.join(all_missing)}")
        print("Consider downloading missing datasets or adjusting experiment plan")
    
    print("="*60)

if __name__ == '__main__':
    main()