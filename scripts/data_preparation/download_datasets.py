"""
Dataset Download Script - FIXED VERSION
Downloads and prepares all datasets for Phase 2 experiments
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directory structure"""
    directories = [
        'data/raw',
        'data/processed',
        'data/meta_features',
        'data/metalearning',
        'results/adult',
        'results/fashion_mnist',
        'results/cifar10_features',
        'results/higgs'
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Directory structure created")

def download_adult():
    """Download Adult (Census Income) dataset"""
    print("\n=== Downloading Adult dataset ===")
    try:
        adult = fetch_openml('adult', version=2, parser='auto', as_frame=True)
        X = adult.data
        y = adult.target
        
        # Convert to numeric
        from sklearn.preprocessing import LabelEncoder
        
        # Encode categorical features
        for col in X.select_dtypes(include=['category', 'object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Convert y to numeric
        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(y)
        
        # Save as npy with allow_pickle=True to avoid issues
        X_array = X.values.astype(np.float32)
        y_array = y_encoded.astype(np.int32)
        
        np.save('data/raw/adult_X.npy', X_array)
        np.save('data/raw/adult_y.npy', y_array)
        
        print(f"✓ Adult: {X_array.shape[0]} samples, {X_array.shape[1]} features, {len(np.unique(y_array))} classes")
        return True
    except Exception as e:
        print(f"✗ Error downloading Adult: {e}")
        return False

def download_fashion_mnist():
    """Download Fashion-MNIST dataset"""
    print("\n=== Downloading Fashion-MNIST ===")
    try:
        # Using OpenML
        fmnist = fetch_openml('Fashion-MNIST', version=1, parser='auto', as_frame=False)
        X = fmnist.data.astype(np.float32)
        y = fmnist.target.astype(np.int32)
        
        # Normalize to [0, 1]
        X = X / 255.0
        
        # Save
        np.save('data/raw/fashion_mnist_X.npy', X)
        np.save('data/raw/fashion_mnist_y.npy', y)
        
        print(f"✓ Fashion-MNIST: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        return True
    except Exception as e:
        print(f"✗ Error downloading Fashion-MNIST: {e}")
        return False

def download_mnist_alternative():
    """Download regular MNIST as fallback for CIFAR-10"""
    print("\n=== Downloading MNIST (alternative to CIFAR-10) ===")
    try:
        mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
        X = mnist.data.astype(np.float32)
        y = mnist.target.astype(np.int32)
        
        # Normalize to [0, 1]
        X = X / 255.0
        
        # Save
        np.save('data/raw/mnist_X.npy', X)
        np.save('data/raw/mnist_y.npy', y)
        
        print(f"✓ MNIST: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        print("  Note: Using MNIST as alternative since CIFAR-10 requires TensorFlow")
        return True
    except Exception as e:
        print(f"✗ Error downloading MNIST: {e}")
        return False

def download_letter_recognition():
    """Download Letter Recognition dataset as alternative"""
    print("\n=== Downloading Letter Recognition (alternative) ===")
    try:
        letter = fetch_openml('letter', version=1, parser='auto', as_frame=True)
        X = letter.data
        y = letter.target
        
        # Encode target
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_array = X.values.astype(np.float32)
        y_array = y_encoded.astype(np.int32)
        
        # Save
        np.save('data/raw/letter_X.npy', X_array)
        np.save('data/raw/letter_y.npy', y_array)
        
        print(f"✓ Letter: {X_array.shape[0]} samples, {X_array.shape[1]} features, {len(np.unique(y_array))} classes")
        return True
    except Exception as e:
        print(f"✗ Error downloading Letter: {e}")
        return False

def download_connect4():
    """Download Connect-4 dataset as alternative"""
    print("\n=== Downloading Connect-4 (alternative) ===")
    try:
        connect4 = fetch_openml('connect-4', version=1, parser='auto', as_frame=True)
        X = connect4.data
        y = connect4.target
        
        # Encode categorical features
        from sklearn.preprocessing import LabelEncoder
        for col in X.select_dtypes(include=['category', 'object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target
        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(y)
        
        X_array = X.values.astype(np.float32)
        y_array = y_encoded.astype(np.int32)
        
        # Save
        np.save('data/raw/connect4_X.npy', X_array)
        np.save('data/raw/connect4_y.npy', y_array)
        
        print(f"✓ Connect-4: {X_array.shape[0]} samples, {X_array.shape[1]} features, {len(np.unique(y_array))} classes")
        return True
    except Exception as e:
        print(f"✗ Error downloading Connect-4: {e}")
        return False

def verify_downloads():
    """Verify all datasets were downloaded successfully"""
    print("\n=== Verification ===")
    
    datasets = {
        'adult': ('data/raw/adult_X.npy', 'data/raw/adult_y.npy'),
        'fashion_mnist': ('data/raw/fashion_mnist_X.npy', 'data/raw/fashion_mnist_y.npy'),
        'mnist': ('data/raw/mnist_X.npy', 'data/raw/mnist_y.npy'),
        'letter': ('data/raw/letter_X.npy', 'data/raw/letter_y.npy'),
        'connect4': ('data/raw/connect4_X.npy', 'data/raw/connect4_y.npy'),
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
            failed.append(name)
    
    print(f"\n{'='*50}")
    print(f"Downloaded successfully: {len(success)} datasets")
    if failed:
        print(f"Not available: {', '.join(failed)}")
    print(f"{'='*50}")
    
    return success, failed

def main():
    """Main execution"""
    print("="*60)
    print("PHASE 2: DATASET DOWNLOAD (FIXED)")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Download datasets
    print("\n[1/5] Adult dataset...")
    adult_ok = download_adult()
    
    print("\n[2/5] Fashion-MNIST dataset...")
    fmnist_ok = download_fashion_mnist()
    
    print("\n[3/5] Alternative dataset 1 (MNIST)...")
    mnist_ok = download_mnist_alternative()
    
    print("\n[4/5] Alternative dataset 2 (Letter Recognition)...")
    letter_ok = download_letter_recognition()
    
    print("\n[5/5] Alternative dataset 3 (Connect-4)...")
    connect4_ok = download_connect4()
    
    # Verify
    success, failed = verify_downloads()
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Successfully downloaded: {len(success)} datasets")
    
    if len(success) >= 4:  # Need Covertype + 3 others minimum
        print("\n✓ Sufficient datasets for Phase 2!")
        print("\nDatasets ready:")
        for ds in success:
            print(f"  - {ds}")
        print("\nYou can now proceed to:")
        print("  1. Meta-feature extraction")
        print("  2. Preprocessing")
        print("  3. Running experiments")
    else:
        print(f"\n✗ Need at least 4 datasets total (have {len(success) + 1} including Covertype)")
    
    print("="*60)

if __name__ == '__main__':
    main()
