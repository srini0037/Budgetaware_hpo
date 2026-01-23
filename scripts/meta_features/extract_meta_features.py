"""
Meta-Feature Extraction Script
Extracts comprehensive meta-features from all datasets including timing features
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Import meta-feature extraction libraries
try:
    from pymfe.mfe import MFE
    PYMFE_AVAILABLE = True
except ImportError:
    print("WARNING: pymfe not installed. Install with: pip install pymfe")
    PYMFE_AVAILABLE = False

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

def load_dataset(dataset_name):
    """Load a dataset from raw files"""
    X = np.load(f'data/raw/{dataset_name}_X.npy', allow_pickle=True)
    y = np.load(f'data/raw/{dataset_name}_y.npy', allow_pickle=True)
    return X, y

def compute_simple_metafeatures(X, y):
    """Compute simple meta-features (fast, always available)"""
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    meta = {
        # Basic dimensions
        'n_instances': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'dimensionality': n_features / n_samples,
        'log_n_instances': np.log10(n_samples),
        'log_n_features': np.log10(n_features),
        'log_dimensionality': np.log10(n_features / n_samples),
    }
    
    # Class distribution
    class_counts = np.bincount(y.astype(int))
    class_probs = class_counts / n_samples
    
    meta['class_prob_min'] = class_probs.min()
    meta['class_prob_max'] = class_probs.max()
    meta['class_prob_mean'] = class_probs.mean()
    meta['class_prob_std'] = class_probs.std()
    
    # Class imbalance ratio
    meta['class_imbalance_ratio'] = class_probs.max() / class_probs.min()
    
    return meta

def compute_statistical_metafeatures(X, y):
    """Compute statistical meta-features"""
    meta = {}
    
    # For each feature, compute statistics
    try:
        # Skewness
        skewness = skew(X, axis=0, nan_policy='omit')
        meta['skewness_min'] = np.nanmin(skewness)
        meta['skewness_max'] = np.nanmax(skewness)
        meta['skewness_mean'] = np.nanmean(skewness)
        meta['skewness_std'] = np.nanstd(skewness)
        
        # Kurtosis
        kurt = kurtosis(X, axis=0, nan_policy='omit')
        meta['kurtosis_min'] = np.nanmin(kurt)
        meta['kurtosis_max'] = np.nanmax(kurt)
        meta['kurtosis_mean'] = np.nanmean(kurt)
        meta['kurtosis_std'] = np.nanstd(kurt)
        
        # Correlation between features
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X.T)
            # Get upper triangle (excluding diagonal)
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            meta['correlation_min'] = np.nanmin(upper_tri)
            meta['correlation_max'] = np.nanmax(upper_tri)
            meta['correlation_mean'] = np.nanmean(upper_tri)
            meta['correlation_std'] = np.nanstd(upper_tri)
        else:
            meta['correlation_min'] = 0
            meta['correlation_max'] = 0
            meta['correlation_mean'] = 0
            meta['correlation_std'] = 0
    except:
        # If statistical features fail, fill with NaN
        for key in ['skewness_min', 'skewness_max', 'skewness_mean', 'skewness_std',
                   'kurtosis_min', 'kurtosis_max', 'kurtosis_mean', 'kurtosis_std',
                   'correlation_min', 'correlation_max', 'correlation_mean', 'correlation_std']:
            meta[key] = np.nan
    
    return meta

def compute_information_theoretic_metafeatures(X, y):
    """Compute information-theoretic meta-features"""
    meta = {}
    
    try:
        from scipy.stats import entropy
        
        # Class entropy
        class_counts = np.bincount(y.astype(int))
        class_probs = class_counts / len(y)
        meta['class_entropy'] = entropy(class_probs, base=2)
        
        # Normalized class entropy
        n_classes = len(class_counts)
        max_entropy = np.log2(n_classes) if n_classes > 1 else 1
        meta['normalized_class_entropy'] = meta['class_entropy'] / max_entropy if max_entropy > 0 else 0
        
    except:
        meta['class_entropy'] = np.nan
        meta['normalized_class_entropy'] = np.nan
    
    return meta

def compute_pca_metafeatures(X, y):
    """Compute PCA-based meta-features"""
    meta = {}
    
    try:
        # Limit to reasonable sample size for PCA
        sample_size = min(5000, X.shape[0])
        X_sample = X[:sample_size]
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        
        # PCA
        n_components = min(50, X.shape[1], sample_size - 1)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        # Variance explained by first PC
        meta['pca_first_pc_variance'] = pca.explained_variance_ratio_[0]
        
        # Cumulative variance for 95%
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_95 = np.argmax(cumsum >= 0.95) + 1
        meta['pca_95_percent_dims'] = n_95
        meta['pca_95_percent_ratio'] = n_95 / X.shape[1]
        
    except Exception as e:
        print(f"    Warning: PCA computation failed - {e}")
        meta['pca_first_pc_variance'] = np.nan
        meta['pca_95_percent_dims'] = np.nan
        meta['pca_95_percent_ratio'] = np.nan
    
    return meta

def compute_timing_metafeatures(X, y):
    """
    Compute timing meta-features - NOVEL CONTRIBUTION!
    Measures how long simple operations take on the dataset
    """
    meta = {}
    
    # Use subsample for timing
    sample_size = min(5000, X.shape[0])
    X_sample = X[:sample_size]
    y_sample = y[:sample_size]
    
    try:
        # Decision Tree time
        start = time.time()
        tree = DecisionTreeClassifier(max_depth=5, random_state=42)
        tree.fit(X_sample, y_sample)
        meta['tree_time'] = time.time() - start
        
        # Also get tree properties
        meta['tree_depth'] = tree.get_depth()
        meta['tree_n_leaves'] = tree.get_n_leaves()
        
    except Exception as e:
        print(f"    Warning: Tree timing failed - {e}")
        meta['tree_time'] = np.nan
        meta['tree_depth'] = np.nan
        meta['tree_n_leaves'] = np.nan
    
    try:
        # 1-NN time
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_sample, y_sample)
        meta['nn_time'] = time.time() - start
        
    except Exception as e:
        print(f"    Warning: KNN timing failed - {e}")
        meta['nn_time'] = np.nan
    
    try:
        # Naive Bayes time
        start = time.time()
        nb = GaussianNB()
        nb.fit(X_sample, y_sample)
        meta['nb_time'] = time.time() - start
        
    except Exception as e:
        print(f"    Warning: NB timing failed - {e}")
        meta['nb_time'] = np.nan
    
    return meta

def compute_pymfe_metafeatures(X, y):
    """Compute meta-features using PyMFE library if available"""
    if not PYMFE_AVAILABLE:
        return {}
    
    meta = {}
    
    try:
        # Use subsample for large datasets
        sample_size = min(10000, X.shape[0])
        X_sample = X[:sample_size]
        y_sample = y[:sample_size]
        
        # Initialize MFE
        mfe = MFE(groups=["general", "statistical", "info-theory"], 
                  summary=["mean", "sd"])
        
        # Fit and extract
        mfe.fit(X_sample, y_sample)
        ft_names, ft_values = mfe.extract()
        
        # Add to meta dict
        for name, value in zip(ft_names, ft_values):
            meta[f'pymfe_{name}'] = value
            
    except Exception as e:
        print(f"    Warning: PyMFE extraction failed - {e}")
    
    return meta

def extract_all_metafeatures(dataset_name):
    """Extract all meta-features for a dataset"""
    print(f"\n{'='*60}")
    print(f"Extracting meta-features: {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    print("  Loading data...")
    X, y = load_dataset(dataset_name)
    print(f"  Shape: {X.shape[0]} samples × {X.shape[1]} features")
    
    # Initialize meta-features dict
    meta = {'dataset': dataset_name}
    
    # Extract different groups
    print("  Computing simple meta-features...")
    meta.update(compute_simple_metafeatures(X, y))
    
    print("  Computing statistical meta-features...")
    meta.update(compute_statistical_metafeatures(X, y))
    
    print("  Computing information-theoretic meta-features...")
    meta.update(compute_information_theoretic_metafeatures(X, y))
    
    print("  Computing PCA meta-features...")
    meta.update(compute_pca_metafeatures(X, y))
    
    print("  Computing timing meta-features (NOVEL)...")
    meta.update(compute_timing_metafeatures(X, y))
    
    if PYMFE_AVAILABLE:
        print("  Computing PyMFE meta-features...")
        meta.update(compute_pymfe_metafeatures(X, y))
    
    print(f"  ✓ Extracted {len(meta)-1} meta-features")
    
    return meta

def main():
    """Main execution"""
    print("="*60)
    print("META-FEATURE EXTRACTION")
    print("="*60)
    
    # Create output directory
    Path('data/meta_features').mkdir(parents=True, exist_ok=True)
    
    # Datasets to process
    datasets = ['adult', 'fashion_mnist', 'mnist', 'letter']
    
    # Extract meta-features for each
    all_metafeatures = []
    
    for dataset_name in datasets:
        try:
            meta = extract_all_metafeatures(dataset_name)
            all_metafeatures.append(meta)
            
            # Save individual dataset meta-features
            df = pd.DataFrame([meta])
            df.to_csv(f'data/meta_features/{dataset_name}_meta_features.csv', index=False)
            print(f"  ✓ Saved to data/meta_features/{dataset_name}_meta_features.csv")
            
        except Exception as e:
            print(f"  ✗ Error extracting meta-features for {dataset_name}: {e}")
    
    # Create comparison table
    if all_metafeatures:
        print(f"\n{'='*60}")
        print("Creating comparison table...")
        df_all = pd.DataFrame(all_metafeatures)
        df_all = df_all.set_index('dataset')
        df_all.to_csv('data/meta_features/all_datasets_metafeatures.csv')
        print("✓ Saved to data/meta_features/all_datasets_metafeatures.csv")
        
        # Display summary
        print(f"\n{'='*60}")
        print("DIVERSITY SUMMARY")
        print(f"{'='*60}")
        
        key_features = ['n_instances', 'n_features', 'n_classes', 'dimensionality', 
                       'class_imbalance_ratio', 'pca_95_percent_ratio',
                       'tree_time', 'nn_time']
        
        for feature in key_features:
            if feature in df_all.columns:
                print(f"\n{feature}:")
                print(df_all[feature].to_string())
        
        print(f"\n{'='*60}")
        print("META-FEATURE EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Processed {len(all_metafeatures)} datasets")
        print("Next step: Preprocessing and running experiments")
        print(f"{'='*60}")

if __name__ == '__main__':
    main()
