"""
Extract Meta-Features for Covertype Only
=========================================
Quick script to extract meta-features for the missing covertype dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import StandardScaler

# Try to import PyMFE
try:
    from pymfe.mfe import MFE
    PYMFE_AVAILABLE = True
    print("✓ PyMFE available")
except ImportError:
    PYMFE_AVAILABLE = False
    print("⚠️  PyMFE not available (some features will be skipped)")

print("=" * 80)
print("EXTRACTING META-FEATURES: COVERTYPE")
print("=" * 80)

def load_dataset(dataset_name):
    """Load dataset from processed directory"""
    ds_dir = Path(f'data/processed/{dataset_name}')
    X = np.load(ds_dir / 'X_train.npy', allow_pickle=True)
    y = np.load(ds_dir / 'y_train.npy', allow_pickle=True)
    return X, y

def compute_simple_metafeatures(X, y):
    """Compute simple meta-features"""
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    meta = {
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
    meta['class_imbalance_ratio'] = class_probs.max() / class_probs.min()
    
    return meta

def compute_statistical_metafeatures(X, y):
    """Compute statistical meta-features"""
    meta = {}
    
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
    except Exception as e:
        print(f"  Warning: Statistical features failed: {e}")
        for key in ['skewness_min', 'skewness_max', 'skewness_mean', 'skewness_std',
                   'kurtosis_min', 'kurtosis_max', 'kurtosis_mean', 'kurtosis_std',
                   'correlation_min', 'correlation_max', 'correlation_mean', 'correlation_std']:
            meta[key] = np.nan
    
    return meta

def compute_information_theoretic_metafeatures(X, y):
    """Compute information-theoretic meta-features"""
    meta = {}
    
    try:
        # Class entropy
        class_counts = np.bincount(y.astype(int))
        class_probs = class_counts / len(y)
        meta['class_entropy'] = entropy(class_probs, base=2)
        
        # Normalized class entropy
        n_classes = len(class_counts)
        max_entropy = np.log2(n_classes) if n_classes > 1 else 1
        meta['normalized_class_entropy'] = meta['class_entropy'] / max_entropy if max_entropy > 0 else 0
    except Exception as e:
        print(f"  Warning: Info-theoretic features failed: {e}")
        meta['class_entropy'] = np.nan
        meta['normalized_class_entropy'] = np.nan
    
    return meta

def compute_pca_metafeatures(X, y):
    """Compute PCA-based meta-features"""
    meta = {}
    
    try:
        # Subsample for PCA
        sample_size = min(5000, X.shape[0])
        X_sample = X[:sample_size]
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        
        # PCA
        n_components = min(50, X.shape[1], sample_size - 1)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        # Variance explained
        meta['pca_first_pc_variance'] = pca.explained_variance_ratio_[0]
        
        # Components for 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_95 = np.searchsorted(cumsum, 0.95) + 1
        meta['pca_95_percent_dims'] = n_95
        meta['pca_95_percent_ratio'] = n_95 / X.shape[1]
    except Exception as e:
        print(f"  Warning: PCA features failed: {e}")
        meta['pca_first_pc_variance'] = np.nan
        meta['pca_95_percent_dims'] = np.nan
        meta['pca_95_percent_ratio'] = np.nan
    
    return meta

def compute_timing_metafeatures(X, y):
    """Compute timing-based meta-features"""
    meta = {}
    
    # Subsample for timing
    sample_size = min(1000, X.shape[0])
    indices = np.random.RandomState(42).choice(X.shape[0], sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    try:
        # Decision Tree timing
        start = time.time()
        dt = DecisionTreeClassifier(max_depth=10, random_state=42)
        dt.fit(X_sample, y_sample)
        tree_time = time.time() - start
        
        meta['tree_time'] = tree_time
        meta['tree_depth'] = dt.get_depth()
        meta['tree_n_leaves'] = dt.get_n_leaves()
    except Exception as e:
        print(f"  Warning: Tree timing failed: {e}")
        meta['tree_time'] = np.nan
        meta['tree_depth'] = np.nan
        meta['tree_n_leaves'] = np.nan
    
    try:
        # Neural network timing
        start = time.time()
        from sklearn.neural_network import MLPClassifier
        nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, random_state=42)
        nn.fit(X_sample, y_sample)
        meta['nn_time'] = time.time() - start
    except Exception as e:
        print(f"  Warning: NN timing failed: {e}")
        meta['nn_time'] = np.nan
    
    try:
        # Naive Bayes timing
        start = time.time()
        nb = GaussianNB()
        nb.fit(X_sample, y_sample)
        meta['nb_time'] = time.time() - start
    except Exception as e:
        print(f"  Warning: NB timing failed: {e}")
        meta['nb_time'] = np.nan
    
    return meta

def compute_pymfe_metafeatures(X, y):
    """Compute PyMFE meta-features if available"""
    meta = {}
    
    if not PYMFE_AVAILABLE:
        return meta
    
    try:
        # Subsample for PyMFE
        sample_size = min(5000, X.shape[0])
        X_sample = X[:sample_size]
        y_sample = y[:sample_size]
        
        print("  Extracting PyMFE features (this may take a few minutes)...")
        mfe = MFE(groups=["general", "statistical"])
        mfe.fit(X_sample, y_sample)
        ft = mfe.extract()
        
        # Add to meta dict with prefix
        for name, value in zip(ft[0], ft[1]):
            meta[f'pymfe_{name}'] = value
            
    except Exception as e:
        print(f"  Warning: PyMFE extraction failed: {e}")
    
    return meta

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\nLoading covertype data...")
X, y = load_dataset('covertype')
print(f"✓ Shape: {X.shape[0]} samples × {X.shape[1]} features")

# Initialize meta-features
meta = {'dataset': 'covertype'}

# Extract all feature groups
print("\nComputing simple meta-features...")
meta.update(compute_simple_metafeatures(X, y))

print("Computing statistical meta-features...")
meta.update(compute_statistical_metafeatures(X, y))

print("Computing information-theoretic meta-features...")
meta.update(compute_information_theoretic_metafeatures(X, y))

print("Computing PCA meta-features...")
meta.update(compute_pca_metafeatures(X, y))

print("Computing timing meta-features...")
meta.update(compute_timing_metafeatures(X, y))

if PYMFE_AVAILABLE:
    print("Computing PyMFE meta-features...")
    meta.update(compute_pymfe_metafeatures(X, y))

print(f"\n✓ Extracted {len(meta)-1} meta-features")

# Save
output_dir = Path('data/meta_features')
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / 'covertype_meta_features.csv'
df = pd.DataFrame([meta])
df.to_csv(output_file, index=False)

print(f"\n✓ Saved to: {output_file}")

print("\n" + "=" * 80)
print("COVERTYPE META-FEATURES COMPLETE!")
print("=" * 80)
print(f"\nNext: Run the combine script to merge all 10 datasets")
