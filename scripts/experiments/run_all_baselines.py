"""
Run MLP Baseline for Multiple Datasets
======================================

Runs default MLP (no HPO) across multiple datasets to establish baseline performance.

Datasets: Adult, Fashion-MNIST, MNIST, Letter
Runs: 10 per dataset (with different random seeds)
Saves to: results/baselines/mlp_baseline_{dataset}.csv
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')


print("=" * 80)
print("MLP BASELINE EXPERIMENTS - MULTIPLE DATASETS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path("/Users/srinivass/Budgetaware_hpo")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets to process
DATASETS = ['adult', 'fashion_mnist', 'mnist', 'letter']

# Experiment settings
N_RUNS = 10  # 10 runs per dataset
RANDOM_SEEDS = list(range(N_RUNS))

print(f"\nDatasets to process: {DATASETS}")
print(f"Runs per dataset: {N_RUNS}")
print(f"Results directory: {RESULTS_DIR}")
print()


# ============================================================================
# DATASET LOADING FUNCTIONS
# ============================================================================

def load_processed_dataset(dataset_name):
    """Load preprocessed dataset from disk"""
    dataset_dir = DATA_DIR / dataset_name
    
    X_train = np.load(dataset_dir / "X_train.npy")
    y_train = np.load(dataset_dir / "y_train.npy")
    X_val = np.load(dataset_dir / "X_val.npy")
    y_val = np.load(dataset_dir / "y_val.npy")
    X_test = np.load(dataset_dir / "X_test.npy")
    y_test = np.load(dataset_dir / "y_test.npy")
    
    print(f"  Loaded {dataset_name}:")
    print(f"    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"    Classes: {len(np.unique(y_train))}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================================
# BASELINE MLP BUILDER
# ============================================================================

def build_baseline_mlp(random_state):
    """
    Build a reasonable baseline MLP with default hyperparameters.
    Not intentionally weak - uses sklearn defaults with some sensible adjustments.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(100,),  # Single hidden layer with 100 neurons
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            alpha=0.0001,  # L2 regularization
            batch_size='auto',
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state,
            verbose=False
        ))
    ])


# ============================================================================
# RUN BASELINES FOR ALL DATASETS
# ============================================================================

for dataset_name in DATASETS:
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Load dataset
        X_train, y_train, X_val, y_val, X_test, y_test = load_processed_dataset(dataset_name)
        
        # Combine train and val for baseline (no HPO, so no need for separate val)
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        print(f"\n  Combined train+val: {X_train_full.shape}")
        
        all_results = []
        
        # Run baseline with different seeds
        for run_idx, seed in enumerate(RANDOM_SEEDS):
            print(f"\n  Run {run_idx + 1}/{N_RUNS} (seed={seed})...")
            
            run_start = time.time()
            
            # Build and train model
            model = build_baseline_mlp(random_state=seed)
            model.fit(X_train_full, y_train_full)
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            run_time = time.time() - run_start
            
            all_results.append({
                'dataset': dataset_name,
                'seed': seed,
                'f1_macro': f1_macro,
                'training_time': run_time
            })
            
            print(f"    F1-macro: {f1_macro:.4f}, Time: {run_time:.1f}s")
        
        # Save results
        df_results = pd.DataFrame(all_results)
        output_path = RESULTS_DIR / f"mlp_baseline_{dataset_name}.csv"
        df_results.to_csv(output_path, index=False)
        
        # Print summary
        total_time = time.time() - start_time
        mean_f1 = df_results['f1_macro'].mean()
        std_f1 = df_results['f1_macro'].std()
        
        print(f"\n  SUMMARY for {dataset_name}:")
        print(f"    Mean F1-macro: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"    Min: {df_results['f1_macro'].min():.4f}")
        print(f"    Max: {df_results['f1_macro'].max():.4f}")
        print(f"    Total time: {total_time:.1f}s")
        print(f"    Saved to: {output_path}")
        
    except Exception as e:
        print(f"\n  ERROR processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ALL BASELINES COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load and summarize all results
print("\nFINAL SUMMARY:")
print("-" * 80)

for dataset_name in DATASETS:
    result_path = RESULTS_DIR / f"mlp_baseline_{dataset_name}.csv"
    if result_path.exists():
        df = pd.read_csv(result_path)
        mean_f1 = df['f1_macro'].mean()
        std_f1 = df['f1_macro'].std()
        print(f"{dataset_name:20s}: {mean_f1:.4f} ± {std_f1:.4f} (n={len(df)})")
    else:
        print(f"{dataset_name:20s}: NOT FOUND")

print("=" * 80)
