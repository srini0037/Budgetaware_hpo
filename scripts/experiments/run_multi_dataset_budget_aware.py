"""
Multi-Dataset Budget-Aware HPO Experiments
==========================================

Runs Random Search, SHA, and Hyperband across multiple datasets at different
budget levels to build the meta-learning dataset.

Datasets: Adult, Fashion-MNIST, MNIST, Letter
Budget levels: 60s, 120s, 300s, 600s
Runs: 10 per method per budget per dataset
HPO Methods: Random Search, SHA, Hyperband

Estimated runtime: 6-10 hours (depending on dataset sizes)
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from scipy.stats import loguniform, randint

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/Users/srinivass/Budgetaware_hpo')
from hpo.hyperband_implementation import Hyperband, get_random_mlp_config


print("=" * 80)
print("MULTI-DATASET BUDGET-AWARE HPO EXPERIMENTS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path("/Users/srinivass/Budgetaware_hpo")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "hpo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets to process
DATASETS = ['adult', 'fashion_mnist', 'mnist', 'letter']

# Budget levels (in seconds)
BUDGET_LEVELS = {
    'very_low': 60,    # 1 minute
    'low': 120,        # 2 minutes
    'medium': 300,     # 5 minutes
    'high': 600,       # 10 minutes
}

# Experiment settings
N_RUNS = 10  # 10 runs per method per budget for statistical validation
RANDOM_SEEDS = list(range(N_RUNS))

# HPO methods to test
HPO_METHODS = ['random_search', 'sha', 'hyperband']

print(f"\nDatasets: {DATASETS}")
print(f"Budget levels: {list(BUDGET_LEVELS.keys())}")
print(f"HPO methods: {HPO_METHODS}")
print(f"Runs per combination: {N_RUNS}")
print(f"Total experiments: {len(DATASETS)} × {len(BUDGET_LEVELS)} × {len(HPO_METHODS)} × {N_RUNS} = {len(DATASETS) * len(BUDGET_LEVELS) * len(HPO_METHODS) * N_RUNS}")
print()


# ============================================================================
# DATASET LOADING
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
# HELPER FUNCTIONS
# ============================================================================

def evaluate_config(config, X_train, y_train, cv=3):
    """Evaluate a single configuration using cross-validation"""
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(**config, random_state=42))
    ])
    
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring='f1_macro', n_jobs=1
    )
    
    return scores.mean()


def run_random_search_with_budget(X_train, y_train, time_budget, seed=42):
    """
    Run Random Search until budget is exhausted.
    
    Returns best config found and number of configurations evaluated.
    """
    np.random.seed(seed)
    
    start_time = time.time()
    best_score = 0
    best_config = None
    configs_evaluated = 0
    
    while True:
        # Check if we've exceeded budget
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            break
        
        # Sample and evaluate random config
        config = get_random_mlp_config()
        
        try:
            score = evaluate_config(config, X_train, y_train)
            
            if score > best_score:
                best_score = score
                best_config = config
            
            configs_evaluated += 1
        except Exception as e:
            # Skip problematic configs
            continue
        
        # Don't start new evaluation if budget almost exhausted
        if time.time() - start_time > time_budget * 0.95:
            break
    
    total_time = time.time() - start_time
    
    return {
        'best_score': best_score,
        'best_config': best_config,
        'configs_evaluated': configs_evaluated,
        'time_used': total_time
    }


def run_sha_with_budget(X_train, y_train, time_budget, seed=42):
    """
    Run Successive Halving Algorithm with budget constraint.
    
    Stops when time budget is exhausted.
    """
    np.random.seed(seed)
    
    start_time = time.time()
    
    # SHA parameters
    n_configs = 27  # Start with 27 configs
    max_iter = 81   # Maximum iterations for best configs
    eta = 3         # Reduction factor
    
    # Generate initial configurations
    configs = [get_random_mlp_config() for _ in range(n_configs)]
    
    # Calculate number of rounds
    s = int(np.floor(np.log(n_configs) / np.log(eta)))
    r = max_iter / (eta ** s)
    
    configs_evaluated = 0
    best_score = 0
    best_config = None
    
    # Successive halving rounds
    for i in range(s + 1):
        # Check budget before starting round
        if time.time() - start_time >= time_budget:
            break
        
        n_i = int(n_configs / (eta ** i))
        r_i = int(r * (eta ** i))
        
        scores = []
        
        for config in configs:
            # Check budget before each evaluation
            if time.time() - start_time >= time_budget:
                break
            
            try:
                config_copy = config.copy()
                config_copy['max_iter'] = r_i
                
                score = evaluate_config(config_copy, X_train, y_train)
                scores.append(score)
                configs_evaluated += 1
                
                if score > best_score:
                    best_score = score
                    best_config = config_copy
            except Exception as e:
                scores.append(0)
                continue
        
        # Keep top configs for next round
        if i < s and len(scores) > 0:
            n_keep = max(1, int(n_i / eta))
            if len(scores) >= n_keep:
                indices = np.argsort(scores)[-n_keep:]
                configs = [configs[j] for j in indices]
    
    total_time = time.time() - start_time
    
    return {
        'best_score': best_score,
        'best_config': best_config,
        'configs_evaluated': configs_evaluated,
        'time_used': total_time
    }


def run_hyperband_with_budget(X_train, y_train, time_budget, seed=42):
    """
    Run Hyperband with STRICT budget enforcement.
    
    Uses SHA-style approach with budget checking before each evaluation.
    This is actually budget-aware Hyperband, not post-hoc checking.
    """
    np.random.seed(seed)
    
    start_time = time.time()
    
    # Hyperband parameters
    max_iter = 81
    eta = 3
    s_max = int(np.floor(np.log(max_iter) / np.log(eta)))
    
    best_score = 0
    best_config = None
    configs_evaluated = 0
    
    # Run brackets in order (from s_max down to 0)
    for s in range(s_max, -1, -1):
        # Check budget before starting bracket
        if time.time() - start_time >= time_budget:
            break
        
        # Calculate n and r for this bracket
        n = int(np.ceil((s_max + 1) * (eta ** s) / (s + 1)))
        r = max_iter * (eta ** (-s))
        
        # Generate configurations for this bracket
        configs = [get_random_mlp_config() for _ in range(n)]
        
        # Run successive halving for this bracket
        for i in range(s + 1):
            # Check budget before starting round
            if time.time() - start_time >= time_budget:
                break
            
            n_i = int(n * (eta ** (-i)))
            r_i = int(r * (eta ** i))
            
            scores = []
            
            for config in configs:
                # CRITICAL: Check budget before EACH evaluation
                if time.time() - start_time >= time_budget:
                    break
                
                try:
                    config_copy = config.copy()
                    config_copy['max_iter'] = r_i
                    
                    score = evaluate_config(config_copy, X_train, y_train)
                    scores.append(score)
                    configs_evaluated += 1
                    
                    if score > best_score:
                        best_score = score
                        best_config = config_copy
                        
                except Exception as e:
                    scores.append(0)
                    continue
            
            # Keep top configs for next round
            if i < s and len(scores) > 0:
                n_keep = max(1, int(n_i / eta))
                if len(scores) >= n_keep:
                    indices = np.argsort(scores)[-n_keep:]
                    configs = [configs[j] for j in indices]
    
    total_time = time.time() - start_time
    
    return {
        'best_score': best_score,
        'best_config': best_config,
        'configs_evaluated': configs_evaluated,
        'time_used': total_time
    }


# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

all_results = []
overall_start = time.time()

for dataset_name in DATASETS:
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 80)
    
    dataset_start = time.time()
    
    try:
        # Load dataset
        X_train, y_train, X_val, y_val, X_test, y_test = load_processed_dataset(dataset_name)
        
        # Combine train and val for HPO (we'll use CV for validation)
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        print(f"  Combined train+val: {X_train_full.shape}")
        
        # Run experiments for each budget level
        for budget_name, budget_seconds in BUDGET_LEVELS.items():
            print(f"\n  Budget: {budget_name} ({budget_seconds}s)")
            print("  " + "-" * 76)
            
            # Run each HPO method
            for method_name in HPO_METHODS:
                print(f"\n  {method_name.upper()}:")
                
                for run_idx, seed in enumerate(RANDOM_SEEDS):
                    print(f"    Run {run_idx + 1}/{N_RUNS} (seed={seed})...", end=" ", flush=True)
                    
                    run_start = time.time()
                    
                    # Run the appropriate HPO method
                    if method_name == 'random_search':
                        result = run_random_search_with_budget(
                            X_train_full, y_train_full, budget_seconds, seed
                        )
                    elif method_name == 'sha':
                        result = run_sha_with_budget(
                            X_train_full, y_train_full, budget_seconds, seed
                        )
                    elif method_name == 'hyperband':
                        result = run_hyperband_with_budget(
                            X_train_full, y_train_full, budget_seconds, seed
                        )
                    
                    # Record result
                    all_results.append({
                        'dataset': dataset_name,
                        'budget_level': budget_name,
                        'budget_seconds': budget_seconds,
                        'method': method_name,
                        'run': run_idx,
                        'seed': seed,
                        'best_score': result['best_score'],
                        'configs_evaluated': result['configs_evaluated'],
                        'time_used': result['time_used']
                    })
                    
                    print(f"Score: {result['best_score']:.4f}, "
                          f"Configs: {result['configs_evaluated']}, "
                          f"Time: {result['time_used']:.1f}s")
        
        dataset_time = time.time() - dataset_start
        print(f"\n  ✅ {dataset_name} complete in {dataset_time/60:.1f} minutes")
        
        # Save intermediate results after each dataset
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_DIR / 'multi_dataset_budget_aware_partial.csv', index=False)
        
    except Exception as e:
        print(f"\n  ❌ ERROR processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue


# ============================================================================
# SAVE FINAL RESULTS
# ============================================================================

results_df = pd.DataFrame(all_results)
output_file = RESULTS_DIR / 'multi_dataset_budget_aware.csv'
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("✅ ALL EXPERIMENTS COMPLETE!")
print("=" * 80)

total_time = time.time() - overall_start
print(f"Total runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
print(f"\n💾 Results saved to: {output_file}")

# Summary statistics
print("\n📊 SUMMARY BY DATASET AND METHOD:")
print("=" * 80)

summary = results_df.groupby(['dataset', 'method'])['best_score'].agg(['mean', 'std', 'count'])
summary = summary.round(4)
print(summary)

print("\n📊 SUMMARY BY BUDGET LEVEL:")
print("=" * 80)

budget_summary = results_df.groupby(['budget_level', 'method'])['best_score'].agg(['mean', 'std'])
budget_summary = budget_summary.round(4)
print(budget_summary)

print("\n🎉 Ready for meta-learning analysis!")
print("=" * 80)
