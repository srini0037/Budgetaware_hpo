"""
CORRECT VERSION - Uses Parallel CV like the existing 4 datasets!
=================================================================
KEY FIX: n_jobs=-1 in cross_val_score (uses all CPU cores)
This matches your existing experiments_generated/ scripts
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from scipy.stats import loguniform

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
print("5 NEW DATASETS - PARALLEL CV (MATCHING EXISTING METHOD)")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)


PROJECT_ROOT = Path("/Users/srinivass/Budgetaware_hpo")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "hpo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['bank', 'shuttle', 'creditcard', 'pendigits', 'satimage']

BUDGET_LEVELS = {
    'very_low': 60,
    'low': 120,
    'medium': 300,
    'high': 600,
}

N_RUNS = 10
RANDOM_SEEDS = list(range(N_RUNS))
HPO_METHODS = ['random_search', 'sha', 'hyperband']

print(f"\nDatasets: {DATASETS}")
print(f"Total: 600 experiments")
print(f"METHOD: Parallel CV with n_jobs=-1 (SAME AS EXISTING!)")
print()


def get_random_mlp_config():
    """Sample MLP config"""
    config = {
        'hidden_layer_sizes': tuple(np.random.choice([50, 100, 200], 
                                     size=np.random.randint(1, 4))),
        'learning_rate_init': loguniform(1e-4, 1e-1).rvs(),
        'alpha': loguniform(1e-5, 1e-2).rvs(),
        'activation': np.random.choice(['relu', 'tanh']),
        'batch_size': np.random.choice([32, 64, 128, 256]),
    }
    return config


def load_processed_dataset(dataset_name):
    """Load dataset"""
    dataset_dir = DATA_DIR / dataset_name
    
    X_train = np.load(dataset_dir / "X_train.npy")
    y_train = np.load(dataset_dir / "y_train.npy")
    X_val = np.load(dataset_dir / "X_val.npy")
    y_val = np.load(dataset_dir / "y_val.npy")
    X_test = np.load(dataset_dir / "X_test.npy")
    y_test = np.load(dataset_dir / "y_test.npy")
    
    print(f"  Loaded {dataset_name}: Train={X_train.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_config(config, X_train, y_train, cv=3):
    """
    CRITICAL FIX: n_jobs=-1 for parallel CV
    This matches experiments_generated/ scripts!
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(**config, random_state=42))
    ])
    
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring='f1_macro', 
        n_jobs=-1  # ← CRITICAL: Use all CPU cores!
    )
    
    return scores.mean()


def run_random_search_with_budget(X_train, y_train, time_budget, seed=42):
    """Random Search"""
    np.random.seed(seed)
    
    start_time = time.time()
    best_score = 0
    best_config = None
    configs_evaluated = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            break
        
        config = get_random_mlp_config()
        
        try:
            score = evaluate_config(config, X_train, y_train)
            
            if score > best_score:
                best_score = score
                best_config = config
            
            configs_evaluated += 1
        except Exception as e:
            continue
        
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
    """SHA"""
    np.random.seed(seed)
    
    start_time = time.time()
    
    n_configs = 27
    max_iter = 81
    eta = 3
    
    configs = [get_random_mlp_config() for _ in range(n_configs)]
    
    s = int(np.floor(np.log(n_configs) / np.log(eta)))
    r = max_iter / (eta ** s)
    
    configs_evaluated = 0
    best_score = 0
    best_config = None
    
    for i in range(s + 1):
        if time.time() - start_time >= time_budget:
            break
        
        n_i = int(n_configs / (eta ** i))
        r_i = int(r * (eta ** i))
        
        scores = []
        
        for config in configs:
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
    """Hyperband"""
    np.random.seed(seed)
    
    start_time = time.time()
    
    max_iter = 81
    eta = 3
    s_max = int(np.floor(np.log(max_iter) / np.log(eta)))
    
    best_score = 0
    best_config = None
    configs_evaluated = 0
    
    for s in range(s_max, -1, -1):
        if time.time() - start_time >= time_budget:
            break
        
        n = int(np.ceil((s_max + 1) * (eta ** s) / (s + 1)))
        r = max_iter * (eta ** (-s))
        
        configs = [get_random_mlp_config() for _ in range(n)]
        
        for i in range(s + 1):
            if time.time() - start_time >= time_budget:
                break
            
            n_i = int(n * (eta ** (-i)))
            r_i = int(r * (eta ** i))
            
            scores = []
            
            for config in configs:
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


# MAIN LOOP
all_results = []
overall_start = time.time()

for dataset_name in DATASETS:
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 80)
    
    dataset_start = time.time()
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_processed_dataset(dataset_name)
        
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        print(f"  Combined: {X_train_full.shape}")
        
        for budget_name, budget_seconds in BUDGET_LEVELS.items():
            print(f"\n  Budget: {budget_name} ({budget_seconds}s)")
            
            for method_name in HPO_METHODS:
                print(f"\n  {method_name.upper()}:")
                
                for run_idx, seed in enumerate(RANDOM_SEEDS):
                    print(f"    Run {run_idx + 1}/{N_RUNS}...", end=" ", flush=True)
                    
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
        print(f"\n  ✅ {dataset_name} complete in {dataset_time/60:.1f} min")
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_DIR / 'new_5_datasets_results.csv', index=False)
        
    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        continue


# SAVE
results_df = pd.DataFrame(all_results)
output_file = RESULTS_DIR / 'new_5_datasets_results.csv'
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("✅ COMPLETE!")
print("=" * 80)

total_time = time.time() - overall_start
print(f"Total: {total_time/3600:.2f} hours")
print(f"Saved: {output_file}")
print("=" * 80)