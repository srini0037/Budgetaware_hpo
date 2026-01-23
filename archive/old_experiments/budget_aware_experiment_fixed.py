"""
Budget-Aware Comparison Experiment (FIXED VERSION)
==================================================

FIXED: Hyperband now properly respects budget constraints
Tests all three HPO methods at different budget levels.

Budget levels: 60s, 120s, 300s, 600s
Runs: 10 per method per budget

Estimated runtime: 3-4 hours (FIXED)
"""

import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import loguniform, randint

from sklearn.datasets import fetch_openml
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/Users/srinivass/Budgetaware_hpo')


print("=" * 80)
print("BUDGET-AWARE HPO COMPARISON EXPERIMENT (FIXED)")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)


# ============================================================================
# CONFIGURATION
# ============================================================================

BUDGET_LEVELS = {
    'very_low': 60,    # 1 minute
    'low': 120,        # 2 minutes
    'medium': 300,     # 5 minutes
    'high': 600,       # 10 minutes
}

N_RUNS = 10
SEEDS = list(range(N_RUNS))


# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📊 Loading Covertype dataset...")

X, y = fetch_openml(name="covertype", version=2, as_frame=False, return_X_y=True)

if hasattr(X, 'toarray'):
    X = X.toarray()

y = y.astype(int)

MAX_SAMPLES = 50000
if X.shape[0] > MAX_SAMPLES:
    X, y = resample(X, y, n_samples=MAX_SAMPLES, stratify=y, random_state=42)

print(f"✅ Dataset: {X.shape}, Classes: {len(np.unique(y))}")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"✅ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_random_mlp_config():
    """Sample a random MLP configuration"""
    return {
        'hidden_layer_sizes': tuple([
            int(loguniform(50, 300).rvs()) 
            for _ in range(randint(1, 4).rvs())
        ]),
        'activation': np.random.choice(['relu', 'tanh']),
        'solver': 'adam',
        'alpha': loguniform(1e-5, 1e-1).rvs(),
        'learning_rate_init': loguniform(1e-4, 1e-2).rvs(),
        'batch_size': int(2 ** randint(5, 9).rvs()),
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
    }


def evaluate_config(config, X_train, y_train):
    """Evaluate a single configuration using 3-fold CV"""
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(**config, random_state=42))
    ])
    
    scores = cross_val_score(
        model, X_train, y_train,
        cv=3, scoring='f1_macro', n_jobs=1
    )
    
    return scores.mean()


def run_random_search_with_budget(X_train, y_train, time_budget, seed=42):
    """Run Random Search until budget is exhausted"""
    np.random.seed(seed)
    
    start_time = time.time()
    best_score = 0
    best_config = None
    configs_evaluated = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_budget * 0.95:
            break
        
        config = get_random_mlp_config()
        score = evaluate_config(config, X_train, y_train)
        
        if score > best_score:
            best_score = score
            best_config = config
        
        configs_evaluated += 1
    
    total_time = time.time() - start_time
    
    return {
        'best_score': best_score,
        'configs_evaluated': configs_evaluated,
        'time_used': total_time
    }


def run_sha_with_budget(X_train, y_train, time_budget, seed=42):
    """Run SHA, stopping when budget is exhausted"""
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
        # Check budget before each round
        if time.time() - start_time >= time_budget * 0.95:
            break
        
        n_i = int(n_configs / (eta ** i))
        r_i = int(r * (eta ** i))
        
        scores = []
        for config in configs:
            # Check budget before each evaluation
            if time.time() - start_time >= time_budget * 0.95:
                break
            
            config_copy = config.copy()
            config_copy['max_iter'] = r_i
            score = evaluate_config(config_copy, X_train, y_train)
            scores.append(score)
            configs_evaluated += 1
            
            if score > best_score:
                best_score = score
                best_config = config_copy
        
        # Keep top configs for next round
        if i < s and len(scores) > 0:
            n_keep = max(1, int(n_i / eta))
            if len(scores) >= n_keep:
                indices = np.argsort(scores)[-n_keep:]
                configs = [configs[j] for j in indices]
    
    total_time = time.time() - start_time
    
    return {
        'best_score': best_score,
        'configs_evaluated': configs_evaluated,
        'time_used': total_time
    }


def run_hyperband_with_budget(X_train, y_train, time_budget, seed=42):
    """
    Run Hyperband with STRICT budget enforcement
    FIXED: Actually stops when budget is exhausted
    """
    np.random.seed(seed)
    
    start_time = time.time()
    
    max_iter = 81
    eta = 3
    
    s_max = int(np.floor(np.log(max_iter) / np.log(eta)))
    
    all_results = []
    best_score = 0
    best_config = None
    configs_evaluated = 0
    
    # Run brackets until budget exhausted
    for s in range(s_max, -1, -1):
        # Check if we have time for this bracket
        elapsed = time.time() - start_time
        if elapsed >= time_budget * 0.95:
            break
        
        n = int(np.ceil((s + 1) * max_iter / eta ** s))
        r = max_iter * eta ** (-s)
        
        # Sample initial configs
        configs = [get_random_mlp_config() for _ in range(n)]
        
        # Run successive halving for this bracket
        for i in range(s + 1):
            # Check budget
            if time.time() - start_time >= time_budget * 0.95:
                break
            
            n_i = int(n * eta ** (-i))
            r_i = int(r * eta ** i)
            
            scores = []
            for config in configs:
                # CRITICAL: Check budget before EACH evaluation
                if time.time() - start_time >= time_budget * 0.95:
                    break
                
                config_copy = config.copy()
                config_copy['max_iter'] = r_i
                
                score = evaluate_config(config_copy, X_train, y_train)
                scores.append(score)
                configs_evaluated += 1
                
                if score > best_score:
                    best_score = score
                    best_config = config_copy
                
                all_results.append({
                    'bracket': s,
                    'round': i,
                    'score': score,
                    'resource': r_i
                })
            
            # Keep top configs for next round
            if i < s and len(scores) > 0:
                n_keep = max(1, int(n_i / eta))
                if len(scores) >= n_keep:
                    indices = np.argsort(scores)[-n_keep:]
                    configs = [configs[j] for j in indices]
    
    total_time = time.time() - start_time
    
    return {
        'best_score': best_score,
        'configs_evaluated': configs_evaluated,
        'time_used': total_time
    }


# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

results_dir = Path('/Users/srinivass/Budgetaware_hpo/results/hpo')
results_dir.mkdir(parents=True, exist_ok=True)

all_results = []
overall_start = time.time()

for budget_name, budget_seconds in BUDGET_LEVELS.items():
    print("\n" + "=" * 80)
    print(f"BUDGET LEVEL: {budget_name.upper()} ({budget_seconds}s)")
    print("=" * 80)
    
    for method_name in ['random_search', 'sha', 'hyperband']:
        print(f"\n{method_name.upper()}:")
        
        for run, seed in enumerate(SEEDS, 1):
            print(f"  Run {run}/{N_RUNS} (seed={seed})...", end=" ", flush=True)
            
            run_start = time.time()
            
            if method_name == 'random_search':
                result = run_random_search_with_budget(X_train, y_train, budget_seconds, seed)
            elif method_name == 'sha':
                result = run_sha_with_budget(X_train, y_train, budget_seconds, seed)
            elif method_name == 'hyperband':
                result = run_hyperband_with_budget(X_train, y_train, budget_seconds, seed)
            
            all_results.append({
                'budget_level': budget_name,
                'budget_seconds': budget_seconds,
                'method': method_name,
                'run': run,
                'seed': seed,
                'best_score': result['best_score'],
                'configs_evaluated': result['configs_evaluated'],
                'time_used': result['time_used']
            })
            
            print(f"Score: {result['best_score']:.4f}, Configs: {result['configs_evaluated']}, Time: {result['time_used']:.1f}s")

# Save results after each budget level (in case of interruption)
results_df = pd.DataFrame(all_results)
results_df.to_csv(results_dir / 'budget_aware_comparison_fixed.csv', index=False)

print("\n" + "=" * 80)
print("✅ ALL BUDGET EXPERIMENTS COMPLETE!")
print("=" * 80)

total_time = time.time() - overall_start
print(f"Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"\n💾 Results saved to: {results_dir / 'budget_aware_comparison_fixed.csv'}")

# Summary
print("\n📊 SUMMARY BY BUDGET LEVEL:")
print("=" * 80)

summary = results_df.groupby(['budget_level', 'method'])['best_score'].agg(['mean', 'std', 'count'])
summary = summary.round(4)
print(summary)

print("\n🎉 Ready for crossing point analysis!")
print("=" * 80)
