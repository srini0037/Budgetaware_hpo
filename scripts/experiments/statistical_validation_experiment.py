"""
Overnight Experiment Script - Statistical Validation
====================================================

Runs all three HPO methods (Random Search, SHA, Hyperband) 
with 10 different random seeds for statistical validation.

Estimated runtime: 3.5 hours
Run this before bed and check results in the morning!

Date: December 23, 2024
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

# Import Hyperband
import sys
sys.path.append('/Users/srinivass/Budgetaware_hpo')
from hpo.hyperband_implementation import Hyperband, get_random_mlp_config


print("=" * 80)
print("OVERNIGHT STATISTICAL VALIDATION EXPERIMENT")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"This will run for approximately 3.5 hours")
print("=" * 80)


# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n📊 STEP 1: Loading Covertype dataset...")

X, y = fetch_openml(
    name="covertype",
    version=2,
    as_frame=False,
    return_X_y=True
)

# Convert sparse to dense
if hasattr(X, 'toarray'):
    X = X.toarray()

y = y.astype(int)

# Subsample to 50K
MAX_SAMPLES = 50000
if X.shape[0] > MAX_SAMPLES:
    X, y = resample(X, y, n_samples=MAX_SAMPLES, stratify=y, random_state=42)

print(f"✅ Dataset: {X.shape}, Classes: {len(np.unique(y))}")

# Train/val/test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"✅ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

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


# ============================================================================
# 3. RANDOM SEARCH FUNCTION
# ============================================================================

def run_random_search(X_train, y_train, n_iter=20, seed=42):
    """Run Random Search with n_iter configurations"""
    np.random.seed(seed)
    
    start_time = time.time()
    best_score = 0
    best_config = None
    
    for i in range(n_iter):
        config = get_random_mlp_config()
        score = evaluate_config(config, X_train, y_train)
        
        if score > best_score:
            best_score = score
            best_config = config
    
    total_time = time.time() - start_time
    
    return {
        'best_score': best_score,
        'best_config': best_config,
        'configs_evaluated': n_iter,
        'time': total_time
    }


# ============================================================================
# 4. SUCCESSIVE HALVING FUNCTION (Simplified)
# ============================================================================

def run_sha(X_train, y_train, n_configs=27, max_iter=81, eta=3, seed=42):
    """Run Successive Halving Algorithm"""
    np.random.seed(seed)
    
    start_time = time.time()
    
    # Sample initial configurations
    configs = [get_random_mlp_config() for _ in range(n_configs)]
    
    # Calculate number of rounds
    s = int(np.floor(np.log(n_configs) / np.log(eta)))
    
    r = max_iter / (eta ** s)  # Initial resource
    
    configs_evaluated = 0
    
    for i in range(s + 1):
        n_i = int(n_configs / (eta ** i))
        r_i = int(r * (eta ** i))
        
        # Evaluate all configs with resource r_i
        scores = []
        for config in configs:
            config_copy = config.copy()
            config_copy['max_iter'] = r_i
            score = evaluate_config(config_copy, X_train, y_train)
            scores.append(score)
            configs_evaluated += 1
        
        # Keep top configs
        if i < s:
            n_keep = int(n_i / eta)
            indices = np.argsort(scores)[-n_keep:]
            configs = [configs[j] for j in indices]
            scores = [scores[j] for j in indices]
    
    # Best config and score
    best_idx = np.argmax(scores)
    best_config = configs[best_idx]
    best_score = scores[best_idx]
    
    total_time = time.time() - start_time
    
    return {
        'best_score': best_score,
        'best_config': best_config,
        'configs_evaluated': configs_evaluated,
        'time': total_time
    }


# ============================================================================
# 5. RUN ALL EXPERIMENTS
# ============================================================================

N_RUNS = 10
SEEDS = list(range(N_RUNS))

results_dir = Path('/Users/srinivass/Budgetaware_hpo/results/hpo')
results_dir.mkdir(parents=True, exist_ok=True)

all_results = {
    'random_search': [],
    'sha': [],
    'hyperband': []
}

overall_start = time.time()


# ============================================================================
# RANDOM SEARCH
# ============================================================================

print("\n" + "=" * 80)
print("METHOD 1/3: RANDOM SEARCH")
print("=" * 80)

for i, seed in enumerate(SEEDS, 1):
    print(f"\n🔄 Run {i}/{N_RUNS} (seed={seed})")
    
    result = run_random_search(X_train, y_train, n_iter=20, seed=seed)
    
    all_results['random_search'].append({
        'run': i,
        'seed': seed,
        'method': 'random_search',
        'best_score': result['best_score'],
        'configs_evaluated': result['configs_evaluated'],
        'time': result['time']
    })
    
    print(f"   ✅ Score: {result['best_score']:.4f}, Time: {result['time']:.1f}s")

# Save Random Search results
rs_df = pd.DataFrame(all_results['random_search'])
rs_df.to_csv(results_dir / 'random_search_multiple_runs.csv', index=False)
print(f"\n💾 Random Search results saved")
print(f"   Mean: {rs_df['best_score'].mean():.4f} (±{rs_df['best_score'].std():.4f})")


# ============================================================================
# SUCCESSIVE HALVING
# ============================================================================

print("\n" + "=" * 80)
print("METHOD 2/3: SUCCESSIVE HALVING (SHA)")
print("=" * 80)

for i, seed in enumerate(SEEDS, 1):
    print(f"\n🔄 Run {i}/{N_RUNS} (seed={seed})")
    
    result = run_sha(X_train, y_train, n_configs=27, max_iter=81, seed=seed)
    
    all_results['sha'].append({
        'run': i,
        'seed': seed,
        'method': 'sha',
        'best_score': result['best_score'],
        'configs_evaluated': result['configs_evaluated'],
        'time': result['time']
    })
    
    print(f"   ✅ Score: {result['best_score']:.4f}, Time: {result['time']:.1f}s")

# Save SHA results
sha_df = pd.DataFrame(all_results['sha'])
sha_df.to_csv(results_dir / 'sha_multiple_runs.csv', index=False)
print(f"\n💾 SHA results saved")
print(f"   Mean: {sha_df['best_score'].mean():.4f} (±{sha_df['best_score'].std():.4f})")


# ============================================================================
# HYPERBAND
# ============================================================================

print("\n" + "=" * 80)
print("METHOD 3/3: HYPERBAND")
print("=" * 80)

for i, seed in enumerate(SEEDS, 1):
    print(f"\n🔄 Run {i}/{N_RUNS} (seed={seed})")
    
    np.random.seed(seed)
    
    hb = Hyperband(
        get_random_config=get_random_mlp_config,
        max_iter=81,
        eta=3,
        verbose=False
    )
    
    result = hb.run(X_train, y_train)
    
    all_results['hyperband'].append({
        'run': i,
        'seed': seed,
        'method': 'hyperband',
        'best_score': result['best_score'],
        'configs_evaluated': result['configs_evaluated'],
        'time': result['total_time']
    })
    
    print(f"   ✅ Score: {result['best_score']:.4f}, Time: {result['total_time']:.1f}s")

# Save Hyperband results
hb_df = pd.DataFrame(all_results['hyperband'])
hb_df.to_csv(results_dir / 'hyperband_multiple_runs.csv', index=False)
print(f"\n💾 Hyperband results saved")
print(f"   Mean: {hb_df['best_score'].mean():.4f} (±{hb_df['best_score'].std():.4f})")


# ============================================================================
# 6. FINAL SUMMARY
# ============================================================================

total_time = time.time() - overall_start

print("\n" + "=" * 80)
print("✅ ALL EXPERIMENTS COMPLETE!")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")

print("\n📊 FINAL COMPARISON:")
print("=" * 80)

comparison_data = []
for method_name, method_results in all_results.items():
    df = pd.DataFrame(method_results)
    comparison_data.append({
        'method': method_name,
        'mean_score': df['best_score'].mean(),
        'std_score': df['best_score'].std(),
        'min_score': df['best_score'].min(),
        'max_score': df['best_score'].max(),
        'mean_configs': df['configs_evaluated'].mean(),
        'mean_time': df['time'].mean()
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('mean_score', ascending=False)

print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv(results_dir / 'methods_comparison_summary.csv', index=False)

print(f"\n💾 All results saved to: {results_dir}")
print("\n🎉 You can now analyze the results and perform statistical tests!")
print("=" * 80)
