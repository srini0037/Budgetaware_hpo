"""
Run Budget-Aware HPO Experiments on Covertype
=============================================

Runs Random Search, SHA, and Hyperband at 4 budget levels on Covertype dataset.
10 runs per (budget, method) combination = 120 experiments total.

Estimated runtime: 2-4 hours
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COVERTYPE BUDGET-AWARE HPO EXPERIMENTS")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================
print("\n1. Loading Covertype data...")

X_train = np.load('data/processed/covertype/X_train.npy')
X_val = np.load('data/processed/covertype/X_val.npy')
X_test = np.load('data/processed/covertype/X_test.npy')
y_train = np.load('data/processed/covertype/y_train.npy')
y_val = np.load('data/processed/covertype/y_val.npy')
y_test = np.load('data/processed/covertype/y_test.npy')

print(f"   Train: {X_train.shape}")
print(f"   Val:   {X_val.shape}")
print(f"   Test:  {X_test.shape}")

# ============================================================================
# HPO Configuration Space
# ============================================================================

def get_random_config(seed):
    """Generate random MLP hyperparameter configuration"""
    np.random.seed(seed)
    
    return {
        'hidden_layer_sizes': tuple(np.random.choice([50, 100, 150, 200], 
                                                      size=np.random.choice([1, 2]))),
        'activation': np.random.choice(['relu', 'tanh']),
        'alpha': 10 ** np.random.uniform(-5, -1),
        'learning_rate_init': 10 ** np.random.uniform(-4, -2),
        'batch_size': np.random.choice([32, 64, 128, 256]),
        'max_iter': 100,
        'early_stopping': True,
        'random_state': seed
    }

def evaluate_config(config, X_train, y_train, X_val, y_val):
    """Train and evaluate MLP with given config"""
    start_time = time.time()
    
    mlp = MLPClassifier(**config)
    mlp.fit(X_train, y_train)
    
    y_pred = mlp.predict(X_val)
    score = f1_score(y_val, y_pred, average='macro')
    
    elapsed = time.time() - start_time
    
    return score, elapsed

# ============================================================================
# HPO Methods
# ============================================================================

def random_search(budget_seconds, seed):
    """Random Search HPO"""
    configs_evaluated = 0
    best_score = 0
    start_time = time.time()
    
    while time.time() - start_time < budget_seconds:
        config = get_random_config(seed + configs_evaluated)
        
        # Check if we have time for another evaluation
        if time.time() - start_time > budget_seconds * 0.9:
            break
            
        score, _ = evaluate_config(config, X_train, y_train, X_val, y_val)
        
        if score > best_score:
            best_score = score
        
        configs_evaluated += 1
    
    time_used = time.time() - start_time
    return best_score, configs_evaluated, time_used

def successive_halving(budget_seconds, seed):
    """Successive Halving Algorithm"""
    n_configs = 81
    eta = 3
    
    configs = [get_random_config(seed + i) for i in range(n_configs)]
    configs_evaluated = 0
    best_score = 0
    start_time = time.time()
    
    # Successive halving rounds
    while len(configs) > 0 and time.time() - start_time < budget_seconds:
        scores = []
        
        for config in configs:
            if time.time() - start_time >= budget_seconds:
                break
                
            score, _ = evaluate_config(config, X_train, y_train, X_val, y_val)
            scores.append(score)
            configs_evaluated += 1
            
            if score > best_score:
                best_score = score
        
        if len(scores) < len(configs):
            break
            
        # Keep top 1/eta
        n_keep = max(1, len(configs) // eta)
        indices = np.argsort(scores)[-n_keep:]
        configs = [configs[i] for i in indices]
    
    time_used = time.time() - start_time
    return best_score, configs_evaluated, time_used

def hyperband(budget_seconds, seed):
    """Hyperband Algorithm"""
    max_iter = 81
    eta = 3
    
    configs_evaluated = 0
    best_score = 0
    start_time = time.time()
    
    # Calculate brackets
    s_max = int(np.log(max_iter) / np.log(eta))
    
    for s in range(s_max, -1, -1):
        if time.time() - start_time >= budget_seconds:
            break
            
        n = int(np.ceil(max_iter / eta**s * eta))
        r = max_iter * eta**(-s)
        
        # Generate configurations
        configs = [get_random_config(seed + configs_evaluated + i) for i in range(n)]
        
        # Successive halving
        for i in range(s + 1):
            if time.time() - start_time >= budget_seconds:
                break
                
            n_i = int(n * eta**(-i))
            r_i = int(r * eta**i)
            
            scores = []
            for config in configs[:n_i]:
                if time.time() - start_time >= budget_seconds:
                    break
                    
                score, _ = evaluate_config(config, X_train, y_train, X_val, y_val)
                scores.append(score)
                configs_evaluated += 1
                
                if score > best_score:
                    best_score = score
            
            if len(scores) < len(configs[:n_i]):
                break
                
            # Keep top 1/eta
            n_keep = max(1, len(scores) // eta)
            indices = np.argsort(scores)[-n_keep:]
            configs = [configs[:n_i][idx] for idx in indices]
    
    time_used = time.time() - start_time
    return best_score, configs_evaluated, time_used

# ============================================================================
# Run Experiments
# ============================================================================

print("\n2. Running experiments...")
print("   Budget levels: 60s, 120s, 300s, 600s")
print("   Methods: Random Search, SHA, Hyperband")
print("   Runs per combination: 10")
print("   Total experiments: 120")
print()

budget_levels = {
    'very_low': 60,
    'low': 120,
    'medium': 300,
    'high': 600
}

methods = {
    'random_search': random_search,
    'sha': successive_halving,
    'hyperband': hyperband
}

# Check for existing results
import os
os.makedirs('results/hpo', exist_ok=True)
results_file = 'results/hpo/covertype_budget_aware.csv'

if os.path.exists(results_file):
    existing_df = pd.read_csv(results_file)
    results = existing_df.to_dict('records')
    print(f"   ⚠️  Found {len(results)} existing results - will skip completed experiments\n")
else:
    results = []

total_experiments = len(budget_levels) * len(methods) * 10
experiment_count = 0

for budget_name, budget_seconds in budget_levels.items():
    for method_name, method_func in methods.items():
        print(f"   Running: {method_name:<15} @ {budget_seconds:>4}s", end=" ")
        
        for run in range(10):
            experiment_count += 1
            seed = run
            
            # Check if already done
            already_done = any(
                r['budget_seconds'] == budget_seconds and 
                r['method'] == method_name and 
                r['run'] == run 
                for r in results
            )
            if already_done:
                print("✓", end="", flush=True)
                continue
            
            # Run HPO
            best_score, configs_evaluated, time_used = method_func(budget_seconds, seed)
            
            # Store result
            results.append({
                'dataset': 'covertype',
                'budget_level': budget_name,
                'budget_seconds': budget_seconds,
                'method': method_name,
                'run': run,
                'seed': seed,
                'best_score': best_score,
                'configs_evaluated': configs_evaluated,
                'time_used': time_used
            })
            
            # Save immediately after each run
            pd.DataFrame(results).to_csv(results_file, index=False)
            
            print(f".", end="", flush=True)
        
        print(f" Done! [{experiment_count}/{total_experiments}]")

# ============================================================================
# Save Results
# ============================================================================

print("\n3. Saving results...")

results_df = pd.DataFrame(results)

# Save standalone covertype results
import os
os.makedirs('results/hpo', exist_ok=True)
results_df.to_csv('results/hpo/covertype_budget_aware.csv', index=False)
print("   ✓ Saved: results/hpo/covertype_budget_aware.csv")

# Append to multi_dataset file if it exists
multi_path = 'results/hpo/multi_dataset_budget_aware.csv'
if os.path.exists(multi_path):
    existing = pd.read_csv(multi_path)
    # Remove covertype if already exists
    existing = existing[existing['dataset'] != 'covertype']
    # Append new covertype results
    updated = pd.concat([existing, results_df], ignore_index=True)
    updated.to_csv(multi_path, index=False)
    print(f"   ✓ Updated: {multi_path} (now {len(updated)} rows)")
else:
    results_df.to_csv(multi_path, index=False)
    print(f"   ✓ Created: {multi_path}")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n4. Summary statistics...")

summary = results_df.groupby(['budget_level', 'method']).agg({
    'best_score': ['mean', 'std'],
    'configs_evaluated': 'mean',
    'time_used': 'mean'
}).round(4)

print("\n   Mean F1-Score by (Budget, Method):")
print(summary)

# Best method per budget
print("\n   Best method per budget level:")
best_per_budget = results_df.groupby(['budget_level', 'budget_seconds', 'method'])['best_score'].mean()
for (budget_name, budget_secs), group in best_per_budget.groupby(level=[0,1]):
    best_method = group.idxmax()[2]
    best_score = group.max()
    print(f"      {budget_name:<10} ({budget_secs:>4}s): {best_method:<15} ({best_score:.4f})")

print("\n" + "=" * 80)
print("COVERTYPE EXPERIMENTS COMPLETE!")
print("=" * 80)
print("\nNext step: Re-run meta-learning with 5 datasets")
print("  Command: python meta_learning/02_train_meta_learner.py")
print("=" * 80)
