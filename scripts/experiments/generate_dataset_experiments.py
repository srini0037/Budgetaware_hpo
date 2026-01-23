"""
Simple Multi-Dataset Experiment Runner
Adapts your existing budget_aware_experiment.py to run on all datasets

This is a quick-start version that reuses your existing code structure.
"""

import sys
import subprocess
from pathlib import Path

# Datasets to process
DATASETS = ['adult', 'fashion_mnist', 'mnist', 'letter']

def create_dataset_experiment_script(dataset_name):
    """
    Create a modified version of your budget_aware_experiment.py for a specific dataset
    """
    
    script_content = f'''"""
Budget-Aware Experiment for {dataset_name}
Auto-generated from your existing budget_aware_experiment.py
"""

import numpy as np
import pandas as pd
import time
import json
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
print("BUDGET-AWARE HPO: {dataset_name.upper()}")
print("=" * 80)
print(f"Start time: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")

# CONFIGURATION
BUDGET_LEVELS = {{
    'very_low': 60,
    'low': 120,
    'medium': 300,
    'high': 600,
}}

N_RUNS = 10
DATASET_NAME = "{dataset_name}"

# LOAD PREPROCESSED DATA
print(f"\\n📊 Loading {{DATASET_NAME}} dataset...")

X_train = np.load(f'data/processed/{{DATASET_NAME}}/X_train.npy')
y_train = np.load(f'data/processed/{{DATASET_NAME}}/y_train.npy')
X_val = np.load(f'data/processed/{{DATASET_NAME}}/X_val.npy')
y_val = np.load(f'data/processed/{{DATASET_NAME}}/y_val.npy')
X_test = np.load(f'data/processed/{{DATASET_NAME}}/X_test.npy')
y_test = np.load(f'data/processed/{{DATASET_NAME}}/y_test.npy')

print(f"✅ Train: {{X_train.shape[0]}}, Val: {{X_val.shape[0]}}, Test: {{X_test.shape[0]}}")

# HELPER FUNCTIONS (from your original code)
def evaluate_config(config, X_train, y_train):
    """Evaluate a single configuration using 3-fold CV"""
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(**config, random_state=42))
    ])
    
    scores = cross_val_score(model, X_train, y_train, cv=3, 
                            scoring='f1_macro', n_jobs=-1)
    return scores.mean()

def run_random_search(X_train, y_train, X_val, y_val, budget_seconds, seed=42):
    """Random Search"""
    np.random.seed(seed)
    start_time = time.time()
    
    best_score = 0
    best_config = None
    n_configs = 0
    
    while (time.time() - start_time) < budget_seconds:
        config = get_random_mlp_config()
        
        try:
            score = evaluate_config(config, X_train, y_train)
            n_configs += 1
            
            if score > best_score:
                best_score = score
                best_config = config
                
        except Exception as e:
            continue
            
        if (time.time() - start_time) >= budget_seconds:
            break
    
    # Evaluate best on validation
    if best_config:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(**best_config, random_state=42))
        ])
        model.fit(X_train, y_train)
        val_score = f1_score(y_val, model.predict(X_val), average='macro')
    else:
        val_score = 0
    
    return {{
        'method': 'random_search',
        'best_val_score': val_score,
        'n_configs_tried': n_configs,
        'time_taken': time.time() - start_time
    }}

def run_sha(X_train, y_train, X_val, y_val, budget_seconds, seed=42):
    """Successive Halving"""
    np.random.seed(seed)
    
    n = 27  # Initial configurations
    r_min = 1
    eta = 3
    
    configs = [get_random_mlp_config() for _ in range(n)]
    
    start_time = time.time()
    r = r_min
    
    while len(configs) > 1 and (time.time() - start_time) < budget_seconds:
        scores = []
        for config in configs:
            if (time.time() - start_time) >= budget_seconds:
                break
            try:
                score = evaluate_config(config, X_train, y_train)
                scores.append((score, config))
            except:
                scores.append((0, config))
        
        scores.sort(reverse=True)
        n_keep = max(1, len(configs) // eta)
        configs = [config for score, config in scores[:n_keep]]
        r *= eta
        
        if (time.time() - start_time) >= budget_seconds:
            break
    
    best_config = configs[0] if configs else None
    
    # Evaluate on validation
    if best_config:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(**best_config, random_state=42))
        ])
        model.fit(X_train, y_train)
        val_score = f1_score(y_val, model.predict(X_val), average='macro')
    else:
        val_score = 0
    
    return {{
        'method': 'sha',
        'best_val_score': val_score,
        'time_taken': time.time() - start_time
    }}

def run_hyperband(X_train, y_train, X_val, y_val, budget_seconds, seed=42):
    """Hyperband"""
    np.random.seed(seed)
    
    hb = Hyperband(
        get_random_config=get_random_mlp_config,
        max_iter=27,
        eta=3,
        verbose=False
    )
    
    start_time = time.time()
    
    # Run Hyperband with time limit
    best_config = None
    best_score = 0
    
    for s in range(hb.s_max, -1, -1):
        if (time.time() - start_time) >= budget_seconds:
            break
            
        n = int(np.ceil(hb.B / hb.max_iter / (s + 1) * hb.eta ** s))
        r = hb.max_iter * hb.eta ** (-s)
        
        configs = [get_random_mlp_config() for _ in range(n)]
        
        for i in range(s + 1):
            if (time.time() - start_time) >= budget_seconds:
                break
                
            n_i = int(n * hb.eta ** (-i))
            r_i = int(r * hb.eta ** i)
            
            scores = []
            for config in configs[:n_i]:
                if (time.time() - start_time) >= budget_seconds:
                    break
                try:
                    score = evaluate_config(config, X_train, y_train)
                    scores.append((score, config))
                except:
                    scores.append((0, config))
            
            scores.sort(reverse=True)
            configs = [config for score, config in scores]
            
            if scores and scores[0][0] > best_score:
                best_score = scores[0][0]
                best_config = scores[0][1]
    
    # Evaluate on validation
    if best_config:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(**best_config, random_state=42))
        ])
        model.fit(X_train, y_train)
        val_score = f1_score(y_val, model.predict(X_val), average='macro')
    else:
        val_score = 0
    
    return {{
        'method': 'hyperband',
        'best_val_score': val_score,
        'time_taken': time.time() - start_time
    }}

# RUN EXPERIMENTS
print("\\n🚀 Starting experiments...")

results = []
methods_to_run = [
    ('random_search', run_random_search),
    ('sha', run_sha),
    ('hyperband', run_hyperband)
]

for budget_name, budget_seconds in BUDGET_LEVELS.items():
    print(f"\\n{'='*60}")
    print(f"Budget: {{budget_name}} ({{budget_seconds}}s)")
    print(f"{'='*60}")
    
    for method_name, method_func in methods_to_run:
        print(f"\\n  {{method_name}}:")
        
        for run in range(N_RUNS):
            print(f"    Run {{run+1}}/{{N_RUNS}}...", end=" ")
            
            result = method_func(X_train, y_train, X_val, y_val, 
                               budget_seconds, seed=run)
            
            result['dataset'] = DATASET_NAME
            result['budget_level'] = budget_name
            result['budget_seconds'] = budget_seconds
            result['run_id'] = run
            result['timestamp'] = datetime.now().isoformat()
            
            results.append(result)
            
            print(f"Score: {{result['best_val_score']:.4f}}")
            
            # Save individual result
            result_dir = Path(f'results/{{DATASET_NAME}}')
            result_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{{method_name}}_budget{{budget_seconds}}_run{{run}}.json"
            with open(result_dir / filename, 'w') as f:
                json.dump(result, f, indent=2)

# Save summary
df = pd.DataFrame(results)
df.to_csv(f'results/{{DATASET_NAME}}/summary.csv', index=False)

print("\\n" + "="*80)
print(f"✅ COMPLETE: {{DATASET_NAME}}")
print("="*80)
print(f"Total experiments: {{len(results)}}")
print(f"Results saved to: results/{{DATASET_NAME}}/")
print("="*80)
'''
    
    # Save the script
    script_path = Path(f'experiments_generated/{dataset_name}_experiment.py')
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def main():
    """Main execution"""
    print("="*70)
    print("SIMPLE MULTI-DATASET EXPERIMENT GENERATOR")
    print("="*70)
    print("\\nThis will create experiment scripts for each dataset")
    print("You can then run them one at a time or in parallel\\n")
    
    # Generate scripts
    scripts = []
    for dataset in DATASETS:
        print(f"Generating script for {dataset}...")
        script_path = create_dataset_experiment_script(dataset)
        scripts.append((dataset, script_path))
        print(f"  ✓ Created: {script_path}")
    
    # Instructions
    print("\\n" + "="*70)
    print("SCRIPTS GENERATED")
    print("="*70)
    print("\\nTo run experiments, execute each script:")
    for dataset, script_path in scripts:
        print(f"  python {script_path}")
    
    print("\\nOr run them sequentially:")
    print("  " + " && ".join([f"python {sp}" for _, sp in scripts]))
    
    print("\\nEstimated time per dataset: 3-4 hours")
    print(f"Total for all {len(DATASETS)} datasets: 12-16 hours")
    print("="*70)
    
    # Ask if user wants to run first one
    response = input("\\nRun first dataset (adult) now? (y/n): ")
    if response.lower() == 'y':
        print("\\nStarting adult dataset experiments...")
        subprocess.run(['python', scripts[0][1]])

if __name__ == '__main__':
    main()
