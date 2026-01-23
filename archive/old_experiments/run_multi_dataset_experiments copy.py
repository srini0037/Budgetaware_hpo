"""
Multi-Dataset Experiment Runner
Runs baseline HPO experiments on all datasets systematically
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import time
import sys

# Import your existing HPO implementations
# Adjust these imports based on your actual file structure
try:
    from hpo.random_search import run_random_search_budget
    from hpo.sha import run_sha_budget
    from hpo.hyperband import run_hyperband_budget
    HPO_AVAILABLE = True
except ImportError:
    print("WARNING: HPO implementations not found in hpo/ directory")
    print("Please ensure you have:")
    print("  - hpo/random_search.py")
    print("  - hpo/sha.py")
    print("  - hpo/hyperband.py")
    HPO_AVAILABLE = False

# Configuration
DATASETS = ['adult', 'fashion_mnist', 'mnist', 'letter']
METHODS = ['random_search', 'sha', 'hyperband']
BUDGETS = [60, 120, 300, 600]  # seconds
N_RUNS = 10

def load_processed_dataset(dataset_name):
    """Load preprocessed dataset"""
    data_dir = Path(f'data/processed/{dataset_name}')
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_single_experiment(dataset, method, budget, run_id):
    """
    Run a single experiment configuration
    
    Returns:
        dict with results including best_score, time_taken, etc.
    """
    print(f"\n  [{dataset}] {method} - {budget}s - Run {run_id}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_dataset(dataset)
    
    # Run the appropriate HPO method
    start_time = time.time()
    
    try:
        if method == 'random_search':
            result = run_random_search_budget(
                X_train, y_train, X_val, y_val,
                budget_seconds=budget,
                random_state=run_id
            )
        elif method == 'sha':
            result = run_sha_budget(
                X_train, y_train, X_val, y_val,
                budget_seconds=budget,
                random_state=run_id
            )
        elif method == 'hyperband':
            result = run_hyperband_budget(
                X_train, y_train, X_val, y_val,
                budget_seconds=budget,
                random_state=run_id
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        elapsed = time.time() - start_time
        
        # Add metadata
        result['dataset'] = dataset
        result['method'] = method
        result['budget'] = budget
        result['run_id'] = run_id
        result['timestamp'] = datetime.now().isoformat()
        result['actual_time'] = elapsed
        
        # Save result
        result_dir = Path(f'results/{dataset}')
        result_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f'{method}_budget{budget}_run{run_id}.json'
        with open(result_dir / filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"    ✓ Score: {result.get('best_val_score', 0):.4f}, Time: {elapsed:.1f}s")
        
        return result
        
    except Exception as e:
        print(f"    ✗ ERROR: {e}")
        return {
            'dataset': dataset,
            'method': method,
            'budget': budget,
            'run_id': run_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def estimate_total_time():
    """Estimate total experiment time"""
    total_experiments = len(DATASETS) * len(METHODS) * len(BUDGETS) * N_RUNS
    
    # Rough time estimates per experiment
    # Budget + overhead (loading, saving, etc.)
    avg_time_per_exp = {
        60: 70,    # 60s budget + 10s overhead
        120: 130,
        300: 310,
        600: 610
    }
    
    total_seconds = 0
    for budget in BUDGETS:
        n_exps_at_budget = len(DATASETS) * len(METHODS) * N_RUNS
        total_seconds += n_exps_at_budget * avg_time_per_exp[budget]
    
    hours = total_seconds / 3600
    
    return total_experiments, hours

def run_all_experiments(resume=True):
    """
    Run all experiments
    
    Args:
        resume: If True, skip experiments that already have results
    """
    
    print("="*70)
    print("PHASE 2: MULTI-DATASET EXPERIMENTS")
    print("="*70)
    
    # Estimate time
    total_exp, est_hours = estimate_total_time()
    print(f"\nTotal experiments: {total_exp}")
    print(f"Estimated time: {est_hours:.1f} hours ({est_hours/24:.1f} days)")
    print(f"\nConfiguration:")
    print(f"  Datasets: {', '.join(DATASETS)}")
    print(f"  Methods: {', '.join(METHODS)}")
    print(f"  Budgets: {BUDGETS}")
    print(f"  Runs per config: {N_RUNS}")
    
    # Check if we can resume
    if resume:
        print(f"\nResume mode: ON (will skip existing results)")
    
    input("\nPress Enter to start experiments (or Ctrl+C to cancel)...")
    
    # Track progress
    completed = 0
    failed = 0
    skipped = 0
    
    start_time = datetime.now()
    
    # Run experiments
    for dataset in DATASETS:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*70}")
        
        for method in METHODS:
            for budget in BUDGETS:
                for run_id in range(N_RUNS):
                    
                    # Check if result already exists
                    result_file = Path(f'results/{dataset}/{method}_budget{budget}_run{run_id}.json')
                    if resume and result_file.exists():
                        skipped += 1
                        continue
                    
                    # Run experiment
                    try:
                        result = run_single_experiment(dataset, method, budget, run_id)
                        if 'error' not in result:
                            completed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"    ✗ CRITICAL ERROR: {e}")
                        failed += 1
                    
                    # Progress update
                    total_done = completed + failed + skipped
                    pct_complete = (total_done / total_exp) * 100
                    
                    if total_done % 10 == 0:  # Update every 10 experiments
                        elapsed = (datetime.now() - start_time).total_seconds() / 3600
                        remaining = (elapsed / total_done) * (total_exp - total_done)
                        print(f"\n  Progress: {total_done}/{total_exp} ({pct_complete:.1f}%)")
                        print(f"  Completed: {completed}, Failed: {failed}, Skipped: {skipped}")
                        print(f"  Elapsed: {elapsed:.1f}h, Est. remaining: {remaining:.1f}h")
    
    # Final summary
    elapsed_total = (datetime.now() - start_time).total_seconds() / 3600
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed_total:.2f} hours")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {completed + failed + skipped}/{total_exp}")
    
    if completed + skipped >= total_exp * 0.9:  # 90% success rate
        print("\n✓ Sufficient results for analysis!")
        print("  Next step: Statistical analysis and Phase 3 preparation")
    else:
        print("\n⚠ Some experiments failed. Review error messages above.")
    
    print("="*70)

def quick_validation():
    """Run quick validation experiments (1 run per config) to test pipeline"""
    print("="*70)
    print("QUICK VALIDATION MODE")
    print("="*70)
    print("Running 1 experiment per dataset/method/budget to validate pipeline")
    print(f"Total: {len(DATASETS) * len(METHODS) * len(BUDGETS)} experiments")
    
    input("\nPress Enter to start...")
    
    for dataset in DATASETS:
        for method in METHODS:
            for budget in BUDGETS:
                run_single_experiment(dataset, method, budget, run_id=0)
    
    print("\n✓ Validation complete!")
    print("If all experiments succeeded, you can run full experiments with:")
    print("  python run_multi_dataset_experiments.py --full")

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multi-dataset HPO experiments')
    parser.add_argument('--full', action='store_true', 
                       help='Run full experiments (all runs)')
    parser.add_argument('--validate', action='store_true',
                       help='Run quick validation (1 run per config)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not resume from existing results')
    
    args = parser.parse_args()
    
    if not HPO_AVAILABLE:
        print("\nERROR: HPO implementations not found!")
        print("Please ensure your HPO code is properly structured in hpo/ directory")
        sys.exit(1)
    
    if args.validate:
        quick_validation()
    else:
        run_all_experiments(resume=not args.no_resume)

if __name__ == '__main__':
    main()
