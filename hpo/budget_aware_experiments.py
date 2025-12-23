"""
Budget-Aware HPO Experiments
=============================

This script implements Harry's budget-aware framework:
1. Test multiple HPO methods at different budget levels
2. Identify crossing points where methods overtake each other
3. Track performance vs budget curves
4. Compare efficiency metrics

Author: Your Name
Date: December 2024
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from scipy.stats import loguniform, randint


class BudgetAwareExperiment:
    """
    Framework for budget-constrained HPO experiments.
    
    Compares different HPO methods under various budget constraints
    to identify crossing points and efficiency trade-offs.
    """
    
    def __init__(self, dataset_name, max_samples=50000, 
                 test_size=0.2, random_state=42):
        """
        Parameters
        ----------
        dataset_name : str
            Name of dataset (e.g., 'covertype', 'adult', 'credit-g')
        max_samples : int
            Maximum number of samples to use
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed
        """
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.test_size = test_size
        self.random_state = random_state
        
        # Will be populated during experiments
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.baseline_score = None
        
    def load_data(self):
        """Load and prepare dataset."""
        print(f"Loading {self.dataset_name}...")
        
        # Load from OpenML
        X, y = fetch_openml(
            name=self.dataset_name,
            version=2 if self.dataset_name == 'covertype' else 1,
            as_frame=False,
            return_X_y=True
        )
        y = y.astype(int)
        
        # Subsample if needed
        if X.shape[0] > self.max_samples:
            X, y = resample(
                X, y,
                n_samples=self.max_samples,
                stratify=y,
                random_state=self.random_state
            )
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"  Dataset: {X.shape}")
        print(f"  Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")
        print(f"  Classes: {len(np.unique(y))}")
        
        return self
    
    def run_baseline(self):
        """Run baseline MLP with default parameters."""
        print("\nRunning baseline (default MLP)...")
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=200,
                random_state=self.random_state
            ))
        ])
        
        start = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(self.X_test)
        self.baseline_score = f1_score(self.y_test, y_pred, average='macro')
        
        print(f"  Baseline F1: {self.baseline_score:.4f}")
        print(f"  Train time: {train_time:.2f}s")
        
        return {
            'method': 'baseline',
            'score': self.baseline_score,
            'time': train_time,
            'configs_evaluated': 1
        }
    
    def run_random_search_with_budget(self, time_budget):
        """
        Run Random Search with time budget constraint.
        
        Parameters
        ----------
        time_budget : float
            Maximum time in seconds
            
        Returns
        -------
        result : dict
            Results including best score, configs evaluated, time used
        """
        print(f"\nRandom Search (budget: {time_budget}s)...")
        
        configs_evaluated = 0
        best_score = 0
        best_config = None
        start_time = time.time()
        all_scores = []
        all_times = []
        
        while True:
            # Check if budget exhausted
            elapsed = time.time() - start_time
            if elapsed >= time_budget:
                break
            
            # Sample random config
            config = self._get_random_config()
            
            # Evaluate
            config_start = time.time()
            score = self._evaluate_config(config)
            config_time = time.time() - config_start
            
            all_scores.append(score)
            all_times.append(elapsed + config_time)
            
            if score > best_score:
                best_score = score
                best_config = config
            
            configs_evaluated += 1
            
            # Don't start new config if budget almost exhausted
            if time.time() - start_time > time_budget * 0.95:
                break
        
        total_time = time.time() - start_time
        improvement = (best_score - self.baseline_score) / self.baseline_score * 100
        
        print(f"  Best F1: {best_score:.4f}")
        print(f"  Improvement: +{improvement:.1f}%")
        print(f"  Configs: {configs_evaluated}")
        print(f"  Time: {total_time:.1f}s")
        
        return {
            'method': 'random_search',
            'budget': time_budget,
            'best_score': best_score,
            'best_config': best_config,
            'configs_evaluated': configs_evaluated,
            'time_used': total_time,
            'improvement_pct': improvement,
            'score_history': all_scores,
            'time_history': all_times
        }
    
    def run_budget_aware_comparison(self, budget_levels):
        """
        Run comparison across multiple budget levels and methods.
        
        Parameters
        ----------
        budget_levels : dict
            Dictionary of {level_name: time_in_seconds}
            
        Returns
        -------
        results_df : DataFrame
            Comprehensive results for analysis
        """
        print("=" * 70)
        print("BUDGET-AWARE HPO COMPARISON")
        print("=" * 70)
        
        # Prepare data
        if self.X_train is None:
            self.load_data()
        
        # Run baseline
        baseline_result = self.run_baseline()
        
        all_results = []
        
        # Run each method at each budget level
        methods = ['random_search']  # Add 'sha', 'hyperband' when implemented
        
        for budget_name, budget_seconds in budget_levels.items():
            print(f"\n{'='*70}")
            print(f"BUDGET LEVEL: {budget_name.upper()} ({budget_seconds}s)")
            print(f"{'='*70}")
            
            for method in methods:
                if method == 'random_search':
                    result = self.run_random_search_with_budget(budget_seconds)
                # elif method == 'sha':
                #     result = self.run_sha_with_budget(budget_seconds)
                # elif method == 'hyperband':
                #     result = self.run_hyperband_with_budget(budget_seconds)
                
                all_results.append({
                    'dataset': self.dataset_name,
                    'budget_level': budget_name,
                    'budget_seconds': budget_seconds,
                    'method': method,
                    'best_score': result['best_score'],
                    'baseline_score': self.baseline_score,
                    'improvement_pct': result['improvement_pct'],
                    'configs_evaluated': result['configs_evaluated'],
                    'time_used': result['time_used'],
                    'efficiency': result['improvement_pct'] / result['time_used']
                })
        
        return pd.DataFrame(all_results)
    
    def _get_random_config(self):
        """Sample random MLP hyperparameter configuration."""
        return {
            'hidden_layer_sizes': tuple([
                int(loguniform(50, 300).rvs()) 
                for _ in range(randint(1, 4).rvs())
            ]),
            'activation': np.random.choice(['relu', 'tanh']),
            'alpha': loguniform(1e-5, 1e-1).rvs(),
            'learning_rate_init': loguniform(1e-4, 1e-2).rvs(),
            'batch_size': int(2 ** randint(5, 9).rvs()),
            'max_iter': 200,
            'early_stopping': True,
            'validation_fraction': 0.1,
        }
    
    def _evaluate_config(self, config):
        """Evaluate a single configuration."""
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(**config, random_state=self.random_state))
        ])
        
        # Use validation split for speed
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=3, scoring='f1_macro', n_jobs=1
        )
        
        return scores.mean()


def analyze_crossing_points(results_df):
    """
    Analyze where different methods start to outperform each other.
    
    This is critical for Harry's requirement of identifying crossing points!
    """
    print("\n" + "=" * 70)
    print("CROSSING POINT ANALYSIS")
    print("=" * 70)
    
    # For each budget level, find which method wins
    for budget_level in results_df['budget_level'].unique():
        subset = results_df[results_df['budget_level'] == budget_level]
        best_method = subset.loc[subset['best_score'].idxmax(), 'method']
        best_score = subset['best_score'].max()
        
        print(f"\n{budget_level.upper()}:")
        print(f"  Winner: {best_method}")
        print(f"  Score: {best_score:.4f}")
        
        # Show all methods
        for _, row in subset.iterrows():
            print(f"    {row['method']:15} | {row['best_score']:.4f} | "
                  f"{row['configs_evaluated']:3} configs | "
                  f"{row['time_used']:6.1f}s")


# Example usage
if __name__ == "__main__":
    
    # Define budget levels (adjust based on your timing measurements)
    budget_levels = {
        'very_low': 60,      # 1 minute
        'low': 120,          # 2 minutes
        'medium': 300,       # 5 minutes
        'high': 600,         # 10 minutes
    }
    
    # Run experiment
    exp = BudgetAwareExperiment('covertype', max_samples=50000)
    results = exp.run_budget_aware_comparison(budget_levels)
    
    # Analyze
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(results)
    
    # Save
    output_path = Path("results/budget_aware_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Analyze crossing points
    analyze_crossing_points(results)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Add SHA and Hyperband to this framework")
    print("2. Run on multiple datasets")
    print("3. Extract meta-features")
    print("4. Train meta-models for crossing point prediction")
