"""
Hyperband Implementation for Budget-Aware HPO
Builds on Successive Halving Algorithm (SHA)

Based on:
- Your existing SHA implementation
- Paper: "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
- AutoML Book Chapter 1
"""

import numpy as np
import time
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform, randint


class Hyperband:
    """
    Hyperband algorithm for hyperparameter optimization.
    
    Hyperband runs multiple brackets of Successive Halving with different
    trade-offs between number of configurations and resources per configuration.
    """
    
    def __init__(self, get_random_config, max_iter=81, eta=3, verbose=True):
        """
        Parameters
        ----------
        get_random_config : callable
            Function that returns a random hyperparameter configuration
        max_iter : int
            Maximum iterations/budget for a single configuration
        eta : int
            Reduction factor (default 3, from Hyperband paper)
        verbose : bool
            Whether to print progress
        """
        self.get_random_config = get_random_config
        self.max_iter = max_iter
        self.eta = eta
        self.verbose = verbose
        
        # Calculate Hyperband parameters
        self.s_max = int(np.floor(np.log(max_iter) / np.log(eta)))
        self.B = (self.s_max + 1) * max_iter
        
        # Tracking
        self.results = []
        self.total_time = 0
        self.configs_evaluated = 0
        
    def successive_halving(self, n_configs, r, s):
        """
        Run one bracket of Successive Halving.
        
        Parameters
        ----------
        n_configs : int
            Number of configurations to start with
        r : int/float
            Initial resource (budget) per configuration
        s : int
            Bracket index
            
        Returns
        -------
        best_config : dict
            Best configuration found in this bracket
        best_score : float
            Best validation score achieved
        """
        # Sample n random configurations
        configs = [self.get_random_config() for _ in range(n_configs)]
        
        if self.verbose:
            print(f"\n  Bracket s={s}: Starting with {n_configs} configs, r={r:.1f}")
        
        for i in range(s + 1):
            # Current number of configs and resource per config
            n_i = int(n_configs * self.eta ** (-i))
            r_i = int(r * self.eta ** i)
            
            if self.verbose:
                print(f"    Round {i}: Evaluating {len(configs)} configs with r={r_i}")
            
            # Evaluate all configurations with resource r_i
            scores = []
            for config in configs:
                score, eval_time = self._evaluate_config(config, r_i)
                scores.append(score)
                self.results.append({
                    'bracket': s,
                    'round': i,
                    'config': config,
                    'resource': r_i,
                    'score': score,
                    'time': eval_time
                })
                self.configs_evaluated += 1
                self.total_time += eval_time
            
            # Keep top eta^(-1) configurations
            if i < s:  # Don't eliminate in last round
                n_keep = int(n_i / self.eta)
                indices = np.argsort(scores)[-n_keep:]  # Top n_keep
                configs = [configs[j] for j in indices]
                scores = [scores[j] for j in indices]
                
                if self.verbose:
                    print(f"      Keeping top {n_keep} configs, best score: {max(scores):.4f}")
        
        # Return best configuration
        best_idx = np.argmax(scores)
        return configs[best_idx], scores[best_idx]
    
    def run(self, X_train, y_train):
        """
        Run Hyperband optimization.
        
        Parameters
        ----------
        X_train : array-like
            Training data
        y_train : array-like
            Training labels
            
        Returns
        -------
        best_config : dict
            Best hyperparameter configuration found
        best_score : float
            Best validation score achieved
        results : list
            Detailed results from all evaluations
        """
        self.X_train = X_train
        self.y_train = y_train
        
        start_time = time.time()
        
        if self.verbose:
            print("=" * 70)
            print("HYPERBAND OPTIMIZATION")
            print(f"Max iterations: {self.max_iter}, eta: {self.eta}")
            print(f"Total brackets: {self.s_max + 1}")
            print("=" * 70)
        
        all_best_configs = []
        all_best_scores = []
        
        # Run each bracket
        for s in range(self.s_max, -1, -1):
            # Calculate n and r for this bracket
            n = int(np.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
            r = self.max_iter * self.eta ** (-s)
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"BRACKET {self.s_max - s + 1}/{self.s_max + 1} (s={s})")
                print(f"{'='*70}")
            
            # Run successive halving for this bracket
            best_config, best_score = self.successive_halving(n, r, s)
            
            all_best_configs.append(best_config)
            all_best_scores.append(best_score)
            
            if self.verbose:
                print(f"\n  Bracket {s} complete. Best score: {best_score:.4f}")
        
        # Find overall best
        best_idx = np.argmax(all_best_scores)
        best_config = all_best_configs[best_idx]
        best_score = all_best_scores[best_idx]
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("HYPERBAND COMPLETE")
            print(f"Best score: {best_score:.4f}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Configs evaluated: {self.configs_evaluated}")
            print("=" * 70)
        
        return {
            'best_config': best_config,
            'best_score': best_score,
            'all_results': self.results,
            'total_time': total_time,
            'configs_evaluated': self.configs_evaluated
        }
    
    def _evaluate_config(self, config, max_iter):
        """
        Evaluate a configuration with given resource budget.
        
        Parameters
        ----------
        config : dict
            Hyperparameter configuration
        max_iter : int
            Maximum training iterations
            
        Returns
        -------
        score : float
            Validation score (higher is better)
        eval_time : float
            Time taken for evaluation
        """
        start = time.time()
        
        # Create model with config
        # Update max_iter to use the budget
        model_config = config.copy()
        model_config['max_iter'] = int(max_iter)
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(**model_config, random_state=42))
        ])
        
        # Cross-validation score (3-fold for speed)
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=3,
            scoring='f1_macro',
            n_jobs=1  # Avoid nested parallelism
        )
        
        eval_time = time.time() - start
        
        return scores.mean(), eval_time


# Example hyperparameter space for MLP
def get_random_mlp_config():
    """
    Sample a random MLP configuration.
    
    This defines your hyperparameter search space.
    Adjust ranges based on your problem.
    """
    return {
        'hidden_layer_sizes': tuple([
            int(loguniform(50, 300).rvs()) 
            for _ in range(randint(1, 4).rvs())
        ]),
        'activation': np.random.choice(['relu', 'tanh']),
        'solver': 'adam',
        'alpha': loguniform(1e-5, 1e-1).rvs(),
        'learning_rate_init': loguniform(1e-4, 1e-2).rvs(),
        'batch_size': int(2 ** randint(5, 9).rvs()),  # 32, 64, 128, 256
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
    }


# Example usage
if __name__ == "__main__":
    # This is just a demonstration
    # In your actual code, load your Covertype data
    
    from sklearn.datasets import make_classification
    
    print("Generating example dataset...")
    X, y = make_classification(
        n_samples=5000,
        n_features=54,
        n_classes=7,
        n_informative=30,
        n_redundant=10,
        random_state=42
    )
    
    print("Running Hyperband...")
    hb = Hyperband(
        get_random_config=get_random_mlp_config,
        max_iter=81,  # Maximum epochs per config
        eta=3,
        verbose=True
    )
    
    result = hb.run(X, y)
    
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Best configuration: {result['best_config']}")
    print(f"Best score: {result['best_score']:.4f}")
    print(f"Total time: {result['total_time']:.2f}s")
    print(f"Configurations evaluated: {result['configs_evaluated']}")


"""
INTEGRATION WITH YOUR EXISTING CODE
====================================

To use this in your mlp_hpo.ipynb:

1. Copy this file to your hpo/ folder

2. In your notebook:

```python
from hpo.hyperband_implementation import Hyperband, get_random_mlp_config

# Load your data
X_train, X_val, y_train, y_val = ...  # Your existing data loading

# Run Hyperband
hb = Hyperband(
    get_random_config=get_random_mlp_config,
    max_iter=81,  # Adjust based on your needs
    eta=3,
    verbose=True
)

result = hb.run(X_train, y_train)

# Save results
results_df = pd.DataFrame(result['all_results'])
results_df.to_csv('results/hpo/hyperband_results.csv', index=False)
```

3. Compare with your existing Random Search and SHA results

BUDGET-AWARE VERSION
====================

To make this budget-aware:

```python
def run_hyperband_with_budget(X, y, time_budget_seconds):
    '''
    Run Hyperband with a time budget constraint.
    '''
    start_time = time.time()
    
    hb = Hyperband(
        get_random_config=get_random_mlp_config,
        max_iter=81,
        eta=3,
        verbose=True
    )
    
    # Modify to stop when budget exhausted
    # (implementation detail - can add timeout mechanism)
    
    result = hb.run(X, y)
    
    actual_time = time.time() - start_time
    
    return {
        **result,
        'budget_used': actual_time,
        'budget_exceeded': actual_time > time_budget_seconds
    }
```
"""
