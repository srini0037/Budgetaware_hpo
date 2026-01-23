"""
Statistical Significance Tests
==============================

Performs comprehensive statistical tests for dissertation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

print("=" * 80)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

PROJECT_ROOT = Path("/Users/srinivass/Budgetaware_hpo")
df_hpo = pd.read_csv(PROJECT_ROOT / "results" / "hpo" / "multi_dataset_budget_aware.csv")

# Load baselines
baselines = {}
baseline_dir = PROJECT_ROOT / "results" / "baselines"
for dataset in ['adult', 'fashion_mnist', 'mnist', 'letter']:
    baseline_file = baseline_dir / f"mlp_baseline_{dataset}.csv"
    if baseline_file.exists():
        df_baseline = pd.read_csv(baseline_file)
        baselines[dataset] = {
            'scores': df_baseline['f1_macro'].values,
            'mean': df_baseline['f1_macro'].mean(),
            'std': df_baseline['f1_macro'].std()
        }

# ============================================================================
# TEST 1: HPO vs Baseline (For Each Dataset and Budget)
# ============================================================================

print("\n" + "=" * 80)
print("TEST 1: HPO vs BASELINE COMPARISONS")
print("=" * 80)

datasets = ['adult', 'fashion_mnist', 'mnist', 'letter']
budgets = ['very_low', 'low', 'medium', 'high']
methods = ['random_search', 'sha', 'hyperband']

hpo_baseline_results = []

for dataset in datasets:
    if dataset not in baselines:
        continue
    
    print(f"\n{dataset.upper()}")
    print("-" * 80)
    
    baseline_scores = baselines[dataset]['scores']
    
    for budget in budgets:
        print(f"\n  Budget: {budget}")
        
        for method in methods:
            # Get HPO scores
            hpo_scores = df_hpo[(df_hpo['dataset'] == dataset) & 
                               (df_hpo['budget_level'] == budget) & 
                               (df_hpo['method'] == method)]['best_score'].values
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(hpo_scores, baseline_scores)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(hpo_scores)**2 + np.std(baseline_scores)**2) / 2)
            cohens_d = (np.mean(hpo_scores) - np.mean(baseline_scores)) / pooled_std
            
            # Significance markers
            if p_value < 0.001:
                sig = "***"
            elif p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"
            else:
                sig = "ns"
            
            # Direction
            direction = "↑" if np.mean(hpo_scores) > np.mean(baseline_scores) else "↓"
            
            print(f"    {method:15s}: p={p_value:.4f} {sig}, d={cohens_d:+.3f} {direction}")
            
            hpo_baseline_results.append({
                'dataset': dataset,
                'budget': budget,
                'method': method,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'hpo_better': np.mean(hpo_scores) > np.mean(baseline_scores)
            })

# Save results
df_hpo_baseline = pd.DataFrame(hpo_baseline_results)
output_file = PROJECT_ROOT / "results" / "statistical_tests_hpo_vs_baseline.csv"
df_hpo_baseline.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# ============================================================================
# TEST 2: Pairwise Method Comparisons (Within Each Budget Level)
# ============================================================================

print("\n" + "=" * 80)
print("TEST 2: PAIRWISE METHOD COMPARISONS")
print("=" * 80)

pairwise_results = []

for dataset in datasets:
    print(f"\n{dataset.upper()}")
    print("-" * 80)
    
    for budget in budgets:
        print(f"\n  Budget: {budget}")
        
        # Get all method pairs
        for method1, method2 in combinations(methods, 2):
            scores1 = df_hpo[(df_hpo['dataset'] == dataset) & 
                            (df_hpo['budget_level'] == budget) & 
                            (df_hpo['method'] == method1)]['best_score'].values
            
            scores2 = df_hpo[(df_hpo['dataset'] == dataset) & 
                            (df_hpo['budget_level'] == budget) & 
                            (df_hpo['method'] == method2)]['best_score'].values
            
            # Paired t-test (same random seeds)
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            
            # Effect size
            diff = scores1 - scores2
            cohens_d = np.mean(diff) / np.std(diff)
            
            # Significance
            if p_value < 0.001:
                sig = "***"
            elif p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"
            else:
                sig = "ns"
            
            # Winner
            winner = method1 if np.mean(scores1) > np.mean(scores2) else method2
            
            print(f"    {method1} vs {method2}: p={p_value:.4f} {sig}, winner={winner}")
            
            pairwise_results.append({
                'dataset': dataset,
                'budget': budget,
                'method1': method1,
                'method2': method2,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'winner': winner
            })

# Save results
df_pairwise = pd.DataFrame(pairwise_results)
output_file = PROJECT_ROOT / "results" / "statistical_tests_pairwise.csv"
df_pairwise.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# ============================================================================
# TEST 3: Budget Effect (Within Each Method)
# ============================================================================

print("\n" + "=" * 80)
print("TEST 3: EFFECT OF BUDGET (ANOVA)")
print("=" * 80)

budget_effect_results = []

for dataset in datasets:
    print(f"\n{dataset.upper()}")
    print("-" * 80)
    
    for method in methods:
        # Get scores for all budget levels
        budget_groups = []
        for budget in budgets:
            scores = df_hpo[(df_hpo['dataset'] == dataset) & 
                           (df_hpo['budget_level'] == budget) & 
                           (df_hpo['method'] == method)]['best_score'].values
            budget_groups.append(scores)
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*budget_groups)
        
        # Significance
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        print(f"  {method:15s}: F={f_stat:.3f}, p={p_value:.4f} {sig}")
        
        budget_effect_results.append({
            'dataset': dataset,
            'method': method,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

# Save results
df_budget_effect = pd.DataFrame(budget_effect_results)
output_file = PROJECT_ROOT / "results" / "statistical_tests_budget_effect.csv"
df_budget_effect.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL TESTS SUMMARY")
print("=" * 80)

print(f"\nHPO vs Baseline:")
print(f"  Total comparisons: {len(df_hpo_baseline)}")
print(f"  Significant (p<0.05): {df_hpo_baseline['significant'].sum()}")
print(f"  HPO better: {df_hpo_baseline['hpo_better'].sum()}")
print(f"  Baseline better: {(~df_hpo_baseline['hpo_better']).sum()}")

print(f"\nPairwise Method Comparisons:")
print(f"  Total comparisons: {len(df_pairwise)}")
print(f"  Significant differences: {df_pairwise['significant'].sum()}")

print(f"\nBudget Effect:")
print(f"  Total tests: {len(df_budget_effect)}")
print(f"  Significant budget effect: {df_budget_effect['significant'].sum()}")

print("\n" + "=" * 80)
print("✅ STATISTICAL ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print(f"  - statistical_tests_hpo_vs_baseline.csv")
print(f"  - statistical_tests_pairwise.csv")
print(f"  - statistical_tests_budget_effect.csv")
print("=" * 80)
