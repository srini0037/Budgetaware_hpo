"""
Comprehensive Analysis of Budget-Aware HPO Experiments
======================================================

Analyzes results from multi-dataset budget-aware HPO experiments.
Generates statistics, comparisons, and prepares data for dissertation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BUDGET-AWARE HPO RESULTS ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

PROJECT_ROOT = Path("/Users/srinivass/Budgetaware_hpo")
results_file = PROJECT_ROOT / "results" / "hpo" / "multi_dataset_budget_aware.csv"
baseline_dir = PROJECT_ROOT / "results" / "baselines"

print("\n📂 Loading data...")
df_hpo = pd.read_csv(results_file)
print(f"✓ Loaded {len(df_hpo)} HPO experiments")

# Load baselines
baselines = {}
for dataset in ['adult', 'fashion_mnist', 'mnist', 'letter']:
    baseline_file = baseline_dir / f"mlp_baseline_{dataset}.csv"
    if baseline_file.exists():
        df_baseline = pd.read_csv(baseline_file)
        baselines[dataset] = {
            'mean': df_baseline['f1_macro'].mean(),
            'std': df_baseline['f1_macro'].std(),
            'n': len(df_baseline)
        }
        print(f"✓ Loaded baseline for {dataset}: {baselines[dataset]['mean']:.4f} ± {baselines[dataset]['std']:.4f}")

# ============================================================================
# 2. DATA VERIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("DATA VERIFICATION")
print("=" * 80)

print(f"\nDataset shape: {df_hpo.shape}")
print(f"Columns: {list(df_hpo.columns)}")

# Check completeness
datasets = sorted(df_hpo['dataset'].unique())
budgets = sorted(df_hpo['budget_level'].unique())
methods = sorted(df_hpo['method'].unique())

print(f"\nDatasets ({len(datasets)}): {datasets}")
print(f"Budget levels ({len(budgets)}): {budgets}")
print(f"Methods ({len(methods)}): {methods}")

expected = len(datasets) * len(budgets) * len(methods) * 10
actual = len(df_hpo)
print(f"\nExpected experiments: {expected}")
print(f"Actual experiments: {actual}")
print(f"Completeness: {100 * actual / expected:.1f}%")

if actual == expected:
    print("✅ All experiments present!")
else:
    print(f"⚠️ Missing {expected - actual} experiments")

# ============================================================================
# 3. SUMMARY STATISTICS BY DATASET AND METHOD
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE BY DATASET AND METHOD (Overall)")
print("=" * 80)

summary_dataset_method = df_hpo.groupby(['dataset', 'method'])['best_score'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max'),
    ('count', 'count')
]).round(4)

print("\n", summary_dataset_method)

# ============================================================================
# 4. COMPARISON WITH BASELINES
# ============================================================================

print("\n" + "=" * 80)
print("HPO vs BASELINE COMPARISON")
print("=" * 80)

for dataset in datasets:
    if dataset not in baselines:
        continue
    
    baseline_mean = baselines[dataset]['mean']
    print(f"\n{dataset.upper()} (Baseline: {baseline_mean:.4f})")
    print("-" * 80)
    
    for method in methods:
        method_data = df_hpo[(df_hpo['dataset'] == dataset) & (df_hpo['method'] == method)]
        method_mean = method_data['best_score'].mean()
        improvement = ((method_mean - baseline_mean) / baseline_mean) * 100
        
        better = "✅" if method_mean > baseline_mean else "❌"
        print(f"  {method:15s}: {method_mean:.4f} ({improvement:+.2f}%) {better}")

# ============================================================================
# 5. PERFORMANCE BY BUDGET LEVEL
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE BY BUDGET LEVEL (All Datasets Combined)")
print("=" * 80)

summary_budget = df_hpo.groupby(['budget_level', 'method'])['best_score'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count')
]).round(4)

print("\n", summary_budget)

# ============================================================================
# 6. BUDGET UTILIZATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("BUDGET UTILIZATION ANALYSIS")
print("=" * 80)

df_hpo['budget_utilization'] = (df_hpo['time_used'] / df_hpo['budget_seconds']) * 100

budget_util = df_hpo.groupby(['budget_level', 'method']).agg({
    'time_used': ['mean', 'std', 'min', 'max'],
    'budget_utilization': ['mean', 'std'],
    'budget_seconds': 'first'
}).round(2)

print("\n", budget_util)

# ============================================================================
# 7. EFFICIENCY METRICS
# ============================================================================

print("\n" + "=" * 80)
print("EFFICIENCY METRICS")
print("=" * 80)

# Configs per second
df_hpo['configs_per_second'] = df_hpo['configs_evaluated'] / df_hpo['time_used']

# Performance per unit time
df_hpo['performance_per_second'] = df_hpo['best_score'] / df_hpo['time_used']

efficiency = df_hpo.groupby(['dataset', 'method']).agg({
    'configs_evaluated': ['mean', 'std'],
    'configs_per_second': ['mean', 'std'],
    'performance_per_second': ['mean', 'std']
}).round(4)

print("\n", efficiency)

# ============================================================================
# 8. STATISTICAL SIGNIFICANCE TESTS
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE TESTS (Example: Adult, High Budget)")
print("=" * 80)

# Compare methods at high budget for adult dataset
adult_high = df_hpo[(df_hpo['dataset'] == 'adult') & (df_hpo['budget_level'] == 'high')]

methods_list = adult_high['method'].unique()
print("\nPairwise t-tests (p-values):")
print("-" * 40)

for i, method1 in enumerate(methods_list):
    for method2 in methods_list[i+1:]:
        scores1 = adult_high[adult_high['method'] == method1]['best_score']
        scores2 = adult_high[adult_high['method'] == method2]['best_score']
        
        t_stat, p_value = stats.ttest_ind(scores1, scores2)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"{method1:15s} vs {method2:15s}: p={p_value:.4f} {sig}")

# ============================================================================
# 9. SAVE PROCESSED RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING PROCESSED RESULTS")
print("=" * 80)

# Summary by dataset, budget, method
summary_full = df_hpo.groupby(['dataset', 'budget_level', 'method']).agg({
    'best_score': ['mean', 'std', 'min', 'max'],
    'configs_evaluated': ['mean', 'std'],
    'time_used': ['mean', 'std'],
    'budget_utilization': ['mean', 'std']
}).round(4)

output_file = PROJECT_ROOT / "results" / "analysis_summary.csv"
summary_full.to_csv(output_file)
print(f"✓ Saved summary to: {output_file}")

# Add baseline comparisons
comparisons = []
for dataset in datasets:
    if dataset not in baselines:
        continue
    
    baseline_mean = baselines[dataset]['mean']
    
    for budget in budgets:
        for method in methods:
            subset = df_hpo[(df_hpo['dataset'] == dataset) & 
                           (df_hpo['budget_level'] == budget) & 
                           (df_hpo['method'] == method)]
            
            if len(subset) > 0:
                hpo_mean = subset['best_score'].mean()
                hpo_std = subset['best_score'].std()
                improvement = ((hpo_mean - baseline_mean) / baseline_mean) * 100
                
                comparisons.append({
                    'dataset': dataset,
                    'budget_level': budget,
                    'method': method,
                    'baseline_score': baseline_mean,
                    'hpo_score': hpo_mean,
                    'hpo_std': hpo_std,
                    'improvement_pct': improvement,
                    'beats_baseline': hpo_mean > baseline_mean
                })

df_comparisons = pd.DataFrame(comparisons)
comparison_file = PROJECT_ROOT / "results" / "baseline_comparisons.csv"
df_comparisons.to_csv(comparison_file, index=False)
print(f"✓ Saved baseline comparisons to: {comparison_file}")

# ============================================================================
# 10. KEY FINDINGS SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)

# Best method overall
best_overall = df_hpo.groupby('method')['best_score'].mean().sort_values(ascending=False)
print(f"\n1. Best Method Overall:")
for i, (method, score) in enumerate(best_overall.items(), 1):
    print(f"   {i}. {method}: {score:.4f}")

# Budget utilization by method
print(f"\n2. Budget Utilization by Method:")
util_by_method = df_hpo.groupby('method')['budget_utilization'].mean().sort_values(ascending=False)
for method, util in util_by_method.items():
    print(f"   {method}: {util:.1f}%")

# Datasets where HPO beats baseline
print(f"\n3. Datasets Where HPO Improves Over Baseline:")
for dataset in datasets:
    if dataset not in baselines:
        continue
    
    baseline_mean = baselines[dataset]['mean']
    dataset_mean = df_hpo[df_hpo['dataset'] == dataset]['best_score'].mean()
    
    if dataset_mean > baseline_mean:
        improvement = ((dataset_mean - baseline_mean) / baseline_mean) * 100
        print(f"   ✅ {dataset}: +{improvement:.2f}%")
    else:
        decline = ((baseline_mean - dataset_mean) / baseline_mean) * 100
        print(f"   ❌ {dataset}: -{decline:.2f}%")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  - {output_file}")
print(f"  - {comparison_file}")
print("\nNext steps:")
print("  1. Review statistical significance tests")
print("  2. Create visualizations")
print("  3. Start writing results section")
print("=" * 80)
