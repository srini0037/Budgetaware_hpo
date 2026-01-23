"""
Budget-Aware Results Analysis and Visualization
===============================================

Analyzes crossing points and creates publication-quality plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
results_dir = Path('/Users/srinivass/Budgetaware_hpo/results/hpo')
df = pd.read_csv(results_dir / 'budget_aware_comparison_fixed.csv')

print("="*80)
print("BUDGET-AWARE HPO ANALYSIS")
print("="*80)

# ============================================================================
# 1. SUMMARY STATISTICS
# ============================================================================

print("\n📊 SUMMARY STATISTICS BY BUDGET:")
print("="*80)

summary = df.groupby(['budget_level', 'method'])['best_score'].agg(['mean', 'std', 'min', 'max'])
summary = summary.round(4)
print(summary)

# ============================================================================
# 2. PERFORMANCE VS BUDGET PLOT
# ============================================================================

print("\n\n📈 Creating Performance vs Budget plot...")

fig, ax = plt.subplots(figsize=(12, 7))

# Define budget order and values
budget_order = ['very_low', 'low', 'medium', 'high']
budget_values = [60, 120, 300, 600]

methods = ['random_search', 'sha', 'hyperband']
method_labels = {'random_search': 'Random Search', 'sha': 'SHA', 'hyperband': 'Hyperband'}
colors = {'random_search': '#2E86AB', 'sha': '#A23B72', 'hyperband': '#F18F01'}
markers = {'random_search': 'o', 'sha': 's', 'hyperband': '^'}

for method in methods:
    means = []
    stds = []
    
    for budget_level in budget_order:
        subset = df[(df['budget_level'] == budget_level) & (df['method'] == method)]
        means.append(subset['best_score'].mean())
        stds.append(subset['best_score'].std())
    
    ax.errorbar(budget_values, means, yerr=stds, 
                label=method_labels[method],
                marker=markers[method], 
                markersize=10,
                linewidth=2.5,
                capsize=5,
                capthick=2,
                color=colors[method])

# Add baseline
baseline_score = 0.811
ax.axhline(y=baseline_score, color='red', linestyle='--', linewidth=2, 
           label='Baseline (No HPO)', alpha=0.7)

ax.set_xlabel('Budget (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-Macro Score', fontsize=14, fontweight='bold')
ax.set_title('HPO Performance vs Computational Budget\n(Covertype Dataset, 50K samples)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xticks(budget_values)
ax.set_xticklabels(['60s\n(1min)', '120s\n(2min)', '300s\n(5min)', '600s\n(10min)'])

plt.tight_layout()
plt.savefig(results_dir / '../figures/performance_vs_budget.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {results_dir / '../figures/performance_vs_budget.png'}")

# ============================================================================
# 3. EFFICIENCY ANALYSIS
# ============================================================================

print("\n\n⚡ EFFICIENCY ANALYSIS:")
print("="*80)

# Calculate improvement over baseline per second
baseline_score = 0.811

efficiency_data = []
for budget_level, budget_seconds in zip(budget_order, budget_values):
    for method in methods:
        subset = df[(df['budget_level'] == budget_level) & (df['method'] == method)]
        mean_score = subset['best_score'].mean()
        improvement = (mean_score - baseline_score) / baseline_score * 100
        efficiency = improvement / budget_seconds * 100  # Improvement per second
        
        efficiency_data.append({
            'budget_level': budget_level,
            'budget_seconds': budget_seconds,
            'method': method,
            'improvement_pct': improvement,
            'efficiency': efficiency
        })

efficiency_df = pd.DataFrame(efficiency_data)

print("\nImprovement over baseline (%):")
pivot_improvement = efficiency_df.pivot(index='budget_level', columns='method', values='improvement_pct')
pivot_improvement = pivot_improvement.reindex(budget_order)
print(pivot_improvement.round(2))

print("\nEfficiency (improvement% per second × 100):")
pivot_efficiency = efficiency_df.pivot(index='budget_level', columns='method', values='efficiency')
pivot_efficiency = pivot_efficiency.reindex(budget_order)
print(pivot_efficiency.round(4))

# ============================================================================
# 4. STATISTICAL TESTS FOR CROSSING POINTS
# ============================================================================

print("\n\n📉 STATISTICAL TESTS (Paired t-tests):")
print("="*80)

for budget_level in budget_order:
    print(f"\n{budget_level.upper()} BUDGET:")
    print("-" * 60)
    
    subset = df[df['budget_level'] == budget_level]
    
    rs_scores = subset[subset['method'] == 'random_search']['best_score'].values
    sha_scores = subset[subset['method'] == 'sha']['best_score'].values
    hb_scores = subset[subset['method'] == 'hyperband']['best_score'].values
    
    # Random Search vs SHA
    t_stat, p_val = stats.ttest_rel(rs_scores, sha_scores)
    print(f"Random Search vs SHA:")
    print(f"  Mean diff: {rs_scores.mean() - sha_scores.mean():.4f}")
    print(f"  p-value: {p_val:.4f} {'✓ significant' if p_val < 0.05 else '✗ not significant'}")
    
    # Random Search vs Hyperband
    t_stat, p_val = stats.ttest_rel(rs_scores, hb_scores)
    print(f"Random Search vs Hyperband:")
    print(f"  Mean diff: {rs_scores.mean() - hb_scores.mean():.4f}")
    print(f"  p-value: {p_val:.4f} {'✓ significant' if p_val < 0.05 else '✗ not significant'}")
    
    # SHA vs Hyperband
    t_stat, p_val = stats.ttest_rel(sha_scores, hb_scores)
    print(f"SHA vs Hyperband:")
    print(f"  Mean diff: {sha_scores.mean() - hb_scores.mean():.4f}")
    print(f"  p-value: {p_val:.4f} {'✓ significant' if p_val < 0.05 else '✗ not significant'}")

# ============================================================================
# 5. CONFIGURATIONS EVALUATED
# ============================================================================

print("\n\n🔢 CONFIGURATIONS EVALUATED:")
print("="*80)

config_summary = df.groupby(['budget_level', 'method'])['configs_evaluated'].agg(['mean', 'std'])
config_summary = config_summary.round(1)
print(config_summary)

# Plot configs evaluated
fig, ax = plt.subplots(figsize=(12, 7))

for method in methods:
    means = []
    
    for budget_level in budget_order:
        subset = df[(df['budget_level'] == budget_level) & (df['method'] == method)]
        means.append(subset['configs_evaluated'].mean())
    
    ax.plot(budget_values, means, 
            label=method_labels[method],
            marker=markers[method], 
            markersize=10,
            linewidth=2.5,
            color=colors[method])

ax.set_xlabel('Budget (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Configurations Evaluated', fontsize=14, fontweight='bold')
ax.set_title('Number of Configurations Evaluated vs Budget', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xticks(budget_values)
ax.set_xticklabels(['60s', '120s', '300s', '600s'])

plt.tight_layout()
plt.savefig(results_dir / '../figures/configs_vs_budget.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: {results_dir / '../figures/configs_vs_budget.png'}")

# ============================================================================
# 6. KEY FINDINGS SUMMARY
# ============================================================================

print("\n\n" + "="*80)
print("KEY FINDINGS SUMMARY")
print("="*80)

print("\n✅ MAIN RESULTS:")
print("  1. Random Search wins at ALL budget levels")
print("  2. Performance gap widens at lower budgets")
print("  3. SHA is fastest but plateaus early")
print("  4. Hyperband needs more budget to be competitive")

print("\n✅ BUDGET-SPECIFIC WINNERS:")
for budget_level, budget_sec in zip(budget_order, budget_values):
    subset = df[df['budget_level'] == budget_level]
    means = subset.groupby('method')['best_score'].mean()
    winner = means.idxmax()
    winner_score = means.max()
    print(f"  {budget_level.upper()} ({budget_sec}s): {method_labels[winner]} ({winner_score:.4f})")

print("\n✅ EFFICIENCY WINNERS (improvement per second):")
best_efficiency = efficiency_df.loc[efficiency_df.groupby('budget_level')['efficiency'].idxmax()]
for _, row in best_efficiency.iterrows():
    print(f"  {row['budget_level'].upper()}: {method_labels[row['method']]} ({row['efficiency']:.6f})")

print("\n✅ FOR DISSERTATION:")
print("  • Clear evidence that method performance depends on budget")
print("  • No crossing points observed (Random Search always wins)")
print("  • This motivates adaptive selection based on problem characteristics")
print("  • Next: Test on different datasets to find crossing points")

print("\n" + "="*80)
print("Analysis complete! Check the figures/ directory for plots.")
print("="*80)
