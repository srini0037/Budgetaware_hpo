"""
Visualization Script for Budget-Aware HPO Results
=================================================

Creates publication-quality figures for dissertation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

print("=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

PROJECT_ROOT = Path("/Users/srinivass/Budgetaware_hpo")
df_hpo = pd.read_csv(PROJECT_ROOT / "results" / "hpo" / "multi_dataset_budget_aware.csv")
df_comp = pd.read_csv(PROJECT_ROOT / "results" / "baseline_comparisons.csv")

# Load baselines
baselines = {}
baseline_dir = PROJECT_ROOT / "results" / "baselines"
for dataset in ['adult', 'fashion_mnist', 'mnist', 'letter']:
    baseline_file = baseline_dir / f"mlp_baseline_{dataset}.csv"
    if baseline_file.exists():
        df_baseline = pd.read_csv(baseline_file)
        baselines[dataset] = df_baseline['f1_macro'].mean()

# Create figures directory
figures_dir = PROJECT_ROOT / "figures" / "budget_aware"
figures_dir.mkdir(parents=True, exist_ok=True)

# Budget level ordering
budget_order = ['very_low', 'low', 'medium', 'high']
budget_labels = ['60s', '120s', '300s', '600s']

# ============================================================================
# FIGURE 1: Performance vs Budget (All Datasets)
# ============================================================================

print("\n📊 Creating Figure 1: Performance vs Budget...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('HPO Performance vs Budget Across Datasets', fontsize=16, y=0.995)

datasets = ['adult', 'fashion_mnist', 'mnist', 'letter']
dataset_titles = ['Adult', 'Fashion-MNIST', 'MNIST', 'Letter Recognition']

for idx, (dataset, title) in enumerate(zip(datasets, dataset_titles)):
    ax = axes[idx // 2, idx % 2]
    
    # Filter data for this dataset
    df_subset = df_hpo[df_hpo['dataset'] == dataset]
    
    # Calculate mean and std for each method at each budget
    for method in ['random_search', 'sha', 'hyperband']:
        means = []
        stds = []
        
        for budget in budget_order:
            data = df_subset[(df_subset['budget_level'] == budget) & 
                           (df_subset['method'] == method)]['best_score']
            means.append(data.mean())
            stds.append(data.std())
        
        # Plot with error bars
        label = method.replace('_', ' ').title()
        ax.errorbar(range(len(budget_order)), means, yerr=stds, 
                   marker='o', linewidth=2, capsize=5, label=label,
                   markersize=8)
    
    # Add baseline
    if dataset in baselines:
        ax.axhline(y=baselines[dataset], color='black', linestyle='--', 
                  linewidth=2, label='Baseline (No HPO)', alpha=0.7)
    
    ax.set_xlabel('Budget Level', fontsize=11)
    ax.set_ylabel('F1-Score (Macro)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(budget_order)))
    ax.set_xticklabels(budget_labels)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(figures_dir / 'performance_vs_budget_all_datasets.png', 
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✓ Saved: {figures_dir / 'performance_vs_budget_all_datasets.png'}")

# ============================================================================
# FIGURE 2: Improvement Over Baseline
# ============================================================================

print("\n📊 Creating Figure 2: Improvement Over Baseline...")

fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data
df_comp_pivot = df_comp.pivot_table(
    values='improvement_pct',
    index=['dataset', 'budget_level'],
    columns='method'
).reset_index()

# Create grouped bar chart
x = np.arange(len(datasets))
width = 0.07
budget_offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

for budget_idx, budget in enumerate(budget_order):
    for method_idx, method in enumerate(['random_search', 'sha', 'hyperband']):
        values = []
        for dataset in datasets:
            val = df_comp[(df_comp['dataset'] == dataset) & 
                         (df_comp['budget_level'] == budget) & 
                         (df_comp['method'] == method)]['improvement_pct'].values
            values.append(val[0] if len(val) > 0 else 0)
        
        offset = budget_offsets[budget_idx] + method_idx * 0.02
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        if budget_idx == 0:  # Only label once
            label = method.replace('_', ' ').title()
            ax.bar(x + offset, values, width, label=label, 
                  color=colors[method_idx], alpha=0.7 + budget_idx*0.075)
        else:
            ax.bar(x + offset, values, width, 
                  color=colors[method_idx], alpha=0.7 + budget_idx*0.075)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Improvement Over Baseline (%)', fontsize=12)
ax.set_title('HPO Improvement Over Baseline Across Budgets', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Adult', 'Fashion-MNIST', 'MNIST', 'Letter'], fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(figures_dir / 'improvement_over_baseline.png', 
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✓ Saved: {figures_dir / 'improvement_over_baseline.png'}")

# ============================================================================
# FIGURE 3: Budget Utilization
# ============================================================================

print("\n📊 Creating Figure 3: Budget Utilization...")

df_hpo['budget_utilization'] = (df_hpo['time_used'] / df_hpo['budget_seconds']) * 100

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Budget Utilization by Method', fontsize=16, y=1.02)

methods = ['random_search', 'sha', 'hyperband']
method_titles = ['Random Search', 'SHA', 'Hyperband']

for idx, (method, title) in enumerate(zip(methods, method_titles)):
    ax = axes[idx]
    
    df_method = df_hpo[df_hpo['method'] == method]
    
    # Box plot for each budget level
    data_to_plot = []
    for budget in budget_order:
        data = df_method[df_method['budget_level'] == budget]['budget_utilization']
        data_to_plot.append(data)
    
    bp = ax.boxplot(data_to_plot, labels=budget_labels, patch_artist=True)
    
    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, 
              label='100% Utilization', alpha=0.5)
    ax.set_xlabel('Budget Level', fontsize=11)
    ax.set_ylabel('Budget Utilization (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 150)
    if idx == 2:
        ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
fig.savefig(figures_dir / 'budget_utilization.png', 
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✓ Saved: {figures_dir / 'budget_utilization.png'}")

# ============================================================================
# FIGURE 4: Configs Evaluated
# ============================================================================

print("\n📊 Creating Figure 4: Configurations Evaluated...")

fig, ax = plt.subplots(figsize=(12, 6))

# Calculate mean configs for each method at each budget
for method in methods:
    df_method = df_hpo[df_hpo['method'] == method]
    
    means = []
    stds = []
    
    for budget in budget_order:
        data = df_method[df_method['budget_level'] == budget]['configs_evaluated']
        means.append(data.mean())
        stds.append(data.std())
    
    label = method.replace('_', ' ').title()
    ax.errorbar(range(len(budget_order)), means, yerr=stds,
               marker='o', linewidth=2.5, capsize=5, label=label,
               markersize=10)

ax.set_xlabel('Budget Level', fontsize=12)
ax.set_ylabel('Configurations Evaluated (Mean)', fontsize=12)
ax.set_title('Number of Configurations Evaluated vs Budget', 
            fontsize=14, fontweight='bold')
ax.set_xticks(range(len(budget_order)))
ax.set_xticklabels(budget_labels)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(figures_dir / 'configs_evaluated.png', 
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✓ Saved: {figures_dir / 'configs_evaluated.png'}")

# ============================================================================
# FIGURE 5: Heatmap of Performance
# ============================================================================

print("\n📊 Creating Figure 5: Performance Heatmap...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Performance Heatmap: Dataset × Budget Level', fontsize=16, y=1.02)

for idx, (method, title) in enumerate(zip(methods, method_titles)):
    ax = axes[idx]
    
    # Create pivot table
    pivot_data = df_hpo[df_hpo['method'] == method].pivot_table(
        values='best_score',
        index='dataset',
        columns='budget_level',
        aggfunc='mean'
    )
    
    # Reorder columns
    pivot_data = pivot_data[budget_order]
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlGnBu',
               cbar_kws={'label': 'F1-Score'},
               ax=ax, vmin=0.75, vmax=0.98)
    
    ax.set_xlabel('Budget Level', fontsize=11)
    ax.set_ylabel('Dataset', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticklabels(budget_labels, rotation=0)
    ax.set_yticklabels(['Adult', 'Fashion-MNIST', 'MNIST', 'Letter'], 
                       rotation=0)

plt.tight_layout()
fig.savefig(figures_dir / 'performance_heatmap.png', 
           bbox_inches='tight', dpi=300)
plt.close()
print(f"✓ Saved: {figures_dir / 'performance_heatmap.png'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ ALL VISUALIZATIONS CREATED")
print("=" * 80)
print(f"\nSaved to: {figures_dir}/")
print("\nGenerated figures:")
print("  1. performance_vs_budget_all_datasets.png - Main results figure")
print("  2. improvement_over_baseline.png - Shows HPO gains")
print("  3. budget_utilization.png - Shows budget usage efficiency")
print("  4. configs_evaluated.png - Shows exploration extent")
print("  5. performance_heatmap.png - Overview of all results")
print("\nThese figures are ready for your dissertation!")
print("=" * 80)
