#!/bin/bash

# Master script to run all analysis
# Run this from the project root directory

echo "================================================================================"
echo "BUDGET-AWARE HPO ANALYSIS PIPELINE"
echo "================================================================================"
echo ""

echo "Step 1: Data Analysis and Summary Statistics"
echo "--------------------------------------------------------------------------------"
python analyze_results.py
echo ""

echo "Step 2: Statistical Significance Tests"
echo "--------------------------------------------------------------------------------"
python statistical_tests.py
echo ""

echo "Step 3: Create Visualizations"
echo "--------------------------------------------------------------------------------"
python create_visualizations.py
echo ""

echo "================================================================================"
echo "✅ ANALYSIS PIPELINE COMPLETE"
echo "================================================================================"
echo ""
echo "Generated outputs:"
echo "  📊 Analysis:"
echo "     - results/analysis_summary.csv"
echo "     - results/baseline_comparisons.csv"
echo ""
echo "  📈 Statistics:"
echo "     - results/statistical_tests_hpo_vs_baseline.csv"
echo "     - results/statistical_tests_pairwise.csv"
echo "     - results/statistical_tests_budget_effect.csv"
echo ""
echo "  📉 Figures:"
echo "     - figures/budget_aware/performance_vs_budget_all_datasets.png"
echo "     - figures/budget_aware/improvement_over_baseline.png"
echo "     - figures/budget_aware/budget_utilization.png"
echo "     - figures/budget_aware/configs_evaluated.png"
echo "     - figures/budget_aware/performance_heatmap.png"
echo ""
echo "Next steps:"
echo "  1. Review the generated figures and statistics"
echo "  2. Start writing the Results section"
echo "  3. Extract meta-features (if not already done)"
echo "  4. Implement meta-learning model"
echo ""
echo "================================================================================"
