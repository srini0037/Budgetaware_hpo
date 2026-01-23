#!/bin/bash
# =============================================================================
# Codebase Reorganisation Script for Budget-Aware HPO Project
# =============================================================================
# This script reorganises the project into a clean, maintainable structure.
# 
# IMPORTANT: Review this script before running. Run with:
#   chmod +x reorganise_codebase.sh
#   ./reorganise_codebase.sh
#
# To do a dry run first (see what would happen without making changes):
#   DRY_RUN=1 ./reorganise_codebase.sh
# =============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if dry run
DRY_RUN=${DRY_RUN:-0}

log_action() {
    if [ "$DRY_RUN" = "1" ]; then
        echo -e "${YELLOW}[DRY RUN]${NC} $1"
    else
        echo -e "${GREEN}[ACTION]${NC} $1"
    fi
}

execute() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "  Would run: $@"
    else
        "$@"
    fi
}

echo "=========================================="
echo "Budget-Aware HPO Codebase Reorganisation"
echo "=========================================="
if [ "$DRY_RUN" = "1" ]; then
    echo -e "${YELLOW}Running in DRY RUN mode - no changes will be made${NC}"
fi
echo ""

# =============================================================================
# STEP 1: Create new directory structure
# =============================================================================
echo "Step 1: Creating directory structure..."

log_action "Creating src/ directories"
execute mkdir -p src/hpo
execute mkdir -p src/meta_learning
execute mkdir -p src/utils

log_action "Creating scripts/ directories"
execute mkdir -p scripts/data_preparation
execute mkdir -p scripts/experiments
execute mkdir -p scripts/meta_features
execute mkdir -p scripts/meta_learning
execute mkdir -p scripts/analysis

log_action "Creating notebooks/ directories"
execute mkdir -p notebooks/baseline
execute mkdir -p notebooks/hpo
execute mkdir -p notebooks/exploratory

log_action "Creating docs/ subdirectories"
execute mkdir -p docs/progress
execute mkdir -p docs/guides

log_action "Creating archive/ directories"
execute mkdir -p archive/old_experiments
execute mkdir -p archive/old_docs

echo ""

# =============================================================================
# STEP 2: Move HPO core implementations to src/hpo/
# =============================================================================
echo "Step 2: Moving core HPO code to src/hpo/..."

log_action "Moving hyperband_implementation.py"
execute cp hpo/hyperband_implementation.py src/hpo/hyperband.py

log_action "Moving budget_aware_experiments.py"
execute cp hpo/budget_aware_experiments.py src/hpo/budget_aware_experiments.py

# Create __init__.py files
log_action "Creating __init__.py files"
execute touch src/__init__.py
execute touch src/hpo/__init__.py
execute touch src/meta_learning/__init__.py
execute touch src/utils/__init__.py

echo ""

# =============================================================================
# STEP 3: Move notebooks to notebooks/
# =============================================================================
echo "Step 3: Moving notebooks..."

log_action "Moving baseline notebooks"
execute mv baseline/*.ipynb notebooks/baseline/ 2>/dev/null || true

log_action "Moving hpo notebooks"
execute mv hpo/*.ipynb notebooks/hpo/ 2>/dev/null || true

log_action "Moving hypothesis_testing.ipynb"
execute mv hypothesis_testing.ipynb notebooks/exploratory/ 2>/dev/null || true

log_action "Moving figure notebook"
execute mv figures/lr_creditg_plot.ipynb notebooks/exploratory/ 2>/dev/null || true

echo ""

# =============================================================================
# STEP 4: Move scripts to scripts/ subdirectories
# =============================================================================
echo "Step 4: Organising scripts..."

# Data preparation scripts
log_action "Moving data preparation scripts"
execute mv download_datasets.py scripts/data_preparation/ 2>/dev/null || true
execute mv download_additional_datasets.py scripts/data_preparation/ 2>/dev/null || true
execute mv preprocess_datasets.py scripts/data_preparation/ 2>/dev/null || true
execute mv preprocess_additional_datasets.py scripts/data_preparation/ 2>/dev/null || true
execute mv add_covertype_dataset.py scripts/data_preparation/ 2>/dev/null || true

# Experiment scripts
log_action "Moving experiment scripts"
execute mv run_covertype_budget_experiments.py scripts/experiments/ 2>/dev/null || true
execute mv run_multi_dataset_budget_aware.py scripts/experiments/ 2>/dev/null || true
execute mv run_new_datasets_budget.py scripts/experiments/ 2>/dev/null || true
execute mv budget_aware_experiment.py scripts/experiments/ 2>/dev/null || true
execute mv generate_dataset_experiments.py scripts/experiments/ 2>/dev/null || true
execute mv run_all_baselines.py scripts/experiments/ 2>/dev/null || true
execute mv statistical_validation_experiment.py scripts/experiments/ 2>/dev/null || true
execute mv measure_training_time.py scripts/experiments/ 2>/dev/null || true

# Meta-feature extraction scripts
log_action "Moving meta-feature scripts"
execute mv extract_meta_features.py scripts/meta_features/ 2>/dev/null || true
execute mv extract_meta_features_10_dataset.py scripts/meta_features/ 2>/dev/null || true
execute mv extract_covertype_metafeatures.py scripts/meta_features/ 2>/dev/null || true
execute mv combine_all_metafeatures.py scripts/meta_features/ 2>/dev/null || true

# Meta-learning scripts
log_action "Moving meta-learning scripts"
execute mv build_meta_learning_dataset.py scripts/meta_learning/ 2>/dev/null || true
execute mv train_meta_learner.py scripts/meta_learning/ 2>/dev/null || true
execute mv train_meta_learner_final.py scripts/meta_learning/ 2>/dev/null || true
execute mv validate_meta_learner.py scripts/meta_learning/ 2>/dev/null || true
execute mv feature_selection_retrain.py scripts/meta_learning/ 2>/dev/null || true

# Analysis scripts
log_action "Moving analysis scripts"
execute mv analyze_results.py scripts/analysis/ 2>/dev/null || true
execute mv analyze_budget_results.py scripts/analysis/ 2>/dev/null || true
execute mv statistical_analysis.py scripts/analysis/ 2>/dev/null || true
execute mv statistical_tests.py scripts/analysis/ 2>/dev/null || true
execute mv create_visualizations.py scripts/analysis/ 2>/dev/null || true
execute mv comprehensive_10_dataset_analysis.py scripts/analysis/ 2>/dev/null || true
execute mv feature_ranking_analysis.py scripts/analysis/ 2>/dev/null || true
execute mv merge_all_results.py scripts/analysis/ 2>/dev/null || true

echo ""

# =============================================================================
# STEP 5: Organise documentation
# =============================================================================
echo "Step 5: Organising documentation..."

log_action "Moving progress docs"
execute mv docs/progress_dec23.md docs/progress/ 2>/dev/null || true
execute mv docs/4_week_timeline_jan18.md docs/progress/ 2>/dev/null || true
execute mv docs/current_status_and_next_steps.md docs/progress/ 2>/dev/null || true

log_action "Moving guide docs"
execute mv docs/integration_guide.md docs/guides/ 2>/dev/null || true
execute mv docs/hyperband_results_comparison.md docs/guides/ 2>/dev/null || true
execute mv docs/overnight_instructions.md docs/guides/ 2>/dev/null || true

log_action "Moving reference docs from root"
execute mv meta_features_reference.md docs/guides/ 2>/dev/null || true
execute mv dataset_selection_strategy.md docs/guides/ 2>/dev/null || true

echo ""

# =============================================================================
# STEP 6: Archive old/duplicate files
# =============================================================================
echo "Step 6: Archiving old/duplicate files..."

log_action "Archiving duplicate experiment file"
execute mv "run_multi_dataset_experiments copy.py" archive/old_experiments/ 2>/dev/null || true

log_action "Archiving fixed version (keep original)"
execute mv budget_aware_experiment_fixed.py archive/old_experiments/ 2>/dev/null || true

log_action "Archiving outdated documentation"
execute mv URGENT_FIX_APPLIED.md archive/old_docs/ 2>/dev/null || true
execute mv ANALYSIS_QUICK_START.md archive/old_docs/ 2>/dev/null || true
execute mv COVERTYPE_INSTRUCTIONS.md archive/old_docs/ 2>/dev/null || true
execute mv EXPERIMENTAL_PIPELINE_STATUS.md archive/old_docs/ 2>/dev/null || true

log_action "Archiving experiment log"
execute mv experiment_log_5new.txt archive/old_docs/ 2>/dev/null || true

echo ""

# =============================================================================
# STEP 7: Move generated experiments to scripts
# =============================================================================
echo "Step 7: Moving generated experiments..."

log_action "Moving generated experiment files"
execute mv experiments_generated/*.py scripts/experiments/ 2>/dev/null || true
execute rmdir experiments_generated 2>/dev/null || true

echo ""

# =============================================================================
# STEP 8: Clean up empty directories
# =============================================================================
echo "Step 8: Cleaning up..."

log_action "Removing empty baseline directory"
execute rmdir baseline 2>/dev/null || true

log_action "Removing empty budget_experiment directory"  
execute rmdir budget_experiment 2>/dev/null || true

# Keep hpo/ but remove __pycache__
log_action "Cleaning pycache"
execute rm -rf hpo/__pycache__ 2>/dev/null || true

echo ""

# =============================================================================
# STEP 9: Move shell script
# =============================================================================
echo "Step 9: Moving utility scripts..."

log_action "Moving shell script to scripts/"
execute mv run_analysis_pipeline.sh scripts/ 2>/dev/null || true

echo ""

# =============================================================================
# DONE
# =============================================================================
echo "=========================================="
echo -e "${GREEN}Reorganisation complete!${NC}"
echo "=========================================="
echo ""
echo "New structure:"
echo "  src/           - Core reusable code (HPO implementations, utils)"
echo "  scripts/       - Executable scripts organised by purpose"
echo "  notebooks/     - All Jupyter notebooks"
echo "  docs/          - Documentation (progress notes, guides)"
echo "  archive/       - Old/deprecated files"
echo "  data/          - Data files (unchanged)"
echo "  results/       - Results files (unchanged)"
echo "  figures/       - Figures (unchanged)"
echo ""
echo "Files kept in root:"
echo "  README.md, requirements.txt, config.json, .gitignore"
echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo -e "${YELLOW}This was a DRY RUN. Run without DRY_RUN=1 to apply changes.${NC}"
fi
