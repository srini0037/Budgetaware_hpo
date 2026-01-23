# Project Structure

```
Budgetaware_hpo/
│
├── README.md                    # Project overview
├── requirements.txt             # Dependencies
├── config.json                  # Configuration
│
├── src/                         # Core reusable code
│   ├── hpo/                     # HPO implementations
│   │   ├── hyperband.py         # Hyperband algorithm
│   │   └── budget_aware_experiments.py
│   ├── meta_learning/           # Meta-learning components
│   └── utils/                   # Shared utilities
│
├── scripts/                     # Executable scripts
│   ├── data_preparation/        # Dataset download & preprocessing
│   │   ├── download_datasets.py
│   │   ├── download_additional_datasets.py
│   │   ├── preprocess_datasets.py
│   │   └── preprocess_additional_datasets.py
│   │
│   ├── experiments/             # Run experiments
│   │   ├── run_covertype_budget_experiments.py
│   │   ├── run_multi_dataset_budget_aware.py
│   │   ├── run_new_datasets_budget.py
│   │   ├── adult_experiment.py
│   │   ├── fashion_mnist_experiment.py
│   │   ├── letter_experiment.py
│   │   └── mnist_experiment.py
│   │
│   ├── meta_features/           # Extract meta-features
│   │   ├── extract_meta_features.py
│   │   ├── extract_meta_features_10_dataset.py
│   │   └── combine_all_metafeatures.py
│   │
│   ├── meta_learning/           # Train & validate meta-learner
│   │   ├── build_meta_learning_dataset.py
│   │   ├── train_meta_learner.py
│   │   ├── train_meta_learner_final.py
│   │   └── validate_meta_learner.py
│   │
│   └── analysis/                # Analyse results
│       ├── analyze_results.py
│       ├── statistical_analysis.py
│       ├── statistical_tests.py
│       ├── create_visualizations.py
│       └── comprehensive_10_dataset_analysis.py
│
├── notebooks/                   # Jupyter notebooks
│   ├── baseline/                # Baseline experiments
│   ├── hpo/                     # HPO experiments
│   └── exploratory/             # Exploratory analysis
│
├── data/                        # Data files
│   ├── raw/                     # Raw downloaded data
│   ├── processed/               # Preprocessed data
│   ├── meta_features/           # Extracted meta-features
│   └── metalearning/            # Meta-learning datasets
│
├── results/                     # Experiment results
│   ├── baselines/               # Baseline results
│   ├── hpo/                     # HPO results
│   ├── meta_learning/           # Meta-learner results
│   └── analysis_10_datasets/    # Comprehensive analysis
│
├── figures/                     # Generated figures
│   ├── budget_aware/
│   └── comprehensive_10_datasets/
│
├── docs/                        # Documentation
│   ├── progress/                # Progress notes
│   └── guides/                  # How-to guides
│
├── reports/                     # Generated reports
│
└── archive/                     # Old/deprecated files
    ├── old_experiments/
    └── old_docs/
```

## Workflow

1. **Data Preparation**: `scripts/data_preparation/`
2. **Run Experiments**: `scripts/experiments/`
3. **Extract Meta-Features**: `scripts/meta_features/`
4. **Train Meta-Learner**: `scripts/meta_learning/`
5. **Analyse Results**: `scripts/analysis/`

## Running Scripts

After reorganisation, run scripts from project root:
```bash
python scripts/experiments/run_multi_dataset_budget_aware.py
python scripts/analysis/create_visualizations.py
```
