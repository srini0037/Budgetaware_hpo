# Adding Covertype Dataset to Meta-Learning
## Complete Instructions

---

## 📋 Overview

You need to run 2 scripts ON YOUR COMPUTER (not Claude's) to add Covertype:

1. **add_covertype_dataset.py** - Downloads data and extracts meta-features (~5 minutes)
2. **run_covertype_budget_experiments.py** - Runs 120 HPO experiments (~2-4 hours)

Then we'll re-run the meta-learning analysis with 5 datasets.

---

## 🚀 Step-by-Step Instructions

### **Step 1: Download and Save the Scripts**

I've created these files for you:
- `add_covertype_dataset.py`  
- `run_covertype_budget_experiments.py`

Download them from Claude and save them to your project root:
```
/Users/srinivass/Budgetaware_hpo/
```

---

### **Step 2: Run Meta-Feature Extraction**

Open terminal in your project directory and run:

```bash
cd /Users/srinivass/Budgetaware_hpo
python add_covertype_dataset.py
```

**What this does:**
- Downloads Covertype from sklearn (~581,012 samples, 54 features)
- Splits into train/val/test (70/15/15)
- Extracts 34 meta-features (same as other datasets)
- Saves processed data to `data/processed/covertype/`
- Updates `data/meta_features/all_datasets_metafeatures.csv`

**Expected output:**
```
================================================================================
ADDING COVERTYPE DATASET
================================================================================

1. Downloading Covertype dataset from sklearn...
   ✓ Downloaded: 581012 samples, 54 features
   Classes: [1 2 3 4 5 6 7]

2. Creating train/val/test splits (70/15/15)...
   Train: 406708 samples (70.0%)
   Val:   87152 samples (15.0%)
   Test:  87152 samples (15.0%)
   ✓ Features standardized

3. Extracting 34 meta-features...
   Computing PCA...
   Computing timing features (tree, NN, NB)...
   ✓ Extracted 34 meta-features

4. Saving meta-features...
   ✓ Saved: data/meta_features/covertype_meta_features.csv
   ✓ Updated: data/meta_features/all_datasets_metafeatures.csv (now 5 datasets)

5. Saving processed train/val/test splits...
   ✓ Saved to: data/processed/covertype/

================================================================================
COVERTYPE DATASET READY!
================================================================================
```

**Time:** ~5 minutes

---

### **Step 3: Run Budget-Aware Experiments**

**IMPORTANT:** This will take 2-4 hours. Consider running overnight or in background.

```bash
python run_covertype_budget_experiments.py
```

**What this does:**
- Runs Random Search, SHA, Hyperband at 60s, 120s, 300s, 600s budgets
- 10 runs per combination = 120 experiments total
- Saves results to `results/hpo/covertype_budget_aware.csv`
- Appends to `results/hpo/multi_dataset_budget_aware.csv`

**Progress tracking:**
```
Running: random_search    @   60s .......... Done! [10/120]
Running: random_search    @  120s .......... Done! [20/120]
Running: random_search    @  300s .......... Done! [30/120]
...
```

**Expected output:**
```
================================================================================
COVERTYPE EXPERIMENTS COMPLETE!
================================================================================

Summary statistics:
   Mean F1-Score by (Budget, Method):
   
   Best method per budget level:
      very_low   (  60s): random_search     (0.XXXX)
      low        ( 120s): random_search     (0.XXXX)
      medium     ( 300s): sha               (0.XXXX)
      high       ( 600s): hyperband         (0.XXXX)

================================================================================
```

**Time:** ~2-4 hours (can run in background)

---

### **Step 4: Re-Run Meta-Learning Analysis**

Once experiments complete, you need to:

1. **Copy updated files back to Claude:**
   - `data/meta_features/all_datasets_metafeatures.csv` (now with 5 datasets)
   - `results/hpo/multi_dataset_budget_aware.csv` (now with 600 rows)

2. **Re-run meta-learning scripts:**

```bash
# On Claude's computer (I'll do this for you)
cd /home/claude/meta_learning
python 01_build_meta_database.py
python 02_train_meta_learner.py
python 03_benchmark_comparison.py
```

---

## 📊 Expected Final Results

After adding Covertype, you'll have:

### **5 Datasets Total:**
1. Adult (48,842 samples, 14 features, 2 classes)
2. Fashion-MNIST (70,000 samples, 784 features, 10 classes)
3. MNIST (70,000 samples, 784 features, 10 classes)
4. Letter (20,000 samples, 16 features, 26 classes)
5. **Covertype (581,012 samples, 54 features, 7 classes)** ← NEW!

### **Meta-Learning Database:**
- **600 experimental results** (up from 480)
- **20 (dataset, budget) combinations** (up from 16)
- **5-fold LODO-CV** (up from 4-fold)

### **Expected Improvements:**
- ✅ **Lower LODO-CV variance** (more training data per fold)
- ✅ **Better generalization** (more diverse dataset types)
- ✅ **Stronger statistical claims** (larger sample size)
- ✅ **More convincing results** for dissertation

---

## ⏰ Timeline Estimation

| Task | Time | When |
|------|------|------|
| Run Step 2 (Meta-features) | 5 min | Now |
| Run Step 3 (Experiments) | 2-4 hours | Tonight/overnight |
| Upload files to Claude | 2 min | Tomorrow morning |
| Re-run meta-learning | 10 min | Tomorrow morning |
| **Total** | **~3-5 hours** | **By tomorrow** |

---

## 🎯 Why This Is Worth It

### **Before (4 datasets):**
- LODO-CV: 62.5% ± 37.5% (high variance)
- Prediction accuracy: 100% (16/16)
- Meta-learner sometimes struggles with outliers

### **After (5 datasets - Expected):**
- LODO-CV: ~65-70% ± 25-30% (lower variance)
- Prediction accuracy: 95-100% (19-20/20)
- Better generalization to diverse datasets
- Stronger dissertation claims

### **Dissertation Benefits:**
- ✅ Can claim "validated on 5 diverse datasets"
- ✅ More robust LODO-CV (5 folds vs 4)
- ✅ Larger sample size for statistical tests
- ✅ Covertype is industry-standard benchmark
- ✅ Covers broader range of problem types

---

## 🔧 Troubleshooting

### **Problem:** Sklearn can't download Covertype
**Solution:** Download manually:
1. Go to: https://archive.ics.uci.edu/ml/datasets/covertype
2. Download covtype.data.gz
3. Extract and load with pandas
4. Modify script to load from local file

### **Problem:** Experiments taking too long
**Solution:** Reduce to 5 runs instead of 10:
- Change `for run in range(10):` to `for run in range(5):`
- Will finish in 1-2 hours instead of 2-4 hours
- Still provides valid results

### **Problem:** Out of memory
**Solution:** Reduce sample sizes:
- In meta-features: change `X_sample[:10000]` to `X_sample[:5000]`
- In experiments: Reduce training sample with stratified sampling

---

## ✅ Checklist

Before running:
- [ ] Downloaded both scripts
- [ ] Saved to project root directory
- [ ] Verified you have ~10GB disk space
- [ ] Checked you have 2-4 hours for experiments

After Step 2:
- [ ] File exists: `data/meta_features/covertype_meta_features.csv`
- [ ] File updated: `data/meta_features/all_datasets_metafeatures.csv` (5 rows)
- [ ] Directory exists: `data/processed/covertype/` with 6 .npy files

After Step 3:
- [ ] File exists: `results/hpo/covertype_budget_aware.csv` (120 rows)
- [ ] File updated: `results/hpo/multi_dataset_budget_aware.csv` (600 rows)
- [ ] Summary shows results for all budget levels

---

## 📧 Next Steps

Once you've completed Steps 2-3 and the files are ready:

1. **Message me:** "Covertype experiments complete, ready to re-run meta-learning"
2. **Share updated files** with Claude
3. **I'll re-run** all meta-learning analysis with 5 datasets
4. **New results** will show improved performance and stronger validation

---

## 🎓 Remember

You have **7 days until submission**. Adding Covertype will take ~4-5 hours but will significantly strengthen your dissertation. The improved LODO-CV and broader dataset coverage are worth the extra time!

**Let me know when you're ready to proceed!**
