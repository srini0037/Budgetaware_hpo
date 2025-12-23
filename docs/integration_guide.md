# Integration Guide: Next Steps for Your Dissertation

## 🎯 IMMEDIATE PRIORITY (This Week)

You've done great work! Now let's add the missing pieces to address Harry's feedback.

---

## Step 1: Measure Your Actual Training Time (30 minutes)

**Why:** Need to know realistic budget levels for 50K Covertype samples

**Action:**
1. Copy `measure_training_time.py` to your repo
2. Run it:
```bash
cd /Users/srinivass/Budgetaware_hpo
python measure_training_time.py
```

**Expected output:**
```
Average training time: X.XX seconds
Recommended budget levels:
  very_low:  60s (~Y configs)
  low:      120s (~Z configs)
  ...
```

**Save these numbers!** You'll use them for all budget experiments.

---

## Step 2: Implement Hyperband (2-3 hours)

**Why:** Harry expects this as your main baseline to beat

**Action:**

### Option A: Create Standalone File
1. Copy `hyperband_implementation.py` to `/Users/srinivass/Budgetaware_hpo/hpo/`
2. Test it:
```python
# In a new notebook or script
from hpo.hyperband_implementation import Hyperband, get_random_mlp_config

# Your existing data loading code
X_train, y_train = ... 

# Run Hyperband
hb = Hyperband(
    get_random_config=get_random_mlp_config,
    max_iter=81,
    eta=3,
    verbose=True
)

result = hb.run(X_train, y_train)
print(f"Best score: {result['best_score']:.4f}")
```

### Option B: Integrate into Existing Notebook
1. Add the Hyperband class to your `mlp_hpo.ipynb`
2. Run it alongside Random Search and SHA
3. Compare all three methods

**Expected Result:**
```
Method              | Val F1  | Configs | Time
--------------------|---------|---------|------
Baseline            | 0.811   | 1       | 5s
Random Search       | 0.876   | 20      | XXs
SHA                 | 0.868   | ~30     | XXs
Hyperband (NEW)     | 0.880   | ~40     | XXs
```

**Success Criteria:**
- ✅ Hyperband ≥ Random Search
- ✅ Hyperband ≥ SHA
- ✅ All results saved to CSV

---

## Step 3: Add Time Tracking to Existing Experiments (1 hour)

**Why:** Can't measure budget efficiency without time data!

**Modify your existing mlp_hpo.ipynb:**

### Before (current code):
```python
# Random Search
for i in range(N_ITER_RANDOM):
    config = sample_config()
    score = evaluate(config)
    results.append({'score': score})
```

### After (with time tracking):
```python
import time

# Random Search with time tracking
start_experiment = time.time()
cumulative_time = 0

for i in range(N_ITER_RANDOM):
    config = sample_config()
    
    # Time this evaluation
    eval_start = time.time()
    score = evaluate(config)
    eval_time = time.time() - eval_start
    
    cumulative_time += eval_time
    
    results.append({
        'config_id': i,
        'score': score,
        'eval_time': eval_time,
        'cumulative_time': cumulative_time,
        'wall_clock_time': time.time() - start_experiment
    })
```

**Apply this to:**
- ✅ Random Search
- ✅ SHA
- ✅ Hyperband (new)

**Save enhanced results:**
```python
# Save with timing info
df = pd.DataFrame(results)
df.to_csv('results/hpo/mlp_hpo_with_timing.csv', index=False)
```

---

## Step 4: Run Budget-Constrained Experiments (2-3 hours)

**Why:** This is the core of "budget-aware" research!

**Setup:**

1. Based on Step 1, define your budget levels:
```python
# In config.json or notebook
BUDGET_LEVELS = {
    'very_low': 60,      # Adjust based on your measurements
    'low': 120,
    'medium': 300,
    'high': 600,
}
```

2. Modify your experiments to respect budgets:

```python
def run_hpo_with_budget(method, X, y, time_budget, seed=42):
    """
    Run HPO method with time budget constraint.
    Stops when budget is exhausted.
    """
    np.random.seed(seed)
    start_time = time.time()
    
    best_score = 0
    best_config = None
    configs_evaluated = 0
    
    if method == 'random_search':
        while time.time() - start_time < time_budget:
            config = sample_random_config()
            score = evaluate_config(config, X, y)
            
            if score > best_score:
                best_score = score
                best_config = config
            
            configs_evaluated += 1
            
    elif method == 'sha':
        # SHA with budget constraint
        result = sha_with_budget(X, y, time_budget)
        best_score = result['best_score']
        best_config = result['best_config']
        configs_evaluated = result['configs_evaluated']
        
    elif method == 'hyperband':
        # Hyperband with budget constraint
        result = hyperband_with_budget(X, y, time_budget)
        best_score = result['best_score']
        best_config = result['best_config']
        configs_evaluated = result['configs_evaluated']
    
    return {
        'method': method,
        'budget': time_budget,
        'best_score': best_score,
        'best_config': best_config,
        'configs_evaluated': configs_evaluated,
        'actual_time': time.time() - start_time
    }
```

3. Run experiments:

```python
results = []

for budget_name, budget_seconds in BUDGET_LEVELS.items():
    for method in ['random_search', 'sha', 'hyperband']:
        for seed in range(10):  # Multiple runs for stats
            result = run_hpo_with_budget(
                method, X_train, y_train, budget_seconds, seed
            )
            result['budget_level'] = budget_name
            result['seed'] = seed
            results.append(result)

# Save
df = pd.DataFrame(results)
df.to_csv('results/budget_aware_experiments.csv', index=False)
```

**Expected Output:**
```
budget_level | method        | best_score | configs | time
-------------|---------------|------------|---------|------
very_low     | random_search | 0.850      | 7       | 60s
very_low     | sha           | 0.845      | 12      | 60s
very_low     | hyperband     | 0.848      | 10      | 60s
low          | random_search | 0.860      | 15      | 120s
low          | sha           | 0.865      | 25      | 120s
low          | hyperband     | 0.870      | 22      | 120s
...
```

**Analysis Questions to Answer:**
1. At what budget does SHA start beating Random Search?
2. At what budget does Hyperband start beating SHA?
3. Are there diminishing returns at higher budgets?
4. Which method is most budget-efficient?

---

## Step 5: Create Performance vs Budget Curves (1 hour)

**Why:** Visual evidence of crossing points for Harry

**Code:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv('results/budget_aware_experiments.csv')

# Group by budget and method
grouped = df.groupby(['budget_level', 'method'])['best_score'].agg(['mean', 'std'])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for method in ['random_search', 'sha', 'hyperband']:
    method_data = grouped.xs(method, level='method')
    x = [BUDGET_LEVELS[bl] for bl in method_data.index]
    y = method_data['mean']
    err = method_data['std']
    
    ax.errorbar(x, y, yerr=err, marker='o', label=method, capsize=5)

ax.axhline(y=baseline_score, color='red', linestyle='--', label='Baseline')
ax.set_xlabel('Budget (seconds)')
ax.set_ylabel('Validation F1-macro Score')
ax.set_title('Performance vs Budget: HPO Methods Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('figures/performance_vs_budget.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Expected Plot:**
- X-axis: Budget (60s, 120s, 300s, 600s)
- Y-axis: F1-macro score
- Lines for each method showing performance scaling
- **Crossing points visible** where lines intersect!

---

## Step 6: Identify Crossing Points (30 minutes)

**Code to find exact crossing points:**

```python
def find_crossing_point(df, method1, method2):
    """
    Find budget level where method2 starts outperforming method1.
    """
    # Get mean scores for each method at each budget
    scores1 = df[df['method'] == method1].groupby('budget_level')['best_score'].mean()
    scores2 = df[df['method'] == method2].groupby('budget_level')['best_score'].mean()
    
    # Find first budget where method2 > method1
    for budget in scores1.index:
        if scores2[budget] > scores1[budget]:
            return budget, BUDGET_LEVELS[budget]
    
    return None, None

# Find crossing points
crossing_rs_sha = find_crossing_point(df, 'random_search', 'sha')
crossing_sha_hb = find_crossing_point(df, 'sha', 'hyperband')

print("Crossing Points:")
print(f"  SHA overtakes Random Search at: {crossing_rs_sha[0]} ({crossing_rs_sha[1]}s)")
print(f"  Hyperband overtakes SHA at: {crossing_sha_hb[0]} ({crossing_sha_hb[1]}s)")
```

**Save these numbers!** They're critical for meta-learning later.

---

## Step 7: Statistical Validation (1 hour)

**Update your hypothesis_testing.ipynb:**

```python
from scipy.stats import ttest_rel, wilcoxon

# Load budget experiment results
df = pd.read_csv('results/budget_aware_experiments.csv')

# For each budget level, compare methods
for budget_level in df['budget_level'].unique():
    subset = df[df['budget_level'] == budget_level]
    
    # Get scores for each method
    rs_scores = subset[subset['method'] == 'random_search']['best_score'].values
    sha_scores = subset[subset['method'] == 'sha']['best_score'].values
    hb_scores = subset[subset['method'] == 'hyperband']['best_score'].values
    
    # Paired t-test: SHA vs Random Search
    t_stat, p_value = ttest_rel(sha_scores, rs_scores)
    print(f"\n{budget_level.upper()}")
    print(f"  SHA vs Random Search: p={p_value:.4f} {'✓ significant' if p_value < 0.05 else '✗ not significant'}")
    
    # Paired t-test: Hyperband vs SHA
    t_stat, p_value = ttest_rel(hb_scores, sha_scores)
    print(f"  Hyperband vs SHA: p={p_value:.4f} {'✓ significant' if p_value < 0.05 else '✗ not significant'}")
```

---

## 📊 WEEK 1 DELIVERABLES

By end of Week 1, you should have:

### ✅ Code:
- `hpo/hyperband_implementation.py` - Working Hyperband
- Updated `mlp_hpo.ipynb` - With time tracking
- `experiments/budget_aware.py` - Budget-constrained framework
- Updated `hypothesis_testing.ipynb` - With budget comparisons

### ✅ Results:
- `results/hpo/hyperband_results.csv` - Hyperband performance
- `results/budget_aware_experiments.csv` - All methods, all budgets
- `results/timing_measurements.txt` - Actual training times

### ✅ Figures:
- `figures/performance_vs_budget.png` - Main comparison plot
- `figures/crossing_points.png` - Visualizing method transitions
- `figures/efficiency_comparison.png` - Improvement per second

### ✅ Documentation:
- Updated README with Phase 2 findings
- Crossing points identified and documented
- Statistical tests showing significance

---

## 🚀 INTEGRATION WITH YOUR EXISTING CODE

### File Structure After Integration:

```
Budgetaware_hpo/
├── baseline/
│   ├── mlp_baseline.ipynb           # ✅ Already done
│   ├── adult_baseline.ipynb         # ✅ Already done
│   └── lr_creditg_baseline.ipynb    # ✅ Already done
├── hpo/
│   ├── mlp_hpo.ipynb                # ✅ Update: add timing
│   ├── hyperband_implementation.py  # 🆕 NEW
│   └── budget_aware_experiments.py  # 🆕 NEW
├── experiments/                      # 🆕 NEW FOLDER
│   ├── measure_timing.py
│   └── run_budget_experiments.py
├── results/
│   ├── baselines/                   # ✅ Already done
│   ├── hpo/
│   │   ├── mlp_hpo_summary.csv      # ✅ Already done
│   │   ├── hyperband_results.csv    # 🆕 NEW
│   │   └── budget_aware_comparison.csv  # 🆕 NEW
│   └── timing_measurements.txt      # 🆕 NEW
├── figures/                          # 🆕 NEW PLOTS
│   ├── performance_vs_budget.png
│   ├── crossing_points.png
│   └── efficiency_comparison.png
├── hypothesis_testing.ipynb          # ✅ Update
├── config.json                       # ✅ Update: add budgets
└── requirements.txt                  # ✅ Check dependencies
```

---

## ⚠️ COMMON PITFALLS TO AVOID

1. **Not tracking time properly**
   - Always use `time.time()` not `time.process_time()`
   - Track both evaluation time AND cumulative budget

2. **Budget enforcement too strict**
   - Don't stop mid-evaluation
   - Allow current evaluation to complete, then check budget

3. **Ignoring variance**
   - Always run multiple seeds (10+)
   - Report mean ± std
   - Use proper statistical tests

4. **Not saving intermediate results**
   - Save after each budget level
   - Don't lose hours of computation!

---

## 🎯 SUCCESS CRITERIA

You'll know you're done with Phase 2 when you can answer:

1. ✅ "What's the training time for one MLP config on 50K Covertype?"
   - Answer: X.XX seconds

2. ✅ "Does Hyperband outperform Random Search and SHA?"
   - Answer: Yes/No, with p-values

3. ✅ "At what budget does SHA overtake Random Search?"
   - Answer: XXX seconds (crossing point 1)

4. ✅ "At what budget does Hyperband overtake SHA?"
   - Answer: XXX seconds (crossing point 2)

5. ✅ "Which method is most budget-efficient?"
   - Answer: [Method] gives X% improvement per minute

6. ✅ "How do results vary across budget levels?"
   - Answer: Performance vs budget plots showing curves

---

## 📞 CHECKING IN WITH HARRY

After completing Week 1, prepare this summary for Harry:

### "Hi Harry, here's my Week 1 progress:

**Completed:**
1. ✅ Implemented Hyperband - outperforms baseline by X%
2. ✅ Added time tracking to all experiments
3. ✅ Ran budget-constrained experiments at 4 levels
4. ✅ Identified crossing points:
   - SHA beats Random Search at: XXX seconds
   - Hyperband beats SHA at: XXX seconds
5. ✅ Statistical validation: p-values all < 0.05

**Key Findings:**
- Hyperband achieves X.XX F1-macro (vs baseline X.XX)
- Budget efficiency varies by method:
  - Very low budget: [Method] wins
  - Medium budget: [Method] wins
  - High budget: [Method] wins
- Crossing points vary as expected

**Next Steps:**
- Expand to 5+ datasets (Adult, Credit-G, MNIST, Fashion-MNIST, etc.)
- Extract meta-features
- Build meta-learning component

**Questions:**
1. [Your question about specific finding]
2. [Request for feedback on approach]"

---

## 🚀 READY TO START?

**Pick one to begin:**

1. **Start with timing measurement** (safest, easiest)
   - Run `measure_training_time.py`
   - Get concrete numbers
   - Update config with budget levels

2. **Implement Hyperband** (most important)
   - Copy code to your repo
   - Test on Covertype
   - Compare with existing results

3. **Add budget tracking** (quick win)
   - Modify existing notebooks
   - Add time tracking
   - Re-run experiments

**Which would you like to tackle first?** I can help you implement it step by step!
