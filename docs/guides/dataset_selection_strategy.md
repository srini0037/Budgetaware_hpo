# Dataset Selection Strategy for Budget-Aware HPO
## Based on Meta-Feature Diversity Requirements

**Goal:** Select 4-5 datasets that maximize diversity across meta-feature space to enable effective meta-learning

---

## DATASET SELECTION FRAMEWORK

### Step 1: Define Diversity Dimensions

Based on the meta-features literature, we need diversity across:

| Dimension | Low | Medium | High | Rationale |
|-----------|-----|--------|------|-----------|
| **Size (n)** | <10K | 10K-100K | >100K | Budget impact |
| **Features (p)** | <20 | 20-100 | >100 | Computational cost |
| **Dimensionality (p/n)** | <0.01 | 0.01-0.1 | >0.1 | Curse of dimensionality |
| **Classes (c)** | 2 | 3-10 | >10 | Complexity |
| **Class Balance** | Balanced | Moderate | Severe | Optimization difficulty |
| **Intrinsic Dim** | Low | Medium | High | True complexity |

### Step 2: Coverage Matrix

Each dataset should occupy a different "cell" in this multidimensional space.

---

## RECOMMENDED DATASETS

### Dataset 1: **Covertype** (ALREADY VALIDATED ✓)
**OpenML ID:** 1596 | **UCI:** Forest Covertype

**Meta-Features:**
- n = 581,012 (LARGE)
- p = 54 (MEDIUM)
- c = 7 (MEDIUM-HIGH)
- p/n = 0.00009 (LOW dimensionality)
- Class balance: Imbalanced (your results show this)
- Domain: Environmental/Geospatial

**Why it's good:**
- ✅ Large-scale dataset tests budget constraints well
- ✅ Already have comprehensive baseline results
- ✅ Imbalanced classes add complexity
- ✅ Real-world application domain

**Your Status:** ✓ Complete - Keep as anchor dataset

---

### Dataset 2: **MNIST/Fashion-MNIST** 
**OpenML ID:** 554 (MNIST) or 40996 (Fashion-MNIST)

**Meta-Features:**
- n = 70,000 (LARGE)
- p = 784 (HIGH)
- c = 10 (HIGH)
- p/n = 0.011 (MEDIUM dimensionality) 
- Class balance: Balanced
- Domain: Computer Vision

**Why it's good:**
- ✅ HIGH feature dimensionality (opposite of Covertype)
- ✅ Balanced classes (contrast to Covertype imbalance)
- ✅ Standard benchmark - well understood
- ✅ Neural network-friendly (MLP performs well)
- ✅ Tests curse of dimensionality effects on HPO

**Recommendation:** Choose Fashion-MNIST over MNIST (slightly harder, more realistic)

---

### Dataset 3: **Adult (Census Income)**
**OpenML ID:** 1590 | **UCI:** Adult

**Meta-Features:**
- n = 48,842 (MEDIUM-LARGE)
- p = 14 (LOW-MEDIUM, 6 continuous + 8 categorical)
- c = 2 (LOW - binary)
- p/n = 0.0003 (LOW dimensionality)
- Class balance: Moderately imbalanced (75%-25%)
- Domain: Social/Economic

**Why it's good:**
- ✅ Binary classification (simpler optimization landscape)
- ✅ Mix of categorical and continuous features
- ✅ Moderate imbalance
- ✅ Medium size - different from Covertype
- ✅ Tests HPO on traditional ML problem

---

### Dataset 4: **CIFAR-10 (Flattened features OR extracted features)**
**OpenML ID:** 40927 (if using raw) or create custom feature extraction

**Option A - Raw Pixels:**
- n = 60,000 (LARGE)
- p = 3,072 (3x32x32) (VERY HIGH)
- c = 10 (HIGH)
- p/n = 0.051 (HIGH dimensionality)

**Option B - Pre-extracted features (ResNet/VGG):**
- n = 60,000 (LARGE)
- p = 512-2048 (HIGH but manageable)
- c = 10 (HIGH)
- p/n = 0.009-0.034 (MEDIUM-HIGH)

**Why it's good:**
- ✅ Extreme dimensionality challenge
- ✅ Multi-class (10 classes)
- ✅ Balanced classes
- ✅ Tests HPO under severe computational constraints
- ✅ Computer vision domain

**Recommendation:** Use Option B (pre-extracted features) to keep training tractable while maintaining high dimensionality

---

### Dataset 5: **HIGGS** (or alternative: **Electricity** or **Nomao**)
**OpenML ID:** 23512 (HIGGS) 

**Option A - HIGGS:**
- n = 11,000,000 (VERY LARGE)
- p = 28 (MEDIUM)
- c = 2 (BINARY)
- p/n = 0.0000025 (VERY LOW)
- Class balance: Relatively balanced (53%-47%)
- Domain: Physics

**Why it's good:**
- ✅ MASSIVE scale tests extreme budget constraints
- ✅ Low-dimensional but huge sample size
- ✅ Binary classification
- ✅ Tests scalability of HPO methods

**Option B - Electricity (if HIGGS too large):**
- n = 45,312 (MEDIUM)
- p = 8 (LOW)
- c = 2 (BINARY)
- Domain: Time series / Concept drift

---

## DIVERSITY COVERAGE MATRIX

| Dataset | Size (n) | Features (p) | p/n | Classes | Balance | Domain |
|---------|----------|--------------|-----|---------|---------|--------|
| **Covertype** | LARGE (580K) | MEDIUM (54) | LOW | 7 | Imbalanced | Environmental |
| **Fashion-MNIST** | LARGE (70K) | HIGH (784) | MEDIUM | 10 | Balanced | Vision |
| **Adult** | MEDIUM (48K) | LOW (14) | LOW | 2 | Moderate Imb. | Social |
| **CIFAR-10** | LARGE (60K) | VERY HIGH (512-2K) | HIGH | 10 | Balanced | Vision |
| **HIGGS** | HUGE (11M) | MEDIUM (28) | VERY LOW | 2 | Balanced | Physics |

### Coverage Analysis:
✅ **Size diversity:** Medium (48K) → Large (60-580K) → Huge (11M)
✅ **Feature diversity:** Low (14) → Medium (28-54) → High (512-784) → Very High (2K+)
✅ **Dimensionality diversity:** Very Low → Low → Medium → High
✅ **Class diversity:** Binary → 7-class → 10-class
✅ **Balance diversity:** Balanced → Moderate → Imbalanced
✅ **Domain diversity:** 5 different domains

---

## ALTERNATIVE DATASET RECOMMENDATIONS

If any of the above don't work, here are backup options:

### Backup Option 1: **Bank Marketing**
- n = 45,211, p = 16, c = 2 (binary)
- Highly imbalanced (11.7% positive class)
- Marketing domain

### Backup Option 2: **Connect-4**
- n = 67,557, p = 42, c = 3
- Game domain, balanced
- Categorical features

### Backup Option 3: **Nomao**
- n = 34,465, p = 118, c = 2
- Medium size, high dimensionality
- Web domain

### Backup Option 4: **KDD Cup 99**
- n = 494,021, p = 41, c = 23 
- Network intrusion detection
- Multi-class, imbalanced

---

## PRACTICAL CONSIDERATIONS

### Computational Budget Allocation:
With 4-5 datasets and budget constraints:

**Per-dataset budget allocation:**
- Covertype: 20% (already complete baseline)
- Fashion-MNIST: 20%
- Adult: 15%
- CIFAR-10 (features): 25% (most expensive)
- HIGGS (subsample): 20%

### Subsampling Strategy:
For very large datasets under tight deadlines:
- **HIGGS:** Sample 500K-1M instances (still large, more tractable)
- **Covertype:** Already using 50K sample (good decision)
- **Others:** Use full datasets

### Implementation Order (Recommended):
1. **Adult** (fastest to run - good quick validation)
2. **Fashion-MNIST** (standard benchmark - medium difficulty)
3. **CIFAR-10 features** (high-dim test)
4. **HIGGS subsample** (scale test)
5. **Keep Covertype as anchor**

---

## DATA ACQUISITION CHECKLIST

### ✅ Sources:
- [ ] OpenML Python API: `from sklearn.datasets import fetch_openml`
- [ ] UCI ML Repository: Direct download
- [ ] Kaggle: Some datasets available
- [ ] TensorFlow Datasets: For MNIST, Fashion-MNIST, CIFAR-10

### ✅ For Each Dataset, Extract:
- [ ] Meta-features (using your meta-feature computation script)
- [ ] Baseline results (default Random Forest / simple MLP)
- [ ] Preprocessing requirements
- [ ] Expected training time on your hardware

---

## VALIDATION OF SELECTION

### Harry's Criteria Check:
✅ "Choose a datasets which give you a diverse range of meta-features"
- Size: 48K → 11M (diverse ✓)
- Features: 14 → 2048 (diverse ✓)
- Dimensionality: Very different ratios (diverse ✓)
- Classes: 2 → 10 (diverse ✓)
- Balance: All variations covered (diverse ✓)

✅ "Choose your meta-features first based on the literature"
- Meta-features defined from AutoML book, Sanders, etc. ✓
- Computational timing features included ✓

✅ "Select your datasets accordingly"
- Datasets selected to maximize coverage ✓
- Each occupies different region of meta-feature space ✓

---

## NEXT ACTIONS

### Immediate (This Week):
1. ✅ Meta-features defined (COMPLETE)
2. ⏳ Confirm dataset selection with Harry (send list + rationale)
3. ⏳ Download and preprocess all datasets
4. ⏳ Compute meta-features for all datasets
5. ⏳ Create diversity visualization (scatter plots showing coverage)

### Following Week:
6. Run baseline experiments (Random Search, Hyperband, SHA) on each dataset
7. Compare with Covertype patterns - do crossing points emerge?
8. Document computational budgets used per dataset

---

## RECOMMENDED: DIVERSITY VISUALIZATION

Create this visualization to show Harry:

```python
import matplotlib.pyplot as plt
import numpy as np

datasets = {
    'Covertype': {'n': 580000, 'p': 54, 'c': 7, 'balance': 0.3},
    'Fashion-MNIST': {'n': 70000, 'p': 784, 'c': 10, 'balance': 0.9},
    'Adult': {'n': 48842, 'p': 14, 'c': 2, 'balance': 0.75},
    'CIFAR-10': {'n': 60000, 'p': 512, 'c': 10, 'balance': 0.9},
    'HIGGS': {'n': 1000000, 'p': 28, 'c': 2, 'balance': 0.9}
}

# Plot 1: Size vs Dimensionality
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Size vs Features
ax1 = axes[0, 0]
for name, props in datasets.items():
    ax1.scatter(props['n'], props['p'], s=props['c']*50, 
               alpha=0.6, label=name)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Number of Instances (n)')
ax1.set_ylabel('Number of Features (p)')
ax1.legend()
ax1.set_title('Dataset Size vs Dimensionality')

# p/n ratio vs number of classes
ax2 = axes[0, 1]
for name, props in datasets.items():
    ratio = props['p'] / props['n']
    ax2.scatter(ratio, props['c'], s=200, alpha=0.6, label=name)
ax2.set_xscale('log')
ax2.set_xlabel('Dimensionality Ratio (p/n)')
ax2.set_ylabel('Number of Classes')
ax2.legend()
ax2.set_title('Dimensionality vs Complexity')

plt.tight_layout()
plt.savefig('dataset_diversity_coverage.png', dpi=300)
```

This shows you're being strategic about dataset selection!

---

**Document prepared for:** Srinivas - Budget-Aware HPO Dissertation  
**Date:** January 6, 2026  
**Status:** Ready for Harry's review
**Next:** Confirm dataset selection → Begin multi-dataset experiments
