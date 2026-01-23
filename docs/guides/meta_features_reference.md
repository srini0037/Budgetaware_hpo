# Meta-Features for Budget-Aware HPO Research
## Comprehensive Literature Review

Based on AutoML Book Chapters 2, 6, 10, Sanders 2017, Cost-Sensitive Meta-Learning 2015, and related papers.

---

## 1. SIMPLE META-FEATURES (Basic Dataset Properties)

### Core Dimensional Features
| Feature | Formula/Description | Rationale | Priority for Budget-Aware HPO |
|---------|-------------------|-----------|------------------------------|
| **Nr instances (n)** | Total number of samples | Speed, Scalability | ⭐⭐⭐ CRITICAL - directly impacts training time |
| **Nr features (p)** | Total number of attributes | Curse of dimensionality | ⭐⭐⭐ CRITICAL - affects computational cost |
| **Nr classes (c)** | Number of target classes | Complexity, imbalance | ⭐⭐ HIGH - impacts optimization difficulty |
| **Dimensionality** | p/n ratio | Data sparseness | ⭐⭐⭐ CRITICAL - high p/n = expensive |
| **Nr missing values (m)** | Count or % missing | Imputation effects | ⭐ MEDIUM |
| **Nr outliers (o)** | Count of outliers | Data noisiness | ⭐ MEDIUM |

### Variants to Consider
- log(n), log(p), log(n/p) - Better for meta-learning as they normalize scale
- % categorical features - Important for model selection
- Ratio min/max class - Class imbalance measure

**Key Insight for Budget-Aware HPO:** Simple features are computationally cheap to calculate and highly predictive of computational requirements. n and p are the primary drivers of training time.

---

## 2. STATISTICAL META-FEATURES

### Distribution Properties
| Feature | Formula | Rationale | Priority |
|---------|---------|-----------|----------|
| **Skewness** | E(X−μX)³/σ³X | Feature normality | ⭐⭐ HIGH |
| **Kurtosis** | E(X−μX)⁴/σ⁴X | Feature normality | ⭐⭐ HIGH |
| **Correlation** | ρX₁X₂ | Feature interdependence | ⭐⭐ HIGH |
| **Covariance** | covX₁X₂ | Feature interdependence | ⭐ MEDIUM |
| **Coefficient of variation** | σY/μY | Variation in target | ⭐ MEDIUM |

### Aggregation Statistics
For each statistical measure, compute: **min, max, mean, standard deviation, q1, q3**

**Key Insight:** Statistical features help predict which HPO methods will converge faster. High feature correlation suggests some methods may be more efficient.

---

## 3. INFORMATION-THEORETIC META-FEATURES

### Entropy-Based Measures
| Feature | Formula | Rationale | Priority |
|---------|---------|-----------|----------|
| **Class entropy** | H(C) | Class imbalance | ⭐⭐⭐ CRITICAL |
| **Normalized entropy** | H(X)/log₂n | Feature informativeness | ⭐⭐ HIGH |
| **Mutual information** | MI(C,X) | Feature importance | ⭐⭐ HIGH |
| **Uncertainty coefficient** | MI(C,X)/H(C) | Feature importance | ⭐⭐ HIGH |
| **Noise-signal ratio** | [H(X)−MI(C,X)]/MI(C,X) | Noisiness of data | ⭐⭐ HIGH |
| **Equiv. nr. features** | H(C)/MI(C,X) | Intrinsic dimensionality | ⭐⭐ HIGH |
| **Conditional entropy** | H(Y|X) | Relationship between attributes and class | ⭐⭐ HIGH |

**Key Insight:** Information-theoretic features predict optimization difficulty. High entropy datasets typically require more HPO budget.

---

## 4. COMPLEXITY META-FEATURES

### Geometric & Topological Properties
| Feature | Description | Rationale | Priority |
|---------|-------------|-----------|----------|
| **Fisher's discriminant** | (μc₁−μc₂)²/(σ²c₁−σ²c₂) | Separability of classes | ⭐⭐ HIGH |
| **Volume of overlap** | - | Class distribution overlap | ⭐⭐ HIGH |
| **Concept variation** | - | Task complexity | ⭐ MEDIUM |
| **Data consistency** | - | Data quality | ⭐ MEDIUM |

**Key Insight:** Complexity features predict whether simple or complex models are needed, influencing HPO method choice.

---

## 5. PCA-BASED META-FEATURES

| Feature | Formula | Rationale | Priority |
|---------|---------|-----------|----------|
| **PCA variance ratio** | λ₁/Σλᵢ | Variance in first PC | ⭐⭐ HIGH |
| **PCA 95%** | dim₉₅%var/p | Intrinsic dimensionality | ⭐⭐⭐ CRITICAL |
| **PCA skewness** | Skewness of first PC | Distribution properties | ⭐ MEDIUM |
| **PCA kurtosis** | Kurtosis of first PC | Distribution properties | ⭐ MEDIUM |

**Key Insight:** PCA features reveal true data complexity. Low intrinsic dimensionality suggests faster training despite high p.

---

## 6. MODEL-BASED META-FEATURES (Decision Tree Properties)

| Feature | Description | Rationale | Priority |
|---------|-------------|-----------|----------|
| **Nr nodes, leaves** | |η|, |ψ| | Concept complexity | ⭐⭐ HIGH |
| **Tree depth** | Maximum depth | Concept complexity | ⭐⭐ HIGH |
| **Branch length** | Average branch length | Concept complexity | ⭐ MEDIUM |
| **Nodes per feature** | |ηX| | Feature importance | ⭐ MEDIUM |
| **Leaves per class** | |ψc|/|ψ| | Class complexity | ⭐ MEDIUM |
| **Leaves agreement** | nψᵢ/n | Class separability | ⭐ MEDIUM |
| **Information gain** | - | Feature importance | ⭐ MEDIUM |

**Key Insight:** Quick to compute, provides insights into problem structure that predict HPO difficulty.

---

## 7. LANDMARKING META-FEATURES

### Standard Landmarkers
| Feature | Description | Rationale | Priority for Budget-Aware |
|---------|-------------|-----------|--------------------------|
| **Landmark 1NN** | Accuracy of 1-nearest neighbor | Data sparsity | ⭐⭐⭐ CRITICAL* |
| **Landmark Decision Tree** | Accuracy of decision tree | Data separability | ⭐⭐⭐ CRITICAL* |
| **Landmark Naive Bayes** | Accuracy of Naive Bayes | Feature independence | ⭐⭐ HIGH* |
| **Landmark Linear** | Accuracy of linear model | Linear separability | ⭐⭐ HIGH* |
| **Landmark Stump** | Accuracy of decision stump | Basic separability | ⭐⭐ HIGH* |

### Advanced Landmarking Variants
- **Elite 1NN** - Enhanced version
- **Random Tree** - Faster to compute
- **Relative Landmarking** - Pairwise performance comparisons (Pa,j − Pb,j)
- **Subsample Landmarking** - Run on data subset for speed

**CRITICAL NOTE:** Auto-sklearn explicitly excluded landmarking features because they were "computationally too expensive to be helpful in the online evaluation phase." However, for budget-aware HPO research:
- **Option 1 (Recommended):** Exclude landmarking to keep meta-feature computation cheap
- **Option 2:** Use subsample landmarking on small data subset
- **Option 3:** Track landmarking time as a meta-feature itself

*Priority marked with asterisk reflects computational cost trade-off

---

## 8. COMPUTATIONAL META-FEATURES (Timing Information)

From Sanders 2017 thesis - these are UNIQUE and highly relevant:

| Feature | Description | Priority for Budget-Aware HPO |
|---------|-------------|------------------------------|
| **tree_time** | Time to build Decision Tree | ⭐⭐⭐ CRITICAL - novel meta-feature |
| **naive_bayes_time** | Time to run Naive Bayes | ⭐⭐⭐ CRITICAL |
| **lda_time** | Time to run LDA | ⭐⭐⭐ CRITICAL |
| **stump_time** | Time to run decision stump | ⭐⭐⭐ CRITICAL |
| **nn_time** | Time to run k-nearest neighbor | ⭐⭐⭐ CRITICAL |
| **simple_time** | Time to calculate simple meta-features | ⭐⭐ HIGH |
| **statistical_time** | Time to calculate statistical meta-features | ⭐⭐ HIGH |
| **inftheo_time** | Time to calculate info-theoretic meta-features | ⭐⭐ HIGH |
| **total_time** | Total meta-feature computation time | ⭐⭐⭐ CRITICAL |

**KEY INNOVATION FOR YOUR RESEARCH:** Including timing meta-features directly addresses the budget-aware aspect! This is a distinguishing element.

---

## RECOMMENDED META-FEATURE SET FOR YOUR DISSERTATION

### Tier 1: MUST INCLUDE (Computationally Cheap + Highly Predictive)
1. **n** (number of instances) - Direct budget impact
2. **p** (number of features) - Direct budget impact  
3. **c** (number of classes) - Complexity indicator
4. **Dimensionality** (p/n) - Computational cost predictor
5. **Class entropy** - Imbalance/difficulty measure
6. **PCA 95%** - Intrinsic dimensionality
7. **Class probability** (min, max, mean) - Imbalance measures

### Tier 2: STRONGLY RECOMMENDED (Good predictive power)
8. **Skewness** (min, max, mean) - Distribution properties
9. **Kurtosis** (min, max, mean) - Distribution properties
10. **Correlation** (min, max, mean) - Feature redundancy
11. **Mutual information** (min, max, mean) - Feature importance
12. **Noise-signal ratio** - Data quality indicator
13. **Tree depth** - Quick complexity measure
14. **Nr nodes** - Concept complexity

### Tier 3: INNOVATIVE ADDITIONS (Novel for Budget-Aware HPO)
15. **simple_time** - Time to compute basic meta-features
16. **tree_time** - Time to build simple decision tree
17. **nn_time** - Time to run 1-NN on small sample

### Total: ~17-20 meta-features with variants

This gives you diversity across:
- ✅ Size dimensions (n, p, c)
- ✅ Statistical properties (skewness, kurtosis, correlation)
- ✅ Information theory (entropy, MI, noise-signal)
- ✅ Geometric complexity (PCA, tree properties)
- ✅ **NOVEL: Computational cost indicators (timing features)**

---

## META-FEATURE COMPUTATION LIBRARIES

### Python Implementation Options:
1. **PyMFE** (Python Meta-Feature Extractor) - Most comprehensive
2. **sklearn** - For statistical and PCA features
3. **scipy.stats** - For distribution measures
4. **OpenML** - Pre-computed for many datasets

### Example Code Structure:
```python
from pymfe.mfe import MFE
import pandas as pd
import time

def extract_meta_features(X, y, include_timing=True):
    """Extract comprehensive meta-features including timing info"""
    
    # Basic meta-features
    mfe = MFE(groups=["general", "statistical", "info-theory", 
                     "model-based", "complexity"])
    
    # Extract without landmarking (too expensive)
    mfe.fit(X, y)
    ft_names, ft_values = mfe.extract()
    
    meta_features = dict(zip(ft_names, ft_values))
    
    if include_timing:
        # Add timing meta-features
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        
        start = time.time()
        DecisionTreeClassifier(max_depth=3).fit(X[:1000], y[:1000])
        meta_features['tree_time'] = time.time() - start
        
        start = time.time()
        KNeighborsClassifier(n_neighbors=1).fit(X[:1000], y[:1000])
        meta_features['nn_time'] = time.time() - start
    
    return meta_features
```

---

## NEXT STEP: DATASET SELECTION STRATEGY

Based on Harry's guidance: "Choose your meta-features first based on the literature and then select your datasets accordingly."

You should now select 4-5 datasets that provide **DIVERSE COVERAGE** across these meta-features:

### Diversity Criteria:
1. **Size diversity:** Small (n<10K), Medium (10K-100K), Large (>100K)
2. **Dimensionality diversity:** Low (p<10), Medium (10-100), High (>100)
3. **Class balance diversity:** Balanced, Moderately imbalanced, Highly imbalanced
4. **Complexity diversity:** Linearly separable, Moderately complex, Highly complex
5. **Domain diversity:** Different application domains

### Coverage Matrix Approach:
Create a table showing where each potential dataset falls on key dimensions:
- n: Small / Medium / Large
- p: Low / Medium / High  
- p/n: <0.01 / 0.01-1 / >1
- Class balance: Balanced / Imbalanced
- Intrinsic complexity: Simple / Medium / Complex

Select datasets to **maximize coverage** across these dimensions.

---

## REFERENCES

- AutoML Book Chapter 2: Meta-Learning (Vanschoren)
- AutoML Book Chapter 6: Auto-sklearn (Feurer et al.)
- AutoML Book Chapter 10: AutoML Challenge Analysis
- Sanders (2017): Informing HPO Through Meta-Learning
- Cost-Sensitive Meta-Learning (2015): Dataset Characterization
- OpenML meta-feature documentation

---

**Document prepared for:** Srinivas - Budget-Aware HPO Dissertation
**Date:** January 6, 2026
**Next Action:** Dataset selection with diversity matrix
