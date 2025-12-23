"""
Quick script to measure actual MLP training time on 50K Covertype samples.
This will help you set realistic budget levels.

Run this first to understand your computational constraints!
"""

import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

print("=" * 70)
print("MEASURING MLP TRAINING TIME ON 50K COVERTYPE")
print("=" * 70)

# Load Covertype (exactly as you do)
print("\n1. Loading Covertype dataset...")
X, y = fetch_openml(
    name="covertype",
    version=2,
    as_frame=False,
    return_X_y=True
)

# Convert sparse matrix to dense if needed
if hasattr(X, 'toarray'):
    print("   Converting sparse matrix to dense...")
    X = X.toarray()

y = y.astype(int)

# Subsample to 50K (exactly as you do)
MAX_SAMPLES = 50000
if X.shape[0] > MAX_SAMPLES:
    X, y = resample(
        X, y,
        n_samples=MAX_SAMPLES,
        stratify=y,
        random_state=42
    )

print(f"   Dataset shape: {X.shape}")
print(f"   Number of classes: {len(np.unique(y))}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train size: {X_train.shape[0]}")
print(f"   Test size: {X_test.shape[0]}")

# Test different MLP configurations
print("\n2. Testing different MLP configurations...")
print("=" * 70)

configs_to_test = [
    {
        'name': 'Small (1 layer, 50 units)',
        'config': {
            'hidden_layer_sizes': (50,),
            'max_iter': 100,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'random_state': 42
        }
    },
    {
        'name': 'Medium (2 layers, 100 units)',
        'config': {
            'hidden_layer_sizes': (100, 100),
            'max_iter': 100,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'random_state': 42
        }
    },
    {
        'name': 'Large (3 layers, 200 units)',
        'config': {
            'hidden_layer_sizes': (200, 200, 200),
            'max_iter': 100,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'random_state': 42
        }
    },
]

timings = []

for i, cfg in enumerate(configs_to_test, 1):
    print(f"\nConfig {i}/3: {cfg['name']}")
    print("-" * 70)
    
    # Create model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(**cfg['config']))
    ])
    
    # Time the training
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Evaluate
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    timings.append({
        'name': cfg['name'],
        'time': train_time,
        'f1_score': f1
    })
    
    print(f"   Training time: {train_time:.2f}s")
    print(f"   F1-macro score: {f1:.4f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY OF TIMINGS")
print("=" * 70)

avg_time = np.mean([t['time'] for t in timings])
min_time = min([t['time'] for t in timings])
max_time = max([t['time'] for t in timings])

print(f"\nAverage training time: {avg_time:.2f}s")
print(f"Min time: {min_time:.2f}s")
print(f"Max time: {max_time:.2f}s")

# Recommend budget levels
print("\n" + "=" * 70)
print("RECOMMENDED BUDGET LEVELS")
print("=" * 70)

# Use average time for estimation
budget_levels = {
    'very_low': 60,
    'low': 120,
    'medium': 300,
    'high': 600,
    'very_high': 1200,
}

print(f"\nBased on avg time of {avg_time:.1f}s per config:\n")
print(f"{'Budget Level':<12} | {'Time':<12} | {'Est. Configs':<15} | {'Use Case'}")
print("-" * 75)

for level, seconds in budget_levels.items():
    configs = int(seconds / avg_time)
    minutes = seconds / 60
    
    use_case = {
        'very_low': 'Quick test',
        'low': 'Fast iteration',
        'medium': 'Standard run',
        'high': 'Thorough search',
        'very_high': 'Comprehensive'
    }[level]
    
    print(f"{level:<12} | {seconds:4}s ({minutes:4.1f}m) | ~{configs:3} configs     | {use_case}")

print("\n" + "=" * 70)
print("SAVE THESE NUMBERS FOR YOUR CONFIG!")
print("=" * 70)
print(f"""
Suggested config.json addition:

{{
  "budget_levels": {{
    "very_low": 60,
    "low": 120,
    "medium": 300,
    "high": 600,
    "very_high": 1200
  }},
  "avg_config_time": {avg_time:.1f}
}}
""")

# Cross-validation timing (more expensive)
print("\n" + "=" * 70)
print("CROSS-VALIDATION TIMING (3-fold)")
print("=" * 70)

from sklearn.model_selection import cross_val_score

# Test medium config with CV
cfg = configs_to_test[1]['config']
model = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(**cfg))
])

print("\nRunning 3-fold CV (this will take ~3x longer)...")
start = time.time()
scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_macro')
cv_time = time.time() - start

print(f"   CV time: {cv_time:.2f}s")
print(f"   Mean F1: {scores.mean():.4f} (±{scores.std():.4f})")
print(f"   Time per fold: {cv_time/3:.2f}s")

print("\n" + "=" * 70)
print("For HPO with 3-fold CV, multiply budget estimates by ~3")
print("=" * 70)
