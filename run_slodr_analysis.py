#!/usr/bin/env python
"""
SLODR (Spearman's Law of Diminishing Returns) Analysis for Epoch Capability Index

This script tests whether benchmark correlation decreases at higher ECI levels.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("SLODR ANALYSIS FOR EPOCH CAPABILITY INDEX")
print("="*70)
print()

# ============================================================================
# STEP 1: LOAD BENCHMARK DATA
# ============================================================================

def load_benchmark_data(bench_dir='benchmark_data', score_col='mean_score', model_col='Model version'):
    """Load all benchmark CSV files from directory."""
    bench_path = Path(bench_dir)
    benchmarks = {}

    csv_files = [f for f in bench_path.glob('*.csv') if 'epoch_capabilities_index' not in f.name]

    print(f"STEP 1: Loading {len(csv_files)} benchmark files...")
    print()

    for csv_file in csv_files:
        benchmark_name = csv_file.stem

        try:
            df = pd.read_csv(csv_file)

            if model_col not in df.columns:
                print(f"  ⚠ Skipping {benchmark_name}: missing '{model_col}' column")
                continue

            if score_col in df.columns:
                score_column = score_col
            elif 'Best score (across scorers)' in df.columns:
                score_column = 'Best score (across scorers)'
            elif 'score' in df.columns:
                score_column = 'score'
            else:
                print(f"  ⚠ Skipping {benchmark_name}: no score column found")
                continue

            df_clean = df[[model_col, score_column]].copy()
            df_clean.columns = ['model', 'score']
            df_clean = df_clean.dropna(subset=['score'])
            df_clean['score'] = pd.to_numeric(df_clean['score'], errors='coerce')
            df_clean = df_clean.dropna(subset=['score'])
            df_clean = df_clean.groupby('model')['score'].max().reset_index()

            if len(df_clean) > 0:
                benchmarks[benchmark_name] = df_clean
                print(f"  ✓ Loaded {benchmark_name}: {len(df_clean)} models")

        except Exception as e:
            print(f"  ✗ Error loading {benchmark_name}: {str(e)}")

    print(f"\n✓ Successfully loaded {len(benchmarks)} benchmarks\n")
    return benchmarks

benchmarks = load_benchmark_data()

# ============================================================================
# STEP 2: LOAD ECI DATA
# ============================================================================

def load_eci_data(eci_csv='benchmark_data/epoch_capabilities_index.csv',
                  model_col='Model version',
                  eci_col='ECI Score'):
    """Load the Epoch Capability Index (ECI) data."""
    print("STEP 2: Loading ECI data...")
    df = pd.read_csv(eci_csv)
    df_eci = df[[model_col, eci_col]].copy()
    df_eci.columns = ['model', 'eci']
    df_eci = df_eci.dropna(subset=['eci'])
    df_eci['eci'] = pd.to_numeric(df_eci['eci'], errors='coerce')
    df_eci = df_eci.dropna(subset=['eci'])

    print(f"✓ Loaded ECI data: {len(df_eci)} models")
    print(f"  ECI range: [{df_eci['eci'].min():.2f}, {df_eci['eci'].max():.2f}]")
    print(f"  ECI mean: {df_eci['eci'].mean():.2f}, median: {df_eci['eci'].median():.2f}\n")

    return df_eci

eci_data = load_eci_data()

# ============================================================================
# STEP 3: CREATE COMBINED DATASET
# ============================================================================

def create_wide_matrix(benchmarks, eci_data):
    """Create a wide matrix with models as rows and benchmarks as columns."""
    print("STEP 3: Creating wide matrix...")
    df_wide = eci_data.copy()
    df_wide = df_wide.set_index('model')

    for bench_name, bench_df in benchmarks.items():
        bench_df_indexed = bench_df.set_index('model')
        df_wide[bench_name] = bench_df_indexed['score']

    df_wide = df_wide.reset_index()

    benchmark_cols = [col for col in df_wide.columns if col not in ['model', 'eci']]
    coverage = df_wide[benchmark_cols].notna().sum() / len(df_wide)

    print(f"✓ Created wide matrix: {len(df_wide)} models × {len(benchmarks)} benchmarks")
    print(f"  Mean benchmark coverage: {coverage.mean():.1%}\n")

    return df_wide

df_wide = create_wide_matrix(benchmarks, eci_data)

# ============================================================================
# STEP 4: PREPROCESSING FUNCTIONS
# ============================================================================

def apply_logit_transform(scores, eps=1e-4):
    """Apply logit transformation to handle ceiling effects."""
    scores_clipped = np.clip(scores, eps, 1 - eps)
    return np.log(scores_clipped / (1 - scores_clipped))

def apply_expit_transform(logit_scores):
    """Apply inverse logit (sigmoid) transformation."""
    return 1 / (1 + np.exp(-logit_scores))

def impute_missing_knn(X, k=5):
    """Impute missing values using k-Nearest Neighbors."""
    imputer = KNNImputer(n_neighbors=k, weights='distance')
    X_imputed = imputer.fit_transform(X)
    n_missing = np.isnan(X).sum()
    print(f"  Imputed {n_missing} missing values using KNN (k={k})")
    return X_imputed

def standardize(X):
    """Z-score standardization."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    std[std == 0] = 1.0
    X_standardized = (X - mean) / std
    return X_standardized, mean, std

def preprocess_data(df_wide, use_logit=True, k=5):
    """Complete preprocessing pipeline."""
    print("STEP 4: Preprocessing data...")

    model_names = df_wide['model'].values
    eci_values = df_wide['eci'].values
    benchmark_cols = [col for col in df_wide.columns if col not in ['model', 'eci']]
    benchmark_names = benchmark_cols

    X = df_wide[benchmark_cols].values

    print(f"  Original shape: {X.shape}")
    print(f"  Missing values: {np.isnan(X).sum()} ({np.isnan(X).mean():.1%})")

    if use_logit:
        print("  Applying logit transformation...")
        X_logit = np.full_like(X, np.nan)
        mask = ~np.isnan(X)
        X_logit[mask] = apply_logit_transform(X[mask])
        X = X_logit

    print("  Performing KNN imputation...")
    X_imputed = impute_missing_knn(X, k=k)

    if use_logit:
        print("  Applying inverse logit transformation...")
        X_imputed = apply_expit_transform(X_imputed)

    print("  Standardizing...")
    X_standardized, _, _ = standardize(X_imputed)

    print(f"  Final shape: {X_standardized.shape}")
    print(f"✓ Preprocessing complete\n")

    return X_standardized, model_names, benchmark_names, eci_values

X_preprocessed, model_names, benchmark_names, eci_values = preprocess_data(df_wide, use_logit=True, k=5)

# ============================================================================
# STEP 5: PCA PER ECI BIN
# ============================================================================

def perform_pca_per_bin(X, eci_values, n_bins=8):
    """Perform PCA within each ECI bin."""
    print(f"STEP 5: Performing PCA per ECI bin ({n_bins} bins)...\n")

    sort_idx = np.argsort(eci_values)
    X_sorted = X[sort_idx]
    eci_sorted = eci_values[sort_idx]

    bin_edges = np.percentile(eci_sorted, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(eci_sorted, bin_edges[1:-1])

    results = []

    for bin_idx in range(n_bins):
        mask = (bin_indices == bin_idx)
        X_bin = X_sorted[mask]
        eci_bin = eci_sorted[mask]

        if len(X_bin) < 2:
            print(f"  Bin {bin_idx+1}: Skipped (only {len(X_bin)} models)")
            continue

        pca = PCA()
        pca.fit(X_bin)

        evr1 = pca.explained_variance_ratio_[0]
        mean_eci = np.mean(eci_bin)

        results.append({
            'bin': bin_idx + 1,
            'mean_eci': mean_eci,
            'min_eci': np.min(eci_bin),
            'max_eci': np.max(eci_bin),
            'evr1': evr1,
            'evr2': pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else np.nan,
            'n_models': len(X_bin)
        })

        print(f"  Bin {bin_idx+1}: ECI [{mean_eci:.1f}], EVR1 = {evr1:.3f}, n = {len(X_bin)}")

    df_results = pd.DataFrame(results)
    print(f"\n✓ PCA complete for {len(df_results)} bins\n")

    return df_results

pca_results = perform_pca_per_bin(X_preprocessed, eci_values, n_bins=8)

# ============================================================================
# STEP 6: LINEAR REGRESSION ANALYSIS
# ============================================================================

def analyze_slodr_hypothesis(pca_results):
    """Test SLODR hypothesis using linear regression."""
    print("STEP 6: Analyzing SLODR hypothesis...\n")

    X = pca_results['mean_eci'].values.reshape(-1, 1)
    y = pca_results['evr1'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    correlation, p_value = stats.pearsonr(pca_results['mean_eci'], pca_results['evr1'])

    results = {
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'mse': mse,
        'correlation': correlation,
        'correlation_p_value': p_value,
        'model': model
    }

    print(f"  Slope: {slope:.6f}")
    print(f"  Intercept: {intercept:.6f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Correlation: {correlation:.4f} (p = {p_value:.4f}")

    if slope < 0:
        print("\n  ✓ SLODR hypothesis supported (negative slope)")
    else:
        print("\n  ✗ SLODR hypothesis NOT supported (positive slope)")

    print("\n✓ Regression analysis complete\n")

    return results

regression_results = analyze_slodr_hypothesis(pca_results)

# ============================================================================
# STEP 7: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_confidence_interval(pca_results, n_iterations=1000, confidence=0.95):
    """Calculate bootstrap confidence intervals for the slope."""
    print(f"STEP 7: Performing bootstrap analysis ({n_iterations} iterations)...\n")

    slopes = []
    n_samples = len(pca_results)

    for i in range(n_iterations):
        sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        sample_data = pca_results.iloc[sample_idx]

        X = sample_data['mean_eci'].values.reshape(-1, 1)
        y = sample_data['evr1'].values

        model = LinearRegression()
        model.fit(X, y)
        slopes.append(model.coef_[0])

    slopes = np.array(slopes)

    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(slopes, lower_percentile)
    upper_bound = np.percentile(slopes, upper_percentile)

    print(f"  Bootstrap slope distribution:")
    print(f"    Mean: {slopes.mean():.6f}")
    print(f"    Std: {slopes.std():.6f}")
    print(f"    {confidence*100:.0f}% CI: [{lower_bound:.6f}, {upper_bound:.6f}]")

    if lower_bound < 0 and upper_bound < 0:
        print(f"\n  ✓ Confidence interval excludes zero (significant negative slope)")
    elif lower_bound > 0 and upper_bound > 0:
        print(f"\n  ! Confidence interval excludes zero (significant positive slope)")
    else:
        print(f"\n  ✗ Confidence interval includes zero (not significant)")

    print("\n✓ Bootstrap complete\n")

    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'slopes': slopes,
        'mean': slopes.mean(),
        'std': slopes.std()
    }

bootstrap_results = bootstrap_confidence_interval(pca_results, n_iterations=1000)

# ============================================================================
# STEP 8: PERMUTATION TEST
# ============================================================================

def permutation_test(pca_results, observed_slope, n_permutations=1000):
    """Perform permutation test to assess significance."""
    print(f"STEP 8: Performing permutation test ({n_permutations} permutations)...\n")

    permuted_slopes = []
    X = pca_results['mean_eci'].values.reshape(-1, 1)
    y_original = pca_results['evr1'].values

    for i in range(n_permutations):
        y_permuted = np.random.permutation(y_original)
        model = LinearRegression()
        model.fit(X, y_permuted)
        permuted_slopes.append(model.coef_[0])

    permuted_slopes = np.array(permuted_slopes)

    if observed_slope < 0:
        p_value = np.mean(permuted_slopes <= observed_slope) * 2
    else:
        p_value = np.mean(permuted_slopes >= observed_slope) * 2

    p_value = min(p_value, 1.0)

    print(f"  Observed slope: {observed_slope:.6f}")
    print(f"  Permuted slopes mean: {permuted_slopes.mean():.6f}")
    print(f"  Permuted slopes std: {permuted_slopes.std():.6f}")
    print(f"  P-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"\n  ✓ Result is statistically significant (p < 0.05)")
    else:
        print(f"\n  ✗ Result is not statistically significant (p >= 0.05)")

    print("\n✓ Permutation test complete\n")

    return {
        'p_value': p_value,
        'permuted_slopes': permuted_slopes
    }

permutation_results = permutation_test(pca_results, regression_results['slope'], n_permutations=1000)

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================

print("STEP 9: Creating visualizations...\n")

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# 9.1 EVR1 vs Mean ECI with Regression Line
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(pca_results['mean_eci'], pca_results['evr1'],
           s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
X_plot = np.linspace(pca_results['mean_eci'].min(), pca_results['mean_eci'].max(), 100)
y_plot = regression_results['model'].predict(X_plot.reshape(-1, 1))
ax.plot(X_plot, y_plot, 'r-', linewidth=2, alpha=0.8,
        label=f"Linear fit: slope = {regression_results['slope']:.6f}")
ax.set_xlabel('Mean ECI', fontsize=12, fontweight='bold')
ax.set_ylabel('First PC Explained Variance Ratio (EVR1)', fontsize=12, fontweight='bold')
ax.set_title('SLODR Analysis: EVR1 vs ECI', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'evr1_vs_eci.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: output/evr1_vs_eci.png")

# 9.2 Permutation Test Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(permutation_results['permuted_slopes'], bins=50, alpha=0.7,
        edgecolor='black', label='Permuted slopes')
ax.axvline(regression_results['slope'], color='red', linewidth=2,
           linestyle='--', label=f"Observed slope = {regression_results['slope']:.6f}")
ax.set_xlabel('Slope', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title(f"Permutation Test Distribution (p = {permutation_results['p_value']:.4f})",
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'permutation_test.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: output/permutation_test.png")

# 9.3 Bootstrap Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(bootstrap_results['slopes'], bins=50, alpha=0.7,
        edgecolor='black', label='Bootstrap slopes')
ax.axvline(regression_results['slope'], color='red', linewidth=2,
           linestyle='--', label=f"Observed slope = {regression_results['slope']:.6f}")
ax.axvline(bootstrap_results['lower_bound'], color='green', linewidth=2,
           linestyle=':', label=f"95% CI: [{bootstrap_results['lower_bound']:.6f}, {bootstrap_results['upper_bound']:.6f}]")
ax.axvline(bootstrap_results['upper_bound'], color='green', linewidth=2, linestyle=':')
ax.set_xlabel('Slope', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Bootstrap Distribution of Slopes', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'bootstrap_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: output/bootstrap_distribution.png")

# 9.4 ECI Distribution Across Bins
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(pca_results['bin'], pca_results['evr1'], alpha=0.7, edgecolor='black')
ax1.set_xlabel('ECI Bin', fontsize=12, fontweight='bold')
ax1.set_ylabel('EVR1', fontsize=12, fontweight='bold')
ax1.set_title('First PC Explained Variance by ECI Bin', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax2.bar(pca_results['bin'], pca_results['n_models'], alpha=0.7,
        edgecolor='black', color='orange')
ax2.set_xlabel('ECI Bin', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Models', fontsize=12, fontweight='bold')
ax2.set_title('Sample Size by ECI Bin', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'bin_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: output/bin_analysis.png\n")

# ============================================================================
# STEP 10: EXPORT RESULTS
# ============================================================================

print("STEP 10: Exporting results...\n")

df_wide.to_csv(output_dir / 'wide_matrix.csv', index=False)
print("✓ Saved: output/wide_matrix.csv")

pca_results.to_csv(output_dir / 'pca_results.csv', index=False)
print("✓ Saved: output/pca_results.csv")

summary = pd.DataFrame({
    'metric': ['slope', 'intercept', 'r2', 'mse', 'correlation', 'correlation_p_value',
               'bootstrap_lower', 'bootstrap_upper', 'bootstrap_mean', 'bootstrap_std',
               'permutation_p_value'],
    'value': [
        regression_results['slope'],
        regression_results['intercept'],
        regression_results['r2'],
        regression_results['mse'],
        regression_results['correlation'],
        regression_results['correlation_p_value'],
        bootstrap_results['lower_bound'],
        bootstrap_results['upper_bound'],
        bootstrap_results['mean'],
        bootstrap_results['std'],
        permutation_results['p_value']
    ]
})
summary.to_csv(output_dir / 'summary_statistics.csv', index=False)
print("✓ Saved: output/summary_statistics.csv\n")

# ============================================================================
# STEP 11: ROBUSTNESS CHECKS
# ============================================================================

print("="*70)
print("ROBUSTNESS CHECK 1: Without Logit Transformation")
print("="*70 + "\n")

X_no_logit, _, _, _ = preprocess_data(df_wide, use_logit=False, k=5)
pca_results_no_logit = perform_pca_per_bin(X_no_logit, eci_values, n_bins=8)
regression_no_logit = analyze_slodr_hypothesis(pca_results_no_logit)

print(f"Comparison:")
print(f"  With logit:    slope = {regression_results['slope']:.6f}, p = {permutation_results['p_value']:.4f}")
print(f"  Without logit: slope = {regression_no_logit['slope']:.6f}")
print(f"  Difference:    {abs(regression_results['slope'] - regression_no_logit['slope']):.6f}\n")

# ============================================================================
# STEP 12: SUMMARY REPORT
# ============================================================================

print("="*70)
print("GENERATING SUMMARY REPORT")
print("="*70 + "\n")

markdown_report = f"""# SLODR Analysis Summary Report

## Dataset Overview
- **Total models**: {len(df_wide)}
- **Total benchmarks**: {len(benchmark_names)}
- **ECI range**: [{eci_values.min():.2f}, {eci_values.max():.2f}]
- **Mean benchmark coverage**: {df_wide[benchmark_names].notna().mean().mean():.1%}

## Main Results

### Linear Regression
- **Slope**: {regression_results['slope']:.6f}
- **Intercept**: {regression_results['intercept']:.6f}
- **R²**: {regression_results['r2']:.4f}
- **Pearson correlation**: {regression_results['correlation']:.4f} (p = {regression_results['correlation_p_value']:.4f})

### Bootstrap Confidence Intervals (95%)
- **Lower bound**: {bootstrap_results['lower_bound']:.6f}
- **Upper bound**: {bootstrap_results['upper_bound']:.6f}
- **Bootstrap mean**: {bootstrap_results['mean']:.6f}
- **Bootstrap std**: {bootstrap_results['std']:.6f}

### Permutation Test
- **P-value**: {permutation_results['p_value']:.4f}
- **Significance level**: {'✓ Significant at p < 0.05' if permutation_results['p_value'] < 0.05 else '✗ Not significant at p < 0.05'}

## Interpretation

### SLODR Hypothesis
The **Spearman's Law of Diminishing Returns** hypothesis predicts that as general capability (ECI) increases,
the first principal component should explain **less variance** in benchmark performance, indicating greater
differentiation of specific abilities.

**Result**: {'✓ **SUPPORTED**' if regression_results['slope'] < 0 and permutation_results['p_value'] < 0.05 else '✗ **NOT SUPPORTED**'}

- The observed slope is **{'negative' if regression_results['slope'] < 0 else 'positive'}** ({regression_results['slope']:.6f})
- The relationship is **{'statistically significant' if permutation_results['p_value'] < 0.05 else 'not statistically significant'}** (p = {permutation_results['p_value']:.4f})
- The 95% confidence interval {'**excludes zero**' if (bootstrap_results['lower_bound'] < 0 and bootstrap_results['upper_bound'] < 0) or (bootstrap_results['lower_bound'] > 0 and bootstrap_results['upper_bound'] > 0) else '**includes zero**'}

## Robustness Checks

### 1. Without Logit Transformation
- **Slope**: {regression_no_logit['slope']:.6f}
- **Difference from main**: {abs(regression_results['slope'] - regression_no_logit['slope']):.6f}

## What to Inspect if Results Look Wrong

### Potential Issues to Check:

1. **Ceiling Effects**
   - Look at score distributions for each benchmark
   - Check if top models cluster near perfect scores (>0.95)
   - Solution: Exclude saturated benchmarks

2. **Model Family Effects**
   - Different model families may have systematic biases
   - Check if results hold within single model families
   - Solution: Perform stratified analysis by organization

3. **Missing Data Patterns**
   - Are higher-ECI models missing specific benchmarks?
   - Check correlation between missingness and ECI
   - Solution: Use only high-coverage benchmarks

4. **Bin Size Effects**
   - Try different numbers of bins (4, 6, 10, 12)
   - Check if results are stable across bin counts
   - Solution: Use sliding windows instead of discrete bins

5. **Outlier Models**
   - Identify models with unusual benchmark patterns
   - Check for evaluation artifacts or data errors
   - Solution: Perform sensitivity analysis excluding outliers

6. **Benchmark Diversity**
   - Are all benchmarks measuring similar skills?
   - Check pairwise correlations between benchmarks
   - Solution: Select diverse, low-correlation benchmarks

## Files Generated

### Data Files
- `output/wide_matrix.csv` - Complete data matrix (models × benchmarks)
- `output/pca_results.csv` - PCA metrics per ECI bin
- `output/summary_statistics.csv` - All statistical results

### Visualizations
- `output/evr1_vs_eci.png` - Main SLODR plot with regression line
- `output/permutation_test.png` - Permutation test distribution
- `output/bootstrap_distribution.png` - Bootstrap confidence intervals
- `output/bin_analysis.png` - EVR1 and sample sizes by bin

## Recommendations for Further Analysis

1. **Sliding Window Analysis**: Use overlapping windows instead of discrete bins for smoother trends
2. **Benchmark Clustering**: Group similar benchmarks and analyze separately
3. **Temporal Analysis**: Check if SLODR effect changes with model release date
4. **Organization-Specific Analysis**: Test hypothesis within individual AI labs
5. **Non-Linear Models**: Try polynomial or spline regression for non-linear trends

---

*Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open(output_dir / 'SLODR_REPORT.md', 'w') as f:
    f.write(markdown_report)

print("✓ Saved: output/SLODR_REPORT.md\n")

print("="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print()
print("All results have been saved to the output/ directory.")
print("Check SLODR_REPORT.md for a comprehensive summary.")
print()
