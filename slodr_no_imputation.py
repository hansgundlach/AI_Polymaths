#!/usr/bin/env python
"""
SLODR Analysis WITHOUT Imputation

This version analyzes SLODR using only observed data:
1. Uses high-coverage benchmarks only
2. Performs PCA only on models with sufficient data
3. Shows what happens without synthetic data

This addresses the concern that KNN imputation may be creating artificial patterns.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("SLODR ANALYSIS - WITHOUT IMPUTATION")
print("="*70)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

def load_benchmark_data(bench_dir='benchmark_data', score_col='mean_score', model_col='Model version'):
    """Load all benchmark CSV files from directory."""
    bench_path = Path(bench_dir)
    benchmarks = {}
    csv_files = [f for f in bench_path.glob('*.csv') if 'epoch_capabilities_index' not in f.name]

    for csv_file in csv_files:
        benchmark_name = csv_file.stem
        try:
            df = pd.read_csv(csv_file)
            if model_col not in df.columns:
                continue
            if score_col in df.columns:
                score_column = score_col
            elif 'Best score (across scorers)' in df.columns:
                score_column = 'Best score (across scorers)'
            elif 'score' in df.columns:
                score_column = 'score'
            else:
                continue

            df_clean = df[[model_col, score_column]].copy()
            df_clean.columns = ['model', 'score']
            df_clean = df_clean.dropna(subset=['score'])
            df_clean['score'] = pd.to_numeric(df_clean['score'], errors='coerce')
            df_clean = df_clean.dropna(subset=['score'])
            df_clean = df_clean.groupby('model')['score'].max().reset_index()

            if len(df_clean) > 0:
                benchmarks[benchmark_name] = df_clean
        except:
            pass

    return benchmarks

def load_eci_data(eci_csv='benchmark_data/epoch_capabilities_index.csv',
                  model_col='Model version', eci_col='ECI Score'):
    """Load the Epoch Capability Index (ECI) data."""
    df = pd.read_csv(eci_csv)
    df_eci = df[[model_col, eci_col]].copy()
    df_eci.columns = ['model', 'eci']
    df_eci = df_eci.dropna(subset=['eci'])
    df_eci['eci'] = pd.to_numeric(df_eci['eci'], errors='coerce')
    return df_eci.dropna(subset=['eci'])

benchmarks = load_benchmark_data()
eci_data = load_eci_data()

# ============================================================================
# STRATEGY 1: HIGH-COVERAGE BENCHMARKS ONLY
# ============================================================================

print("STRATEGY 1: Using only high-coverage benchmarks (>30% coverage)")
print("="*70)
print()

df_wide = eci_data.copy().set_index('model')
for bench_name, bench_df in benchmarks.items():
    bench_df_indexed = bench_df.set_index('model')
    df_wide[bench_name] = bench_df_indexed['score']
df_wide = df_wide.reset_index()

# Filter to high-coverage benchmarks
benchmark_cols = [col for col in df_wide.columns if col not in ['model', 'eci']]
coverage = {}
for col in benchmark_cols:
    cov = df_wide[col].notna().mean()
    coverage[col] = cov

min_coverage = 0.30
high_cov_benchmarks = [col for col, cov in coverage.items() if cov >= min_coverage]

print(f"Benchmarks with ≥{min_coverage*100}% coverage:")
for col in high_cov_benchmarks:
    print(f"  {col}: {coverage[col]*100:.1f}%")
print()

if len(high_cov_benchmarks) < 3:
    print("ERROR: Not enough high-coverage benchmarks for PCA")
    exit(1)

# Keep only high-coverage benchmarks
df_filtered = df_wide[['model', 'eci'] + high_cov_benchmarks].copy()

# For each model, keep only if they have at least 2 benchmark scores
min_benchmarks_per_model = 2
benchmark_count = df_filtered[high_cov_benchmarks].notna().sum(axis=1)
df_filtered = df_filtered[benchmark_count >= min_benchmarks_per_model].copy()

print(f"After filtering:")
print(f"  Models: {len(df_filtered)}")
print(f"  Benchmarks: {len(high_cov_benchmarks)}")
print(f"  Missing data: {df_filtered[high_cov_benchmarks].isna().sum().sum()} / {len(df_filtered) * len(high_cov_benchmarks)} ({df_filtered[high_cov_benchmarks].isna().mean().mean()*100:.1f}%)")
print()

# ============================================================================
# STRATEGY 2: PCA ON PAIRWISE COMPLETE DATA WITHIN BINS
# ============================================================================

def standardize(X):
    """Z-score standardization, handling NaN."""
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0, ddof=1)
    std[std == 0] = 1.0
    X_standardized = (X - mean) / std
    return X_standardized

def pca_with_missingness(X, n_components=None):
    """
    Simple PCA with missing data using available data only.
    This is NOT optimal but avoids imputation.
    """
    # Fill NaN with column mean for PCA (simple approach)
    X_filled = X.copy()
    col_means = np.nanmean(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X_filled[mask, i] = col_means[i]

    pca = PCA(n_components=n_components)
    pca.fit(X_filled)
    return pca

print("STRATEGY 2: PCA with minimal mean-imputation per bin")
print("="*70)
print()

eci_values = df_filtered['eci'].values
X = df_filtered[high_cov_benchmarks].values

# Standardize
X_std = standardize(X)

# Bin by ECI (equal-count)
n_bins = 6  # Fewer bins due to reduced sample size
sort_idx = np.argsort(eci_values)
X_sorted = X_std[sort_idx]
eci_sorted = eci_values[sort_idx]

bin_edges = np.percentile(eci_sorted, np.linspace(0, 100, n_bins + 1))
bin_indices = np.digitize(eci_sorted, bin_edges[1:-1])

results = []

for bin_idx in range(n_bins):
    mask = (bin_indices == bin_idx)
    X_bin = X_sorted[mask]
    eci_bin = eci_sorted[mask]

    if len(X_bin) < 3:
        print(f"  Bin {bin_idx+1}: Skipped (only {len(X_bin)} models)")
        continue

    # Count how much real data in this bin
    n_total = X_bin.size
    n_missing = np.isnan(X_bin).sum()
    pct_real = ((n_total - n_missing) / n_total) * 100

    pca = pca_with_missingness(X_bin)
    evr1 = pca.explained_variance_ratio_[0]
    mean_eci = np.mean(eci_bin)

    results.append({
        'bin': bin_idx + 1,
        'mean_eci': mean_eci,
        'min_eci': np.min(eci_bin),
        'max_eci': np.max(eci_bin),
        'evr1': evr1,
        'n_models': len(X_bin),
        'pct_real_data': pct_real
    })

    print(f"  Bin {bin_idx+1}: ECI [{mean_eci:.1f}], EVR1 = {evr1:.3f}, n = {len(X_bin)}, real data = {pct_real:.1f}%")

pca_results = pd.DataFrame(results)
print(f"\n✓ PCA complete for {len(pca_results)} bins\n")

if len(pca_results) < 3:
    print("ERROR: Not enough bins for meaningful regression")
    exit(1)

# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================

print("LINEAR REGRESSION ANALYSIS")
print("="*70)
print()

X_reg = pca_results['mean_eci'].values.reshape(-1, 1)
y_reg = pca_results['evr1'].values

model = LinearRegression()
model.fit(X_reg, y_reg)
y_pred = model.predict(X_reg)

slope = model.coef_[0]
intercept = model.intercept_
r2 = r2_score(y_reg, y_pred)
mse = mean_squared_error(y_reg, y_pred)

if len(pca_results) >= 3:
    correlation, p_value = stats.pearsonr(pca_results['mean_eci'], pca_results['evr1'])
else:
    correlation, p_value = np.nan, np.nan

print(f"  Slope: {slope:.6f}")
print(f"  Intercept: {intercept:.6f}")
print(f"  R²: {r2:.4f}")
print(f"  MSE: {mse:.6f}")
if not np.isnan(correlation):
    print(f"  Correlation: {correlation:.4f} (p = {p_value:.4f})")
print()

if slope < 0:
    print("  ✓ SLODR hypothesis supported (negative slope)")
else:
    print("  ✗ SLODR hypothesis NOT supported (positive slope)")
print()

# ============================================================================
# VISUALIZATION
# ============================================================================

output_dir = Path('output_no_imputation')
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))

# Size by % real data
sizes = pca_results['pct_real_data'] * 3
colors = pca_results['pct_real_data']

scatter = ax.scatter(pca_results['mean_eci'], pca_results['evr1'],
                     s=sizes, alpha=0.6, edgecolors='black', linewidth=2,
                     c=colors, cmap='RdYlGn', vmin=0, vmax=100)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('% Real Data', fontsize=10)

X_plot = np.linspace(pca_results['mean_eci'].min(), pca_results['mean_eci'].max(), 100)
y_plot = model.predict(X_plot.reshape(-1, 1))
ax.plot(X_plot, y_plot, 'r-', linewidth=2, alpha=0.8,
        label=f"Linear fit: slope = {slope:.6f}")

ax.set_xlabel('Mean ECI', fontsize=12, fontweight='bold')
ax.set_ylabel('First PC Explained Variance Ratio (EVR1)', fontsize=12, fontweight='bold')
ax.set_title('SLODR Analysis: NO IMPUTATION (High-Coverage Benchmarks Only)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'evr1_vs_eci_no_imputation.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: output_no_imputation/evr1_vs_eci_no_imputation.png")

# ============================================================================
# EXPORT RESULTS
# ============================================================================

pca_results.to_csv(output_dir / 'pca_results_no_imputation.csv', index=False)
print("✓ Saved: output_no_imputation/pca_results_no_imputation.csv")

summary = pd.DataFrame({
    'metric': ['slope', 'intercept', 'r2', 'mse', 'correlation', 'correlation_p_value',
               'n_bins', 'n_benchmarks', 'avg_pct_real_data'],
    'value': [slope, intercept, r2, mse, correlation, p_value,
              len(pca_results), len(high_cov_benchmarks),
              pca_results['pct_real_data'].mean()]
})
summary.to_csv(output_dir / 'summary_no_imputation.csv', index=False)
print("✓ Saved: output_no_imputation/summary_no_imputation.csv")

# ============================================================================
# REPORT
# ============================================================================

report = f"""# SLODR Analysis - Without KNN Imputation

## Motivation

The original analysis uses KNN imputation to fill in 77.5% of missing data. This raises concerns:
1. **Circular logic**: KNN uses correlations to impute, then PCA measures those correlations
2. **Synthetic data dominance**: Analysis is based on 3/4 fabricated numbers
3. **Inflated patterns**: Imputation can create or hide true patterns

This analysis avoids imputation by:
- Using only high-coverage benchmarks (>30% coverage)
- Using simple mean imputation within bins (transparent, minimal assumptions)
- Clearly reporting how much data is real vs. imputed

## Data Filtering

**Original dataset:**
- 282 models × 8 benchmarks
- 77.5% missing data
- 0 models with complete data

**Filtered dataset:**
- {len(df_filtered)} models × {len(high_cov_benchmarks)} benchmarks (>30% coverage)
- {df_filtered[high_cov_benchmarks].isna().mean().mean()*100:.1f}% missing data
- Models kept only if they have ≥{min_benchmarks_per_model} benchmark scores

**High-coverage benchmarks:**
{chr(10).join(f'- {col}: {coverage[col]*100:.1f}% coverage' for col in high_cov_benchmarks)}

## Results

### Regression

- **Slope**: {slope:.6f}
- **Intercept**: {intercept:.6f}
- **R²**: {r2:.4f}
- **Correlation**: {correlation:.4f} (p = {p_value:.4f})

### SLODR Hypothesis

**Result**: {'✓ **SUPPORTED**' if slope < 0 else '✗ **NOT SUPPORTED**'}

The slope is **{'negative' if slope < 0 else 'positive'}** ({slope:.6f}), suggesting that
as models become more capable, the first principal component explains
{'less' if slope < 0 else 'more'} variance in benchmark performance.

### Per-Bin Results

| Bin | Mean ECI | EVR1 | N Models | % Real Data |
|-----|----------|------|----------|-------------|
{chr(10).join(f"| {row['bin']} | {row['mean_eci']:.1f} | {row['evr1']:.4f} | {row['n_models']} | {row['pct_real_data']:.1f}% |" for _, row in pca_results.iterrows())}

**Average real data per bin**: {pca_results['pct_real_data'].mean():.1f}%

## Comparison with KNN Imputation

The key question: **Does the SLODR pattern hold without massive imputation?**

| Method | Slope | Interpretation |
|--------|-------|----------------|
| KNN imputation (original) | ~-0.01 | SLODR supported |
| No imputation (this) | {slope:.6f} | SLODR {'supported' if slope < 0 else 'not supported'} |

**Conclusion**:
{
'The negative slope persists even without KNN imputation, suggesting the SLODR pattern is real and not an artifact of imputation.'
if slope < 0 else
'The slope is no longer clearly negative without imputation, suggesting the original SLODR finding may be partially driven by imputation artifacts.'
}

## Recommendations

1. **Report both methods**: Show results with and without heavy imputation
2. **Use high-coverage benchmarks**: Prioritize benchmarks that most models are tested on
3. **Transparency**: Always report % of real vs. imputed data
4. **Sensitivity analysis**: Test multiple imputation methods (mean, median, KNN with different k)
5. **Consider alternatives**: Direct correlation analysis may be more robust than PCA with missing data

---

*Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open(output_dir / 'NO_IMPUTATION_REPORT.md', 'w') as f:
    f.write(report)

print("✓ Saved: output_no_imputation/NO_IMPUTATION_REPORT.md")
print()
print("="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print()
print("Key finding:")
if slope < 0:
    print("  ✓ SLODR pattern persists WITHOUT heavy imputation")
    print("    This suggests the effect is real, not an imputation artifact")
else:
    print("  ⚠ SLODR pattern does NOT persist without imputation")
    print("    This suggests the original finding may be driven by KNN imputation")
print()
