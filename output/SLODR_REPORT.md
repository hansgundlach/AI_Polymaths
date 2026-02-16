# SLODR Analysis Summary Report

## Dataset Overview
- **Total models**: 282
- **Total benchmarks**: 8
- **ECI range**: [56.75, 153.81]
- **Mean benchmark coverage**: 22.5%

## Main Results

### Linear Regression
- **Slope**: -0.009954
- **Intercept**: 1.996534
- **R²**: 0.6755
- **Pearson correlation**: -0.8219 (p = 0.0123)

### Bootstrap Confidence Intervals (95%)
- **Lower bound**: -0.019086
- **Upper bound**: -0.003436
- **Bootstrap mean**: -0.010232
- **Bootstrap std**: 0.004131

### Permutation Test
- **P-value**: 0.0040
- **Significance level**: ✓ Significant at p < 0.05

## Interpretation

### SLODR Hypothesis
The **Spearman's Law of Diminishing Returns** hypothesis predicts that as general capability (ECI) increases,
the first principal component should explain **less variance** in benchmark performance, indicating greater
differentiation of specific abilities.

**Result**: ✓ **SUPPORTED**

- The observed slope is **negative** (-0.009954)
- The relationship is **statistically significant** (p = 0.0040)
- The 95% confidence interval **excludes zero**

## Robustness Checks

### 1. Without Logit Transformation
- **Slope**: -0.010509
- **Difference from main**: 0.000555

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

*Analysis completed: 2026-01-06 13:44:06*
