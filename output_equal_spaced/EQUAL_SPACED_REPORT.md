# SLODR Analysis - Equally-Spaced Bins Method

## Binning Method

**Equally-Spaced Bins:**
- Creates bins with equal width in ECI space
- Bin edges: `np.linspace(min_ECI, max_ECI, n_bins + 1)`
- Each bin covers the same ECI range
- Bins may have **different numbers of models**

**Comparison with Equal-Count Bins:**
- Equal-count bins ensure each bin has approximately the same number of models
- Equal-count bins use percentiles: `np.percentile(ECI, [0, 12.5, 25, ...])`
- Equal-spaced bins may have very small samples at the extremes

## Results Summary

### Sample Size Distribution

| Bin | Min ECI | Max ECI | N Models | EVR1 |
|-----|---------|---------|----------|------|
| 3.0 | 82.23 | 92.59 | 11.0 | 1.0000 |
| 4.0 | 93.23 | 104.50 | 17.0 | 1.0000 |
| 5.0 | 105.52 | 117.09 | 29.0 | 0.9446 |
| 6.0 | 117.50 | 129.43 | 51.0 | 0.7901 |
| 7.0 | 130.00 | 141.41 | 69.0 | 0.4402 |
| 8.0 | 141.68 | 153.81 | 103.0 | 0.5870 |

**Observations:**
- Total models across bins: 280
- Mean models per bin: 46.7
- Std dev of bin sizes: 35.1
- Smallest bin: 11 models
- Largest bin: 103 models

### Regression Results

- **Slope**: -0.009688
- **Intercept**: 1.933601
- **R²**: 0.8073
- **Pearson correlation**: -0.8985 (p = 0.0149)

### Bootstrap Confidence Intervals (95%)
- **Lower bound**: -0.016353
- **Upper bound**: -0.004958
- **Bootstrap mean**: -0.010006

### Permutation Test
- **P-value**: 0.0200
- **Significance**: ✓ Significant at p < 0.05

## Interpretation

**SLODR Hypothesis:** ✓ **SUPPORTED**

The observed slope is **negative** (-0.009688),
indicating that as models become more capable (higher ECI), the first principal component explains
less variance in their benchmark performance.

## Advantages and Disadvantages of Equally-Spaced Bins

### Advantages:
1. **Intuitive interpretation**: Each bin represents the same ECI range
2. **No compression**: Doesn't artificially group very different capability levels
3. **Captures density**: Shows where models are concentrated in ECI space

### Disadvantages:
1. **Unequal sample sizes**: Some bins may have very few models, reducing statistical power
2. **Extreme bins**: Lowest and highest bins often have small samples
3. **Noise sensitivity**: Small bins are more susceptible to outliers
4. **PCA instability**: PCA with few samples may not be reliable

## Recommendations

- If bin sizes vary greatly (>3x difference), consider equal-count bins for more stable estimates
- If interested in density effects, equally-spaced bins show where models cluster
- For publication, report both methods to demonstrate robustness
- Weight bins by sample size in meta-analyses

---

*Analysis completed: 2026-02-17 22:37:20*
