# Code for Analysis of AI Model Intelligence

This repository contains code to test the **Spearman's Law of Diminishing Returns (SLODR)** hypothesis on AI benchmark data. SLODR predicts that as general capability increases, benchmark performance becomes more differentiated (i.e., the first principal component explains less variance).

## Statistical Procedures

### 1. Data Preparation

**Loading:**
- Benchmark scores are loaded from CSV files in `benchmark_data/`
- The Epoch Capability Index (ECI) serves as the measure of general capability
- Data is organized into a wide matrix with models as rows and benchmarks as columns

**Missing Data Handling:**
- Missing benchmark scores are imputed using **k-Nearest Neighbors (KNN)** imputation (k=5, distance-weighted)
- The imputation is performed in logit space to better handle bounded scores

### 2. Data Transformations

**Logit Transformation:**
- Scores (bounded between 0 and 1) are transformed using the logit function: `logit(p) = log(p / (1-p))`
- This handles ceiling effects where top models cluster near perfect scores
- Small epsilon values (1e-4) prevent numerical issues at boundaries

**Standardization:**
- After imputation and inverse logit transformation, each benchmark is z-score standardized
- Formula: `(X - mean) / std`
- This ensures all benchmarks contribute equally to the PCA regardless of their original variance

### 3. Principal Component Analysis (PCA)

**Binning Strategy:**
- Models are sorted by ECI and divided into equal-count bins (default: 8 bins)
- This creates groups with similar general capability levels

**PCA per Bin:**
- Within each bin, PCA is performed on the standardized benchmark scores
- The **first principal component's explained variance ratio (EVR1)** is extracted
- EVR1 represents how much variance a single "general factor" explains within that capability level

### 4. Regression Analysis

**Linear Regression:**
- EVR1 is regressed on mean ECI per bin
- Model: `EVR1 = intercept + slope × mean_ECI`
- SLODR predicts a **negative slope** (higher capability → lower EVR1 → more differentiation)

**Correlation:**
- Pearson correlation coefficient between mean ECI and EVR1
- P-value computed to assess linear relationship strength

**Metrics:**
- R² (coefficient of determination)
- MSE (mean squared error)

### 5. Statistical Significance Testing

**Bootstrap Confidence Intervals:**
- 1000 bootstrap iterations with replacement
- Resamples bins and refits linear regression each time
- Computes 95% confidence interval for the slope
- If the CI excludes zero, the effect is considered significant

**Permutation Test:**
- 1000 random permutations of EVR1 values
- Refits regression on each permutation to create null distribution
- Two-tailed p-value: proportion of permuted slopes as extreme as observed
- Significance threshold: p < 0.05

### 6. Robustness Checks

The analysis includes three robustness checks to validate findings:

1. **Without Logit Transformation:**
   - Repeats the entire pipeline without logit/expit transformations
   - Tests whether ceiling effects substantially impact results

2. **Excluding Near-Ceiling Benchmarks:**
   - Removes benchmarks where top 10% of models have median score > 0.98
   - Addresses saturation effects in easy benchmarks

3. **High-Coverage Benchmarks Only:**
   - Keeps only benchmarks with ≥70% non-missing data
   - Reduces influence of imputation on results

### 7. Visualizations

The analysis generates four main plots:

1. **EVR1 vs ECI Scatter Plot:**
   - Shows relationship between capability level and explained variance
   - Includes fitted regression line with slope annotation

2. **Permutation Test Distribution:**
   - Histogram of slopes under the null hypothesis
   - Vertical line shows observed slope
   - Illustrates statistical significance

3. **Bootstrap Distribution:**
   - Histogram of bootstrap slope estimates
   - Shows 95% confidence interval bounds
   - Demonstrates estimation uncertainty

4. **Bin Analysis:**
   - Bar charts showing EVR1 and sample size per bin
   - Helps identify if results are driven by specific capability ranges

## Running the Analysis

**Main Script:**
```bash
python run_slodr_analysis.py
```

This executes the full pipeline and saves results to `output/`:
- `wide_matrix.csv` - Complete data matrix
- `pca_results.csv` - PCA metrics per bin
- `summary_statistics.csv` - All regression and test results
- `SLODR_REPORT.md` - Comprehensive analysis summary
- Four PNG visualization files

**Jupyter Notebook:**
The analysis can also be run interactively in `slodr_analysis.ipynb`

## Interpretation

**If SLODR is supported:**
- Slope is negative and statistically significant
- Bootstrap CI excludes zero
- As models become more capable (higher ECI), their benchmark scores are explained less by a single general factor
- This suggests increasing specialization and differentiation at higher capability levels




