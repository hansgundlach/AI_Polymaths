# CLAUDE.md - Project Context for Claude Code

## Project Overview

This repository tests **Spearman's Law of Diminishing Returns (SLODR)** on AI model benchmark data. The hypothesis: as AI models become more capable (higher general intelligence), their performance across different benchmarks becomes more differentiated rather than uniformly correlated.

**Core Question:** Do more capable models show greater specialization in their abilities?

## Repository Structure

```
diminshing_returns/
├── benchmark_data/          # CSV files with benchmark scores
│   ├── *.csv               # Individual benchmark results
│   └── epoch_capabilities_index.csv  # ECI scores (general capability metric)
├── slodr_analysis.py        # Main analysis functions (Jupyter notebook format)
├── run_slodr_analysis.py    # Command-line script version
├── benchmark_dimensions.py  # Alternative analysis exploring benchmark dimensionality
├── slodr_analysis.ipynb     # Interactive Jupyter notebook
├── output/                  # Generated results (created by scripts)
│   ├── evr1_vs_eci.png
│   ├── permutation_test.png
│   ├── bootstrap_distribution.png
│   ├── bin_analysis.png
│   ├── wide_matrix.csv
│   ├── pca_results.csv
│   ├── summary_statistics.csv
│   └── SLODR_REPORT.md
└── README.md                # User-facing documentation
```

## Key Files

### `run_slodr_analysis.py`
The main executable script. Runs the complete SLODR analysis pipeline:
1. Loads benchmark data and ECI scores
2. Preprocesses with logit transformation and KNN imputation
3. Performs PCA within ECI bins
4. Tests SLODR via linear regression
5. Validates with bootstrap and permutation tests
6. Generates visualizations and reports

**Run with:** `python run_slodr_analysis.py`

### `slodr_analysis.py`
Contains all the analysis functions but in Jupyter notebook cell format (with `%matplotlib inline`, etc.). This is essentially the same logic as `run_slodr_analysis.py` but meant for interactive use.

### `benchmark_dimensions.py`
Alternative analysis notebook exploring:
- Benchmark correlation heatmaps
- Pairwise benchmark scatter plots
- Outlier detection using sigmoid regression
- PCA dimensionality via parallel analysis
- Model projections in PC1-PC2 space

## Statistical Pipeline

### 1. Data Preparation
- Load benchmark CSVs from `benchmark_data/`
- Create models × benchmarks matrix
- Extract ECI (Epoch Capability Index) as capability measure

### 2. Preprocessing
- **Logit transform** scores to handle ceiling effects
- **KNN imputation** (k=5, distance-weighted) for missing values in logit space
- **Inverse logit** back to probability space
- **Z-score standardization** of each benchmark

### 3. Analysis Strategy
- Bin models by ECI into equal-count groups (default: 8 bins)
- Within each bin, perform PCA on benchmark scores
- Extract **EVR1** (explained variance ratio of first PC)
- Higher EVR1 = more variance explained by single factor = less differentiation

### 4. SLODR Test
- Regress EVR1 on mean ECI per bin
- **Expected:** Negative slope (higher capability → lower EVR1 → more differentiation)
- Validate with bootstrap CI and permutation test

### 5. Robustness
- Test without logit transformation
- Exclude near-ceiling benchmarks
- Use only high-coverage benchmarks

## Key Concepts

**ECI (Epoch Capability Index):**
A composite measure of general AI capability across multiple benchmarks. Higher ECI = more capable model.

**EVR1 (Explained Variance Ratio, PC1):**
The proportion of total variance explained by the first principal component. High EVR1 means a single "general factor" dominates performance differences.

**SLODR Prediction:**
As capability increases, EVR1 should decrease because:
- Lower capability: all abilities improve together (high correlation)
- Higher capability: abilities differentiate (lower correlation)

**Why Logit Space?**
Benchmark scores are bounded [0, 1]. At high capability levels, scores cluster near 1.0 (ceiling effect). Logit transformation expands this compressed range, making relationships more linear.

## Common Tasks

### Running a Complete Analysis
```bash
python run_slodr_analysis.py
```
Results saved to `output/` directory.

### Modifying Parameters
Edit these variables in `run_slodr_analysis.py`:
- `n_bins`: Number of ECI bins (default: 8)
- `k`: Number of neighbors for KNN (default: 5)
- `use_logit`: Whether to apply logit transformation (default: True)
- `n_iterations`: Bootstrap iterations (default: 1000)
- `n_permutations`: Permutation test iterations (default: 1000)

### Adding New Benchmarks
1. Place CSV file in `benchmark_data/`
2. Ensure columns: `Model version`, `mean_score` (or `Best score (across scorers)`)
3. Re-run analysis

### Interpreting Results
Check `output/SLODR_REPORT.md` for:
- Regression slope and significance
- Bootstrap confidence intervals
- Permutation test p-value
- Robustness check comparisons

**SLODR Supported if:**
- Slope < 0 (negative)
- p-value < 0.05
- Bootstrap CI excludes zero

## Dependencies

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

Optional for interactive notebooks:
```python
jupyter
plotly  # For benchmark_dimensions.py interactivity
```

## Typical Workflow

1. **Add/update benchmark data** in `benchmark_data/`
2. **Run analysis:** `python run_slodr_analysis.py`
3. **Review results:** Check `output/SLODR_REPORT.md` and visualizations
4. **Iterate:** Adjust parameters or add robustness checks as needed

## Notes for Claude

- The main analysis logic is duplicated between `slodr_analysis.py` (notebook format) and `run_slodr_analysis.py` (script format)
- `benchmark_dimensions.py` is a separate exploratory analysis with different focus
- When making changes, consider which file(s) need updating:
  - Analysis logic changes → both `slodr_analysis.py` and `run_slodr_analysis.py`
  - New visualizations → likely `run_slodr_analysis.py` only
  - Documentation → `README.md` and potentially this file

## Questions to Ask When Modifying

1. **Data changes:** Do new benchmarks have the expected column names?
2. **Statistical changes:** Should robustness checks be updated to match?
3. **Visualization changes:** Are axes labels and titles clear?
4. **Performance:** Is KNN imputation reasonable for the data size?
5. **Interpretation:** Does the change affect how results should be interpreted in `SLODR_REPORT.md`?

## Testing the Analysis

After modifications, verify:
```bash
# Run main analysis
python run_slodr_analysis.py

# Check outputs exist
ls output/

# Verify report was generated
cat output/SLODR_REPORT.md | head -20
```

Expected outputs: 4 PNG files, 3 CSV files, 1 markdown report.
