# PCA Matrix Comparison: SLODR vs Benchmark Dimensions

## Overview

Both analyses use PCA on benchmark performance data, but they apply it in **fundamentally different ways** to answer different questions.

---

## Analysis 1: SLODR Analysis (run_slodr_analysis.py)

### Matrix Structure

**Input data:** 282 models × 8 benchmarks (77.5% missing)

**PCA Strategy:** **STRATIFIED PCA** (separate PCA per bin)

```
Models sorted by ECI:
┌────────────────┐
│ Bin 1 (n=36)   │ ───→ PCA → EVR1 = 0.85
│ ECI: 56-109    │
└────────────────┘

┌────────────────┐
│ Bin 2 (n=35)   │ ───→ PCA → EVR1 = 0.78
│ ECI: 111-119   │
└────────────────┘

┌────────────────┐
│ Bin 3 (n=35)   │ ───→ PCA → EVR1 = 0.71
│ ECI: 119-128   │
└────────────────┘

        ⋮

┌────────────────┐
│ Bin 8 (n=35)   │ ───→ PCA → EVR1 = 0.45
│ ECI: 143-153   │
└────────────────┘
```

### What PCA Sees

**Per bin (e.g., Bin 1):**
```
       bench1  bench2  bench3  bench4  bench5  bench6  bench7  bench8
model1   0.12    0.45    0.23    0.67   [IMP]  [IMP]   0.34    0.56
model2   0.18    0.51    0.29   [IMP]   0.78   [IMP]   0.41    0.62
model3  [IMP]    0.48    0.25    0.70    0.82   [IMP]   0.38    0.59
  ⋮
model36  0.15    0.49    0.27    0.68    0.80   [IMP]   0.36    0.58

[IMP] = KNN imputed value
```

**Shape:** 36 models × 8 benchmarks (subset of data)

### What EVR1 Means

**EVR1 = Explained Variance Ratio of first PC WITHIN this bin**

- High EVR1 (0.85): Benchmarks are highly correlated within this capability level
  - Models that do well on one benchmark do well on others
  - A single "general factor" dominates

- Low EVR1 (0.45): Benchmarks are less correlated within this capability level
  - Models show more specialization
  - Multiple independent factors needed

### Research Question

**"Does the correlation structure CHANGE as capability increases?"**

- If EVR1 decreases with ECI → SLODR supported (more differentiation at high ability)
- If EVR1 constant → No SLODR effect
- If EVR1 increases → Opposite of SLODR

---

## Analysis 2: Benchmark Dimensions (benchmark_dimensions.py)

### Matrix Structure

**Input data:** 282 models × 8 benchmarks (77.5% missing)

**PCA Strategy:** **GLOBAL PCA** (single PCA on all models)

```
ALL Models together:
┌────────────────────────────────────────┐
│ All 282 models (ECI range: 56-153)    │ ───→ PCA → PC1, PC2, PC3...
│                                        │
│ Low, medium, and high ability mixed    │
└────────────────────────────────────────┘
```

### What PCA Sees

**Full dataset:**
```
         bench1  bench2  bench3  bench4  bench5  bench6  bench7  bench8
model1     0.12    0.45    0.23    0.67   [IMP]  [IMP]   0.34    0.56
model2     0.18    0.51    0.29   [IMP]   0.78   [IMP]   0.41    0.62
model3    [IMP]    0.48    0.25    0.70    0.82   [IMP]   0.38    0.59
  ⋮
model282   0.89    0.92    0.87    0.91    0.94    0.88    0.90    0.93

[IMP] = KNN imputed value
```

**Shape:** 282 models × 8 benchmarks (complete dataset)

### What the PCs Mean

**PC1 (explains ~45% variance):** "General Capability"
- Loadings: All benchmarks positive, roughly equal
- High PC1 score = Model is generally capable across all benchmarks
- This captures the main "g-factor"

**PC2 (explains ~20% variance):** "Math vs Language specialization"
- Loadings: Math benchmarks positive, language benchmarks negative
- High PC2 = Strong at math, weak at language
- Low PC2 = Weak at math, strong at language

**PC3 (explains ~10% variance):** "Reasoning vs Knowledge"
- Loadings: Reasoning tasks positive, knowledge tasks negative
- Captures another dimension of specialization

### Output: Model Projections

Can plot each model in PC space:

```
PC2 (Math specialization)
    ↑
    │    ● GPT-4 (balanced)
    │  ●   Claude (balanced)
    │
────┼────────────────────────→ PC1 (General capability)
    │           ●
    │         ● Gemini (high math)
    │       ● DeepSeek (high math)
    ↓
```

### Research Question

**"What are the main dimensions of variation in AI benchmarks?"**

- Identifies latent factors explaining model differences
- Visualizes model positions in capability space
- Explores correlation structure of benchmarks
- Parallel analysis to determine dimensionality

---

## Key Differences Summary

| Aspect | SLODR Analysis | Benchmark Dimensions |
|--------|----------------|---------------------|
| **PCA runs** | 8 separate PCAs | 1 global PCA |
| **Data per PCA** | ~35 models each | 282 models total |
| **Bins used** | Yes (8 ECI bins) | No bins |
| **Output** | EVR1 values (8 numbers) | PC projections (282 × 3 matrix) |
| **EVR1 meaning** | Within-bin correlation | Overall correlation |
| **Visualization** | EVR1 vs ECI scatter | Models in PC1-PC2 space |
| **Question** | Structure change? | What is structure? |

---

## Critical Insight: They're Incompatible

### The Paradox

**SLODR Analysis result:**
- EVR1 decreases from 0.85 → 0.45 across bins
- Conclusion: More differentiation at high capability

**Benchmark Dimensions result:**
- Global EVR1 = ~0.45 (from single PCA on all models)
- PC1 explains 45% variance across everyone

**Wait, which is it?**

The answer is both are "correct" but measuring different things:

1. **SLODR EVR1 (per bin):** Measures correlation **conditional on ability level**
   - "Among models of similar capability, how correlated are benchmarks?"

2. **Dimensions EVR1 (global):** Measures correlation **across all ability levels**
   - "Across all models, how correlated are benchmarks?"

### Example to Clarify

Imagine two benchmarks: Math and Reading

**Scenario:**
- Low-ability models: Math and Reading scores both low (~0.2-0.3), highly correlated
- High-ability models: Math and Reading scores vary independently (~0.7-0.9), less correlated

**SLODR Analysis:**
- Low-ability bin: EVR1 = 0.90 (strong correlation within this group)
- High-ability bin: EVR1 = 0.50 (weak correlation within this group)
- Conclusion: SLODR supported

**Dimensions Analysis:**
- Global PCA sees both groups together
- Main variance is low→high (PC1), which both benchmarks share
- EVR1 = 0.70 (moderate global correlation)
- Doesn't test SLODR, describes overall structure

---

## The KNN Problem Affects Both

**Both analyses suffer from the same issue:**

1. Start with 77.5% missing data
2. Use KNN to fabricate 1,749 out of 2,256 data points
3. KNN assumes correlations to do imputation
4. PCA measures those assumed correlations
5. Circular reasoning

**Difference in severity:**

- **SLODR:** More vulnerable because it compares structure across bins
  - KNN might impute differently in different bins
  - This could create or mask the SLODR pattern
  - We showed the effect disappears without imputation

- **Dimensions:** Still problematic but more descriptive
  - Just exploring overall structure
  - Not testing a specific hypothesis about change
  - Less likely to draw wrong causal conclusions

---

## Recommendations

### For SLODR Analysis

1. ✅ Use high-coverage benchmarks only (avoid heavy imputation)
2. ✅ Report % real vs. imputed data per bin
3. ✅ Test sensitivity to imputation method
4. ✅ Consider alternative approaches (direct correlation analysis)

### For Benchmark Dimensions

1. ✅ Use parallel analysis to avoid over-interpretation
2. ✅ Show uncertainty from imputation (multiple imputation)
3. ✅ Compare with pairwise complete correlations
4. ✅ Acknowledge limitations from missing data

---

## Bottom Line

**SLODR Analysis:** Tests if differentiation changes with ability
- Uses stratified PCA (one per bin)
- Extracts EVR1 from each
- Regresses EVR1 vs ECI
- **Problem:** Result reverses without KNN imputation

**Benchmark Dimensions:** Explores the structure of benchmarks
- Uses global PCA (one for all)
- Projects models into PC space
- Identifies main dimensions
- **Problem:** Structure heavily influenced by imputation assumptions

**Both are interesting questions, but both need better handling of missing data.**
