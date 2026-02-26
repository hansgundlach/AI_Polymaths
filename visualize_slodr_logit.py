#!/usr/bin/env python
"""
Direct Visualization of SLODR - ECI vs LOGIT-TRANSFORMED Benchmark Scores

This version uses logit transformation on benchmark scores to handle ceiling effects,
matching the original analysis methodology.

logit(p) = log(p / (1-p))

This expands scores near 0 and 1, making relationships more linear.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("DIRECT SLODR VISUALIZATION - ECI vs LOGIT-TRANSFORMED Benchmark Scores")
print("="*80)
print()

# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================

def logit(p, eps=1e-4):
    """Apply logit transformation to handle ceiling effects."""
    p_clipped = np.clip(p, eps, 1 - eps)
    return np.log(p_clipped / (1 - p_clipped))

def expit(x):
    """Inverse logit (for reference)."""
    return 1 / (1 + np.exp(-x))

# ============================================================================
# LOAD DATA
# ============================================================================

def load_benchmark_data(bench_dir='benchmark_data', score_col='mean_score', model_col='Model version'):
    bench_path = Path(bench_dir)
    benchmarks = {}
    csv_files = [f for f in bench_path.glob('*.csv') if 'epoch_capabilities_index' not in f.name]

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if model_col not in df.columns:
                continue
            if score_col in df.columns:
                score_column = score_col
            elif 'Best score (across scorers)' in df.columns:
                score_column = 'Best score (across scorers)'
            else:
                continue

            df_clean = df[[model_col, score_column]].copy()
            df_clean.columns = ['model', 'score']
            df_clean = df_clean.dropna()
            df_clean['score'] = pd.to_numeric(df_clean['score'], errors='coerce')
            df_clean = df_clean.dropna()
            df_clean = df_clean.groupby('model')['score'].max().reset_index()

            if len(df_clean) > 0:
                benchmarks[csv_file.stem] = df_clean
        except:
            pass
    return benchmarks

def load_eci_data():
    df = pd.read_csv('benchmark_data/epoch_capabilities_index.csv')
    df_eci = df[['Model version', 'ECI Score']].copy()
    df_eci.columns = ['model', 'eci']
    df_eci = df_eci.dropna()
    df_eci['eci'] = pd.to_numeric(df_eci['eci'], errors='coerce')
    return df_eci.dropna()

benchmarks = load_benchmark_data()
eci_data = load_eci_data()

# Create wide matrix with RAW scores
df_wide = eci_data.copy().set_index('model')
for bench_name, bench_df in benchmarks.items():
    df_wide[bench_name] = bench_df.set_index('model')['score']
df_wide = df_wide.reset_index()

benchmark_cols = [col for col in df_wide.columns if col not in ['model', 'eci']]

# Apply logit transformation to all benchmark columns
print("Applying logit transformation to benchmark scores...")
df_wide_logit = df_wide.copy()
for col in benchmark_cols:
    df_wide_logit[col] = logit(df_wide[col].values)
print()

print(f"Loaded {len(benchmarks)} benchmarks")
print(f"Total models: {len(df_wide)}")
print()

# ============================================================================
# COMPUTE CORRELATIONS BY ECI LEVEL
# ============================================================================

def compute_binned_correlations(df, benchmark_col, n_bins=4):
    """Compute correlations between ECI and benchmark in bins."""
    df_valid = df[['eci', benchmark_col]].dropna()

    if len(df_valid) < 10:
        return None

    # Create bins
    df_valid = df_valid.sort_values('eci')
    bin_edges = np.percentile(df_valid['eci'], np.linspace(0, 100, n_bins + 1))
    df_valid['bin'] = pd.cut(df_valid['eci'], bins=bin_edges, labels=range(n_bins), include_lowest=True)

    results = []
    for bin_idx in range(n_bins):
        bin_data = df_valid[df_valid['bin'] == bin_idx]

        if len(bin_data) >= 3:
            corr, p_val = stats.pearsonr(bin_data['eci'], bin_data[benchmark_col])
            results.append({
                'bin': bin_idx,
                'mean_eci': bin_data['eci'].mean(),
                'correlation': corr,
                'p_value': p_val,
                'n': len(bin_data)
            })

    return pd.DataFrame(results) if results else None

# Compute correlations for all benchmarks (LOGIT-transformed)
all_correlations_logit = {}
for bench in benchmark_cols:
    corr_df = compute_binned_correlations(df_wide_logit, bench, n_bins=4)
    if corr_df is not None:
        all_correlations_logit[bench] = corr_df

# Also compute for RAW scores for comparison
all_correlations_raw = {}
for bench in benchmark_cols:
    corr_df = compute_binned_correlations(df_wide, bench, n_bins=4)
    if corr_df is not None:
        all_correlations_raw[bench] = corr_df

print("Computed correlations by ECI level (both logit and raw scores)")
print()

# ============================================================================
# VISUALIZATION 1: PANEL PLOT - ALL BENCHMARKS (LOGIT)
# ============================================================================

print("Creating panel plot of all benchmarks (logit-transformed)...")

n_benchmarks = len(benchmark_cols)
n_cols = 3
n_rows = int(np.ceil(n_benchmarks / n_cols))

fig = plt.figure(figsize=(16, 4*n_rows))
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25)

for idx, bench_name in enumerate(benchmark_cols):
    row = idx // n_cols
    col = idx % n_cols
    ax = fig.add_subplot(gs[row, col])

    # Get data (LOGIT)
    df_plot = df_wide_logit[['eci', bench_name]].dropna()

    if len(df_plot) < 3:
        ax.text(0.5, 0.5, f'{bench_name}\n\nInsufficient data',
                ha='center', va='center', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        continue

    # Define ECI bins for coloring
    eci_bins = pd.qcut(df_plot['eci'], q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'], duplicates='drop')
    colors = {'Low': '#d62728', 'Med-Low': '#ff7f0e', 'Med-High': '#2ca02c', 'High': '#1f77b4'}

    # Scatter plot
    for bin_name in ['Low', 'Med-Low', 'Med-High', 'High']:
        if bin_name in eci_bins.values:
            mask = (eci_bins == bin_name)
            ax.scatter(df_plot.loc[mask, 'eci'], df_plot.loc[mask, bench_name],
                      alpha=0.6, s=40, c=colors[bin_name], label=bin_name, edgecolors='black', linewidth=0.5)

    # Add smoothed trend line
    if len(df_plot) >= 10:
        try:
            df_sorted = df_plot.sort_values('eci')
            window = max(5, len(df_plot) // 10)
            eci_smooth = df_sorted['eci'].rolling(window=window, center=True, min_periods=3).mean()
            score_smooth = df_sorted[bench_name].rolling(window=window, center=True, min_periods=3).mean()

            mask_valid = ~eci_smooth.isna() & ~score_smooth.isna()
            ax.plot(eci_smooth[mask_valid], score_smooth[mask_valid], 'k-', linewidth=2, alpha=0.5, label='Trend')
        except:
            pass

    # Overall correlation
    corr_all, p_all = stats.pearsonr(df_plot['eci'], df_plot[bench_name])

    # Formatting
    ax.set_xlabel('ECI', fontsize=9)
    ax.set_ylabel('Logit(Score)', fontsize=9)

    # Shorten benchmark name for title
    short_name = bench_name.replace('_', ' ').title()
    if len(short_name) > 30:
        short_name = short_name[:27] + '...'

    ax.set_title(f'{short_name}\nr={corr_all:.3f}, n={len(df_plot)}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if idx == 0:
        ax.legend(fontsize=8, loc='lower right')

plt.suptitle('Direct SLODR Test: ECI vs Logit(Benchmark Scores)\n(If SLODR: scatter should increase at high ECI)',
             fontsize=14, fontweight='bold', y=0.995)

output_dir = Path('output_direct_slodr')
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'all_benchmarks_panel_logit.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: output_direct_slodr/all_benchmarks_panel_logit.png")
print()

# ============================================================================
# VISUALIZATION 2: CORRELATION DECAY PLOTS (LOGIT)
# ============================================================================

print("Creating correlation decay plots (logit-transformed)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Plot 1: Correlations by ECI level for each benchmark (LOGIT)
ax = axes[0]
for bench_name, corr_df in all_correlations_logit.items():
    if len(corr_df) >= 3:
        short_name = bench_name.replace('_', ' ')[:20]
        ax.plot(corr_df['mean_eci'], corr_df['correlation'],
               marker='o', alpha=0.7, linewidth=2, label=short_name)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Mean ECI', fontsize=11, fontweight='bold')
ax.set_ylabel('Correlation (ECI vs Logit-Benchmark)', fontsize=11, fontweight='bold')
ax.set_title('SLODR Test: Correlation by ECI Level (Logit Transform)\n(Should decrease if SLODR is true)',
            fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='best', ncol=2)
ax.grid(True, alpha=0.3)

# Plot 2: Average correlation trend (LOGIT)
ax = axes[1]
if len(all_correlations_logit) > 0:
    all_bins = []
    for bench_name, corr_df in all_correlations_logit.items():
        for _, row in corr_df.iterrows():
            all_bins.append({
                'mean_eci': row['mean_eci'],
                'correlation': row['correlation'],
                'benchmark': bench_name
            })

    df_all_bins = pd.DataFrame(all_bins)
    eci_bins = pd.cut(df_all_bins['mean_eci'], bins=4)
    grouped = df_all_bins.groupby(eci_bins)['correlation'].agg(['mean', 'std', 'count'])
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
    bin_centers = [interval.mid for interval in grouped.index]

    ax.errorbar(bin_centers, grouped['mean'], yerr=grouped['sem'],
               marker='o', markersize=10, linewidth=2, capsize=5, capthick=2,
               color='steelblue', ecolor='steelblue', alpha=0.8, label='Logit Transform')

    # Fit linear trend
    mask_valid = ~np.isnan(grouped['mean'])
    if mask_valid.sum() >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.array(bin_centers)[mask_valid],
            grouped['mean'].values[mask_valid]
        )
        x_line = np.linspace(min(bin_centers), max(bin_centers), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7,
               label=f'Trend: slope={slope:.5f}\n(p={p_value:.3f})')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('ECI Level', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Correlation', fontsize=11, fontweight='bold')
    ax.set_title('Average Correlation (Logit Transform)\n(SLODR predicts negative slope)',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

# Plot 3: Comparison - Logit vs Raw
ax = axes[2]
if len(all_correlations_logit) > 0 and len(all_correlations_raw) > 0:
    # Logit
    all_bins_logit = []
    for bench_name, corr_df in all_correlations_logit.items():
        for _, row in corr_df.iterrows():
            all_bins_logit.append({'mean_eci': row['mean_eci'], 'correlation': row['correlation']})
    df_logit = pd.DataFrame(all_bins_logit)
    df_logit['eci_quartile'] = pd.qcut(df_logit['mean_eci'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Raw
    all_bins_raw = []
    for bench_name, corr_df in all_correlations_raw.items():
        for _, row in corr_df.iterrows():
            all_bins_raw.append({'mean_eci': row['mean_eci'], 'correlation': row['correlation']})
    df_raw = pd.DataFrame(all_bins_raw)
    df_raw['eci_quartile'] = pd.qcut(df_raw['mean_eci'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Means by quartile
    means_logit = df_logit.groupby('eci_quartile')['correlation'].mean()
    means_raw = df_raw.groupby('eci_quartile')['correlation'].mean()

    x = np.arange(4)
    width = 0.35

    ax.bar(x - width/2, means_logit, width, label='Logit Transform', alpha=0.7, color='steelblue')
    ax.bar(x + width/2, means_raw, width, label='Raw Scores', alpha=0.7, color='coral')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])
    ax.set_xlabel('ECI Quartile', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Correlation', fontsize=11, fontweight='bold')
    ax.set_title('Comparison: Logit vs Raw Scores\n(Does transformation affect SLODR pattern?)',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Distribution of correlations by quartile (LOGIT)
ax = axes[3]
if len(all_correlations_logit) > 0:
    df_all_bins['eci_quartile'] = pd.qcut(df_all_bins['mean_eci'], q=4,
                                           labels=['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4\n(Highest)'])

    positions = [1, 2, 3, 4]
    data_by_quartile = [df_all_bins[df_all_bins['eci_quartile'] == q]['correlation'].values
                        for q in ['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4\n(Highest)']]

    bp = ax.boxplot(data_by_quartile, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(marker='D', markerfacecolor='green', markersize=8))

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4\n(Highest)'])
    ax.set_xlabel('ECI Quartile', fontsize=11, fontweight='bold')
    ax.set_ylabel('Correlation Distribution (Logit)', fontsize=11, fontweight='bold')
    ax.set_title('Correlation Distributions (Logit Transform)\n(SLODR: Q4 should be lower than Q1)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for i, data in enumerate(data_by_quartile):
        ax.text(i+1, ax.get_ylim()[0] + 0.05, f'n={len(data)}',
               ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'correlation_decay_analysis_logit.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: output_direct_slodr/correlation_decay_analysis_logit.png")
print()

# ============================================================================
# VISUALIZATION 3: DETAILED PLOTS (LOGIT)
# ============================================================================

print("Creating detailed plots for high-coverage benchmarks (logit)...")

coverage = {col: df_wide[col].notna().mean() for col in benchmark_cols}
high_cov_benchmarks = sorted([(cov, col) for col, cov in coverage.items() if cov >= 0.25], reverse=True)[:6]

if len(high_cov_benchmarks) > 0:
    n_plots = len(high_cov_benchmarks)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (cov, bench_name) in enumerate(high_cov_benchmarks):
        ax = axes[idx]

        df_plot = df_wide_logit[['eci', bench_name]].dropna()

        df_plot['eci_quartile'] = pd.qcut(df_plot['eci'], q=4,
                                          labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
                                          duplicates='drop')

        colors_q = {'Q1 (Low)': '#d62728', 'Q2': '#ff7f0e', 'Q3': '#2ca02c', 'Q4 (High)': '#1f77b4'}

        for q_name in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
            if q_name in df_plot['eci_quartile'].values:
                q_data = df_plot[df_plot['eci_quartile'] == q_name]
                ax.scatter(q_data['eci'], q_data[bench_name],
                          alpha=0.6, s=60, c=colors_q[q_name], label=q_name,
                          edgecolors='black', linewidth=0.5)

                if len(q_data) >= 3:
                    slope, intercept = np.polyfit(q_data['eci'], q_data[bench_name], 1)
                    x_line = np.array([q_data['eci'].min(), q_data['eci'].max()])
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color=colors_q[q_name],
                           linestyle='--', linewidth=2, alpha=0.7)

        corr_all, p_all = stats.pearsonr(df_plot['eci'], df_plot[bench_name])

        corr_text = "Correlations by quartile:\n"
        for q_name in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
            if q_name in df_plot['eci_quartile'].values:
                q_data = df_plot[df_plot['eci_quartile'] == q_name]
                if len(q_data) >= 3:
                    corr_q, _ = stats.pearsonr(q_data['eci'], q_data[bench_name])
                    corr_text += f"{q_name}: r={corr_q:.3f}\n"

        ax.text(0.02, 0.98, corr_text.strip(), transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('ECI', fontsize=10, fontweight='bold')
        ax.set_ylabel('Logit(Score)', fontsize=10, fontweight='bold')

        short_name = bench_name.replace('_', ' ').title()
        ax.set_title(f'{short_name}\nOverall: r={corr_all:.3f}, cov={cov*100:.0f}%, n={len(df_plot)}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    for idx in range(len(high_cov_benchmarks), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Detailed SLODR Analysis: Logit-Transformed Scores\n(If SLODR: correlations should weaken in Q4)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_high_coverage_logit.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Saved: output_direct_slodr/detailed_high_coverage_logit.png")

print()

# ============================================================================
# SUMMARY STATISTICS - LOGIT vs RAW
# ============================================================================

print("="*80)
print("SUMMARY STATISTICS - LOGIT vs RAW COMPARISON")
print("="*80)
print()

# LOGIT stats
if len(all_correlations_logit) > 0:
    all_bin_corrs_logit = []
    for bench_name, corr_df in all_correlations_logit.items():
        for _, row in corr_df.iterrows():
            all_bin_corrs_logit.append({
                'mean_eci': row['mean_eci'],
                'correlation': row['correlation']
            })

    df_trend_logit = pd.DataFrame(all_bin_corrs_logit)
    slope_logit, intercept_logit, r_value_logit, p_value_logit, std_err_logit = stats.linregress(
        df_trend_logit['mean_eci'], df_trend_logit['correlation']
    )

    df_trend_logit['quartile'] = pd.qcut(df_trend_logit['mean_eci'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    q1_mean_logit = df_trend_logit[df_trend_logit['quartile'] == 'Q1']['correlation'].mean()
    q4_mean_logit = df_trend_logit[df_trend_logit['quartile'] == 'Q4']['correlation'].mean()

# RAW stats
if len(all_correlations_raw) > 0:
    all_bin_corrs_raw = []
    for bench_name, corr_df in all_correlations_raw.items():
        for _, row in corr_df.iterrows():
            all_bin_corrs_raw.append({
                'mean_eci': row['mean_eci'],
                'correlation': row['correlation']
            })

    df_trend_raw = pd.DataFrame(all_bin_corrs_raw)
    slope_raw, intercept_raw, r_value_raw, p_value_raw, std_err_raw = stats.linregress(
        df_trend_raw['mean_eci'], df_trend_raw['correlation']
    )

    df_trend_raw['quartile'] = pd.qcut(df_trend_raw['mean_eci'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    q1_mean_raw = df_trend_raw[df_trend_raw['quartile'] == 'Q1']['correlation'].mean()
    q4_mean_raw = df_trend_raw[df_trend_raw['quartile'] == 'Q4']['correlation'].mean()

print("LOGIT-TRANSFORMED SCORES:")
print(f"  Slope: {slope_logit:.6f}")
print(f"  R²: {r_value_logit**2:.4f}")
print(f"  P-value: {p_value_logit:.4f}")
print(f"  Q1 mean correlation: {q1_mean_logit:.3f}")
print(f"  Q4 mean correlation: {q4_mean_logit:.3f}")
print(f"  Difference (Q4 - Q1): {q4_mean_logit - q1_mean_logit:.3f}")
print()

print("RAW SCORES:")
print(f"  Slope: {slope_raw:.6f}")
print(f"  R²: {r_value_raw**2:.4f}")
print(f"  P-value: {p_value_raw:.4f}")
print(f"  Q1 mean correlation: {q1_mean_raw:.3f}")
print(f"  Q4 mean correlation: {q4_mean_raw:.3f}")
print(f"  Difference (Q4 - Q1): {q4_mean_raw - q1_mean_raw:.3f}")
print()

print("COMPARISON:")
print(f"  Slope change (logit - raw): {slope_logit - slope_raw:.6f}")
print(f"  R² change: {r_value_logit**2 - r_value_raw**2:.4f}")
print(f"  P-value change: {p_value_logit - p_value_raw:.4f}")
print()

if abs(slope_logit) > abs(slope_raw):
    print(f"  ⚠ Logit transformation STRENGTHENS SLODR signal")
    print(f"    (More negative slope with logit)")
else:
    print(f"  ⚠ Logit transformation WEAKENS SLODR signal")
    print(f"    (Less negative slope with logit)")
print()

if p_value_logit < p_value_raw:
    print(f"  ⚠ Logit transformation makes result MORE significant")
else:
    print(f"  ⚠ Logit transformation makes result LESS significant")

print()

# Save comparison summary
summary_df = pd.DataFrame({
    'metric': ['slope', 'r_squared', 'p_value', 'q1_correlation', 'q4_correlation', 'q4_minus_q1'],
    'logit': [slope_logit, r_value_logit**2, p_value_logit, q1_mean_logit, q4_mean_logit, q4_mean_logit - q1_mean_logit],
    'raw': [slope_raw, r_value_raw**2, p_value_raw, q1_mean_raw, q4_mean_raw, q4_mean_raw - q1_mean_raw],
    'difference': [slope_logit - slope_raw, r_value_logit**2 - r_value_raw**2, p_value_logit - p_value_raw,
                   q1_mean_logit - q1_mean_raw, q4_mean_logit - q4_mean_raw,
                   (q4_mean_logit - q1_mean_logit) - (q4_mean_raw - q1_mean_raw)]
})
summary_df.to_csv(output_dir / 'logit_vs_raw_comparison.csv', index=False)
print("✓ Saved: output_direct_slodr/logit_vs_raw_comparison.csv")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
