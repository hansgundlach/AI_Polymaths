#!/usr/bin/env python
"""
Direct Visualization of SLODR - ECI vs Logit(Benchmark Scores)

Panel of scatter plots: ECI (x) vs logit-transformed benchmark score (y).
Uses logit transformation to handle ceiling effects near 0 and 1.

If SLODR is true, we should see:
- Strong linear relationship at low ECI (high correlation)
- Weaker, more scattered relationship at high ECI (low correlation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================================
# CONFIGURABLE BENCHMARK LIST
# Comment out any benchmark you want to exclude.
# The key is the CSV filename stem in benchmark_data/; the value is the
# display name used in plot titles.
# ============================================================================

BENCHMARKS = {
    # --- Math & Reasoning ---
    "gpqa_diamond": "GPQA Diamond",
    "frontiermath": "FrontierMath",
    "frontiermath_tier_4": "FrontierMath Tier 4",
    "otis_mock_aime_2024_2025": "OTIS Mock AIME 2024-2025",
    "math_level_5": "MATH Level 5",
    "bbh_external": "BIG-Bench Hard",
    "arc_agi_external": "ARC-AGI",
    "chess_puzzles": "Chess Puzzles",
    # --- Coding ---
    "aider_polyglot_external": "Aider Polyglot",
    "swe_bench_verified": "SWE-Bench Verified",
    "swe_bench_bash": "SWE-Bench Bash",
    "cybench_external": "Cybench",
    "terminalbench_external": "Terminal Bench",
    "webdev_arena_external": "WebDev Arena",
    "cad_eval_external": "CadEval",
    # --- Agents & Autonomy ---
    "os_world_external": "OSWorld",
    "the_agent_company_external": "The Agent Company",
    "metr_time_horizons_external": "METR Autonomy",
    # --- General Knowledge & QA ---
    "live_bench_external": "LiveBench",
    "simplebench_external": "SimpleBench",
    "simpleqa_verified": "SimpleQA Verified",
    "mmlu_external": "MMLU",
    "trivia_qa_external": "TriviaQA",
    "science_qa_external": "ScienceQA",
    "open_book_qa_external": "OpenBookQA",
    # --- NLI & Commonsense ---
    "hella_swag_external": "HellaSwag",
    "piqa_external": "PIQA",
    "wino_grande_external": "WinoGrande",
    "bool_q_external": "BoolQ",
    "common_sense_qa_2_external": "CommonsenseQA 2",
    "arc_ai2_external": "ARC (AI2)",
    "superglue_external": "SuperGLUE",
    "lambada_external": "LAMBADA",
    "adversarial_nli_external": "Adversarial NLI",
    "gsm8k_external": "GSM8K",
    # --- Creative & Multimodal ---
    "fictionlivebench_external": "Fiction.LiveBench",
    "lech_mazur_writing_external": "Lech Mazur Writing",
    "video_mme_external": "Video-MME",
    "deepresearchbench_external": "DeepResearch Bench",
    # --- Other ---
    "balrog_external": "Balrog",
    "weirdml_external": "WeirdML",
    "vpct_external": "VPCT",
    "geobench_external": "GeoBench",
    "gso_external": "GSO",
}

# Minimum number of models with both ECI and a benchmark score to include
# the benchmark in a plot. Set low to be inclusive.
MIN_MODELS_PER_BENCHMARK = 5

# ============================================================================
# TRANSFORMS
# ============================================================================

EPS = 1e-4


def logit(p, eps=EPS):
    p_clipped = np.clip(p, eps, 1 - eps)
    return np.log(p_clipped / (1 - p_clipped))


def expit(x):
    return 1 / (1 + np.exp(-x))


# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("DIRECT SLODR VISUALIZATION - ECI vs Logit(Benchmark Scores)")
print("=" * 80)
print()


SCORE_COL_PRIORITY = [
    "mean_score",
    "Best score (across scorers)",
    "Score",
    "Accuracy",
    "EM",
    "Correct",
    "Average",
    "Average progress",
    "Average score",
    "average_score",
    "Global average",
    "Mean score",
    "Accuracy mean",
    "Overall accuracy",
    "Percent correct",
    "Score (AVG@5)",
    "Score OPT@1",
    "% Resolved",
    "% Score",
    "ACW Avg Score",
    "Challenge score",
    "Unguided % Solved",
    "Overall pass (%)",
    "Arena Score",
    "120k token score",
    "Overall (no subtitles)",
]


def load_benchmark_data(bench_dir="benchmark_data", model_col="Model version"):
    bench_path = Path(bench_dir)
    benchmarks = {}
    csv_files = [
        f for f in bench_path.glob("*.csv") if "epoch_capabilities_index" not in f.name
    ]

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if model_col not in df.columns:
                continue

            score_column = None
            for candidate in SCORE_COL_PRIORITY:
                if candidate in df.columns:
                    score_column = candidate
                    break
            if score_column is None:
                continue

            df_clean = df[[model_col, score_column]].copy()
            df_clean.columns = ["model", "score"]
            df_clean = df_clean.dropna()
            df_clean["score"] = pd.to_numeric(df_clean["score"], errors="coerce")
            df_clean = df_clean.dropna()

            # Normalise percentage-style scores to [0,1].
            # Skip benchmarks on non-probability scales (e.g. Elo).
            smax = df_clean["score"].max()
            if smax > 100:
                continue
            if smax > 1.0:
                df_clean["score"] = df_clean["score"] / 100.0

            df_clean = df_clean.groupby("model")["score"].max().reset_index()

            if len(df_clean) > 0:
                benchmarks[csv_file.stem] = df_clean
        except Exception:
            pass
    return benchmarks


def load_eci_data():
    df = pd.read_csv("benchmark_data/epoch_capabilities_index.csv")
    df_eci = df[["Model version", "ECI Score"]].copy()
    df_eci.columns = ["model", "eci"]
    df_eci = df_eci.dropna()
    df_eci["eci"] = pd.to_numeric(df_eci["eci"], errors="coerce")
    return df_eci.dropna()


all_benchmarks = load_benchmark_data()
eci_data = load_eci_data()

# Create wide matrix — no model-count filter; every model with an ECI is kept
df_wide = eci_data.copy().set_index("model")
for bench_stem, bench_df in all_benchmarks.items():
    df_wide[bench_stem] = bench_df.set_index("model")["score"]
df_wide = df_wide.reset_index()

print(f"Loaded {len(all_benchmarks)} benchmark CSVs")
print(f"Total models with ECI: {len(df_wide)}")

# Resolve which requested benchmarks actually exist
bench_stems_to_plot = []
for stem, display in BENCHMARKS.items():
    if stem not in all_benchmarks:
        print(f"  [skip] {display} ({stem}.csv) — file not found or unreadable")
        continue
    n_valid = df_wide[["eci", stem]].dropna().shape[0]
    if n_valid < MIN_MODELS_PER_BENCHMARK:
        print(
            f"  [skip] {display} — only {n_valid} models (need {MIN_MODELS_PER_BENCHMARK})"
        )
        continue
    bench_stems_to_plot.append(stem)

print(f"\nPlotting {len(bench_stems_to_plot)} benchmarks")
print()

# ============================================================================
# VISUALIZATION 1: PANEL — ECI vs logit(score)
# ============================================================================

print("Creating panel plot: ECI vs logit(score) ...")

n_bench = len(bench_stems_to_plot)
n_cols = 4
n_rows = int(np.ceil(n_bench / n_cols))

fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.30)

for idx, stem in enumerate(bench_stems_to_plot):
    ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
    display = BENCHMARKS[stem]

    df_plot = df_wide[["eci", stem]].dropna().copy()
    df_plot["logit_score"] = logit(df_plot[stem].values)

    # Color by ECI quartile
    try:
        q_labels = ["Low", "Med-Low", "Med-High", "High"]
        df_plot["q"] = pd.qcut(df_plot["eci"], q=4, labels=q_labels, duplicates="drop")
    except ValueError:
        df_plot["q"] = "All"
        q_labels = ["All"]

    colors = {
        "Low": "#d62728",
        "Med-Low": "#ff7f0e",
        "Med-High": "#2ca02c",
        "High": "#1f77b4",
        "All": "#1f77b4",
    }

    for ql in q_labels:
        mask = df_plot["q"] == ql
        if mask.any():
            ax.scatter(
                df_plot.loc[mask, "eci"],
                df_plot.loc[mask, "logit_score"],
                alpha=0.6,
                s=35,
                c=colors.get(ql, "#1f77b4"),
                edgecolors="black",
                linewidth=0.4,
                label=ql if idx == 0 else None,
            )

    # Regression line
    if len(df_plot) >= 3:
        sl, ic, r, p, se = stats.linregress(df_plot["eci"], df_plot["logit_score"])
        xs = np.linspace(df_plot["eci"].min(), df_plot["eci"].max(), 50)
        ax.plot(xs, ic + sl * xs, "r--", linewidth=1.5)
        r_label = f"r={r:.2f}, p={p:.3f}"
    else:
        r_label = "n<3"

    ax.set_xlabel("ECI", fontsize=8)
    ax.set_ylabel("logit(score)", fontsize=8)
    ax.set_title(
        f"{display}\n{r_label}, n={len(df_plot)}", fontsize=9, fontweight="bold"
    )
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)

# Legend on first subplot only
if n_bench > 0:
    fig.axes[0].legend(fontsize=7, loc="best")

# Hide unused subplots
for idx in range(n_bench, n_rows * n_cols):
    fig.add_subplot(gs[idx // n_cols, idx % n_cols]).set_visible(False)

fig.suptitle(
    "ECI vs Logit(Benchmark Score)\nColors = ECI quartile",
    fontsize=14,
    fontweight="bold",
    y=1.005,
)

output_dir = Path("output_direct_slodr")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "all_benchmarks_panel_logit.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: output_direct_slodr/all_benchmarks_panel_logit.png")

# ============================================================================
# VISUALIZATION 2: SAME PANEL BUT RAW SCORES (for comparison)
# ============================================================================

print("Creating panel plot: ECI vs raw score ...")

fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.30)

for idx, stem in enumerate(bench_stems_to_plot):
    ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
    display = BENCHMARKS[stem]

    df_plot = df_wide[["eci", stem]].dropna().copy()

    try:
        q_labels = ["Low", "Med-Low", "Med-High", "High"]
        df_plot["q"] = pd.qcut(df_plot["eci"], q=4, labels=q_labels, duplicates="drop")
    except ValueError:
        df_plot["q"] = "All"
        q_labels = ["All"]

    for ql in q_labels:
        mask = df_plot["q"] == ql
        if mask.any():
            ax.scatter(
                df_plot.loc[mask, "eci"],
                df_plot.loc[mask, stem],
                alpha=0.6,
                s=35,
                c=colors.get(ql, "#1f77b4"),
                edgecolors="black",
                linewidth=0.4,
                label=ql if idx == 0 else None,
            )

    if len(df_plot) >= 3:
        sl, ic, r, p, se = stats.linregress(df_plot["eci"], df_plot[stem])
        xs = np.linspace(df_plot["eci"].min(), df_plot["eci"].max(), 50)
        ax.plot(xs, ic + sl * xs, "r--", linewidth=1.5)
        r_label = f"r={r:.2f}, p={p:.3f}"
    else:
        r_label = "n<3"

    ax.set_xlabel("ECI", fontsize=8)
    ax.set_ylabel("Score", fontsize=8)
    ax.set_title(
        f"{display}\n{r_label}, n={len(df_plot)}", fontsize=9, fontweight="bold"
    )
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)

if n_bench > 0:
    fig.axes[0].legend(fontsize=7, loc="best")

for idx in range(n_bench, n_rows * n_cols):
    fig.add_subplot(gs[idx // n_cols, idx % n_cols]).set_visible(False)

fig.suptitle(
    "ECI vs Raw Benchmark Score\nColors = ECI quartile",
    fontsize=14,
    fontweight="bold",
    y=1.005,
)

plt.savefig(output_dir / "all_benchmarks_panel.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: output_direct_slodr/all_benchmarks_panel.png")

# ============================================================================
# VISUALIZATION 3: CORRELATION DECAY — binned correlations
# ============================================================================

print("Creating correlation decay analysis ...")


def compute_binned_correlations(df, benchmark_col, n_bins=4, use_logit=True):
    df_valid = df[["eci", benchmark_col]].dropna().copy()
    if len(df_valid) < 8:
        return None

    if use_logit:
        df_valid["y"] = logit(df_valid[benchmark_col].values)
    else:
        df_valid["y"] = df_valid[benchmark_col]

    df_valid = df_valid.sort_values("eci")
    bin_edges = np.percentile(df_valid["eci"], np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6
    df_valid["bin"] = pd.cut(
        df_valid["eci"], bins=bin_edges, labels=range(n_bins), include_lowest=True
    )

    results = []
    for b in range(n_bins):
        bd = df_valid[df_valid["bin"] == b]
        if len(bd) >= 3:
            corr, p_val = stats.pearsonr(bd["eci"], bd["y"])
            results.append(
                {
                    "bin": b,
                    "mean_eci": bd["eci"].mean(),
                    "correlation": corr,
                    "p_value": p_val,
                    "n": len(bd),
                }
            )
    return pd.DataFrame(results) if results else None


all_correlations = {}
for stem in bench_stems_to_plot:
    corr_df = compute_binned_correlations(df_wide, stem, n_bins=4, use_logit=True)
    if corr_df is not None:
        all_correlations[stem] = corr_df

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes = axes.flatten()

# Panel 1: per-benchmark correlation traces
ax = axes[0]
for stem, corr_df in all_correlations.items():
    if len(corr_df) >= 3:
        ax.plot(
            corr_df["mean_eci"],
            corr_df["correlation"],
            marker="o",
            alpha=0.7,
            linewidth=2,
            label=BENCHMARKS.get(stem, stem)[:22],
        )
ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Mean ECI", fontsize=11, fontweight="bold")
ax.set_ylabel("Correlation (ECI vs logit(score))", fontsize=11, fontweight="bold")
ax.set_title(
    "Correlation by ECI Level per Benchmark\n(SLODR: should decrease)",
    fontsize=12,
    fontweight="bold",
)
ax.legend(fontsize=6, loc="best", ncol=2)
ax.grid(True, alpha=0.3)

# Panel 2: average correlation trend
ax = axes[1]
if all_correlations:
    all_bins = []
    for stem, corr_df in all_correlations.items():
        for _, row in corr_df.iterrows():
            all_bins.append(
                {
                    "mean_eci": row["mean_eci"],
                    "correlation": row["correlation"],
                    "benchmark": stem,
                }
            )
    df_all_bins = pd.DataFrame(all_bins)

    eci_bins = pd.cut(df_all_bins["mean_eci"], bins=4)
    grouped = df_all_bins.groupby(eci_bins)["correlation"].agg(["mean", "std", "count"])
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
    bin_centers = [iv.mid for iv in grouped.index]

    ax.errorbar(
        bin_centers,
        grouped["mean"],
        yerr=grouped["sem"],
        marker="o",
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        color="steelblue",
        ecolor="steelblue",
        alpha=0.8,
    )

    mask_valid = ~np.isnan(grouped["mean"])
    if mask_valid.sum() >= 2:
        sl, ic, rv, pv, se = stats.linregress(
            np.array(bin_centers)[mask_valid], grouped["mean"].values[mask_valid]
        )
        xline = np.linspace(min(bin_centers), max(bin_centers), 100)
        ax.plot(
            xline,
            ic + sl * xline,
            "r--",
            linewidth=2,
            alpha=0.7,
            label=f"slope={sl:.5f}, p={pv:.3f}",
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("ECI Level", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Correlation", fontsize=11, fontweight="bold")
    ax.set_title(
        "Average Correlation Across Benchmarks\n(SLODR predicts negative slope)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

# Panel 3: box plots by quartile
ax = axes[2]
if all_correlations:
    df_all_bins["eci_q"] = pd.qcut(
        df_all_bins["mean_eci"], q=4, labels=["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"]
    )
    data_by_q = [
        df_all_bins[df_all_bins["eci_q"] == q]["correlation"].values
        for q in ["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"]
    ]
    bp = ax.boxplot(
        data_by_q,
        positions=[1, 2, 3, 4],
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
        meanprops=dict(marker="D", markerfacecolor="green", markersize=8),
    )
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"])
    ax.set_xlabel("ECI Quartile", fontsize=11, fontweight="bold")
    ax.set_ylabel("Correlation", fontsize=11, fontweight="bold")
    ax.set_title(
        "Correlation Distributions by ECI Level\n(SLODR: Q4 < Q1)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    for i, d in enumerate(data_by_q):
        ax.text(i + 1, ax.get_ylim()[0] + 0.05, f"n={len(d)}", ha="center", fontsize=9)

# Panel 4: residual variance by quartile
ax = axes[3]
variance_data = []
for stem in bench_stems_to_plot:
    df_plot = df_wide[["eci", stem]].dropna().copy()
    if len(df_plot) < 8:
        continue
    df_plot["y"] = logit(df_plot[stem].values)
    try:
        df_plot["q"] = pd.qcut(
            df_plot["eci"], q=4, labels=[1, 2, 3, 4], duplicates="drop"
        )
    except ValueError:
        continue
    for q in [1, 2, 3, 4]:
        qd = df_plot[df_plot["q"] == q]
        if len(qd) >= 3:
            if len(qd) >= 5:
                sl, ic = np.polyfit(qd["eci"], qd["y"], 1)
                resid = qd["y"] - (sl * qd["eci"] + ic)
                var = resid.var()
            else:
                var = qd["y"].var()
            variance_data.append(
                {
                    "quartile": q,
                    "variance": var,
                    "benchmark": stem,
                    "mean_eci": qd["eci"].mean(),
                }
            )

if variance_data:
    df_var = pd.DataFrame(variance_data)
    gv = df_var.groupby("quartile")["variance"].agg(["mean", "std", "count"])
    gv["sem"] = gv["std"] / np.sqrt(gv["count"])
    ax.errorbar(
        gv.index,
        gv["mean"],
        yerr=gv["sem"],
        marker="s",
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        color="coral",
        ecolor="coral",
        alpha=0.8,
    )
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"])
    ax.set_xlabel("ECI Quartile", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Residual Variance", fontsize=11, fontweight="bold")
    ax.set_title(
        "Residual Variance by ECI Level\n(SLODR: should increase at high ECI)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    output_dir / "correlation_decay_analysis_logit.png", dpi=300, bbox_inches="tight"
)
plt.close()
print("  Saved: output_direct_slodr/correlation_decay_analysis_logit.png")

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"Benchmarks plotted: {len(bench_stems_to_plot)}")
print(f"Models in dataset:  {len(df_wide)}")
print()

if all_correlations:
    all_bin_corrs = []
    for stem, corr_df in all_correlations.items():
        for _, row in corr_df.iterrows():
            all_bin_corrs.append(
                {"mean_eci": row["mean_eci"], "correlation": row["correlation"]}
            )
    df_trend = pd.DataFrame(all_bin_corrs)
    sl, ic, rv, pv, se = stats.linregress(df_trend["mean_eci"], df_trend["correlation"])
    print(f"Overall correlation-vs-ECI trend (logit scores):")
    print(f"  Slope: {sl:.6f}   R²: {rv**2:.4f}   p: {pv:.4f}")
    if sl < 0:
        print(f"  -> NEGATIVE slope: SUPPORTS SLODR")
    else:
        print(f"  -> POSITIVE slope: CONTRADICTS SLODR")
    print()

    df_trend["q"] = pd.qcut(df_trend["mean_eci"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    q1 = df_trend[df_trend["q"] == "Q1"]["correlation"].mean()
    q4 = df_trend[df_trend["q"] == "Q4"]["correlation"].mean()
    print(f"  Q1 (lowest ECI)  mean corr: {q1:.3f}")
    print(f"  Q4 (highest ECI) mean corr: {q4:.3f}")
    print(f"  Δ (Q4 − Q1): {q4 - q1:+.3f}")

print()
print("=" * 80)
print("DONE")
print("=" * 80)
print()
print("Generated:")
print("  1. output_direct_slodr/all_benchmarks_panel_logit.png")
print("  2. output_direct_slodr/all_benchmarks_panel.png")
print("  3. output_direct_slodr/correlation_decay_analysis_logit.png")
print()
