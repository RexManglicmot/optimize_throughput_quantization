# app/plot_results.py
"""
Generate plots & a summary table from benchmark results.

- Reads CSV path from config.yaml:
    cfg.paths.results_csv  (e.g., outputs/results.csv or outputs/results_quant.csv)
- Saves figures to: cfg.paths.plots_dir or 'outputs/figures'
- Writes summary tables: summary_table.csv and summary_table.md

Run:
    python -m app.plot_results
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from app.config import cfg


# --------------------------- THEME ---------------------------

def _apply_theme(name: str = "pro"):
    """
    Global color scheme + rcParams (no grids, subtle spines).
    Options: 'pro', 'pastel', 'colorblind'
    """
    if name == "pastel":
        colors = ["#AEC6CF", "#FFB3BA", "#C2C2F0", "#B4E6A8", "#FFD1BA", "#F1E3A0"]
    elif name == "colorblind":
        colors = ["#000000", "#E69F00", "#56B4E9", "#009E73",
                  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    else:
        colors = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]

    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=colors),

        # Background + text
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#AAAAAA",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",

        # No grid lines
        "axes.grid": False,
        "axes.axisbelow": True,

        # Line defaults
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,

        # Fonts
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,

        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#AAAAAA",
    })


# --------------------------- HELPERS ---------------------------

def _label_bars(ax, bars, fmt="%.1f", pad=2, size=9):
    """Add numeric labels on top of bars."""
    try:
        ax.bar_label(bars, fmt=fmt, padding=pad, fontsize=size)
    except Exception:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2.0, h, (fmt % h),
                    ha="center", va="bottom", fontsize=size, clip_on=True)

BAR_KW = dict(edgecolor="black", linewidth=0.8)


def _resolve_paths() -> Tuple[Path, Path]:
    csv_path = Path(getattr(cfg.paths, "results_csv", "outputs/results.csv"))
    base_plots = Path(getattr(cfg.paths, "plots_dir", "outputs"))
    figs_dir = base_plots / "figures" if base_plots.name != "figures" else base_plots
    figs_dir.mkdir(parents=True, exist_ok=True)
    return csv_path, figs_dir


def _load(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    defaults = {
        "run_id": None,
        "precision": None, "batch_size": None,
        "throughput_qps": np.nan, "tokens_per_sec": np.nan,
        "latency_ms_avg": np.nan, "latency_ms_p50": np.nan, "latency_ms_p95": np.nan,
        "vram_gb_peak": np.nan, "elapsed_s": np.nan,
        "speedup_vs_fp32": np.nan, "batch_efficiency": np.nan,
    }
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v

    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce").astype("Int64")
    for c in (
        "throughput_qps","tokens_per_sec","latency_ms_avg","latency_ms_p50",
        "latency_ms_p95","vram_gb_peak","elapsed_s","speedup_vs_fp32","batch_efficiency"
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask_fp32 = (df["precision"].astype(str).str.lower() == "fp32")
    df.loc[mask_fp32 & df["speedup_vs_fp32"].isna(), "speedup_vs_fp32"] = 1.0
    df.loc[(df["batch_size"] == 1) & df["batch_efficiency"].isna(), "batch_efficiency"] = 1.0
    df.loc[df["latency_ms_p50"].isna(), "latency_ms_p50"] = df["latency_ms_avg"]

    order_prec = ["fp32","fp16","int8","int4"]
    df["precision"] = df["precision"].astype(str)
    df["precision"] = pd.Categorical(df["precision"], categories=order_prec, ordered=True)

    return df.sort_values(["precision","batch_size"], kind="stable").reset_index(drop=True)


def _sla_threshold_ms() -> Optional[float]:
    try:
        return float(getattr(cfg.sla, "ms_p95", None))
    except Exception:
        return None


def _savefig(fig, figs_dir: Path, name: str):
    path = figs_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _legend_outside(ax):
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)


# --------------------------- PLOTS ---------------------------

def plot_throughput_vs_precision(df: pd.DataFrame, figs_dir: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    precisions = [p for p in df["precision"].cat.categories if p in df["precision"].unique().tolist()]
    bszs = sorted(df["batch_size"].dropna().unique())
    x = np.arange(len(precisions))
    width = min(0.8 / max(1, len(bszs)), 0.25)

    for i, b in enumerate(bszs):
        sub = df[df["batch_size"] == b]
        y = [sub[sub["precision"] == p]["throughput_qps"].mean() for p in precisions]
        bars = ax.bar(x + (i - (len(bszs)-1)/2)*width, y, width=width,
                      label=f"bsz={int(b)}", **BAR_KW)
        _label_bars(ax, bars)

    ax.set_xticks(x); ax.set_xticklabels(precisions)
    ax.set_ylabel("Throughput (QPS)")
    ax.set_title("Throughput vs Precision")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "throughput_vs_precision")


def plot_latency_vs_batch_avg(df: pd.DataFrame, figs_dir: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    for p, sub in df.groupby("precision", observed=True):
        if pd.isna(p): 
            continue
        sub = sub.sort_values("batch_size")
        ax.plot(sub["batch_size"], sub["latency_ms_avg"], marker="o", label=f"{p}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms, avg)")
    ax.set_title("Latency (avg) vs Batch Size")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "latency_vs_batch_avg")


def plot_vram_vs_precision(df: pd.DataFrame, figs_dir: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    precisions = [p for p in df["precision"].cat.categories if p in df["precision"].unique().tolist()]
    bszs = sorted(df["batch_size"].dropna().unique())
    x = np.arange(len(precisions))
    width = min(0.8 / max(1, len(bszs)), 0.25)

    for i, b in enumerate(bszs):
        sub = df[df["batch_size"] == b]
        y = [sub[sub["precision"] == p]["vram_gb_peak"].mean() for p in precisions]
        bars = ax.bar(x + (i - (len(bszs)-1)/2)*width, y, width=width,
                      label=f"bsz={int(b)}", **BAR_KW)
        _label_bars(ax, bars)

    ax.set_xticks(x); ax.set_xticklabels(precisions)
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("VRAM vs Precision")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "vram_vs_precision")


def plot_memory_usage_vs_precision(df: pd.DataFrame, figs_dir: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    precisions = [p for p in df["precision"].cat.categories if p in df["precision"].unique().tolist()]
    y = [df[df["precision"] == p]["vram_gb_peak"].mean() for p in precisions]
    bars = ax.bar(precisions, y, **BAR_KW)
    _label_bars(ax, bars)
    ax.set_ylabel("Memory (GB, peak VRAM)")
    ax.set_title("Memory Usage vs Precision")
    _savefig(fig, figs_dir, "memory_usage_vs_precision")


def plot_latency_grouped_p50_p95(df: pd.DataFrame, figs_dir: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    precisions = [p for p in df["precision"].cat.categories if p in df["precision"].unique().tolist()]
    p50_means = [df[df["precision"] == p]["latency_ms_p50"].mean() for p in precisions]
    p95_means = [df[df["precision"] == p]["latency_ms_p95"].mean() for p in precisions]

    x = np.arange(len(precisions))
    width = 0.38
    bars1 = ax.bar(x - width/2, p50_means, width=width, label="p50", **BAR_KW)
    bars2 = ax.bar(x + width/2, p95_means, width=width, label="p95", **BAR_KW)
    _label_bars(ax, bars1)
    _label_bars(ax, bars2)

    ax.set_xticks(x); ax.set_xticklabels(precisions)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency (p50 & p95) by Precision")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "latency_grouped_p50_p95")


def plot_token_share_by_precision(df: pd.DataFrame, figs_dir: Path):
    summary = df.groupby("precision", observed=True).agg(
        tokens_per_sec=("tokens_per_sec","mean"),
        speedup=("speedup_vs_fp32","mean")
    ).reset_index()

    fp32_tps = summary.loc[summary["precision"]=="fp32","tokens_per_sec"].values
    fp32_tps = fp32_tps[0] if len(fp32_tps) else np.nan

    def _spd(row):
        if pd.notna(row["speedup"]):
            return row["speedup"]
        if pd.notna(fp32_tps) and fp32_tps > 0 and pd.notna(row["tokens_per_sec"]):
            return row["tokens_per_sec"] / fp32_tps
        return np.nan

    summary["speedup_eff"] = summary.apply(_spd, axis=1)

    base = np.where(summary["precision"].notna(), 1.0, np.nan)
    inc = np.clip(summary["speedup_eff"] - 1.0, a_min=0.0, a_max=None)

    fig, ax = plt.subplots(figsize=(8,5))
    bars_base = ax.bar(summary["precision"], base, label="FP32 baseline (1×)", **BAR_KW)
    bars_inc  = ax.bar(summary["precision"], inc, bottom=base, label="Increment over FP32", **BAR_KW)

    totals = base + inc
    for rect, total in zip(bars_inc, totals):
        x = rect.get_x() + rect.get_width()/2
        y = rect.get_y() + rect.get_height()
        ax.text(x, y, f"{total:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Relative Token Share (× FP32)")
    ax.set_title("Token Share by Precision (Stacked: baseline + increment)")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "token_share_by_precision_stacked")


# (Other plots unchanged: plot_speedup_vs_fp32, plot_batch_efficiency,
#  plot_sla_frontier, plot_tokens_vs_latency95, plot_throughput_vs_vram,
#  plot_speedup_vs_vram_savings_bubble, write_summary_tables)


def plot_speedup_vs_fp32(df: pd.DataFrame, figs_dir: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    for p, sub in df.groupby("precision", observed=True):
        if pd.isna(p): continue
        sub = sub.sort_values("batch_size")
        ax.plot(sub["batch_size"], sub["speedup_vs_fp32"], marker="o", label=str(p))
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup vs FP32 (×)")
    ax.set_title("Speedup vs FP32 by Batch Size")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "speedup_vs_fp32")


def plot_batch_efficiency(df: pd.DataFrame, figs_dir: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    for p, sub in df.groupby("precision", observed=True):
        if pd.isna(p): continue
        sub = sub.sort_values("batch_size")
        ax.plot(sub["batch_size"], sub["batch_efficiency"], marker="o", label=str(p))
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Batch Efficiency (QPS / QPS@bsz=1)")
    ax.set_title("Batch Efficiency by Precision")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "batch_efficiency")


def plot_sla_frontier(df: pd.DataFrame, figs_dir: Path, sla_ms: Optional[float]):
    fig, ax = plt.subplots(figsize=(8,5))
    for p, sub in df.groupby("precision", observed=True):
        if pd.isna(p): continue
        ok = sub if sla_ms is None else sub[sub["latency_ms_p95"] <= sla_ms]
        ax.plot(ok["batch_size"], ok["throughput_qps"], marker="o", label=f"{p} (≤ SLA)")
        if sla_ms is not None:
            bad = sub[sub["latency_ms_p95"] > sla_ms]
            ax.scatter(bad["batch_size"], bad["throughput_qps"], marker="x", label=f"{p} (> SLA)")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (QPS)")
    title = "SLA Frontier (p95 latency)" + (f" — SLA ≤ {int(sla_ms)} ms" if sla_ms else "")
    ax.set_title(title)
    _legend_outside(ax)
    _savefig(fig, figs_dir, "sla_frontier")


def plot_tokens_vs_latency95(df: pd.DataFrame, figs_dir: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    for p, sub in df.groupby("precision", observed=True):
        if pd.isna(p): continue
        ax.scatter(sub["latency_ms_p95"], sub["tokens_per_sec"], label=str(p))
        for _, r in sub.iterrows():
            if pd.notna(r["latency_ms_p95"]) and pd.notna(r["tokens_per_sec"]):
                ax.annotate(f"bsz={int(r['batch_size'])}", (r["latency_ms_p95"], r["tokens_per_sec"]),
                            fontsize=8, xytext=(4,4), textcoords="offset points")
    ax.set_xlabel("Latency p95 (ms)")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Tokens/sec vs Latency p95 (Pareto)")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "tokens_vs_latency95")


def plot_throughput_vs_vram(df: pd.DataFrame, figs_dir: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    for p, sub in df.groupby("precision", observed=True):
        if pd.isna(p): continue
        ax.scatter(sub["vram_gb_peak"], sub["throughput_qps"], label=str(p))
        for _, r in sub.iterrows():
            if pd.notna(r["vram_gb_peak"]) and pd.notna(r["throughput_qps"]):
                ax.annotate(f"bsz={int(r['batch_size'])}", (r["vram_gb_peak"], r["throughput_qps"]),
                            fontsize=8, xytext=(4,4), textcoords="offset points")
    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_ylabel("Throughput (QPS)")
    ax.set_title("Throughput vs VRAM")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "throughput_vs_vram")


# --------------------------- NEW plots you asked for ---------------------------

# (2) Memory Usage vs Precision (explicit name; similar to VRAM vs Precision)
def plot_memory_usage_vs_precision(df: pd.DataFrame, figs_dir: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    precisions = [p for p in df["precision"].cat.categories if p in df["precision"].unique().tolist()]
    y = [df[df["precision"] == p]["vram_gb_peak"].mean() for p in precisions]
    ax.bar(precisions, y)
    ax.set_ylabel("Memory (GB, peak VRAM)")
    ax.set_title("Memory Usage vs Precision")
    _savefig(fig, figs_dir, "memory_usage_vs_precision")


# (3) Latency grouped bars: p50 and p95 per precision
def plot_latency_grouped_p50_p95(df: pd.DataFrame, figs_dir: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    precisions = [p for p in df["precision"].cat.categories if p in df["precision"].unique().tolist()]
    p50_means = [df[df["precision"] == p]["latency_ms_p50"].mean() for p in precisions]
    p95_means = [df[df["precision"] == p]["latency_ms_p95"].mean() for p in precisions]

    x = np.arange(len(precisions))
    width = 0.38
    ax.bar(x - width/2, p50_means, width=width, label="p50")
    ax.bar(x + width/2, p95_means, width=width, label="p95")
    ax.set_xticks(x); ax.set_xticklabels(precisions)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency (p50 & p95) by Precision")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "latency_grouped_p50_p95")


# (4) Stacked Bar: Token Share by Precision
#    Visualize speedup over FP32 as "baseline 1× + incremental"
def plot_token_share_by_precision(df: pd.DataFrame, figs_dir: Path):
    # Compute mean speedup per precision; if missing, derive from tokens/sec vs FP32 mean
    summary = df.groupby("precision", observed=True).agg(
        tokens_per_sec=("tokens_per_sec","mean"),
        speedup=("speedup_vs_fp32","mean")
    ).reset_index()
    # Compute fallback speedup from FP32 tokens/sec
    fp32_tps = summary.loc[summary["precision"]=="fp32","tokens_per_sec"].values
    fp32_tps = fp32_tps[0] if len(fp32_tps) else np.nan
    def _spd(row):
        if pd.notna(row["speedup"]):
            return row["speedup"]
        if pd.notna(fp32_tps) and fp32_tps > 0 and pd.notna(row["tokens_per_sec"]):
            return row["tokens_per_sec"] / fp32_tps
        return np.nan
    summary["speedup_eff"] = summary.apply(_spd, axis=1)

    # Build stacked bars: base=1 for all, inc=max(speedup-1,0)
    base = np.where(summary["precision"].notna(), 1.0, np.nan)
    inc = np.clip(summary["speedup_eff"] - 1.0, a_min=0.0, a_max=None)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(summary["precision"], base, label="FP32 baseline (1×)")
    ax.bar(summary["precision"], inc, bottom=base, label="Increment over FP32")
    ax.set_ylabel("Relative Token Share (× FP32)")
    ax.set_title("Token Share by Precision (Stacked: baseline + increment)")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "token_share_by_precision_stacked")


# (5) Speedup vs VRAM Reduction (Bubble Plot)
#     x: speedup_vs_fp32, y: VRAM savings %, size: QPS
def plot_speedup_vs_vram_savings_bubble(df: pd.DataFrame, figs_dir: Path):
    # Build FP32 baselines by (batch_size) and (optionally run_id) if present
    # Prefer matching run_id + batch_size; else fall back to batch_size only
    key_cols = ["batch_size"]
    if "run_id" in df.columns and df["run_id"].notna().any():
        key_cols = ["run_id","batch_size"]

    fp32 = df[df["precision"]=="fp32"].groupby(key_cols, observed=True).agg(
        tps=("tokens_per_sec","mean"),
        vram=("vram_gb_peak","mean"),
    )

    # Join baselines back
    merged = df.merge(fp32, how="left", left_on=key_cols, right_index=True, suffixes=("","_fp32"))
    # Compute metrics
    merged["speedup_x"] = merged["speedup_vs_fp32"]
    # Fallback speedup if missing
    need = merged["speedup_x"].isna() & merged["tokens_per_sec"].notna() & merged["tps"].notna() & (merged["tps"]>0)
    merged.loc[need, "speedup_x"] = merged.loc[need, "tokens_per_sec"] / merged.loc[need, "tps"]

    merged["vram_savings_pct"] = np.where(
        (merged["vram_gb_peak"].notna()) & (merged["vram"].notna()) & (merged["vram"]>0),
        (1.0 - (merged["vram_gb_peak"] / merged["vram"])) * 100.0,
        np.nan
    )

    # Filter valid points and drop FP32 itself (speedup=1, savings=0) to declutter
    pts = merged[
        merged["precision"].notna()
        & merged["speedup_x"].notna()
        & merged["vram_savings_pct"].notna()
    ].copy()

    if pts.empty:
        # still make an empty plot
        fig, ax = plt.subplots(figsize=(8,5))
        ax.set_xlabel("Speedup vs FP32 (×)")
        ax.set_ylabel("VRAM savings (%)")
        ax.set_title("Speedup vs VRAM Reduction (no valid data)")
        _savefig(fig, figs_dir, "speedup_vs_vram_savings_bubble")
        return

    # Bubble sizes scaled by QPS
    qps = pts["throughput_qps"].fillna(0).clip(lower=0)
    max_qps = qps.max() if qps.max() > 0 else 1.0
    sizes = 50 + 300 * (qps / max_qps)  # perceptible bubbles

    fig, ax = plt.subplots(figsize=(8,5))
    for p, sub in pts.groupby("precision", observed=True):
        ax.scatter(sub["speedup_x"], sub["vram_savings_pct"], s=sizes.loc[sub.index], alpha=0.7, label=str(p))
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Speedup vs FP32 (×)")
    ax.set_ylabel("VRAM savings (%)  (positive is better)")
    ax.set_title("Speedup vs VRAM Reduction (Bubble size = QPS)")
    _legend_outside(ax)
    _savefig(fig, figs_dir, "speedup_vs_vram_savings_bubble")




# --------------------------- MAIN ---------------------------

def main():
    _apply_theme("pro")
    csv_path, figs_dir = _resolve_paths()
    print(f"Reading: {csv_path}")
    print(f"Figures: {figs_dir}")
    df = _load(csv_path)
    sla = _sla_threshold_ms()

    plot_throughput_vs_precision(df, figs_dir)
    plot_latency_vs_batch_avg(df, figs_dir)
    plot_vram_vs_precision(df, figs_dir)

    plot_memory_usage_vs_precision(df, figs_dir)
    plot_latency_grouped_p50_p95(df, figs_dir)
    plot_token_share_by_precision(df, figs_dir)

    # keep other analytical views
    # (unchanged implementations in your file)
    plot_speedup_vs_fp32(df, figs_dir)
    plot_batch_efficiency(df, figs_dir)
    plot_sla_frontier(df, figs_dir, sla_ms=sla)
    plot_tokens_vs_latency95(df, figs_dir)
    plot_throughput_vs_vram(df, figs_dir)
    plot_speedup_vs_vram_savings_bubble(df, figs_dir)

    # write_summary_tables(df, figs_dir, sla_ms=sla)

    print("✅ Plots and tables written.")


if __name__ == "__main__":
    main()
