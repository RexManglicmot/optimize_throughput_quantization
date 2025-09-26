# app/make_tables_v3.py
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

RESULTS_CSV = Path("outputs/results.csv")                    # kept for parity
STATS_CSV   = Path("outputs/stats_throughput_results.csv")   # from stats script
OUT_MD      = Path("outputs/README_tables.md")

BASELINE = "fp32"
METRIC   = "throughput_qps"

def _coerce_float(s):
    try: return float(s)
    except Exception: return np.nan

def _fmt_ratio(x: float) -> str:
    if not np.isfinite(x): return "—"
    v = float(x)
    return f"×{v:.3g}" if (v >= 10 or v < 0.1) else f"×{v:.2f}"

def _fmt_ci(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)): return "—"
    def _f(v): return f"{v:.3g}" if (v >= 10 or v < 0.1) else f"{v:.2f}"
    return f"{_f(lo)}–{_f(hi)}"

def _fmt_p(p: float) -> str:
    if not np.isfinite(p): return "—"
    return f"{p:.1e}" if p < 1e-6 else f"{p:.4f}"

def _se_from_ci95(lo: float, hi: float) -> float:
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= 0:
        return np.nan
    return (math.log(hi) - math.log(lo)) / (2 * 1.96)

def _combine_fixed_effects_log(ratios, ci_los, ci_his):
    logs, ses = [], []
    for r, lo, hi in zip(ratios, ci_los, ci_his):
        if not (np.isfinite(r) and r > 0): continue
        se = _se_from_ci95(lo, hi)
        if not (np.isfinite(se) and se > 0): continue
        logs.append(math.log(r)); ses.append(se)
    if not logs:
        return (np.nan, np.nan, np.nan, np.nan)
    w = np.array([1/(s*s) for s in ses], float)
    L = np.array(logs, float)
    L_bar = float(np.sum(w * L) / np.sum(w))
    se_pool = float(1 / math.sqrt(np.sum(w)))
    lo = math.exp(L_bar - 1.96 * se_pool)
    hi = math.exp(L_bar + 1.96 * se_pool)
    return (math.exp(L_bar), lo, hi, se_pool)

def _combine_pvalues_stouffer(pvals):
    zs = []
    for p in pvals:
        if not (np.isfinite(p) and 0 < p <= 1): continue
        zs.append(abs(norm.ppf(1 - p/2.0)))
    if not zs: return np.nan
    Z = float(np.sum(zs) / math.sqrt(len(zs)))
    return 2 * (1 - norm.cdf(abs(Z)))

def main():
    if not STATS_CSV.exists():
        raise FileNotFoundError(f"Missing {STATS_CSV}")
    stats = pd.read_csv(STATS_CSV)

    # Normalize expected column names
    if "ratio" in stats.columns and "ratio_precision_over_baseline" not in stats.columns:
        stats = stats.rename(columns={"ratio": "ratio_precision_over_baseline"})

    # Keep QPS rows only
    stats = stats[stats["metric"].astype(str) == METRIC].copy()

    # Add numeric batch column; non-numeric (e.g., "ALL") -> NaN
    stats["_batch_num"] = pd.to_numeric(stats.get("batch_size"), errors="coerce")

    # ---------------- Per-batch table (numeric batches only) ----------------
    per_batch = stats[stats["_batch_num"].notna()].copy()

    per_batch_rows = []
    for (bsz, prec), sub in per_batch.groupby(["_batch_num", "precision"], observed=True):
        prec = str(prec)
        if prec.lower() == BASELINE: 
            continue
        r = sub.iloc[0]
        ratio = _coerce_float(r.get("ratio_precision_over_baseline"))
        lo    = _coerce_float(r.get("ci95_low"))
        hi    = _coerce_float(r.get("ci95_high"))
        p2    = _coerce_float(r.get("p_two_sided"))
        per_batch_rows.append({
            "Batch": int(bsz),
            "Precision": prec,
            "× vs FP32": _fmt_ratio(ratio),
            "95% CI": _fmt_ci(lo, hi),
            "p(two-sided)": _fmt_p(p2),
        })
    per_batch_df = pd.DataFrame(per_batch_rows).sort_values(["Batch","Precision"])

    # ---------------- Across-batches summary (geo-mean) ----------------
    # Use numeric batches only to avoid double counting if you also stored "ALL"
    summary_rows = []
    for prec, sub in per_batch.groupby("precision", observed=True):
        prec = str(prec)
        if prec.lower() == BASELINE:
            continue
        ratios = pd.to_numeric(sub["ratio_precision_over_baseline"], errors="coerce").values
        ci_los = pd.to_numeric(sub.get("ci95_low"), errors="coerce").values
        ci_his = pd.to_numeric(sub.get("ci95_high"), errors="coerce").values
        pvals  = pd.to_numeric(sub.get("p_two_sided"), errors="coerce").values

        gm, lo, hi, _ = _combine_fixed_effects_log(ratios, ci_los, ci_his)
        p_comb = _combine_pvalues_stouffer(pvals)

        summary_rows.append({
            "Precision": prec,
            "× vs FP32 (geo-mean)": _fmt_ratio(gm),
            "95% CI": _fmt_ci(lo, hi),
            "p(combined)": _fmt_p(p_comb),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("Precision")

    # ---------------- Markdown ----------------
    md = []
    md.append("# Quantization — Throughput (QPS) Tables\n")
    md.append("## QPS vs FP32 — per batch\n")
    if per_batch_df.empty:
        md.append("_No data found for throughput_qps._\n")
    else:
        md.append("| Batch | Precision | × vs FP32 | 95% CI | p(two-sided) |\n")
        md.append("| ---: | --- | ---: | --- | ---: |\n")
        for _, row in per_batch_df.iterrows():
            md.append(f"| {row['Batch']} | {row['Precision']} | {row['× vs FP32']} | {row['95% CI']} | {row['p(two-sided)']} |\n")

    md.append("\n## QPS vs FP32 — across batches (geo-mean)\n")
    if summary_df.empty:
        md.append("_No data available to summarize._\n")
    else:
        md.append("| Precision | × vs FP32 (geo-mean) | 95% CI | p(combined) |\n")
        md.append("| --- | ---: | --- | ---: |\n")
        for _, row in summary_df.iterrows():
            md.append(f"| {row['Precision']} | {row['× vs FP32 (geo-mean)']} | {row['95% CI']} | {row['p(combined)']} |\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print("".join(md))
    print(f"\n✅ Wrote tables to: {OUT_MD}")

if __name__ == "__main__":
    main()
