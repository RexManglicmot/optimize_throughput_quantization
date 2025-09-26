# app/make_tables_v4.py
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

STATS_CSV = Path("outputs/stats_throughput_results.csv")
OUT_MD    = Path("outputs/README_throughput_tables.md")

METRIC = "throughput_qps"   # QPS (secondary you chose to keep)
BASELINE = "fp32"

def _fmt_ratio(r: float) -> str:
    return f"×{r:.2f}"

def _fmt_ci(lo: float, hi: float) -> str:
    return f"{lo:.2f}–{hi:.2f}"

def _fmt_p(p: float) -> str:
    if not np.isfinite(p): 
        return "nan"
    if p < 1e-12:
        return "<1e-12"
    return f"{p:.1e}"

def _direction(r: float, eps: float = 1e-12) -> str:
    return "faster" if r > 1.0 + eps else "slower"

def _geo_mean(vals: np.ndarray) -> float:
    vals = np.asarray(vals, float)
    vals = vals[(vals > 0) & np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(vals))))

def _geo_ci(vals: np.ndarray, alpha: float = 0.05) -> tuple[float,float]:
    """95% CI on geometric mean via t-interval on log-values."""
    vals = np.asarray(vals, float)
    vals = vals[(vals > 0) & np.isfinite(vals)]
    n = vals.size
    if n < 2:
        gm = _geo_mean(vals)
        return (gm, gm)
    l = np.log(vals)
    m, s = l.mean(), l.std(ddof=1)
    # t critical ~ normal for small n; use 1.96 for simplicity/robustness
    tcrit = 1.96
    lo, hi = np.exp(m - tcrit * s / math.sqrt(n)), np.exp(m + tcrit * s / math.sqrt(n))
    return (float(lo), float(hi))

def _combine_pvalues_fisher(pvals: list[float]) -> float:
    pvals = [p for p in pvals if np.isfinite(p) and p > 0]
    if len(pvals) == 0:
        return np.nan
    stat = -2.0 * np.sum(np.log(pvals))
    df = 2 * len(pvals)
    return float(1.0 - chi2.cdf(stat, df))

def build_tables(df_stats: pd.DataFrame) -> tuple[str, str]:
    # Filter to QPS metric
    d = df_stats[df_stats["metric"] == METRIC].copy()
    # Expect columns:
    # ['metric','batch_size','precision','baseline','ratio_precision_over_baseline',
    #  'ci95_low','ci95_high','p_two_sided', ...]
    d = d.dropna(subset=["batch_size","precision","ratio_precision_over_baseline"])
    d["batch_size"] = pd.to_numeric(d["batch_size"], errors="coerce")

    # ----- Table 1: per-batch -----
    per_rows = []
    for bsz, sub in d.groupby("batch_size"):
        for _, r in sub.iterrows():
            ratio = float(r["ratio_precision_over_baseline"])
            per_rows.append({
                "Batch": int(bsz),
                "Precision": str(r["precision"]),
                "Direction": _direction(ratio),
                "× vs FP32": _fmt_ratio(ratio),
                "95% CI": _fmt_ci(float(r["ci95_low"]), float(r["ci95_high"])),
                "p(two-sided)": _fmt_p(float(r["p_two_sided"])),
            })
    per_df = pd.DataFrame(per_rows).sort_values(["Batch","Precision"], kind="stable")

    # ----- Table 2: across-batches (geo-mean) -----
    geo_rows = []
    for prec, sub in d.groupby("precision"):
        ratios = sub["ratio_precision_over_baseline"].astype(float).to_numpy()
        gm = _geo_mean(ratios)
        lo, hi = _geo_ci(ratios)
        p_comb = _combine_pvalues_fisher(sub["p_two_sided"].astype(float).tolist())
        geo_rows.append({
            "Precision": str(prec),
            "Direction": _direction(gm),
            "× vs FP32 (geo-mean)": _fmt_ratio(gm),
            "95% CI": _fmt_ci(lo, hi),
            "p(combined)": _fmt_p(p_comb),
        })
    geo_df = pd.DataFrame(geo_rows).sort_values("Precision", kind="stable")

    # ----- Render Markdown -----
    per_md = [
        "# Quantization — Throughput (QPS) Tables",
        "## QPS vs FP32 — per batch",
        "| Batch | Precision | Direction | × vs FP32 | 95% CI | p(two-sided) |",
        "| ---: | --- | --- | ---: | --- | ---: |",
    ]
    for _, r in per_df.iterrows():
        per_md.append(f"| {r['Batch']} | {r['Precision']} | {r['Direction']} | {r['× vs FP32']} | {r['95% CI']} | {r['p(two-sided)']} |")

    geo_md = [
        "",
        "## QPS vs FP32 — across batches (geo-mean)",
        "| Precision | Direction | × vs FP32 (geo-mean) | 95% CI | p(combined) |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for _, r in geo_df.iterrows():
        geo_md.append(f"| {r['Precision']} | {r['Direction']} | {r['× vs FP32 (geo-mean)']} | {r['95% CI']} | {r['p(combined)']} |")

    per_md_str = "\n".join(per_md)
    geo_md_str = "\n".join(geo_md)
    return per_md_str, geo_md_str

def main():
    if not STATS_CSV.exists():
        raise FileNotFoundError(f"Missing {STATS_CSV}")
    df_stats = pd.read_csv(STATS_CSV)

    md1, md2 = build_tables(df_stats)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md1 + "\n" + md2 + "\n", encoding="utf-8")

    print(md1)
    print(md2)
    print(f"\n✅ Wrote {OUT_MD}")

if __name__ == "__main__":
    main()
