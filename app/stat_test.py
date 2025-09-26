#!/usr/bin/env python3
# analyze_quant_stats.py
# Runs 4 analyses per batch size to highlight quantization benefits.
# - Input:  outputs/results_n.csv  (replicate rows; FP32 + quantized)
# - Output: outputs/stats_summary.csv
#
# Tests:
# 1) Throughput improvement — log-Welch t (ratio > 1)
# 2) Cost improvement       — log-Welch t (ratio < 1) on usd_per_1k_tokens (or tokens_per_usd fallback)
# 3) Latency non-inferiority — one-sided Welch t on latency_ms_p95 with tolerance Δ
# 4) VRAM savings — Welch t on vram_gb_peak + mean % reduction
#
# Edit constants below if you want different thresholds/columns.

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
except Exception as e:
    raise SystemExit("This script needs SciPy. Install with: pip install scipy") from e

# --------------------------------------------------------------------------------------
# CONFIG (no flags)
# --------------------------------------------------------------------------------------
IN_CSV  = Path("outputs/results_n.csv")
OUT_CSV = Path("outputs/stats_summary.csv")

# Cost metric preference: use the first that exists in your CSV columns
COST_METRIC_PREFERENCE = ["usd_per_1k_tokens", "tokens_per_usd"]  # lower-better first, higher-better fallback
LATENCY_METRIC = "latency_ms_p95"
THROUGHPUT_METRIC = "tokens_per_sec"
VRAM_METRIC = "vram_gb_peak"

# Non-inferiority tolerance Δ (ms): quantized p95 can be at most Δ higher than FP32
NONINFERIORITY_DELTA_MS = 100.0

# SLA for simple compliance readout (optional; set to None to disable)
SLA_MS = 1500.0


# --------------------------------------------------------------------------------------
# Welch helpers
# --------------------------------------------------------------------------------------
def _welch_core(a: np.ndarray, b: np.ndarray):
    """Welch test core on raw values. Returns (mean_a, mean_b, se, df, tstat, p_two_sided)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = a.size, b.size
    if n_a < 2 or n_b < 2:
        raise ValueError("Welch test needs at least 2 replicates per group.")

    mean_a, mean_b = a.mean(), b.mean()
    s2_a, s2_b = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(s2_a / n_a + s2_b / n_b)

    num = (s2_a / n_a + s2_b / n_b) ** 2
    den = (s2_a**2) / (n_a**2 * (n_a - 1)) + (s2_b**2) / (n_b**2 * (n_b - 1))
    df = num / den if den > 0 else (n_a + n_b - 2)

    tstat = (mean_a - mean_b) / se if se > 0 else np.inf
    p_two = 2.0 * min(student_t.cdf(tstat, df), 1 - student_t.cdf(tstat, df))
    return mean_a, mean_b, se, df, tstat, p_two


def welch_log_ratio(a: np.ndarray, b: np.ndarray):
    """
    Welch on log-values. Returns ratio of geometric means (a/b) and 95% CI.
    a: quantized group, b: fp32 group
    """
    la = np.log(np.asarray(a, dtype=float))
    lb = np.log(np.asarray(b, dtype=float))
    mean_a, mean_b, se, df, tstat, p_two = _welch_core(la, lb)
    diff = mean_a - mean_b  # log ratio
    tcrit = student_t.ppf(0.975, df)
    lo, hi = diff - tcrit * se, diff + tcrit * se
    return {
        "ratio": float(np.exp(diff)),
        "ci_low": float(np.exp(lo)),
        "ci_high": float(np.exp(hi)),
        "df": float(df),
        "tstat": float(tstat),
        "p_two_sided": float(p_two),
        "log_diff": float(diff),
        "log_se": float(se),
    }


def welch_noninferiority(a: np.ndarray, b: np.ndarray, delta: float):
    """
    One-sided NI on means: H0: (mean_a - mean_b) >= delta  vs  H1: < delta
    a: quantized group, b: fp32 group
    """
    mean_a, mean_b, se, df, _, _ = _welch_core(a, b)
    diff = mean_a - mean_b
    tstat_NI = (diff - delta) / se
    p_one = student_t.cdf(tstat_NI, df)  # small p supports non-inferiority (diff < delta)

    # 95% upper bound for (mean_a - mean_b), one-sided
    tcrit_one = student_t.ppf(0.95, df)
    upper_bound = diff + tcrit_one * se
    return {
        "diff_ms": float(diff),
        "upper95_ms": float(upper_bound),
        "df": float(df),
        "tstat_NI": float(tstat_NI),
        "p_one_sided": float(p_one),
        "non_inferior": bool(upper_bound < delta),
    }


# --------------------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------------------
def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Input not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    # Basic guards
    need_cols = {"precision", "batch_size", THROUGHPUT_METRIC, LATENCY_METRIC, VRAM_METRIC}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {IN_CSV}: {missing}")

    # Pick cost metric
    cost_col = None
    for c in COST_METRIC_PREFERENCE:
        if c in df.columns:
            cost_col = c
            break
    if cost_col is None:
        print("WARN: No cost metric found; cost tests will be skipped.")
    else:
        print(f"[info] Using cost metric: {cost_col}")

    # Normalize types
    df["precision"] = df["precision"].astype(str).str.lower()
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")

    precisions = sorted([p for p in df["precision"].unique() if p != "fp32"])
    batch_sizes = sorted(df["batch_size"].dropna().unique())

    out_rows: List[Dict] = []

    for bsz in batch_sizes:
        base = df[(df["precision"] == "fp32") & (df["batch_size"] == bsz)]
        base_tps = base[THROUGHPUT_METRIC].dropna().to_numpy()
        base_lat = base[LATENCY_METRIC].dropna().to_numpy()
        base_vram = base[VRAM_METRIC].dropna().to_numpy()

        if base_tps.size < 2 or base_lat.size < 2 or base_vram.size < 2:
            print(f"[skip] Not enough FP32 replicates for batch_size {bsz}")
            continue

        sla_note = ""
        if SLA_MS is not None:
            fp32_sla = float(np.mean(base_lat <= SLA_MS))
            sla_note = f"SLA≤{int(SLA_MS)}ms: fp32={int(fp32_sla*base_lat.size)}/{base_lat.size}"

        for prec in precisions:
            sub = df[(df["precision"] == prec) & (df["batch_size"] == bsz)]

            qtps = sub[THROUGHPUT_METRIC].dropna().to_numpy()
            qlat = sub[LATENCY_METRIC].dropna().to_numpy()
            qvram = sub[VRAM_METRIC].dropna().to_numpy()

            if min(qtps.size, qlat.size, qvram.size) < 2:
                print(f"[skip] Not enough {prec} replicates for batch_size {bsz}")
                continue

            # 1) Throughput improvement — log-Welch t (ratio > 1)
            thr = welch_log_ratio(qtps, base_tps)
            out_rows.append({
                "batch_size": int(bsz),
                "precision": prec,
                "test": "throughput_ratio_vs_fp32",
                "effect": thr["ratio"],
                "ci_low": thr["ci_low"],
                "ci_high": thr["ci_high"],
                "p_value": thr["p_two_sided"],
                "df": thr["df"],
                "note": "ratio > 1 better",
            })

            # 2) Cost improvement — log-Welch t on chosen cost metric
            if cost_col is not None:
                qc = sub[cost_col].dropna().to_numpy()
                bc = base[cost_col].dropna().to_numpy()
                if qc.size >= 2 and bc.size >= 2:
                    cost = welch_log_ratio(qc, bc)  # ratio = quant / fp32
                    # If metric is "lower is better" (usd_per_1k_tokens), convert to savings %
                    if cost_col == "usd_per_1k_tokens":
                        savings_pct = (1.0 - cost["ratio"]) * 100.0
                        note = "ratio < 1 better; savings % shown"
                        effect = savings_pct
                        ci_low = (1.0 - cost["ci_high"]) * 100.0
                        ci_high = (1.0 - cost["ci_low"]) * 100.0
                    else:
                        # tokens_per_usd: higher is better; keep ratio as-is
                        effect = cost["ratio"]
                        ci_low = cost["ci_low"]
                        ci_high = cost["ci_high"]
                        note = "ratio > 1 better"
                    out_rows.append({
                        "batch_size": int(bsz),
                        "precision": prec,
                        "test": f"cost_{cost_col}_vs_fp32",
                        "effect": effect,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": cost["p_two_sided"],
                        "df": cost["df"],
                        "note": note,
                    })

            # 3) Latency non-inferiority — one-sided Welch t
            lat = welch_noninferiority(qlat, base_lat, NONINFERIORITY_DELTA_MS)
            # SLA compliance (simple readout)
            sla_part = ""
            if SLA_MS is not None:
                q_sla = float(np.mean(qlat <= SLA_MS))
                sla_part = f"; {prec}={int(q_sla*qlat.size)}/{qlat.size}"
            out_rows.append({
                "batch_size": int(bsz),
                "precision": prec,
                "test": "latency_noninferiority_vs_fp32",
                "effect": lat["diff_ms"],          # mean difference (quant - fp32), ms
                "ci_low": None,
                "ci_high": lat["upper95_ms"],      # one-sided 95% upper bound
                "p_value": lat["p_one_sided"],
                "df": lat["df"],
                "note": f"Δ={NONINFERIORITY_DELTA_MS} ms; non_inferior={lat['non_inferior']} | {sla_note}{sla_part}",
            })

            # 4) VRAM savings — mean % reduction + Welch t on raw GB
            q_mean_vram = float(qvram.mean())
            b_mean_vram = float(base_vram.mean())
            savings_pct = (1.0 - q_mean_vram / b_mean_vram) * 100.0 if b_mean_vram > 0 else np.nan
            _, _, _, df_v, t_v, p_v = _welch_core(qvram, base_vram)
            out_rows.append({
                "batch_size": int(bsz),
                "precision": prec,
                "test": "vram_savings_vs_fp32",
                "effect": savings_pct,   # % savings (positive is better)
                "ci_low": None,
                "ci_high": None,
                "p_value": p_v,
                "df": df_v,
                "note": "positive = less VRAM than FP32",
            })

    out = pd.DataFrame(out_rows, columns=[
        "batch_size","precision","test","effect","ci_low","ci_high","p_value","df","note"
    ])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # Pretty console summary
    if not out.empty:
        def fmt(x):
            return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.4g}" if isinstance(x, float) else str(x))
        print("\n=== Quantization Stats Summary ===")
        for (bsz, prec), grp in out.groupby(["batch_size","precision"]):
            print(f"\nBatch {int(bsz)} | {prec}")
            for _, r in grp.iterrows():
                if "throughput_ratio" in r["test"]:
                    print(f"  Throughput: ratio vs FP32 = {fmt(r['effect'])} (CI {fmt(r['ci_low'])}-{fmt(r['ci_high'])}), p={fmt(r['p_value'])}")
                elif r["test"].startswith("cost_"):
                    label = "Cost (usd/1k)" if "usd_per_1k_tokens" in r["test"] else "Value (tokens/$)"
                    if "usd_per_1k_tokens" in r["test"]:
                        print(f"  {label}: savings = {fmt(r['effect'])}% (from ratio), p={fmt(r['p_value'])}")
                    else:
                        print(f"  {label}: ratio vs FP32 = {fmt(r['effect'])} (CI {fmt(r['ci_low'])}-{fmt(r['ci_high'])}), p={fmt(r['p_value'])}")
                elif "latency_noninferiority" in r["test"]:
                    print(f"  Latency NI: Δ={NONINFERIORITY_DELTA_MS}ms, diff={fmt(r['effect'])} ms, upper95={fmt(r['ci_high'])} ms, p(one-sided)={fmt(r['p_value'])}  [{r['note']}]")
                elif "vram_savings" in r["test"]:
                    print(f"  VRAM: savings={fmt(r['effect'])}% (Welch p={fmt(r['p_value'])})")
        print(f"\nSaved: {OUT_CSV}")
    else:
        print("No results to summarize (check input file and required columns).")


if __name__ == "__main__":
    main()
