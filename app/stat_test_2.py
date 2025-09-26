#!/usr/bin/env python3
# analyze_quant_stats_min.py
# Minimal stats: Welch log t-tests (throughput & cost), one-sided NI on latency, VRAM savings.
# Input : outputs/results_n.csv  (replicate rows)
# Output: outputs/stats_summary.csv

from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

IN_CSV  = Path("outputs/results_n.csv")
OUT_CSV = Path("outputs/stats_summary.csv")

THROUGHPUT_COL = "tokens_per_sec"
LATENCY_COL    = "latency_ms_p95"
VRAM_COL       = "vram_gb_peak"
# Use the first present cost KPI: lower-better then higher-better
COST_PREF = ["usd_per_1k_tokens", "tokens_per_usd"]
# Non-inferiority tolerance (ms): quantized can be up to +Δ vs FP32
NI_DELTA_MS = 100.0
SLA_MS = 1500.0  # optional readout (set None to disable)

# ---------- tiny helpers ----------
def _welch_stats(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2: raise ValueError("Need ≥2 replicates per group.")
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(s1/n1 + s2/n2)
    df = (s1/n1 + s2/n2)**2 / ((s1**2)/(n1**2*(n1-1)) + (s2**2)/(n2**2*(n2-1)))
    t = (m1 - m2) / se if se > 0 else np.inf
    p_two = 2.0 * min(student_t.cdf(t, df), 1 - student_t.cdf(t, df))
    return m1, m2, se, df, t, p_two

def welch_log_ratio(a, b):
    la, lb = np.log(a), np.log(b)
    m1, m2, se, df, t, p_two = _welch_stats(la, lb)
    diff = m1 - m2
    tcrit = student_t.ppf(0.975, df)
    lo, hi = diff - tcrit*se, diff + tcrit*se
    return np.exp(diff), np.exp(lo), np.exp(hi), float(p_two), float(df)

def welch_noninferiority(a, b, delta):
    m1, m2, se, df, _, _ = _welch_stats(a, b)
    diff = m1 - m2
    t_NI = (diff - delta) / se
    p_one = student_t.cdf(t_NI, df)                 # small p → non-inferior
    ub95 = diff + student_t.ppf(0.95, df) * se      # one-sided 95% upper bound
    return float(diff), float(ub95), float(p_one), float(df), bool(ub95 < delta)

# ---------- main ----------
def main():
    if not IN_CSV.exists():
      raise SystemExit(f"Missing {IN_CSV}")
    df = pd.read_csv(IN_CSV)
    for c in [THROUGHPUT_COL, LATENCY_COL, VRAM_COL, "precision", "batch_size"]:
        if c not in df.columns: raise SystemExit(f"Missing column: {c}")

    # choose cost column if available
    cost_col = next((c for c in COST_PREF if c in df.columns), None)

    df["precision"] = df["precision"].astype(str).str.lower()
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")
    batch_sizes = sorted(df["batch_size"].dropna().unique())
    quant_precisions = sorted([p for p in df["precision"].unique() if p != "fp32"])

    rows = []
    for bsz in batch_sizes:
        base = df[(df["precision"]=="fp32") & (df["batch_size"]==bsz)]
        btps = base[THROUGHPUT_COL].dropna().to_numpy()
        blat = base[LATENCY_COL].dropna().to_numpy()
        bvrm = base[VRAM_COL].dropna().to_numpy()
        if min(len(btps), len(blat), len(bvrm)) < 2: continue

        for prec in quant_precisions:
            sub = df[(df["precision"]==prec) & (df["batch_size"]==bsz)]
            qtps = sub[THROUGHPUT_COL].dropna().to_numpy()
            qlat = sub[LATENCY_COL].dropna().to_numpy()
            qvrm = sub[VRAM_COL].dropna().to_numpy()
            if min(len(qtps), len(qlat), len(qvrm)) < 2: continue

            # 1) Throughput ratio (log-Welch t): ratio > 1 better
            ratio, lo, hi, p_thr, df_thr = welch_log_ratio(qtps, btps)
            rows.append([bsz, prec, "throughput_ratio_vs_fp32", ratio, lo, hi, p_thr, df_thr, "ratio>1 better"])

            # 2) Cost improvement (log-Welch t)
            if cost_col is not None:
                qc = sub[cost_col].dropna().to_numpy()
                bc = base[cost_col].dropna().to_numpy()
                if len(qc) >= 2 and len(bc) >= 2:
                    r, lo, hi, p_cost, df_cost = welch_log_ratio(qc, bc)  # quant / fp32
                    if cost_col == "usd_per_1k_tokens":  # lower-better → report savings %
                        rows.append([bsz, prec, f"cost_{cost_col}_vs_fp32",
                                    (1-r)*100.0, (1-hi)*100.0, (1-lo)*100.0, p_cost, df_cost,
                                    "savings % (pos=cheaper)"])
                    else:  # tokens_per_usd: higher-better
                        rows.append([bsz, prec, f"cost_{cost_col}_vs_fp32",
                                    r, lo, hi, p_cost, df_cost, "ratio>1 better"])

            # 3) Latency non-inferiority (one-sided Welch)
            diff, ub95, p_NI, df_NI, ok = welch_noninferiority(qlat, blat, NI_DELTA_MS)
            note = f"Δ={NI_DELTA_MS}ms; non-inferior={ok}"
            if SLA_MS is not None:
                note += f"; SLA {int(SLA_MS)}ms: {prec} {int((qlat<=SLA_MS).sum())}/{len(qlat)}, fp32 {int((blat<=SLA_MS).sum())}/{len(blat)}"
            rows.append([bsz, prec, "latency_noninferiority_vs_fp32", diff, None, ub95, p_NI, df_NI, note])

            # 4) VRAM savings: % reduction + Welch p on raw GB (two-sided)
            sv = (1.0 - qvrm.mean()/bvrm.mean())*100.0 if bvrm.mean() > 0 else np.nan
            # quick p-value using Welch on raw GB
            _, _, _, df_v, t_v, p_v = _welch_stats(qvrm, bvrm)
            rows.append([bsz, prec, "vram_savings_vs_fp32", sv, None, None, p_v, df_v, "positive=less VRAM than FP32"])

    out = pd.DataFrame(rows, columns=["batch_size","precision","test","effect","ci_low","ci_high","p_value","df","note"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # tiny console summary
    if not out.empty:
        print("\n=== Quantization Stats Summary ===")
        for (b, p), g in out.groupby(["batch_size","precision"]):
            print(f"\nBatch {int(b)} | {p}")
            for _, r in g.iterrows():
                name = r['test']
                if "throughput_ratio" in name:
                    print(f"  Throughput: ×{r.effect:.3g} (CI {r.ci_low:.3g}-{r.ci_high:.3g}), p={r.p_value:.3g}")
                elif name.startswith("cost_"):
                    if "usd_per_1k_tokens" in name:
                        print(f"  Cost: {r.effect:.2f}% savings (p={r.p_value:.3g})")
                    else:
                        print(f"  Value: ×{r.effect:.3g} (CI {r.ci_low:.3g}-{r.ci_high:.3g}), p={r.p_value:.3g}")
                elif "latency_noninferiority" in name:
                    ub = "—" if pd.isna(r.ci_high) else f"{r.ci_high:.1f} ms"
                    print(f"  Latency NI: Δ={NI_DELTA_MS}ms, diff={r.effect:.1f} ms, upper95={ub}, p(one-sided)={r.p_value:.3g} [{r.note}]")
                elif "vram_savings" in name:
                    print(f"  VRAM: {r.effect:.1f}% savings (Welch p={r.p_value:.3g})")
        print(f"\nSaved: {OUT_CSV}")
    else:
        print("No valid groups found. Check your input and replicates.")

if __name__ == "__main__":
    main()
