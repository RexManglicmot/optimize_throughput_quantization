# app/stats_throughput.py
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ---- I/O ----
CSV_INPUT  = Path("outputs/results_n.csv")                 # per-replicate results
CSV_OUTPUT = Path("outputs/stats_throughput_results.csv")  # summary table to write

# ---- Settings ----
BASELINE   = "fp32"
METRIC     = "throughput_qps"  # QPS only (as requested)

def _welch_log_test(baseline_vals: np.ndarray, quant_vals: np.ndarray):
    """
    Welch t on log-values (quant vs baseline).
    Returns dict with ratio, CI, p-values, df (or None if not enough data).
    """
    lx = np.log(np.asarray(baseline_vals, float))
    ly = np.log(np.asarray(quant_vals, float))
    n1, n2 = len(lx), len(ly)
    if n1 < 2 or n2 < 2:
        return None

    m1, m2 = lx.mean(), ly.mean()
    v1, v2 = lx.var(ddof=1), ly.var(ddof=1)
    diff = m2 - m1
    se = math.sqrt(v1 / n1 + v2 / n2)

    if se == 0:
        return {
            "ratio": math.exp(diff),
            "ci_low": math.exp(diff),
            "ci_high": math.exp(diff),
            "p_two_sided": 0.0,
            "p_one_sided_gt": 0.0,
            "df": n1 + n2 - 2,
        }

    # Welch-Satterthwaite df
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = (v1**2) / (n1**2 * (n1 - 1)) + (v2**2) / (n2**2 * (n2 - 1))
    df = df_num / df_den if df_den > 0 else (n1 + n2 - 2)

    t = diff / se
    p_two = 2 * (1 - stats.t.cdf(abs(t), df))
    p_gt = 1 - stats.t.cdf(t, df)  # H1: quant > baseline

    tcrit = stats.t.ppf(0.975, df)
    lo, hi = math.exp(diff - tcrit * se), math.exp(diff + tcrit * se)

    return {
        "ratio": math.exp(diff),
        "ci_low": lo,
        "ci_high": hi,
        "p_two_sided": p_two,
        "p_one_sided_gt": p_gt,
        "df": df,
    }

def _geom_mean(a: np.ndarray) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.exp(np.log(a).mean())) if a.size else float("nan")

def _analyze_qps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-batch comparison of each precision vs FP32 using Welch t on log(QPS).
    Also appends an 'ALL' row per precision with geo-mean ratio across batches and a 95% CI.
    """
    # Keep needed columns & coerce numeric
    df = df[["precision", "batch_size", METRIC]].dropna()
    df["precision"] = df["precision"].astype(str)
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")
    df = df.dropna(subset=["batch_size"])

    rows = []

    # ---- Per-batch tests ----
    for bsz, sub in df.groupby("batch_size"):
        base = sub[sub["precision"].str.lower() == BASELINE][METRIC].dropna().values
        if len(base) == 0:
            continue
        gm_base = _geom_mean(base)
        n_base = len(base)

        for prec, subp in sub.groupby("precision"):
            if prec.lower() == BASELINE:
                continue
            vals = subp[METRIC].dropna().values
            if len(vals) == 0:
                continue
            gm_quant = _geom_mean(vals)
            n_quant = len(vals)

            res = _welch_log_test(base, vals)
            if res is None:
                continue

            direction = "faster" if res["ratio"] > 1 else "slower"

            rows.append({
                "metric": METRIC,
                "batch_size": int(bsz),
                "precision": prec,
                "baseline": BASELINE,
                "n_baseline": n_base,
                "n_precision": n_quant,
                "geom_mean_baseline": gm_base,
                "geom_mean_precision": gm_quant,
                "ratio_precision_over_baseline": res["ratio"],
                "ci95_low": res["ci_low"],
                "ci95_high": res["ci_high"],
                "p_two_sided": res["p_two_sided"],
                "p_one_sided_gt": res["p_one_sided_gt"],
                "df": res["df"],
                "direction": direction,
            })

            # Console line
            print(
                f"Batch {int(bsz)} | {prec} on {METRIC}: "
                f"×{res['ratio']:.3g} (CI {res['ci_low']:.3g}-{res['ci_high']:.3g}), "
                f"{direction} than {BASELINE} "
                f"(p={res['p_two_sided']:.2g} two-sided; one-sided p={res['p_one_sided_gt']:.2g})"
            )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    # ---- Across-batch geometric-mean summary per precision ----
    summaries = []
    for prec, sub in out.groupby("precision", observed=True):
        ratios = sub["ratio_precision_over_baseline"].to_numpy(dtype=float)
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        if ratios.size == 0:
            continue

        # Geo-mean of ratios across batches
        logs = np.log(ratios)
        n = logs.size
        gm_ratio = float(np.exp(logs.mean()))
        # CI across batches (t-based on batch-level log ratios)
        if n > 1:
            se = float(logs.std(ddof=1) / math.sqrt(n))
            tcrit = float(stats.t.ppf(0.975, n - 1))
            lo, hi = math.exp(logs.mean() - tcrit * se), math.exp(logs.mean() + tcrit * se)
        else:
            lo = hi = gm_ratio

        summaries.append({
            "metric": METRIC,
            "batch_size": "ALL",                    # <-- overall across batches
            "precision": prec,
            "baseline": BASELINE,
            "n_batches": int(n),
            "geo_mean_ratio_across_batches": gm_ratio,
            "geo_ci95_low": lo,
            "geo_ci95_high": hi,
        })

    out_all = pd.DataFrame(summaries)

    # Merge: put overall rows after the per-batch rows
    out = pd.concat([out, out_all], ignore_index=True)

    # Stable sort by precision then batch (with ALL at the end)
    def _batch_key(v):
        try:
            return (0, int(v))
        except Exception:
            return (1, float("inf"))
    out = out.sort_values(
        by=["precision", "batch_size"],
        key=lambda s: s.map(_batch_key),
        kind="stable"
    ).reset_index(drop=True)

    return out

def main():
    if not CSV_INPUT.exists():
        raise FileNotFoundError(f"Missing {CSV_INPUT}")
    df = pd.read_csv(CSV_INPUT)

    print("\n=== QPS (PRIMARY) ===")
    summary = _analyze_qps(df)

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(CSV_OUTPUT, index=False)
    print(f"\n✅ Saved: {CSV_OUTPUT} ({len(summary)} rows)")

if __name__ == "__main__":
    main()
