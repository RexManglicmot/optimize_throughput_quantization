# app/stats_throughput.py
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

CSV_INPUT  = Path("outputs/results_n.csv")              # per-replicate results
CSV_OUTPUT = Path("outputs/stats_throughput_results.csv")
BASELINE   = "fp32"
PRIMARY    = "tokens_per_sec"
SECONDARY  = "throughput_qps"

def welch_log_test(x: np.ndarray, y: np.ndarray):
    """Welch t on log-values (quant vs baseline). Returns dict with ratio, CI, p-values, df."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    n1, n2 = len(lx), len(ly)
    if n1 < 2 or n2 < 2:
        return None
    m1, m2 = lx.mean(), ly.mean()
    v1, v2 = lx.var(ddof=1), ly.var(ddof=1)
    diff = m2 - m1
    se = math.sqrt(v1/n1 + v2/n2)
    if se == 0:
        return {"ratio": math.exp(diff), "ci_low": math.exp(diff), "ci_high": math.exp(diff),
                "p_two_sided": 0.0, "p_one_sided_gt": 0.0, "df": n1+n2-2}
    df_num = (v1/n1 + v2/n2) ** 2
    df_den = (v1**2) / (n1**2 * (n1-1)) + (v2**2) / (n2**2 * (n2-1))
    df = df_num / df_den if df_den > 0 else (n1 + n2 - 2)
    t = diff / se
    p_two = 2 * (1 - stats.t.cdf(abs(t), df))
    p_gt  = 1 - stats.t.cdf(t, df)  # H1: quant > baseline
    tcrit = stats.t.ppf(0.975, df)
    lo, hi = math.exp(diff - tcrit*se), math.exp(diff + tcrit*se)
    return {"ratio": math.exp(diff), "ci_low": lo, "ci_high": hi,
            "p_two_sided": p_two, "p_one_sided_gt": p_gt, "df": df}

def geom_mean(a: np.ndarray) -> float:
    a = np.asarray(a, float)
    return float(np.exp(np.log(a).mean()))

def analyze_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    # Keep needed cols, coerce numeric
    df = df[["precision","batch_size",metric]].dropna()
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")
    df = df.dropna(subset=["batch_size"])
    for bsz, sub in df.groupby("batch_size"):
        base = sub[sub["precision"].str.lower() == BASELINE][metric].dropna().values
        if len(base) == 0:
            continue
        gm_base = geom_mean(base)
        n_base  = len(base)
        for prec, subp in sub.groupby("precision"):
            if str(prec).lower() == BASELINE:
                continue
            vals = subp[metric].dropna().values
            if len(vals) == 0:
                continue
            gm_quant = geom_mean(vals)
            n_quant  = len(vals)
            res = welch_log_test(base, vals)
            if res is None:
                continue
            direction = "faster" if res["ratio"] > 1 else "slower"
            rows.append({
                "metric": metric,
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
                f"Batch {int(bsz)} | {prec} on {metric}: "
                f"×{res['ratio']:.3g} (CI {res['ci_low']:.3g}-{res['ci_high']:.3g}), "
                f"{direction} than {BASELINE} "
                f"(p={res['p_two_sided']:.2g} two-sided; one-sided p={res['p_one_sided_gt']:.2g})"
            )
    return pd.DataFrame(rows)

def main():
    if not CSV_INPUT.exists():
        raise FileNotFoundError(f"Missing {CSV_INPUT}")
    df = pd.read_csv(CSV_INPUT)

    print("\n=== Tokens/sec (PRIMARY) ===")
    d1 = analyze_metric(df, PRIMARY)

    print("\n=== QPS (SECONDARY) ===")
    d2 = analyze_metric(df, SECONDARY)

    out = pd.concat([d1, d2], ignore_index=True)
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV_OUTPUT, index=False)
    print(f"\n✅ Saved: {CSV_OUTPUT} ({len(out)} rows)")

if __name__ == "__main__":
    main()
