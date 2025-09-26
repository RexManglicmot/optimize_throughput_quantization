# app/make_tables.py
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import stats

INPUT  = Path("outputs/stats_throughput_results.csv")
OUTPUT = Path("outputs/README_tables.md")

BASELINE = "fp32"
PRIMARY  = "tokens_per_sec"
SECONDARY= "throughput_qps"

def _gm_ci_from_ratios(r: np.ndarray, conf: float = 0.95) -> tuple[float,float,float]:
    """Geo-mean ratio & t-interval on log(ratios) across batches."""
    a = np.asarray([x for x in r if np.isfinite(x) and x > 0], dtype=float)
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    l = np.log(a)
    mean = float(l.mean())
    n = l.size
    if n < 2:
        gm = float(np.exp(mean))
        return (gm, gm, gm)
    sd = float(l.std(ddof=1))
    if sd == 0.0:
        gm = float(np.exp(mean))
        return (gm, gm, gm)
    se = sd / math.sqrt(n)
    alpha = 1 - conf
    tcrit = stats.t.ppf(1 - alpha/2, df=n-1)
    lo, hi = mean - tcrit*se, mean + tcrit*se
    return (float(np.exp(mean)), float(np.exp(lo)), float(np.exp(hi)))

def _build_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per precision (excluding fp32), geo-mean ratio across batches."""
    sub = df[(df["metric"] == metric) & (df["baseline"].str.lower() == BASELINE)].copy()
    rows = []
    for prec, grp in sub.groupby("precision", observed=True):
        if str(prec).lower() == BASELINE:
            continue
        ratios = grp["ratio_precision_over_baseline"].astype(float).values
        gm, lo, hi = _gm_ci_from_ratios(ratios)
        rows.append({
            "Precision": str(prec),
            "× vs FP32 (geo-mean)": gm,
            "95% CI": f"{lo:.2f}–{hi:.2f}" if np.isfinite(lo) and np.isfinite(hi) else "—",
        })
    out = pd.DataFrame(rows)
    # nice ordering
    order = ["int4","int8","fp16","bf16"]
    out["__ord"] = out["Precision"].str.lower().map({k:i for i,k in enumerate(order)})
    out = out.sort_values(["__ord","Precision"], na_position="last").drop(columns="__ord")
    # format ratio
    out["× vs FP32 (geo-mean)"] = out["× vs FP32 (geo-mean)"].apply(
        lambda x: f"×{x:.2f}" if np.isfinite(x) else "—"
    )
    return out

def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "\n_(no data)_\n"
    # plain markdown table
    header = "| " + " | ".join(df.columns) + " |"
    sep    = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows   = ["| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy()]
    return "\n".join([header, sep] + rows) + "\n"

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}")
    df = pd.read_csv(INPUT)

    tps_tbl = _build_table(df, PRIMARY)
    qps_tbl = _build_table(df, SECONDARY)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        f.write("# Quantization — Throughput Tables\n\n")
        f.write("## Tokens/sec vs FP32 (geo-mean across batches)\n")
        f.write(_df_to_md(tps_tbl))
        f.write("\n## QPS vs FP32 (geo-mean across batches)\n")
        f.write(_df_to_md(qps_tbl))

    print(f"✅ Wrote {OUTPUT}")

if __name__ == "__main__":
    main()
