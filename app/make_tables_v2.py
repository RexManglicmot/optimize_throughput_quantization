# app/make_tables.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

IN_CSV  = Path("outputs/stats_throughput_results.csv")  # produced by stats_throughput.py
OUT_MD  = Path("outputs/throughput_tables.md")

# Keep headers EXACT
COL_HDRS = ["Precision", "× vs FP32 (geo-mean)", "95% CI"]

# Order precisions for display (only those present will be shown)
PRECISION_ORDER = ["fp16", "int8", "int4"]

def _fmt_ratio(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"×{x:.2f}"

def _fmt_ci(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "—"
    return f"{lo:.2f}–{hi:.2f}"

def _one_metric_tables(df: pd.DataFrame, metric: str) -> str:
    """
    Build markdown for one metric:
      - section title
      - per-batch subtable with the 3 required columns
    """
    out_lines = []
    title = "Tokens/sec vs FP32 (per batch)" if metric == "tokens_per_sec" else "QPS vs FP32 (per batch)"
    out_lines.append(f"## {title}\n")

    d = df[df["metric"] == metric].copy()
    if d.empty:
        out_lines.append("_No data_\n")
        return "\n".join(out_lines)

    # Ensure required cols exist
    needed = {"batch_size","precision","ratio_precision_over_baseline","ci95_low","ci95_high"}
    missing = needed - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns in {IN_CSV}: {sorted(missing)}")

    # Normalize types
    d["batch_size"] = pd.to_numeric(d["batch_size"], errors="coerce")
    d = d.dropna(subset=["batch_size"])
    d["precision"] = d["precision"].astype(str)

    # Loop batches; one table per batch
    for bsz in sorted(d["batch_size"].unique()):
        sub = d[d["batch_size"] == bsz].copy()

        # Show only quantized (exclude fp32 rows if present)
        sub = sub[sub["precision"].str.lower() != "fp32"]

        # Display order
        sub["__order__"] = sub["precision"].map({p:i for i,p in enumerate(PRECISION_ORDER)})
        sub["__order__"] = sub["__order__"].fillna(999)
        sub = sub.sort_values(["__order__","precision"], kind="stable")

        # Build rows
        rows = []
        for _, r in sub.iterrows():
            rows.append([
                r["precision"],
                _fmt_ratio(r["ratio_precision_over_baseline"]),
                _fmt_ci(r["ci95_low"], r["ci95_high"]),
            ])

        # Write table
        out_lines.append(f"### Batch {int(bsz)}")
        out_lines.append(f"| {COL_HDRS[0]} | {COL_HDRS[1]} | {COL_HDRS[2]} |")
        out_lines.append("| --- | --- | --- |")
        if rows:
            for prc, ratio_s, ci_s in rows:
                out_lines.append(f"| {prc} | {ratio_s} | {ci_s} |")
        else:
            out_lines.append("| _none_ | — | — |")
        out_lines.append("")  # blank line

    return "\n".join(out_lines)

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input: {IN_CSV}")
    df = pd.read_csv(IN_CSV)

    # Compose markdown
    lines = ["# Quantization — Throughput Tables", ""]
    lines.append(_one_metric_tables(df, "tokens_per_sec"))
    lines.append("")
    lines.append(_one_metric_tables(df, "throughput_qps"))
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Wrote {OUT_MD}")

if __name__ == "__main__":
    main()
