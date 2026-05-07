"""
analysis/table_baseline_comparison.py — Load results and print comparison table.

Usage
-----
  python table_baseline_comparison.py --run_dir ../results/baseline_comparison/run_1
  python table_baseline_comparison.py --run_dir ../results/baseline_comparison/run_1 --latex
"""

import argparse
import os

import numpy as np
import pandas as pd

ORDER  = ["baseline", "curb_l05", "pdp", "dpdp"]
LABELS = {
    "baseline": "Baseline (no intervention)",
    "curb_l05": "CURB λ=0.5",
    "pdp":      "PDP (γ=1)",
    "dpdp":     "DPDP (γ=1, adv=0.3)",
}


def load(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "comparison_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No comparison_results.csv in {run_dir}")
    return pd.read_csv(path)


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    price_cols = [c for c in df.columns if c.startswith("price_agent")]
    rows = []
    for cond in ORDER:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        row = {"Condition": LABELS.get(cond, cond)}
        for pc in price_cols:
            label = pc.replace("price_agent", "Price agent ")
            row[f"{label} mean"] = sub[pc].mean()
            row[f"{label} std"]  = sub[pc].std()
        row["CI mean"] = sub["CI"].mean()
        row["CI std"]  = sub["CI"].std()
        rows.append(row)
    return pd.DataFrame(rows)


def print_plain(tbl: pd.DataFrame):
    price_cols = [c for c in tbl.columns if "Price agent" in c and "mean" in c]
    agents     = [c.replace(" mean", "") for c in price_cols]

    header_parts = [f"{'Condition':<30}"]
    for a in agents:
        header_parts.append(f"  {a:>18}")
    header_parts.append(f"  {'CI':>14}")
    header = "".join(header_parts)
    print(header)
    print("-" * len(header))

    for _, row in tbl.iterrows():
        line = f"{row['Condition']:<30}"
        for a in agents:
            m, s = row[f"{a} mean"], row[f"{a} std"]
            line += f"  {m:6.4f} ± {s:6.4f}"
        line += f"  {row['CI mean']:6.4f} ± {row['CI std']:6.4f}"
        print(line)
    print()


def print_latex(tbl: pd.DataFrame):
    price_cols = [c for c in tbl.columns if "Price agent" in c and "mean" in c]
    agents     = [c.replace(" mean", "") for c in price_cols]
    n_agents   = len(agents)

    col_spec = "l" + "r" * (n_agents + 1)
    print(r"\begin{tabular}{" + col_spec + r"}")
    print(r"\toprule")

    hdr = "Condition & " + " & ".join(agents) + " & CI \\\\"
    print(hdr)
    print(r"\midrule")

    for _, row in tbl.iterrows():
        cells = [row["Condition"]]
        for a in agents:
            m, s = row[f"{a} mean"], row[f"{a} std"]
            cells.append(f"${m:.4f} \\pm {s:.4f}$")
        cells.append(f"${row['CI mean']:.4f} \\pm {row['CI std']:.4f}$")
        print(" & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--latex",   action="store_true")
    args = parser.parse_args()

    df  = load(args.run_dir)
    tbl = build_table(df)

    print_plain(tbl)
    if args.latex:
        print_latex(tbl)

    out = os.path.join(args.run_dir, "summary_table.txt")
    import io, sys
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    print_plain(tbl)
    sys.stdout = old
    with open(out, "w") as f:
        f.write(buf.getvalue())
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
