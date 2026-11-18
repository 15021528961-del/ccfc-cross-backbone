"""把 results/*.json 汇总成论文用的 LaTeX 表格与 CSV。

用法：python -m src.aggregate --results ./results --out ./tables
"""
import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import load_json

METHOD_ORDER = ["zeroshot", "linear", "bitfit", "lora", "adapter", "paa", "full"]
METHOD_LABEL = {
    "zeroshot": "Zero-shot",
    "linear": "Linear probe",
    "bitfit": "BitFit",
    "lora": "LoRA",
    "adapter": "Adapter",
    "paa": "PAA (ours)",
    "full": "Full fine-tune",
}


def collect(results_dir: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        d = load_json(f)
        rows.append({
            "dataset": d["dataset"], "method": d["method"], "ratio": d["ratio"],
            "seed": d["seed"], "acc": d["acc"],
            "macro_acc": d.get("macro_acc", float("nan")),
            "trainable_params": d.get("trainable_params", 0),
            "train_time_sec": d.get("train_time_sec", float("nan")),
        })
    if not rows:
        raise SystemExit(f"{results_dir} 下没有找到 json 结果，先把实验跑起来")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "ratio", "method"]).agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        macro_mean=("macro_acc", "mean"),
        params=("trainable_params", "max"),
        time_mean=("train_time_sec", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    g["acc_std"] = g["acc_std"].fillna(0.0)
    return g


def to_latex(summary: pd.DataFrame, dataset: str) -> str:
    sub = summary[summary["dataset"] == dataset]
    ratios = sorted(sub["ratio"].unique())
    methods = [m for m in METHOD_ORDER if m in set(sub["method"])]

    head = " & ".join([f"{int(r*100)}\\% labels" if r < 1 else "100\\% labels" for r in ratios])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Top-1 accuracy (\\%) on {dataset} under different label budgets. "
        "Mean $\\pm$ std over seeds.}",
        f"\\label{{tab:{dataset}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "c" * len(ratios) + "r}",
        "\\toprule",
        f"Method & {head} & \\#Trainable \\\\",
        "\\midrule",
    ]
    for m in methods:
        cells = []
        params = 0
        for r in ratios:
            row = sub[(sub["method"] == m) & (np.isclose(sub["ratio"], r))]
            if row.empty:
                cells.append("--")
            else:
                cells.append(f"{row.acc_mean.iloc[0]*100:.2f}\\,$\\pm$\\,{row.acc_std.iloc[0]*100:.2f}")
                params = max(params, int(row.params.iloc[0]))
        label = METHOD_LABEL.get(m, m)
        if m == "paa":
            label = "\\textbf{PAA (ours)}"
        lines.append(f"{label} & " + " & ".join(cells) + f" & {params/1e6:.2f}M \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--out", default="./tables")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = collect(args.results)
    summary = summarize(df)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    for ds in sorted(summary["dataset"].unique()):
        tex = to_latex(summary, ds)
        with open(os.path.join(args.out, f"table_{ds}.tex"), "w", encoding="utf-8") as f:
            f.write(tex)
    print(f"已写出 {len(summary)} 行汇总到 {args.out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
