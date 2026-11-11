"""生成论文图：标签预算曲线 + 参数量-精度散点。

用法：python -m src.plot --results ./results --out ./figures
"""
import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import load_json

METHOD_ORDER = ["zeroshot", "linear", "bitfit", "lora", "adapter", "paa", "full"]
NICE = {"zeroshot": "Zero-shot", "linear": "Linear", "bitfit": "BitFit", "lora": "LoRA",
        "adapter": "Adapter", "paa": "PAA (ours)", "full": "Full FT"}
COLORS = {"zeroshot": "#999999", "linear": "#4C72B0", "bitfit": "#55A868", "lora": "#C44E52",
          "adapter": "#8172B2", "paa": "#DD8452", "full": "#937860"}


def load(results_dir: str) -> List[Dict]:
    return [load_json(f) for f in sorted(glob.glob(os.path.join(results_dir, "*.json")))]


def curve(rows: List[Dict], dataset: str, out: str, backbone: str = "clip-vit-b16") -> None:
    agg: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["dataset"] != dataset or r.get("backbone", "clip-vit-b16") != backbone:
            continue
        agg[r["method"]][r["ratio"]].append(r["acc"] * 100)

    plt.figure(figsize=(5.0, 3.4))
    for m in METHOD_ORDER:
        if m not in agg:
            continue
        xs = sorted(agg[m])
        ys = [np.mean(agg[m][x]) for x in xs]
        es = [np.std(agg[m][x]) for x in xs]
        plt.errorbar([x * 100 for x in xs], ys, yerr=es, marker="o", ms=4, capsize=3,
                     label=NICE[m], color=COLORS[m],
                     lw=2.2 if m == "paa" else 1.4)
    plt.xlabel("Label budget (%)")
    plt.ylabel("Top-1 accuracy (%)")
    plt.xscale("log")
    plt.xticks([1, 5, 10, 25, 50, 100], ["1", "5", "10", "25", "50", "100"])
    plt.grid(alpha=0.3)
    plt.legend(fontsize=6.5, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"fig_curve_{dataset}.pdf"))
    plt.savefig(os.path.join(out, f"fig_curve_{dataset}.png"), dpi=300)
    plt.close()


def params_vs_acc(rows: List[Dict], out: str) -> None:
    """100% 标签预算下，可训练参数量（对数轴）与精度的关系。"""
    pts: Dict[str, List] = defaultdict(lambda: [[], []])
    for r in rows:
        if r["ratio"] < 0.999 or r.get("trainable_params", 0) <= 0:
            continue
        pts[r["method"]][0].append(r["trainable_params"] / 1e6)
        pts[r["method"]][1].append(r["acc"] * 100)

    plt.figure(figsize=(4.6, 3.2))
    for m in METHOD_ORDER:
        if m not in pts:
            continue
        plt.scatter(pts[m][0], pts[m][1], s=46, label=NICE[m], color=COLORS[m],
                    edgecolor="k", linewidth=0.4, zorder=3)
    plt.xscale("log")
    plt.xlabel("Trainable parameters (M)")
    plt.ylabel("Top-1 accuracy (%)")
    plt.grid(alpha=0.3, zorder=0)
    plt.legend(fontsize=6.5, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_params_acc.pdf"))
    plt.savefig(os.path.join(out, "fig_params_acc.png"), dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--out", default="./figures")
    ap.add_argument("--curve-dataset", default="eurosat")
    ap.add_argument("--backbone", default="clip-vit-b16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = load(args.results)
    curve(rows, args.curve_dataset, args.out, args.backbone)
    params_vs_acc(rows, args.out)
    print(f"图已写入 {args.out}")


if __name__ == "__main__":
    main()
