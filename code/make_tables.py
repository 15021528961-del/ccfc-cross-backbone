"""按"每方法验证集选最优学习率"重算论文里的四张表。

用法（在 code 目录下）：
    python make_tables.py                 # 用 results/ + results_lr/ 生成
    python make_tables.py --lr 1e-3       # 只看单一学习率（等价于当前论文）
    python make_tables.py --outdir tables

输出：
    tables/table_eurosat_auto.tex    对应正文 Table 1
    tables/table_dtd_auto.tex        对应正文 Table 2
    tables/table_collapse_auto.tex   对应正文 Table 3
    tables/table_stats_auto.tex      对应正文 Table 4
    tables/table_app_vits_auto.tex   对应附录 A.8
    tables/table_convnext_auto.tex   对应附录 A.9
    tables/selection.csv             每个格子选了哪一档学习率及其验证集准确率

选择规则：同一 (dataset, backbone, method, ratio) 下，对每档学习率取三个种子的
平均验证集准确率（best_val_acc），取最大的那一档；并列时取较大的学习率
（即更接近主网格 1e-3 的那档），避免因为噪声把结论推向"必须调参"。
测试集只用于报告，不参与任何选择。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

METHOD_ORDER = ["zeroshot", "coop", "linear", "bitfit", "lora", "adapter", "paa", "full"]
PEFT = ["linear", "bitfit", "lora", "adapter", "paa"]
PRETTY = {
    "zeroshot": "Zero-shot",
    "coop": "CoOp",
    "linear": "Linear probe",
    "bitfit": "BitFit",
    "lora": "LoRA",
    "adapter": "Adapter",
    "paa": "PAA",
    "full": "Full FT",
}
MAIN_BACKBONES = ["clip-vit-b16", "vit-b16", "resnet50"]
BACKBONE_HEAD = {
    "clip-vit-b16": "CLIP ViT-B/16 (contrastive)",
    "vit-b16": "ViT-B/16 (supervised)",
    "vit-small": "ViT-S/16 (supervised)",
    "convnext-tiny": "ConvNeXt-T",
    "resnet50": "ResNet-50",
}
DATASET_HEAD = {"eurosat": "EuroSAT", "dtd": "DTD"}
PARAMS = {
    "clip-vit-b16": "86.0M",
    "vit-b16": "86.0M",
    "vit-small": "21.7M",
    "convnext-tiny": "28.6M",
    "resnet50": "25.0M",
}


def load_patterns(main_dir: str, lr_dir: str) -> list[str]:
    """主结果目录取全部 json；补跑目录只取带 _lr 后缀的，避免与主结果重复计数。"""
    pats = [os.path.join(main_dir, "*.json")]
    if os.path.isdir(lr_dir):
        pats += [os.path.join(lr_dir, f"*_lr*.json")]
    return pats


def load(patterns: list[str]) -> dict:
    """返回 runs[(ds,bb,method,ratio)][seed][lr] = dict(acc, val)"""
    runs = defaultdict(lambda: defaultdict(dict))
    for pat in patterns:
        for f in glob.glob(pat):
            name = os.path.basename(f)[:-5]
            lr = 1e-3
            if "_lr" in name:
                base, lrs = name.rsplit("_lr", 1)
                try:
                    lr = float(lrs)
                except ValueError:
                    continue
                name = base
            if any(name.endswith(sfx) for sfx in ("_noproto", "_anchor_noproto", "_anchor")):
                continue
            try:
                prefix, s = name.rsplit("_s", 1)
                head, r = prefix.rsplit("_r", 1)
                seed, ratio = int(s), float(r)
            except ValueError:
                continue
            parts = head.split("_")
            if len(parts) < 3:
                continue
            method, backbone, dataset = parts[-1], parts[-2], "_".join(parts[:-2])
            try:
                d = json.load(open(f, encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"[warn] 跳过损坏的结果文件 {f}: {e}")
                continue
            runs[(dataset, backbone, method, ratio)][seed][round(lr, 8)] = {
                "acc": d["acc"] * 100,
                "val": d.get("best_val_acc", float("nan")),
            }
    return runs


def choose_lr(runs, cell, allowed=None):
    """按平均验证集准确率选学习率；返回 (lr, {seed: acc})"""
    per_seed = runs.get(cell, {})
    if not per_seed:
        return None, {}
    lrs = set()
    for s, d in per_seed.items():
        lrs |= set(d)
    if allowed:
        lrs &= set(allowed)
    if not lrs:
        return None, {}
    best_lr, best_val = None, -1.0
    for lr in sorted(lrs, reverse=True):  # 并列时优先大学习率
        vals = [d[lr]["val"] for d in per_seed.values() if lr in d]
        if len(vals) < len(per_seed):  # 该档没跑全所有种子，不参与选择
            continue
        v = st.mean(vals)
        if v > best_val + 1e-9:
            best_lr, best_val = lr, v
        elif abs(v - best_val) <= 1e-9 and lr > best_lr:
            best_lr = lr
    if best_lr is None:  # 都跑不全，退回到有结果的那档
        best_lr = max(lrs)
    accs = {s: d[best_lr]["acc"] for s, d in per_seed.items() if best_lr in d}
    return best_lr, accs


def fmt(accs):
    if not accs:
        return "--"
    vals = list(accs.values())
    if len(vals) == 1:
        return f"{vals[0]:.2f} (s{next(iter(accs))})"
    return f"{st.mean(vals):.2f}$\\pm${st.stdev(vals):.2f}"


def grid_table(runs, datasets, backbones, allowed_lr, caption, label, selection):
    """datasets/backbones 各取一个时，列按 backbone 分组；单 backbone 多 dataset 时按 dataset 分组。"""
    group_by = "backbone" if len(datasets) == 1 else "dataset"
    cols = []
    for bb in backbones:
        for ds in datasets:
            ratios = sorted({c[3] for c in runs if c[0] == ds and c[1] == bb})
            for r in ratios:
                cols.append((ds, bb, r))

    groups = []
    for c in cols:
        key = c[1] if group_by == "backbone" else c[0]
        if not groups or groups[-1][0] != key:
            groups.append([key, 0])
        groups[-1][1] += 1

    head1 = []
    for key, n in groups:
        name = BACKBONE_HEAD[key] if group_by == "backbone" else DATASET_HEAD[key]
        head1.append(f"\\multicolumn{{{n}}}{{c}}{{{name}}}")
    span, pos = [], 2
    for _, n in groups:
        span.append(f"\\cmidrule(lr){{{pos}-{pos+n-1}}}")
        pos += n

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        f"\\begin{{tabular}}{{l{'c'*len(cols)}}}",
        "\\toprule",
        " & " + " & ".join(head1) + " \\\\",
        " ".join(span),
        "Method & " + " & ".join(f"{int(round(r*100))}\\%" for _, _, r in cols) + " \\\\",
        "\\midrule",
    ]
    for m in METHOD_ORDER:
        cells = []
        for ds, bb, r in cols:
            lr, accs = choose_lr(runs, (ds, bb, m, r), allowed_lr)
            if lr is None:
                cells.append("--")
            else:
                cells.append(fmt(accs))
                selection.append((ds, bb, m, r, lr, len(accs)))
        if all(c == "--" for c in cells):
            continue
        lines.append(f"{PRETTY[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def collapse_table(runs, allowed_lr, selection):
    out = []
    for dataset in ("eurosat", "dtd"):
        for ratio in (0.01, 0.1):
            for bb in ["clip-vit-b16", "vit-b16", "vit-small", "convnext-tiny", "resnet50"]:
                _, acc_full = choose_lr(runs, (dataset, bb, "full", ratio), allowed_lr)
                if not acc_full:
                    continue
                best, best_name = -1, ""
                for m in PEFT:
                    lr, accs = choose_lr(runs, (dataset, bb, m, ratio), allowed_lr)
                    if accs and st.mean(accs.values()) > best:
                        best, best_name = st.mean(accs.values()), m
                out.append((dataset, bb, ratio, st.mean(acc_full.values()), best, best_name))
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Full fine-tuning against the strongest parameter-efficient method, "
        "with the learning rate chosen per method on the validation split.}",
        "\\label{tab:collapse-auto}", "\\small", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llcccc}", "\\toprule",
        "Dataset & Backbone & Params & $\\rho$ & Full FT & Best PEFT ($\\Delta$) \\\\",
        "\\midrule",
    ]
    prev = None
    for ds, bb, r, full, best, name in out:
        if prev and (ds, r) != prev:
            lines.append("\\midrule")
        prev = (ds, r)
        lines.append(
            f"{ds.capitalize()} & {bb} & {PARAMS.get(bb,'--')} & {int(round(r*100))}\\% & "
            f"{full:.2f} & {best:.2f} ({full-best:+.2f}) \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def paired_stats(runs, allowed_lr, selection):
    rows = []
    for dataset in ("eurosat", "dtd"):
        for ratio in (0.01, 0.1, 1.0):
            for bb in ["clip-vit-b16", "vit-b16", "resnet50"]:
                _, paa = choose_lr(runs, (dataset, bb, "paa", ratio), allowed_lr)
                if not paa:
                    continue
                best, best_name = -1, ""
                for m in ["adapter", "lora", "bitfit", "linear", "full"]:
                    _, accs = choose_lr(runs, (dataset, bb, m, ratio), allowed_lr)
                    if accs and st.mean(accs.values()) > best:
                        best, best_name = st.mean(accs.values()), m
                seeds = sorted(set(paa) & set(
                    choose_lr(runs, (dataset, bb, best_name, ratio), allowed_lr)[1]
                ))
                if not best_name or len(seeds) < 2:
                    continue
                other = choose_lr(runs, (dataset, bb, best_name, ratio), allowed_lr)[1]
                diffs = [paa[s] - other[s] for s in seeds]
                md = st.mean(diffs)
                sd = st.stdev(diffs)
                t = md / (sd / math.sqrt(len(diffs))) if sd > 0 else float("inf")
                p = 1 - abs(t) / math.sqrt(2 + t * t)  # df = 2 的闭式解
                rows.append((dataset, bb, ratio, md, t, p, best - st.mean(paa.values()), best_name))
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Paired comparison of PAA against the strongest baseline in the cell "
        "($df=2$; $p$ two-sided).}", "\\label{tab:stats-auto}", "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llcccc l}", "\\toprule",
        "Dataset & Backbone & $\\rho$ & $\\Delta$ (pts) & $t$ & $p$ & baseline \\\\",
        "\\midrule",
    ]
    for ds, bb, r, md, t, p, _, name in rows:
        mark = "\\textbf" if p < 0.05 else ""
        pc = f"{mark}{{{p:.3f}}}" if mark else f"{p:.2f}"
        lines.append(
            f"{ds.capitalize()} & {bb} & {int(round(r*100))}\\% & ${md:+.2f}$ & "
            f"${t:.2f}$ & {pc} & {name} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--lr-results", default="results_lr")
    ap.add_argument("--outdir", default="tables")
    ap.add_argument("--lr", type=float, nargs="*", default=None,
                    help="限制只用这些学习率（默认全部可用档）")
    args = ap.parse_args()

    runs = load(load_patterns(args.results, args.lr_results))
    if not runs:
        raise SystemExit("没读到结果，检查 --results 路径")
    allowed = set(round(x, 8) for x in args.lr) if args.lr else None
    os.makedirs(args.outdir, exist_ok=True)
    selection: list = []

    files = {
        "table_eurosat_auto.tex": grid_table(
            runs, ["eurosat"], MAIN_BACKBONES, allowed,
            "EuroSAT: top-1 accuracy (\\%) for three of the five backbones, learning "
            "rate chosen per method on the validation split.", "tab:eurosat-auto", selection),
        "table_dtd_auto.tex": grid_table(
            runs, ["dtd"], MAIN_BACKBONES, allowed,
            "DTD: top-1 accuracy (\\%) for three of the five backbones.",
            "tab:dtd-auto", selection),
        "table_app_vits_auto.tex": grid_table(
            runs, ["eurosat", "dtd"], ["vit-small"], allowed,
            "ViT-S/16 (supervised, 21.7M): top-1 accuracy (\\%).",
            "tab:app-vits-auto", selection),
        "table_convnext_auto.tex": grid_table(
            runs, ["eurosat", "dtd"], ["convnext-tiny"], allowed,
            "ConvNeXt-T (28.6M): top-1 accuracy (\\%).",
            "tab:app-convnext-auto", selection),
        "table_collapse_auto.tex": collapse_table(runs, allowed, selection),
        "table_stats_auto.tex": paired_stats(runs, allowed, selection),
    }
    for name, tex in files.items():
        with open(os.path.join(args.outdir, name), "w", encoding="utf-8") as fh:
            fh.write(tex + "\n")
        print(f"[write] {args.outdir}/{name}")

    with open(os.path.join(args.outdir, "selection.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "backbone", "method", "ratio", "lr", "n_seeds"])
        for row in sorted(set(selection)):
            w.writerow(row)
    print(f"[write] {args.outdir}/selection.csv")

    changed = [s for s in selection if abs(s[4] - 1e-3) > 1e-12]
    print(f"\n选了非 1e-3 学习率的格子数：{len(changed)} / {len(set(selection))}")
    print("（为 0 说明补跑结果没有改变任何选择，主表数字不变）")


if __name__ == "__main__":
    main()
