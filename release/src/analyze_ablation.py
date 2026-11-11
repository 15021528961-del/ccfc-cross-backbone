"""把消融实验的结果汇总成表。

用法：cd code && python analyze_ablation.py
"""
import glob
import json
import os
import statistics as st

COND = [
    # (method, no_proto_init, anchor, 标签)
    # 注意：filename 里的 _noproto 表示"关掉原型初始化"，
    # 默认（无后缀）反而是开着的，标签别写反。
    ("adapter", True, False, "Adapter only"),
    ("adapter", False, False, "+ prototypical init"),
    ("adapter", True, True, "+ anchoring"),
    ("paa", False, False, "PAA (both)"),
]


def parse(name: str):
    method = "paa" if "_paa_" in name else "adapter"
    npi = name.endswith("_noproto")
    anc = "_anchor" in name
    ratio = float(name.split("_r")[1].split("_")[0])
    return method, npi, anc, ratio


def main() -> None:
    rows = {}
    for f in glob.glob("results_ablation/*.json"):
        name = os.path.basename(f)[:-5]
        method, npi, anc, ratio = parse(name)
        d = json.load(open(f, encoding="utf-8"))
        rows.setdefault((method, npi, anc, ratio), []).append(d["acc"] * 100)

    if not rows:
        print("results_ablation 目录里还没有结果，先跑 run_ablation.bat")
        return

    print(f"{'variant':<24}{'1%':>16}{'10%':>16}")
    print("-" * 56)
    for method, npi, anc, label in COND:
        cells = []
        for ratio in (0.01, 0.1):
            v = rows.get((method, npi, anc, ratio))
            if not v:
                cells.append("--")
            elif len(v) > 1:
                cells.append(f"{st.mean(v):.2f}±{st.stdev(v):.2f}")
            else:
                cells.append(f"{st.mean(v):.2f}")
        print(f"{label:<24}" + "".join(f"{c:>16}" for c in cells))

    print("\n各格子的种子数：")
    for method, npi, anc, label in COND:
        ns = [len(rows.get((method, npi, anc, r), [])) for r in (0.01, 0.1)]
        print(f"  {label:<24} {ns}")


if __name__ == "__main__":
    main()
