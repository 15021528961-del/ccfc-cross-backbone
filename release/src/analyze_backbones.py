"""三个主干横向对比：CLIP ViT-B/16 vs ViT-B/16 (监督) vs ResNet-50。

用法：cd code && python analyze_backbones.py
"""
import glob
import json
import os
import statistics as st

BACKBONES = [("clip-vit-b16", "CLIP ViT-B/16"), ("vit-b16", "ViT-B/16 sup."),
             ("vit-small", "ViT-S/16 21.7M"), ("convnext-tiny", "ConvNeXt-T"), ("resnet50", "ResNet-50")]
METHODS = [("linear", "Linear"), ("bitfit", "BitFit"), ("lora", "LoRA"),
           ("adapter", "Adapter"), ("paa", "PAA"), ("full", "Full FT")]


def load():
    rows = {}
    for f in glob.glob("results/*.json"):
        if "_lr" in os.path.basename(f):
            continue
        d = json.load(open(f))
        rows.setdefault((d["dataset"], d["backbone"], d["method"], d["ratio"]), []).append(d["acc"] * 100)
    return rows


def main() -> None:
    rows = load()
    for ds, dsname in [("eurosat", "EuroSAT"), ("dtd", "DTD")]:
        for ratio in (0.01, 0.1):
            print(f"\n=== {dsname} @ rho={ratio:.0%} ===")
            header = f"{'method':<10}" + "".join(f"{n:>18}" for _, n in BACKBONES)
            print(header)
            print("-" * len(header))
            for m, disp in METHODS:
                cells = []
                for bb, _ in BACKBONES:
                    v = rows.get((ds, bb, m, ratio))
                    cells.append(f"{st.mean(v):.2f}±{st.stdev(v):.2f}" if v and len(v) > 1
                                 else (f"{st.mean(v):.2f}" if v else "--"))
                print(f"{disp:<10}" + "".join(f"{c:>18}" for c in cells))

            # 关键对比：三个主干上"全量微调 vs 最好的参数高效方法"
            print()
            for bb, name in BACKBONES:
                fv = rows.get((ds, bb, "full", ratio))
                if not fv:
                    continue
                best = max((m for m, _ in METHODS if m != "full"),
                           key=lambda m: st.mean(rows[(ds, bb, m, ratio)]) if rows.get((ds, bb, m, ratio)) else -1)
                bv = rows[(ds, bb, best, ratio)]
                print(f"  {name:<14} full={st.mean(fv):6.2f}   最好PEFT={best}({st.mean(bv):6.2f})   "
                      f"差={st.mean(fv)-st.mean(bv):+6.2f}")


if __name__ == "__main__":
    main()
