"""跨平台实验网格运行器（Windows / Linux / macOS 通用）。

用法
----
python code/run_grid.py --quick                      # 冒烟：EuroSAT + 2 方法 + 1 epoch
python code/run_grid.py                              # 全量网格
python code/run_grid.py --datasets eurosat isic2019  # 指定数据集
python code/run_grid.py --methods paa linear --ratios 0.01 0.1

特性：已完成的配置自动跳过，中途断了直接重跑。
"""
import argparse
import os
import subprocess
import sys
import time

ALL_DATASETS = ["eurosat", "isic2019", "plantvillage"]
ALL_METHODS = ["zeroshot", "linear", "bitfit", "lora", "adapter", "paa", "full"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader 进程数。内存 16G 以上用 8，8G 用 4，Windows 上太高反而慢")
    ap.add_argument("--backbone", default="clip-vit-b16")
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.datasets = ["eurosat"]
        args.methods = ["linear", "paa"]
        args.ratios = [0.01]
        args.seeds = [0]
        args.epochs = 1

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 数据可用性预检 ----
    def _ready(ds: str) -> bool:
        # eurosat 会在运行时自动下载，不做预检
        if ds == "eurosat":
            return True
        root = os.path.join(here, args.data_root)
        return os.path.isdir(os.path.join(root, ds))

    usable = []
    for ds in args.datasets:
        if _ready(ds):
            usable.append(ds)
        else:
            print(f"[跳过] 数据集 {ds} 未就绪，路径应为 {os.path.join(here, args.data_root, ds)}")
    if not usable:
        print("\n没有任何可用的数据集。请先按 README 第二步准备数据。")
        sys.exit(1)
    args.datasets = usable
    print(f"将使用数据集：{', '.join(args.datasets)}\n")

    total = len(args.datasets) * len(args.methods) * len(args.ratios) * len(args.seeds)
    done = 0
    t_start = time.time()
    failed = []

    for ds in args.datasets:
        for m in args.methods:
            for r in args.ratios:
                for s in args.seeds:
                    done += 1
                    tag = f"{ds}_{args.backbone}_{m}_r{r}_s{s}"
                    if os.path.exists(os.path.join(out_dir, f"{tag}.json")):
                        print(f"[{done}/{total}] skip  {tag}")
                        continue
                    if m == "zeroshot" and (r != 1.0 or s != 0):
                        continue
                    # 只有 CLIP 这类图像-文本模型能做零样本；纯视觉主干没有文本编码器
                    if m == "zeroshot" and not args.backbone.startswith("clip"):
                        continue

                    print(f"[{done}/{total}] run   {tag}", flush=True)
                    cmd = [
                        sys.executable, "-m", "src.train",
                        "--dataset", ds, "--backbone", args.backbone, "--method", m,
                        "--ratio", str(r), "--seed", str(s), "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--num-workers", str(args.num_workers),
                        "--out", out_dir, "--data-root", os.path.join(here, args.data_root),
                        "--cache-dir", os.path.join(here, "splits"),
                    ]
                    t0 = time.time()
                    rc = subprocess.call(cmd, cwd=here)
                    if rc != 0:
                        failed.append(tag)
                        print(f"        [失败] {tag}（返回码 {rc}），继续下一个")
                    else:
                        print(f"        用时 {time.time()-t0:.0f}s")

    print("\n" + "=" * 60)
    print(f"完成 {total - len(failed)}/{total}，总耗时 {(time.time()-t_start)/3600:.1f} 小时")
    if failed:
        print("失败配置：")
        for f in failed:
            print("  -", f)
    print("\n下一步：")
    print(f"  python -m src.aggregate --results {args.out} --out tables")
    print(f"  python -m src.plot      --results {args.out} --out figures")


if __name__ == "__main__":
    main()
