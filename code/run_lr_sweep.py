"""补跑主网格缺少的学习率，为重算表格做准备。

用法（在 code 目录下，或直接双击根目录的 run_lr_sweep.bat）：
    python run_lr_sweep.py --status       # 先看进度/还剩多少
    python run_lr_sweep.py --dry-run      # 只打印要跑什么
    python run_lr_sweep.py                # 开跑；可随时 Ctrl-C，重跑会自动跳过已完成的

补哪些、为什么这么补：
    所有方法 × {1e-4, 3e-4}            —— 让"按方法选学习率"这件事有得选
    仅 full × {1e-5, 3e-5}             —— §5.7 的结论是"网格下界压在最优点的边界上"，
                                          这一档决定全量微调真正的天花板，是全文最关键的
                                          待补实验，所以排在最前面跑

执行顺序（停在任何时候都有可用结果）：
    第 1 段  full  @ 1e-5, 3e-5   约 130 次，最 decisive
    第 2 段  全部  @ 1e-4         约 410 次
    第 3 段  全部  @ 3e-4         约 410 次

其它说明：
* 待跑清单由 results/*.json 反推，主网格跑过哪些格子就补哪些格子。
* 结果写入 results_lr/，文件名带 _lr 后缀（src/train.py 自动加），不覆盖主网格。
* 已存在同名 json 就跳过，所以可以断点续跑，也可以分几个晚上跑。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- 计划

# 所有方法都补这两档
COMMON_LRS = [1e-4, 3e-4]
# 只对全量微调补这两档（关键：包住它的最优点）
FULL_ONLY_LRS = [1e-5, 3e-5]
# 可选：CNN 上的全量微调看起来还在 1e-3 处上升，补两档更高的把它的最优点也包住
CNN_HIGH_LRS = [3e-3, 1e-2]
CNN_BACKBONES = {"resnet50", "convnext-tiny"}

EPOCHS = 20
EPOCHS_FULL = 10  # ρ=100% 时主网格用的是 10

# 与各自 run_*.bat 保持一致
NO_AMP_BACKBONES = {"vit-small", "convnext-tiny"}


def tag(dataset: str, backbone: str, method: str, ratio: float, seed: int) -> str:
    """必须与 src/train.py 的 tag 逐字一致。
    注意那里用的是 f"r{args.ratio}"，所以 ρ=1.0 会写成 r1.0 而不是 r1。"""
    return f"{dataset}_{backbone}_{method}_r{ratio}_s{seed}"


def tag_legacy(dataset: str, backbone: str, method: str, ratio: float, seed: int) -> str:
    """旧版脚本用过 :g 格式（r1），保留一个兼容检查，免得已有结果被当没跑过。"""
    return f"{dataset}_{backbone}_{method}_r{ratio:g}_s{seed}"


_LOG_FH = None


def log(msg: str = "") -> None:
    """同时打印到控制台和 results_lr/sweep.log，方便挂机后回看。"""
    print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


def parse_name(name: str):
    """把 eurosat_clip-vit-b16_paa_r0.01_s0 拆成五元组，认不出返回 None。"""
    try:
        prefix, s = name.rsplit("_s", 1)
        head, r = prefix.rsplit("_r", 1)
        seed, ratio = int(s), float(r)
    except ValueError:
        return None
    parts = head.split("_")
    if len(parts) < 3:
        return None
    method, backbone, dataset = parts[-1], parts[-2], "_".join(parts[:-2])
    return dataset, backbone, method, ratio, seed


def collect_cells(results_dir: Path):
    """返回 (cells, durations)。cells 是主网格里真实跑过的格子集合，
    durations[(ds,bb,method,ratio)] 是已知单次耗时（秒），用于估算总时长。"""
    cells, durations = set(), {}
    for f in sorted(results_dir.glob("*.json")):
        if f.stem.endswith(("_lr",)) or "_lr" in f.stem:
            continue
        key = parse_name(f.stem)
        if key is None:
            continue
        cells.add(key)
        try:
            t = json.load(open(f, encoding="utf-8")).get("train_time_sec")
        except Exception:
            t = None
        if t:
            k = key[:4]
            durations.setdefault(k, []).append(t)
    return cells, durations


def planned(cells, cnn_high: bool = False):
    """生成 (cell, lr, phase) 列表，按 phase 排序。"""
    out = []
    for cell in cells:
        ds, bb, m, r, s = cell
        lrs = list(COMMON_LRS)
        if m == "full":
            lrs += FULL_ONLY_LRS
            if cnn_high and bb in CNN_BACKBONES:
                lrs += CNN_HIGH_LRS
        for lr in lrs:
            if m == "full" and lr in FULL_ONLY_LRS:
                phase = 0
            elif lr == 1e-4:
                phase = 1
            elif cnn_high and m == "full" and bb in CNN_BACKBONES and lr in CNN_HIGH_LRS:
                phase = 2
            else:
                phase = 3
            out.append((cell, lr, phase))
    out.sort(key=lambda x: (x[2], x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[1]))
    return out


def out_path(out_dir: Path, cell, lr: float) -> Path:
    return out_dir / f"{tag(*cell)}_lr{lr:g}.json"


def is_done(out_dir: Path, cell, lr: float) -> bool:
    """两种命名都认，避免因为 r1 / r1.0 的写法差异把已完成的格子当成没跑。"""
    return out_path(out_dir, cell, lr).exists() or (out_dir / f"{tag_legacy(*cell)}_lr{lr:g}.json").exists()


def build_command(cell, lr: float, out_dir: str) -> list[str]:
    dataset, backbone, method, ratio, seed = cell
    epochs = EPOCHS_FULL if ratio >= 1.0 else EPOCHS
    if method == "coop":
        # CoOp 有独立入口，且写死了 clip-vit-b16，不认 --backbone/--method
        return [
            sys.executable, "-m", "src.train_coop",
            "--dataset", dataset, "--ratio", str(ratio), "--seed", str(seed),
            "--epochs", str(epochs), "--lr", str(lr), "--out", out_dir,
            "--num-workers", "2",
        ]
    cmd = [
        sys.executable, "-m", "src.train",
        "--dataset", dataset,
        "--backbone", backbone,
        "--method", method,
        "--ratio", str(ratio),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--out", out_dir,
        "--num-workers", "2",
    ]
    if backbone in NO_AMP_BACKBONES:
        cmd.append("--no-amp")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results_lr")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（调试用）")
    ap.add_argument("--cnn-high", action="store_true",
                    help="额外补 CNN 全量微调的 3e-3 / 1e-2 两档，用于包住 CNN 的最优点")
    ap.add_argument("--retries", type=int, default=2,
                    help="单个格子失败后重试几次（默认 2 = 总共跑 2 次）")
    ap.add_argument("--only", choices=["full", "common", "cnn", "hi"], default=None,
                    help="full=段1；common=段2；cnn=段3；hi=段4（3e-4）")
    args = ap.parse_args()

    results_dir, out_dir = Path(args.results), Path(args.out)
    if not results_dir.is_dir():
        sys.exit(f"找不到 {results_dir}，请在 code/ 目录下运行")
    out_dir.mkdir(exist_ok=True)

    cells, durations = collect_cells(results_dir)
    if not cells:
        sys.exit("results/ 里没有可解析的结果文件")

    plan = planned(cells, cnn_high=args.cnn_high)
    wanted = {"full": [0], "common": [1], "cnn": [2], "hi": [3]}
    if args.only:
        plan = [p for p in plan if p[2] in wanted[args.only]]

    todo = [(c, lr, ph) for c, lr, ph in plan if not is_done(out_dir, c, lr)]
    done = len(plan) - len(todo)

    # 用实测耗时估总时长
    est_sec = 0.0
    for cell, lr, _ in todo:
        ts = durations.get(cell[:4])
        est_sec += st.median(ts) if ts else 150.0

    global _LOG_FH
    _LOG_FH = open(out_dir / "sweep.log", "a", encoding="utf-8")
    log(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start =====")

    log(f"主网格格子数        : {len(cells)}")
    log(f"计划补跑            : {len(plan)}")
    log(f"已完成 / 剩余       : {done} / {len(todo)}")
    log(f"剩余粗估耗时        : {est_sec/3600:.1f} 小时")
    for ph, name in enumerate(["段1 full@1e-5,3e-5", "段2 全部@1e-4", "段3 CNN full@3e-3,1e-2", "段4 全部@3e-4"]):
        n = sum(1 for _, _, p in todo if p == ph)
        log(f"  {name:<22}: {n}")

    if args.status:
        return
    if args.dry_run:
        for cell, lr, _ in todo[:15]:
            log(f"  {tag(*cell)}  lr={lr:g}")
        if len(todo) > 15:
            log(f"  ... 共 {len(todo)} 条")
        return
    if not todo:
        log("没有待跑的，全部完成。接着运行 make_tables.py")
        return

    if args.limit:
        todo = todo[: args.limit]

    t_start = time.time()
    for i, (cell, lr, _) in enumerate(todo, 1):
        log(f"\n[{i}/{len(todo)}] {tag(*cell)} lr={lr:g}")
        t1 = time.time()
        err_file = out_dir / f"{tag(*cell)}_lr{lr:g}.err"
        res = None
        for attempt in range(1, args.retries + 1):
            # 捕获子进程输出：失败时把尾部写进 sweep.log 并单独存 .err，方便定位
            res = subprocess.run(
                build_command(cell, lr, args.out),
                cwd=Path.cwd(), capture_output=True, text=True, errors="replace",
            )
            if res.returncode == 0:
                break
            blob = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            err_file.write_text(blob, encoding="utf-8", errors="replace")
            if attempt < args.retries:
                # 多为 HF 镜像的 SSL 抖动，隔一会儿重试一次通常就过了
                log(f"  .. attempt {attempt} failed (rc={res.returncode}), retrying in 20s")
                time.sleep(20)
        elapsed = (time.time() - t1) / 60
        if res.returncode != 0:
            tail = "\n".join((res.stderr or res.stdout or "").strip().splitlines()[-15:])
            log(f"  -> FAILED({res.returncode}) in {elapsed:.1f} min; see {err_file.name}")
            if tail:
                log("  ---- last lines ----")
                for line in tail.splitlines():
                    log("  | " + line)
                log("  --------------------")
        else:
            log(f"  -> ok in {elapsed:.1f} min")
            if err_file.exists():
                err_file.unlink()

    log(f"\n全部结束，用时 {(time.time()-t_start)/3600:.2f} 小时")


if __name__ == "__main__":
    main()
