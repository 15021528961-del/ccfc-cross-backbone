"""单次实验入口。

用法示例
--------
python -m src.train --dataset eurosat --backbone clip-vit-b16 --method paa \
    --ratio 0.01 --seed 0 --epochs 30 --out results
"""
import argparse
import os
import sys
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters import METHODS, AdaptModel  # noqa: E402
from src.backbones import BACKBONES, build_transforms, get_backbone  # noqa: E402
from src.data import KNOWN_DATASETS, build_datasets  # noqa: E402
from src.engine import evaluate, train  # noqa: E402
from src.utils import get_logger, save_json, seed_everything  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="eurosat",
                   help="数据目录名：{data_root}/{dataset}/<类别>/*.jpg")
    p.add_argument("--backbone", default="clip-vit-b16", choices=BACKBONES)
    p.add_argument("--method", default="paa", choices=METHODS)
    p.add_argument("--ratio", type=float, default=1.0, help="训练集标签比例，1.0 表示全量")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head-lr-mult", type=float, default=10.0)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--val-subset", type=int, default=1000,
                   help="训练中用于早停的验证子集大小，最终测试仍用完整测试集")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--r", type=int, default=8, help="LoRA 秩")
    p.add_argument("--alpha", type=int, default=16, help="LoRA 缩放")
    p.add_argument("--bottleneck", type=int, default=32)
    p.add_argument("--paa-lambda", type=float, default=0.5, help="原型锚定正则权重")
    p.add_argument("--anchor", action="store_true", help="强制开启原型锚定（消融用）")
    p.add_argument("--no-proto-init", action="store_true", help="分类头不用原型初始化（消融用）")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--cache-dir", default="./splits")
    p.add_argument("--out", default="./results")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def eval_zeroshot(backbone, loader, classes, device) -> Dict[str, float]:
    backbone.to(device).eval()
    correct = total = 0
    t0 = time.time()
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = backbone.zero_shot(x, classes)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return {"acc": correct / max(total, 1), "macro_acc": float("nan"),
            "eval_time_sec": time.time() - t0, "n": total}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{args.dataset}_{args.backbone}_{args.method}_r{args.ratio}_s{args.seed}"
    if abs(args.lr - 1e-3) > 1e-9:
        tag += f"_lr{args.lr:g}"  # 换学习率的实验单独存，避免互相覆盖
    if args.anchor:
        tag += "_anchor"
    if args.no_proto_init:
        tag += "_noproto"
    os.makedirs(args.out, exist_ok=True)
    logger = get_logger(os.path.join(args.out, tag))
    logger.info(f"=== {tag} device={device} ===")

    tf_train = build_transforms(args.backbone, True, args.image_size)
    tf_eval = build_transforms(args.backbone, False, args.image_size)
    ds_train, ds_val, ds_test, classes = build_datasets(
        args.dataset, args.data_root, tf_train, tf_eval, args.cache_dir,
        args.ratio, args.seed,
    )
    logger.info(f"类别数={len(classes)} train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")

    common = dict(num_workers=args.num_workers, pin_memory=(device == "cuda"),
                  persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False, **common)

    # 训练过程中用一个固定的验证子集做模型选择，把每轮评估开销降下来；
    # 最终测试仍然在完整测试集上进行（见下方 test_loader）。
    if len(ds_val) > args.val_subset:
        step = max(1, len(ds_val) // args.val_subset)
        val_idx = list(range(0, len(ds_val), step))[: args.val_subset]
    else:
        val_idx = list(range(len(ds_val)))
    val_loader = DataLoader(Subset(ds_val, val_idx), batch_size=args.batch_size, shuffle=False, **common)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, **common)

    backbone = get_backbone(args.backbone)

    if args.method == "zeroshot":
        metrics = eval_zeroshot(backbone, test_loader, classes, device)
        metrics.update({"method": args.method, "dataset": args.dataset,
                        "backbone": args.backbone, "ratio": args.ratio, "seed": args.seed,
                        "trainable_params": 0,
                        "total_params": sum(p.numel() for p in backbone.parameters())})
    else:
        model = AdaptModel(
            backbone, args.method, len(classes),
            r=args.r, alpha=args.alpha, bottleneck=args.bottleneck,
            l2sp_lambda=args.paa_lambda,
            use_anchor=(True if args.anchor else None),
        )
        model.to(device)
        protos = model.estimate_prototypes(train_loader, device)
        if args.method != "full" and not args.no_proto_init:
            model.head.init_from_prototypes(protos)
        detach_grads = []
        if args.method == "bitfit":
            import torch.nn as _nn
            for _m in model.modules():
                if isinstance(_m, _nn.LayerNorm) or type(_m).__name__.startswith("LayerNorm"):
                    w = getattr(_m, "weight", None)
                    if w is not None and not w.requires_grad:
                        w.requires_grad_(True)
                        detach_grads.append(w)
            if detach_grads:
                logger.info(f"BitFit: {len(detach_grads)} 个 LayerNorm weight 参与反向但不更新")

        ckpt_file = os.path.join(args.out, f"ckpt_{tag}.pt")
        tr = train(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, head_lr_mult=args.head_lr_mult,
            wd=args.wd, patience=args.patience, amp=not args.no_amp,
            prototypes=protos if model.use_anchor else None,
            log_fn=logger.info, ckpt_path=ckpt_file, detach_grads=detach_grads,
        )
        ev = evaluate(model, test_loader, device)
        metrics = {"acc": ev["acc"], "macro_acc": ev["macro_acc"], **tr, "method": args.method,
                   "dataset": args.dataset, "backbone": args.backbone,
                   "ratio": args.ratio, "seed": args.seed}
        try:
            os.remove(ckpt_file)  # 正常跑完就删掉断点
        except OSError:
            pass
        del model

    save_json(os.path.join(args.out, f"{tag}.json"), metrics)
    logger.info(f"TEST acc={metrics['acc']:.4f} macro={metrics.get('macro_acc', float('nan')):.4f}")


if __name__ == "__main__":
    main()
