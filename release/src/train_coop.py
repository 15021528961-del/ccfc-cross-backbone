"""CoOp 基线的训练入口。

用法：
    python -m src.train_coop --dataset eurosat --ratio 0.01 --seed 0 --epochs 20

结果写到 results/ 下，命名与其它方法一致（method=coop），因此
aggregate.py / plot.py / analyze_backbones.py 都能直接读到。
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backbones import build_transforms, _ensure_hf_endpoint  # noqa: E402
from src.coop import CoOp  # noqa: E402
from src.data import build_datasets  # noqa: E402
from src.utils import AverageMeter, get_logger, save_json, seed_everything  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)["logits"]
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="eurosat")
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--n-ctx", type=int, default=16, help="上下文向量个数")
    p.add_argument("--val-subset", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--cache-dir", default="./splits")
    p.add_argument("--out", default="./results")
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{args.dataset}_clip-vit-b16_coop_r{args.ratio}_s{args.seed}"
    if abs(args.lr - 1e-3) > 1e-9:
        tag += f"_lr{args.lr:g}"
    os.makedirs(args.out, exist_ok=True)
    log = get_logger(os.path.join(args.out, tag))
    log.info(f"=== {tag} device={device} ===")

    tf_train = build_transforms("clip-vit-b16", True, args.image_size)
    tf_eval = build_transforms("clip-vit-b16", False, args.image_size)
    ds_tr, ds_va, ds_te, classes = build_datasets(
        args.dataset, args.data_root, tf_train, tf_eval, args.cache_dir, args.ratio, args.seed
    )
    log.info(f"类别数={len(classes)} train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}")

    common = dict(num_workers=args.num_workers, pin_memory=(device == "cuda"),
                  persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, **common)
    if len(ds_va) > args.val_subset:
        step = max(1, len(ds_va) // args.val_subset)
        val_idx = list(range(0, len(ds_va), step))[: args.val_subset]
    else:
        val_idx = list(range(len(ds_va)))
    val_loader = DataLoader(Subset(ds_va, val_idx), batch_size=args.batch_size, shuffle=False, **common)
    test_loader = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, **common)

    _ensure_hf_endpoint()
    from transformers import CLIPModel, CLIPTokenizer

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    model = CoOp(clip, tok, classes, n_ctx=args.n_ctx).to(device)

    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_acc, best_state, stale = -1.0, None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        meter = AverageMeter()
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = model(x, y)["loss"]
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            meter.update(loss.item(), x.size(0))
        sched.step()

        acc = evaluate(model, val_loader, device)
        log.info(f"  ep{ep:02d} loss={meter.avg:.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, stale = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    log.info(f"TEST acc={test_acc:.4f}")

    save_json(os.path.join(args.out, f"{tag}.json"), {
        "dataset": args.dataset, "backbone": "clip-vit-b16", "method": "coop",
        "ratio": args.ratio, "seed": args.seed, "acc": test_acc, "macro_acc": float("nan"),
        "trainable_params": sum(p.numel() for p in params),
        "total_params": sum(p.numel() for p in model.parameters()),
        "train_time_sec": time.time() - t0,
        "best_val_acc": best_acc,
    })


if __name__ == "__main__":
    main()
