"""训练与评估循环。"""
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import AverageMeter, count_trainable


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    per_class_correct: Dict[int, int] = {}
    per_class_total: Dict[int, int] = {}
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        logits = out["logits"]
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for t, p in zip(y.tolist(), pred.tolist()):
            per_class_total[t] = per_class_total.get(t, 0) + 1
            per_class_correct[t] = per_class_correct.get(t, 0) + int(t == p)
    accs = [per_class_correct.get(c, 0) / n for c, n in per_class_total.items() if n > 0]
    return {
        "acc": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "macro_acc": sum(accs) / max(len(accs), 1),
        "n": total,
    }


def build_optimizer(model: nn.Module, lr: float, head_lr_mult: float, wd: float):
    head_ids = {id(p) for p in model.head.parameters()}
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head_params, "lr": lr * head_lr_mult}]
    if rest:
        groups.append({"params": rest, "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=wd)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 30,
    lr: float = 1e-3,
    head_lr_mult: float = 10.0,
    wd: float = 0.05,
    patience: int = 8,
    amp: bool = True,
    prototypes: Optional[torch.Tensor] = None,
    log_fn=None,
    ckpt_path: Optional[str] = None,
    detach_grads=None,
) -> Dict[str, object]:
    model.to(device)
    opt = build_optimizer(model, lr, head_lr_mult, wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device == "cuda")

    best = {"acc": -1.0, "epoch": -1, "state": None}
    history: List[Dict[str, float]] = []
    t0 = time.time()
    stale = 0
    start_ep = 0

    # ---------------- 断点续跑：断电/关机后从上次的 epoch 接着来 ----------------
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_ep = int(ck["epoch"]) + 1
            best = ck["best"]
            history = ck.get("history", [])
            stale = ck.get("stale", 0)
            if log_fn:
                log_fn(f"  [断点续跑] 已恢复到第 {start_ep} 轮")
        except Exception as e:  # noqa: BLE001
            if log_fn:
                log_fn(f"  [断点读取失败，从头开始] {e}")
            start_ep = 0

    for ep in range(start_ep, epochs):
        model.train()
        meter = AverageMeter()
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and device == "cuda"):
                out = model(x, y, prototypes)
                loss = out["loss"]
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            # BitFit 用：这些参数参与反向是为了绕开融合 LayerNorm 的形状检查，
            # 但它们不应被更新，所以每步清掉梯度，优化器会跳过。
            if detach_grads:
                for per in detach_grads:
                    per.grad = None
            meter.update(loss.item(), x.size(0))
        sched.step()

        val = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "train_loss": meter.avg, "val_acc": val["acc"]})
        if log_fn:
            log_fn(f"  ep{ep:02d} loss={meter.avg:.4f} val_acc={val['acc']:.4f}")
        if val["acc"] > best["acc"]:
            best = {
                "acc": val["acc"],
                "epoch": ep,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

        # 每个 epoch 落一次盘，断电最多损失一轮
        if ckpt_path:
            try:
                torch.save(
                    {
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "epoch": ep,
                        "best": best,
                        "history": history,
                        "stale": stale,
                    },
                    ckpt_path,
                )
            except Exception:  # noqa: BLE001
                pass

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {
        "best_val_acc": best["acc"],
        "best_epoch": best["epoch"],
        "history": history,
        "train_time_sec": time.time() - t0,
        "trainable_params": count_trainable(model),
        "total_params": sum(p.numel() for p in model.parameters()),
    }
