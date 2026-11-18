"""通用工具：随机种子、日志、指标、IO。"""
import json
import logging
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "run") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    log_path = f"{name}.log"
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)
    return logger


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class Timer:
    def __enter__(self) -> "Timer":
        self.t0 = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.time() - self.t0


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
