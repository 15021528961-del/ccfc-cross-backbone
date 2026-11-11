"""冒烟测试：不依赖真实数据集与预训练权重，验证每种方法能前向、能回传、参数量合理。

用法：cd code && python -m src.smoke_test
"""
import torch
import torch.nn as nn

from src.adapters import METHODS, AdaptModel, inject, set_adapter_enabled
from src.backbones import Backbone
from src.engine import evaluate
from src.utils import count_trainable


class DummyBackbone(Backbone):
    """小型 CNN + 两层 Linear，模拟真实主干的接口。"""

    def _build(self) -> None:
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self._dim = 64

    @property
    def feature_dim(self) -> int:
        return self._dim

    @property
    def trunk(self) -> nn.Module:
        return self.encoder

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def run() -> None:
    torch.manual_seed(0)
    B, C, H, W = 8, 3, 32, 32
    num_classes = 10
    x = torch.randn(B, C, H, W)
    y = torch.randint(0, num_classes, (B,))
    ok = True

    for method in [m for m in METHODS if m != "zeroshot"]:
        bb = DummyBackbone("dummy")
        kw = dict(patterns=("",)) if method in ("lora", "adapter", "paa") else {}
        model = AdaptModel(bb, method, num_classes, r=4, alpha=8, bottleneck=16,
                           l2sp_lambda=0.5, **kw)
        protos = torch.randn(num_classes, model.backbone.feature_dim)
        out = model(x, y, protos if method == "paa" else None)
        loss = out["loss"]
        loss.backward()

        trainable = count_trainable(model)
        n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        assert torch.isfinite(loss), f"{method}: loss 非有限值"
        assert n_grad > 0, f"{method}: 没有任何参数拿到梯度"
        assert out["logits"].shape == (B, num_classes), f"{method}: logits 形状错误"
        print(f"[ok] {method:9s} loss={loss.item():.4f} trainable={trainable:>8,} "
              f"grad_tensors={n_grad}")

        # 适配器开关：关闭后应回到主干原始输出
        if method in ("lora", "adapter", "paa"):
            with torch.no_grad():
                f_on = model.features(x, adapt=True)
                f_off = model.features(x, adapt=False)
            zero_init_ok = torch.allclose(f_on, f_off, atol=1e-5)
            print(f"       适配器零初始化一致: {zero_init_ok}")
            ok &= zero_init_ok

    # ---------------------------------------------------------------- #
    # 注入数量与梯度隔离
    bb = DummyBackbone("dummy")
    n = inject(bb.trunk, "adapter", ("",), bottleneck=8)
    assert n == 3, f"期望注入 3 层（1 Conv2d + 2 Linear），实际 {n}"
    bb2 = DummyBackbone("dummy")
    m = AdaptModel(bb2, "paa", num_classes, patterns=("",))
    frozen = [p for n_, p in m.backbone.named_parameters() if "up" not in n_ and "down" not in n_]
    assert all(not p.requires_grad for p in frozen), "主干参数未被冻结"
    print(f"[ok] inject 替换层数={n}，主干参数已冻结")

    # ---------------------------------------------------------------- #
    # 评估循环
    from torch.utils.data import DataLoader, TensorDataset

    class Wrap(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.x[i], self.y[i], i

    loader = DataLoader(Wrap(x, y), batch_size=4)
    res = evaluate(m, loader, "cpu")
    print(f"[ok] evaluate acc={res['acc']:.3f} macro={res['macro_acc']:.3f} n={res['n']}")
    assert res["n"] == B

    # 适配器全关时 set_adapter_enabled 行为
    set_adapter_enabled(m.backbone, False)
    res2 = evaluate(m, loader, "cpu")
    print(f"[ok] 适配器关闭后 acc={res2['acc']:.3f}")

    # 原型估计
    protos = m.estimate_prototypes(loader, "cpu")
    assert protos.shape == (num_classes, m.backbone.feature_dim)
    print("[ok] estimate_prototypes 形状正确")

    print("\n所有冒烟测试通过。" if ok else "\n有测试未通过，请检查上方输出。")


if __name__ == "__main__":
    run()
