"""参数高效适配方法的统一实现，包含本文提出的 PAA。

实现的方法
----------
zeroshot   : CLIP 零样本（不训练）
linear     : 冻结主干 + 线性/余弦分类头
full       : 全参数微调
bitfit     : 仅训练偏置项（Ben Zaken et al., 2022）
lora       : 低秩分解适配（Hu et al., 2022）
adapter    : 残差瓶颈适配器（Houlsby et al., 2019 的残差变体）
paa        : 本文方法 Prototype-Anchored Adapters
             = 残差适配器 + 余弦分类头（原型初始化、可学习温度）+ 原型锚定正则
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

METHODS = ("zeroshot", "linear", "full", "bitfit", "lora", "adapter", "paa")


# --------------------------------------------------------------------------- #
# 基础适配模块
# --------------------------------------------------------------------------- #
class LoRALayer(nn.Module):
    """在冻结的 Linear/Conv2d 旁路并联一个低秩增量。"""

    def __init__(self, base: nn.Module, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.enabled = True
        for p in self.base.parameters():
            p.requires_grad_(False)

        if isinstance(base, nn.Linear):
            self.down = nn.Linear(base.in_features, r, bias=False)
            self.up = nn.Linear(r, base.out_features, bias=False)
            self.kind = "linear"
        elif isinstance(base, nn.Conv2d):
            # 下行卷积必须复刻 base 的核/步长/填充，否则输出尺寸对不上
            self.down = nn.Conv2d(
                base.in_channels, r, base.kernel_size,
                stride=base.stride, padding=base.padding, bias=False,
            )
            self.up = nn.Conv2d(r, base.out_channels, 1, bias=False)
            self.kind = "conv"
        else:
            raise TypeError(f"LoRA 不支持 {type(base)}")

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)  # 初始时增量恒为 0，训练起点等价于原模型

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + self.scale * self.up(self.down(x))
        return out


class ResidualAdapter(nn.Module):
    """瓶颈残差适配：out = base(x) + up(act(down(x)))，up 零初始化。"""

    def __init__(self, base: nn.Module, bottleneck: int = 32):
        super().__init__()
        self.base = base
        self.enabled = True
        for p in self.base.parameters():
            p.requires_grad_(False)

        if isinstance(base, nn.Linear):
            self.down = nn.Linear(base.in_features, bottleneck, bias=True)
            self.up = nn.Linear(bottleneck, base.out_features, bias=True)
        elif isinstance(base, nn.Conv2d):
            self.down = nn.Conv2d(
                base.in_channels, bottleneck, base.kernel_size,
                stride=base.stride, padding=base.padding, bias=True,
            )
            self.up = nn.Conv2d(bottleneck, base.out_channels, 1, bias=True)
        else:
            raise TypeError(f"Adapter 不支持 {type(base)}")

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + self.up(F.gelu(self.down(x)))
        return out


def _default_targets(backbone: str) -> Tuple[str, ...]:
    """不同主干上适配器插入的位置（按子模块名子串匹配）。"""
    if backbone.startswith("clip"):
        return ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
    if backbone == "resnet50":
        return ("layer4.",)
    if backbone == "vit-small":
        # timm 的 ViT 每层是 attn.qkv / attn.proj / mlp.fc1 / mlp.fc2。
        # 只改 MLP，保持与 torchvision ViT 那一组可比。
        return ("mlp.fc",)
    if backbone == "convnext-tiny":
        # torchvision 的 ConvNeXt 把每个 block 放在 Sequential 里，层名是纯数字
        # （block.0 是 depthwise conv、block.3 和 block.5 是两个 Linear），
        # 所以按路径 ".block." 匹配，并且只改 Linear——depthwise 卷积带 groups，
        # 我们的适配器不支持分组卷积。
        return (".block.",)
    if backbone == "vit-b16":
        # 注意：torchvision 的 ViT 用 nn.MultiheadAttention，其 forward 直接读
        # self.out_proj.weight，把 out_proj 换成适配器会报 AttributeError。
        # 因此这里只改 MLP 的两层。
        return ("mlp",)
    return ()


def inject(
    module: nn.Module,
    kind: str,
    patterns: Sequence[str],
    r: int = 8,
    alpha: int = 16,
    bottleneck: int = 32,
    prefix: str = "",
    only_linear: bool = False,
) -> int:
    """递归地把匹配的 Linear/Conv2d 替换成适配版本，返回替换数量。

    prefix 记录从根到当前的完整路径。必须带上它：ResNet 里层名全是
    conv1/conv2，只有靠 "layer4." 这样的前缀才能定位，丢了前缀就一个都匹配不上。
    """
    n = 0
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        eligible = isinstance(child, nn.Linear) or (
            isinstance(child, nn.Conv2d) and not only_linear
        )
        if eligible and any(p in full for p in patterns):
            wrapper = (
                LoRALayer(child, r, alpha)
                if kind == "lora"
                else ResidualAdapter(child, bottleneck)
            )
            setattr(module, name, wrapper)
            n += 1
        else:
            n += inject(child, kind, patterns, r, alpha, bottleneck, full, only_linear)
    return n


def set_adapter_enabled(module: nn.Module, flag: bool) -> None:
    for m in module.modules():
        if isinstance(m, (LoRALayer, ResidualAdapter)):
            m.enabled = flag


def apply_bitfit(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad_("bias" in n)


# --------------------------------------------------------------------------- #
# 分类头
# --------------------------------------------------------------------------- #
class CosineHead(nn.Module):
    """余弦分类器，可学习温度；支持用类别原型初始化。"""

    def __init__(self, dim: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(scale)))

    @torch.no_grad()
    def init_from_prototypes(self, protos: torch.Tensor) -> None:
        self.weight.copy_(F.normalize(protos, dim=-1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f = F.normalize(f, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return self.logit_scale.exp() * (f @ w.t())


# --------------------------------------------------------------------------- #
# 统一模型
# --------------------------------------------------------------------------- #
class AdaptModel(nn.Module):
    """主干 + 分类头 + 适配器，训练与评估的唯一入口。"""

    def __init__(
        self,
        backbone,
        method: str,
        num_classes: int,
        r: int = 8,
        alpha: int = 16,
        bottleneck: int = 32,
        l2sp_lambda: float = 0.0,
        patterns: Optional[Sequence[str]] = None,
        use_anchor: Optional[bool] = None,
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"未知方法 {method}，可选 {METHODS}")
        self.method = method
        self.backbone = backbone
        self.num_classes = num_classes
        self.l2sp_lambda = l2sp_lambda
        # use_anchor=None 时按方法默认（只有 paa 打开）；显式传入则用于消融实验
        self.use_anchor = (method == "paa") if use_anchor is None else use_anchor
        self.head = CosineHead(backbone.feature_dim, num_classes)

        if method in ("zeroshot", "linear"):
            backbone.freeze_all()
        elif method == "full":
            for p in backbone.parameters():
                p.requires_grad_(True)
        elif method == "bitfit":
            backbone.freeze_all()
            apply_bitfit(backbone)
        elif method in ("lora", "adapter", "paa"):
            backbone.freeze_all()
            kind = "lora" if method == "lora" else "adapter"
            self.n_injected = inject(
                backbone.trunk,
                kind,
                patterns if patterns is not None else _default_targets(backbone.name),
                r=r,
                alpha=alpha,
                bottleneck=bottleneck,
                only_linear=(backbone.name == "convnext-tiny"),
            )
            if self.n_injected == 0:
                raise RuntimeError(f"在 {backbone.name} 上未注入任何适配器，请检查 patterns")

    # ------------------------------------------------------------------ #
    def features(self, x: torch.Tensor, adapt: bool = True) -> torch.Tensor:
        if not adapt:
            set_adapter_enabled(self.backbone, False)
            try:
                return self.backbone.features(x)
            finally:
                set_adapter_enabled(self.backbone, True)
        return self.backbone.features(x)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        f = self.features(x)
        logits = self.head(f)
        out = {"logits": logits, "feat": f}
        if y is not None:
            loss = F.cross_entropy(logits, y)
            if self.use_anchor and prototypes is not None:
                # 原型锚定正则：把适配后特征拉向冻结特征空间的类别原型
                proto = F.normalize(prototypes, dim=-1)[y]
                cosine = F.cosine_similarity(F.normalize(f, dim=-1), proto, dim=-1)
                loss = loss + self.l2sp_lambda * (1.0 - cosine).mean()
            out["loss"] = loss
        return out

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def estimate_prototypes(self, loader, device: str = "cuda") -> torch.Tensor:
        """用未适配的主干统计每类特征均值，用于初始化分类头与锚定正则。"""
        sums = torch.zeros(self.num_classes, self.backbone.feature_dim, device=device)
        counts = torch.zeros(self.num_classes, device=device)
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            f = self.features(x, adapt=False)
            sums.index_add_(0, y, f)
            counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
        return sums / counts.clamp(min=1).unsqueeze(1)
