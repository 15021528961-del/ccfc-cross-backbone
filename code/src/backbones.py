"""主干网络封装：CLIP ViT-B/16（支持零样本）、ResNet-50、ViT-B/16。

统一接口
--------
features(x) -> [B, D]     图像特征（未归一化）
logits(x)   -> [B, C]     仅线性分类头
trunk                    需要注入适配器的子模块（Linear 所在处）
zero_shot(x, classnames) 仅 CLIP 支持
"""
import os
import socket
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models as tvm
from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)

BACKBONES = ("clip-vit-b16", "resnet50", "vit-b16", "convnext-tiny", "vit-small")


def build_transforms(backbone: str, train: bool, image_size: int = 224) -> transforms.Compose:
    mean, std = (CLIP_MEAN, CLIP_STD) if backbone.startswith("clip") else (IN_MEAN, IN_STD)
    ops: List = []
    if train:
        ops += [
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops += [transforms.Resize(int(image_size * 256 / 224)), transforms.CenterCrop(image_size)]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


class Backbone(nn.Module):
    """所有主干的公共外壳，子类实现 _build。"""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._build()

    def _build(self) -> None:
        raise NotImplementedError

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    @property
    def trunk(self) -> nn.Module:
        """适配器注入位置。"""
        raise NotImplementedError

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """默认：不训练时返回 None（仅适配器范式使用）。"""
        raise NotImplementedError

    def zero_shot(self, x: torch.Tensor, classnames: List[str]) -> Optional[torch.Tensor]:
        return None

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)


HF_MIRROR = "https://hf-mirror.com"


def _pin_hf_endpoint(endpoint: str) -> None:
    """huggingface_hub 在 import 时就把 ENDPOINT 读成常量了，
    只改环境变量对已导入的库无效，所以这里连模块常量一起改掉。"""
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        import huggingface_hub.constants as C

        C.ENDPOINT = endpoint
        if hasattr(C, "HUGGINGFACE_CO_URL_TEMPLATE"):
            C.HUGGINGFACE_CO_URL_TEMPLATE = (
                endpoint + "/{repo_id}/resolve/{revision}/{filename}"
            )
    except Exception:  # noqa: BLE001
        pass


def _ensure_hf_endpoint() -> None:
    """国内网络常常连不上 huggingface.co。先探测，不通就自动切到镜像。"""
    if os.environ.get("HF_ENDPOINT"):
        _pin_hf_endpoint(os.environ["HF_ENDPOINT"])
        return
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        _pin_hf_endpoint(HF_MIRROR)
        print("[hf] huggingface.co 不可达，已切换到 hf-mirror.com")


class CLIPBackbone(Backbone):
    def _build(self) -> None:
        from transformers import CLIPModel

        _ensure_hf_endpoint()
        ckpt = "openai/clip-vit-base-patch16"
        self.model = CLIPModel.from_pretrained(ckpt)
        self.vision = self.model.vision_model
        self._text_feats: Optional[torch.Tensor] = None

    @property
    def feature_dim(self) -> int:
        return self.model.config.projection_dim

    @property
    def trunk(self) -> nn.Module:
        return self.vision

    def features(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.vision(pixel_values=x).pooler_output
        return self.model.visual_projection(pooled)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x) @ self._text_feats.t()

    @torch.no_grad()
    def prepare_text(self, classnames: List[str], templates: Optional[List[str]] = None) -> None:
        from transformers import CLIPTokenizer

        _ensure_hf_endpoint()
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        templates = templates or ["a photo of a {}."]
        feats = []
        for c in classnames:
            texts = [t.format(c.replace("_", " ")) for t in templates]
            inp = tok(texts, padding=True, truncation=True, return_tensors="pt").to(
                next(self.model.parameters()).device
            )
            tf = self.model.get_text_features(**inp)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            feats.append((tf.mean(0)))
        emb = torch.stack(feats)
        self._text_feats = emb / emb.norm(dim=-1, keepdim=True)

    def zero_shot(self, x: torch.Tensor, classnames: List[str]) -> Optional[torch.Tensor]:
        if self._text_feats is None:
            self.prepare_text(classnames)
        f = self.features(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return self.model.logit_scale.exp() * (f @ self._text_feats.t())


class TimmBackbone(Backbone):
    """timm 主干。用于参数量与卷积网络可比的小型 ViT 对照。"""

    TIMM_NAME = {
        "vit-small": "vit_small_patch16_224.augreg_in1k",
    }

    def _build(self) -> None:
        import timm

        name = self.TIMM_NAME[self.name]
        self.encoder = timm.create_model(name, pretrained=True, num_classes=0)
        self._dim = self.encoder.num_features

    @property
    def feature_dim(self) -> int:
        return self._dim

    @property
    def trunk(self) -> nn.Module:
        return self.encoder

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("timm 主干不提供零样本分类")


class TVBackbone(Backbone):
    """torchvision 主干：ResNet-50 / ViT-B/16 / ConvNeXt-Tiny。"""

    def _build(self) -> None:
        if self.name == "resnet50":
            w = tvm.ResNet50_Weights.IMAGENET1K_V2
            m = tvm.resnet50(weights=w)
            self._dim = m.fc.in_features
            m.fc = nn.Identity()
            self.encoder = m
        elif self.name == "vit-b16":
            w = tvm.ViT_B_16_Weights.IMAGENET1K_V1
            m = tvm.vit_b_16(weights=w)
            self._dim = m.heads.head.in_features
            m.heads.head = nn.Identity()
            self.encoder = m
        elif self.name == "convnext-tiny":
            w = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            m = tvm.convnext_tiny(weights=w)
            self._dim = m.classifier[2].in_features
            m.classifier[2] = nn.Identity()
            self.encoder = m
        else:
            raise ValueError(self.name)

    @property
    def feature_dim(self) -> int:
        return self._dim

    @property
    def trunk(self) -> nn.Module:
        return self.encoder

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("torchvision 主干不提供零样本分类")


def get_backbone(name: str) -> Backbone:
    name = name.lower()
    if name == "clip-vit-b16":
        return CLIPBackbone(name)
    if name in ("resnet50", "vit-b16", "convnext-tiny"):
        return TVBackbone(name)
    if name in TimmBackbone.TIMM_NAME:
        return TimmBackbone(name)
    raise ValueError(f"未知主干 {name}，可选：{BACKBONES}")
