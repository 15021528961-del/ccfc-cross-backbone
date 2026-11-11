"""CoOp 基线：Context Optimization（Zhou et al., IJCV 2022）。

与 PEFT 方法的区别
------------------
适配器/LoRA 改的是**图像**分支，CoOp 改的是**文本**分支：它学一组可训练的
上下文向量，拼在类别名前面，通过冻结的文本编码器得到类别嵌入。
图像编码器全程冻结。

因此它不能复用 AdaptModel，这里单独实现。
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoOp(nn.Module):
    def __init__(
        self,
        clip_model,
        tokenizer,
        classnames: List[str],
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
    ):
        super().__init__()
        self.model = clip_model
        self.n_ctx = n_ctx
        self.num_classes = len(classnames)

        # 全部冻结，只训练 ctx 和 logit_scale
        for p in self.model.parameters():
            p.requires_grad_(False)

        text_dim = self.model.text_model.config.hidden_size  # ViT-B/16 为 512
        self.embed_dim = text_dim

        # 可训练的上下文向量
        if ctx_init:
            ids = tokenizer(ctx_init, add_special_tokens=False).input_ids
            if len(ids) > n_ctx:
                ids = ids[:n_ctx]
            self.n_ctx = len(ids)
            init = self.model.text_model.embeddings.token_embedding(
                torch.tensor(ids)
            ).detach()
            self.ctx = nn.Parameter(init)
        else:
            self.ctx = nn.Parameter(torch.empty(n_ctx, text_dim))
            nn.init.normal_(self.ctx, std=0.02)

        # 类别名的词嵌入，冻结（标准 CoOp 只学上下文）
        max_len = 0
        class_ids = []
        for c in classnames:
            ids = tokenizer(c.replace("_", " "), add_special_tokens=False).input_ids
            class_ids.append(ids)
            max_len = max(max_len, len(ids))
        self.register_buffer("class_ids", torch.zeros(self.num_classes, max_len, dtype=torch.long))
        self.register_buffer("class_mask", torch.zeros(self.num_classes, max_len, dtype=torch.long))
        for i, ids in enumerate(class_ids):
            self.class_ids[i, : len(ids)] = torch.tensor(ids)
            self.class_mask[i, : len(ids)] = 1

        # 占位 input_ids：实际嵌入由上文的注入替换，这里只为满足接口
        self.register_buffer(
            "dummy_ids", torch.zeros(self.num_classes, self.n_ctx + max_len, dtype=torch.long)
        )

        # CLIP 的 logit_scale 存的本来就是 log(1/temperature)，直接沿用，不要再取指数
        self.logit_scale = nn.Parameter(self.model.logit_scale.detach().clone())

    # ------------------------------------------------------------------ #
    def text_features(self) -> torch.Tensor:
        """拼 [ctx ; class tokens] 过冻结的文本编码器，返回归一化的类别嵌入。

        transformers 的 CLIPTextTransformer 只接受 input_ids、不接受 inputs_embeds，
        所以这里临时把嵌入层换掉，让内部照常走因果掩码和编码器。
        """
        C, L = self.class_ids.shape
        class_emb = self.model.text_model.embeddings.token_embedding(self.class_ids)  # [C,L,d]
        ctx = self.ctx.unsqueeze(0).expand(C, -1, -1)                                  # [C,n_ctx,d]
        emb = torch.cat([ctx, class_emb], dim=1)                                       # [C,n_ctx+L,d]
        mask = torch.cat(
            [torch.ones(C, self.n_ctx, device=emb.device, dtype=self.class_mask.dtype), self.class_mask],
            dim=1,
        )

        emb_module = self.model.text_model.embeddings
        original = emb_module.forward
        emb_module.forward = lambda input_ids=None, position_ids=None, **kw: emb  # 注入
        try:
            out = self.model.text_model(input_ids=self.dummy_ids, attention_mask=mask)
        finally:
            emb_module.forward = original

        hidden = out.last_hidden_state                                                # [C,n_ctx+L,d]
        last = mask.sum(dim=1) - 1                                                    # 每条序列最后一个有效位置
        pooled = hidden[torch.arange(C, device=emb.device), last]                     # [C,d]
        feats = self.model.text_projection(pooled)                                    # [C,D]
        return F.normalize(feats, dim=-1)

    def image_features(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.model.vision_model(pixel_values=x).pooler_output
        feats = self.model.visual_projection(pooled)
        return F.normalize(feats, dim=-1)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> dict:
        img = self.image_features(x)
        txt = self.text_features()
        logits = self.logit_scale.exp() * (img @ txt.t())
        out = {"logits": logits}
        if y is not None:
            out["loss"] = F.cross_entropy(logits, y)
        return out

    def trainable_parameters(self):
        return [self.ctx, self.logit_scale]
