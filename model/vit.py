#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vision Transformer backbone using timm (robust and compatible)."""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter  # 添加这行导入
import timm
from core.config import cfg


class ViTBackbone(nn.Module):
    """ViT backbone: 返回 (global_feature, patch_feature_map)
    - global_feature: [B, reduction_dim]
    - patch_feature_map: [B, embed_dim, H_p, W_p]  (未经过local conv)
    """

    def __init__(self):
        super(ViTBackbone, self).__init__()
        model_name = cfg.VIT.MODEL_NAME
        pretrained = bool(cfg.VIT.PRETRAINED)

        print(f"[ViT] Loading model: {model_name}")
        print(f"[ViT] Pretrained: {pretrained}")

        # 创建模型（不要强制传 global_pool，避免与部分模型冲突）
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # 兼容不同 timm 版本：优先取 embed_dim，再取 num_features
        self.embed_dim = getattr(self.vit, "embed_dim", None)
        if self.embed_dim is None:
            self.embed_dim = getattr(self.vit, "num_features", None)
        if self.embed_dim is None:
            raise RuntimeError("无法从 timm 模型中获取 embed_dim / num_features，请检查模型类型：{}".format(model_name))

        # 用线性层把 cls token 投影到 reduction dim（等价于 ResNet 的 GlobalHead 的 fc 部分）
        self.feature_proj = nn.Linear(self.embed_dim, cfg.MODEL.HEADS.REDUCTION_DIM, bias=True)

        # 方便：记录 patch size 与输出 patch 数的期望计算
        # timm ViT 通常有 patch_embed 属性，包含 patch_size (tuple)
        patch_size = None
        if hasattr(self.vit, "patch_embed") and hasattr(self.vit.patch_embed, "patch_size"):
            p = self.vit.patch_embed.patch_size
            # patch_size 可能是 int 或 tuple
            if isinstance(p, tuple):
                patch_size = p[0]
            else:
                patch_size = int(p)
        self.patch_size = patch_size  # 可能为 None（多模型兼容）

    def forward(self, x):
        features = self.vit.forward_features(x)  # [B, N+1, D]
        cls_token = features[:, 0]
        global_feature = self.feature_proj(cls_token)
        return global_feature, features

class LocalFeatureProcessor(nn.Module):
    """把 ViT 的 patch embedding map 转成 local feature map（与原 DELG pipeline 对齐）"""

    def __init__(self, in_dim):
        super(LocalFeatureProcessor, self).__init__()
        # 输出通道 1024（与原 DELG 保持一致），你可以调整
        out_c = 1024
        # self.local_conv = nn.Sequential(
        #     nn.Conv2d(in_dim, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(out_c, eps=cfg.BN.EPS, momentum=cfg.BN.MOM),
        #     nn.ReLU(inplace=True)
        # )

        # self.local_conv = nn.Sequential(
        #     nn.Conv2d(in_dim, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(out_c, eps=cfg.BN.EPS, momentum=cfg.BN.MOM),
        #     nn.ReLU(inplace=True),
        #     # 新增：特征增强层
        #     nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(out_c, eps=cfg.BN.EPS, momentum=cfg.BN.MOM),
        #     nn.ReLU(inplace=True)
        # )

        self.local_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0, bias=True)    

        # 初始化
        for m in self.local_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, embed_dim, h, w]
        return self.local_conv(x)

class ViTAttentionHead(nn.Module):
    def __init__(self, w_in, nc):
        super(ViTAttentionHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 🔥 极简结构：直接分类
        self.fc = nn.Linear(w_in, nc, bias=True)
        
        # 保守初始化
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class ViTLocalAttention(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        
        # 🔥 简化但保留一定复杂度的注意力
        self.spatial_attention = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 512)
        )
        
        self.classifier = nn.Linear(512, num_classes, bias=True)
        self._epoch = 1
        
        # 保守初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def set_epoch(self, epoch):
        self._epoch = epoch
        
    def forward(self, patch_tokens, targets=None):
        patch_tokens = self.norm(patch_tokens)
        
        # 🔥 混合策略：注意力加权 + mean pooling
        attention_weights = self.spatial_attention(patch_tokens)
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)
        
        # 注意力特征
        attended_feature = torch.sum(
            patch_tokens * attention_weights.unsqueeze(-1), 
            dim=1
        )
        
        # 全局特征
        global_feature = patch_tokens.mean(dim=1)
        
        # 🔥 自适应融合
        if self.training:
            weight = min(0.8, self._epoch / 50.0)
        else:
            weight = 0.7
            
        local_feature = weight * attended_feature + (1 - weight) * global_feature
        
        local_feature = self.feature_proj(local_feature)
        
        # 🔥 恢复关键的归一化
        local_feature = F.normalize(local_feature, p=2, dim=1)
        
        normalized_weight = F.normalize(self.classifier.weight, p=2, dim=1)
        logits = F.linear(local_feature, normalized_weight, self.classifier.bias)
        
        # 🔥 动态scale策略
        if self.training:
            scale = 12.0 if self._epoch < 30 else 15.0
        else:
            scale = 18.0
        logits = logits * scale
        
        return logits
        