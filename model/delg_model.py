#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by yangmin09 (modified for ViT)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import core.net as net
from core.config import cfg


class IdentityLocalProc(nn.Module):
    """在 ResNet 分支中作为占位：直接透传 feature map"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Delg(nn.Module):
    def __init__(self):
        super(Delg, self).__init__()

        if cfg.MODEL.TYPE.lower() == "vit":
            from model.vit import ViTBackbone, ViTLocalAttention, LocalFeatureProcessor  # 改：导入新的类
            self.globalmodel = ViTBackbone()

            # 使用新的ViTLocalAttention
            embed_dim = getattr(self.globalmodel, "embed_dim", 768)

            self.local_feature_proc = LocalFeatureProcessor(embed_dim)

            self.desc_cls = Arcface(cfg.MODEL.HEADS.REDUCTION_DIM, cfg.MODEL.NUM_CLASSES)
            self.localmodel = ViTLocalAttention(embed_dim, cfg.MODEL.NUM_CLASSES)
        else:
            # ResNet分支保持不变
            from model.resnet import ResNet, ResHead
            self.globalmodel = ResNet()
            self.local_feature_proc = IdentityLocalProc()
            self.desc_cls = Arcface(cfg.MODEL.HEADS.REDUCTION_DIM, cfg.MODEL.NUM_CLASSES)
            self.localmodel = SpatialAttention2d(1024)
            self.att_cls = ResHead(512, cfg.MODEL.NUM_CLASSES)

    def forward(self, x, targets):
        if cfg.MODEL.TYPE.lower() == "vit":
            # ViT分支：全新的处理流程
            features = self.globalmodel.vit.forward_features(x)  # [B, N+1, D]
            cls_token = features[:, 0]  # [B, D]
            patch_tokens = features[:, 1:]  # [B, N, D]
            B, N, D = patch_tokens.shape
            
            # 全局特征
            global_feature = self.globalmodel.feature_proj(cls_token)
            global_logits = self.desc_cls(global_feature, targets)
            
            p = self.globalmodel.patch_size
            H = x.shape[2] // p
            W = x.shape[3] // p
            patch_map = patch_tokens.transpose(1, 2).reshape(B, D, H, W).contiguous()
            patch_map = self.local_feature_proc(patch_map)        # [B, D, H, W]，D 不变
            patch_tokens = patch_map.flatten(2).transpose(1, 2)   # [B, N, D]
            
            # 局部特征：传入targets
            local_logits = self.localmodel(patch_tokens, targets)
            
            return global_feature, global_logits, patch_tokens, local_logits, None
        else:
            # ResNet分支保持原逻辑
            global_feature, local_map = self.globalmodel(x)
            global_logits = self.desc_cls(global_feature, targets)
            
            feamap = self.local_feature_proc(local_map)
            block3 = feamap
            local_feature, att_score = self.localmodel(block3)
            local_logits = self.att_cls(local_feature)
            
            return global_feature, global_logits, local_feature, local_logits, att_score

    def set_epoch(self, epoch):
        """设置当前epoch，用于动态调整训练策略"""
        self._current_epoch = epoch
            # 传递给 ViTLocalAttention
        if cfg.MODEL.TYPE.lower() == "vit" and hasattr(self.localmodel, 'set_epoch'):
            self.localmodel.set_epoch(epoch)

class SpatialAttention2d(nn.Module):
    def __init__(self, in_c, act_fn='relu'):
        super(SpatialAttention2d, self).__init__()
        # 简化attention模块
        self.conv1 = nn.Conv2d(in_c, 512, 1, 1, bias=True)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1, bias=True)
        self.softplus = nn.Softplus(beta=1, threshold=20)
        
        # 保守初始化
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        # 移除BatchNorm，简化流程
        x = self.conv1(x)
        feature_map_norm = F.normalize(x, p=2, dim=1)
        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        x = att * feature_map_norm
        return x, att_score

class Arcface(nn.Module):
    def __init__(self, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN

        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))

        # 添加这行！
        nn.init.xavier_uniform_(self.weight)

        self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets):
        # get cos(theta)
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self._s

        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )

