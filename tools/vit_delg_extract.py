#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ViT-DELG特征提取脚本"""

import sys
import os
import numpy as np
import cv2
import torch
import pickle
from tqdm import tqdm
from scipy.io import savemat
import gc

# 将父目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import cfg
import core.config as config

# 在命令行参数解析之前先临时加载配置
if __name__ == '__main__':
    config.load_cfg_fom_args("Extract feature.")
    config.assert_and_infer_cfg()
    cfg.freeze()

""" 全局设置 """
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

# 配置路径
# MODEL_WEIGHTS = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/output/8gpu_newzzdata300w_vit_20251025/checkpoints/model_epoch_0065.pyth'
# INFER_DIR = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2'  # 修改为newtest目录
# GND_FILE = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2/newtest2.pkl'
# OUTPUT_DIR = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/features/8gpu_newzzdata300w_vit_20251025'
# GLOBAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_global.mat')
# LOCAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_local_global_fea.pickle')

# MODEL_WEIGHTS = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/output/8gpu_newzzdata300w_vit_casdata9w_V2/checkpoints/model_epoch_0023.pyth'
# INFER_DIR = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_no_chart3_test200'  # 修改为newtest目录
# GND_FILE = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_no_chart3_test200/no_chart3_test20.pkl'
# OUTPUT_DIR = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/features/8gpu_newzzdata300w_vit_casdata9w_V2'
# GLOBAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_global_casdata200.mat')
# LOCAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_local_global_fea_casdata200.pickle')

MODEL_WEIGHTS = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/output/8gpu_newzzdata300w_vit_20251121_CNN/checkpoints/model_epoch_0008.pyth'
INFER_DIR = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2'  # 修改为newtest目录
GND_FILE = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2/newtest2.pkl'
OUTPUT_DIR = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/features/8gpu_newzzdata300w_vit_251121_CNN'
GLOBAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_global_epoch08.mat')
LOCAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_local_global_fea_epoch08.pickle')

# ViT模型参数
TARGET_SIZE = 224  # ViT标准输入尺寸
TOP_K_PATCHES = 100  # 每张图最多提取的关键patch数量

def setup_model():
    """设置ViT-DELG模型"""
    from model.delg_model import Delg
    model = Delg()
    load_checkpoint(MODEL_WEIGHTS, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

# def extract_vit_features(input_data, model):
#     """ViT专用的特征提取"""
#     with torch.no_grad():
#         # 直接调用ViT的forward
#         features = model.globalmodel.vit.forward_features(input_data)
#         cls_token = features[:, 0]  # [B, D]
#         patch_tokens = features[:, 1:]  # [B, N, D]
        
#         # 全局特征
#         global_feature = model.globalmodel.feature_proj(cls_token)  # [B, 512]
        
#         # 局部特征处理
#         batch_size, num_patches, embed_dim = patch_tokens.shape
#         grid_size = int(num_patches ** 0.5)  # 通常是14 (224/16)
        
#         # 使用cross-attention计算patch重要性
#         attended_tokens, attn_weights = model.localmodel.cross_attention(
#             patch_tokens, patch_tokens, patch_tokens
#         )
        
#         # 计算每个patch的重要性分数 [B, N]
#         # attn_weights shape: [B, num_heads, N, N]
#         patch_importance = attn_weights.mean(dim=1)  # [B, N, N]
#         patch_importance = patch_importance.mean(dim=-1)  # [B, N] - 每个patch作为query的平均attention
        
#         # 投影到512维
#         projected_patches = model.localmodel.feature_proj(attended_tokens)  # [B, N, 512]
        
#         # 为每个batch生成局部特征
#         local_features_list = []
#         local_scores_list = []
#         local_locations_list = []
        
#         for b in range(batch_size):
#             # 选择top-k重要的patches
#             top_k = min(TOP_K_PATCHES, num_patches)
#             top_scores, top_indices = torch.topk(patch_importance[b], k=top_k, dim=0)
            
#             # 提取对应的特征
#             selected_features = projected_patches[b, top_indices]  # [K, 512]
#             selected_scores = top_scores  # [K]
            
#             # 计算patch的空间位置
#             patch_locations = []
#             for idx in top_indices:
#                 row = idx // grid_size
#                 col = idx % grid_size
#                 # 转换为相对坐标 (0-1)
#                 x = (col.float() + 0.5) / grid_size
#                 y = (row.float() + 0.5) / grid_size
#                 patch_locations.append([x.item(), y.item()])
            
#             local_features_list.append(selected_features.cpu().numpy())
#             local_scores_list.append(selected_scores.cpu().numpy())
#             local_locations_list.append(np.array(patch_locations, dtype=np.float32))
        
#         # 转换全局特征
#         global_feature_np = global_feature.cpu().numpy()
        
#         return global_feature_np, local_features_list, local_scores_list, local_locations_list

# def extract_vit_features(input_data, model):
#     """ViT专用的特征提取 - 修复版本"""
#     with torch.no_grad():
#         try:
#             # 直接调用ViT的forward
#             features = model.globalmodel.vit.forward_features(input_data)
#             cls_token = features[:, 0]  # [B, D]
#             patch_tokens = features[:, 1:]  # [B, N, D]
            
#             # 调试信息
#             batch_size, num_patches, embed_dim = patch_tokens.shape
#             print(f"[DEBUG] Batch: {batch_size}, Patches: {num_patches}, Embed: {embed_dim}")
            
#             # 全局特征
#             global_feature = model.globalmodel.feature_proj(cls_token)  # [B, 512]
            
#             # 检查patch数量
#             if num_patches == 0:
#                 print("警告: 没有patch tokens")
#                 return None, None, None, None
            
#             grid_size = int(num_patches ** 0.5)  # 通常是14 (224/16)
            
#             # 使用cross-attention计算patch重要性
#             attended_tokens, attn_weights = model.localmodel.cross_attention(
#                 patch_tokens, patch_tokens, patch_tokens
#             )
            
#             # 🔥 修复：更安全的attention处理
#             if attn_weights.dim() == 4:  # [B, num_heads, N, N]
#                 patch_importance = attn_weights.mean(dim=1)  # [B, N, N]
#                 patch_importance = patch_importance.mean(dim=-1)  # [B, N]
#             elif attn_weights.dim() == 3:  # [B, N, N]
#                 patch_importance = attn_weights.mean(dim=-1)  # [B, N]
#             else:
#                 print(f"警告: 意外的attention维度: {attn_weights.shape}")
#                 # 使用均匀分布作为fallback
#                 patch_importance = torch.ones(batch_size, num_patches, device=input_data.device)
            
#             # 投影到512维
#             projected_patches = model.localmodel.feature_proj(attended_tokens)  # [B, N, 512]
            
#             # 为每个batch生成局部特征
#             local_features_list = []
#             local_scores_list = []
#             local_locations_list = []
            
#             for b in range(batch_size):
#                 # 🔥 修复：动态确定top_k
#                 available_patches = patch_importance[b].numel()
#                 top_k = min(TOP_K_PATCHES, available_patches, num_patches)
                
#                 print(f"[DEBUG] Batch {b}: available={available_patches}, top_k={top_k}")
                
#                 if top_k <= 0:
#                     print(f"警告: Batch {b} 没有可用的patches")
#                     # 添加空特征
#                     local_features_list.append(np.zeros((0, 512), dtype=np.float32))
#                     local_scores_list.append(np.zeros(0, dtype=np.float32))
#                     local_locations_list.append(np.zeros((0, 2), dtype=np.float32))
#                     continue
                
#                 try:
#                     # 选择top-k重要的patches
#                     top_scores, top_indices = torch.topk(patch_importance[b], k=top_k, dim=0)
                    
#                     # 提取对应的特征
#                     selected_features = projected_patches[b, top_indices]  # [K, 512]
#                     selected_scores = top_scores  # [K]
                    
#                     # 计算patch的空间位置
#                     patch_locations = []
#                     for idx in top_indices:
#                         row = idx // grid_size
#                         col = idx % grid_size
#                         # 转换为相对坐标 (0-1)
#                         x = (col.float() + 0.5) / grid_size
#                         y = (row.float() + 0.5) / grid_size
#                         patch_locations.append([x.item(), y.item()])
                    
#                     local_features_list.append(selected_features.cpu().numpy())
#                     local_scores_list.append(selected_scores.cpu().numpy())
#                     local_locations_list.append(np.array(patch_locations, dtype=np.float32))
                    
#                 except Exception as e:
#                     print(f"错误: Batch {b} topk失败: {e}")
#                     # 添加空特征作为fallback
#                     local_features_list.append(np.zeros((0, 512), dtype=np.float32))
#                     local_scores_list.append(np.zeros(0, dtype=np.float32))
#                     local_locations_list.append(np.zeros((0, 2), dtype=np.float32))
            
#             # 转换全局特征
#             global_feature_np = global_feature.cpu().numpy()
            
#             return global_feature_np, local_features_list, local_scores_list, local_locations_list
            
#         except Exception as e:
#             print(f"特征提取过程中发生错误: {e}")
#             return None, None, None, None
        
def extract_vit_features(input_data, model):
    """ViT专用的特征提取 - 修复版本"""
    with torch.no_grad():
        try:
            #  # 🔥 添加：检查输入维度
            # print(f"[DEBUG] 输入数据维度: {input_data.shape}")

            # 直接调用ViT的forward
            features = model.globalmodel.vit.forward_features(input_data)

            # # 🔥 添加：检查ViT原始输出
            # print(f"[DEBUG] ViT原始输出维度: {features.shape}")

            cls_token = features[:, 0]  # [B, D]
            patch_tokens = features[:, 1:]  # [B, N, D]

            # # 🔥 添加：检查token维度
            # print(f"[DEBUG] CLS token维度: {cls_token.shape}")
            # print(f"[DEBUG] Patch tokens维度: {patch_tokens.shape}")
            
            # 调试信息
            batch_size, num_patches, embed_dim = patch_tokens.shape
            print(f"[DEBUG] Batch: {batch_size}, Patches: {num_patches}, Embed: {embed_dim}")
            
            # 全局特征
            global_feature = model.globalmodel.feature_proj(cls_token)  # [B, 512]

            # # 🔥 添加：检查投影后的全局特征维度
            # print(f"[DEBUG] 投影后全局特征维度: {global_feature.shape}")
            
            # 检查patch数量
            if num_patches == 0:
                print("警告: 没有patch tokens")
                return None, None, None, None
            
            grid_size = int(num_patches ** 0.5)  # 通常是14 (224/16)
            
            # 🔥 修改：使用特征范数作为patch重要性分数
            patch_norms = torch.norm(patch_tokens, dim=-1)  # [B, N]
            
            # 🔥 修改：使用 localmodel 的处理方式
            normed_patches = model.localmodel.norm(patch_tokens)  # [B, N, D]
            projected_patches = model.localmodel.feature_proj(normed_patches)  # [B, N, 512]
            
            # 为每个batch生成局部特征
            local_features_list = []
            local_scores_list = []
            local_locations_list = []
            
            for b in range(batch_size):
                # 🔥 修复：动态确定top_k
                available_patches = patch_norms[b].numel()
                top_k = min(TOP_K_PATCHES, available_patches, num_patches)
                
                print(f"[DEBUG] Batch {b}: available={available_patches}, top_k={top_k}")
                
                if top_k <= 0:
                    print(f"警告: Batch {b} 没有可用的patches")
                    # 添加空特征
                    local_features_list.append(np.zeros((0, 512), dtype=np.float32))
                    local_scores_list.append(np.zeros(0, dtype=np.float32))
                    local_locations_list.append(np.zeros((0, 2), dtype=np.float32))
                    continue
                
                try:
                    # 选择top-k重要的patches（基于特征范数）
                    top_scores, top_indices = torch.topk(patch_norms[b], k=top_k, dim=0)
                    
                    # 提取对应的特征
                    selected_features = projected_patches[b, top_indices]  # [K, 512]
                    selected_scores = top_scores  # [K]
                    
                    # 计算patch的空间位置
                    patch_locations = []
                    for idx in top_indices:
                        row = idx // grid_size
                        col = idx % grid_size
                        # 转换为相对坐标 (0-1)
                        x = (col.float() + 0.5) / grid_size
                        y = (row.float() + 0.5) / grid_size
                        patch_locations.append([x.item(), y.item()])
                    
                    local_features_list.append(selected_features.cpu().numpy())
                    local_scores_list.append(selected_scores.cpu().numpy())
                    local_locations_list.append(np.array(patch_locations, dtype=np.float32))
                    
                except Exception as e:
                    print(f"错误: Batch {b} topk失败: {e}")
                    # 添加空特征作为fallback
                    local_features_list.append(np.zeros((0, 512), dtype=np.float32))
                    local_scores_list.append(np.zeros(0, dtype=np.float32))
                    local_locations_list.append(np.zeros((0, 2), dtype=np.float32))
            
            # 转换全局特征
            global_feature_np = global_feature.cpu().numpy()
            
            return global_feature_np, local_features_list, local_scores_list, local_locations_list
            
        except Exception as e:
            print(f"特征提取过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

# def vit_delg_extract(img, model):
#     """ViT版本的DELG特征提取"""
#     print(f"输入图像尺寸: {img.shape}")
    
#     try:
#         # ViT只处理标准尺寸，不做多尺度
#         im = preprocess(img.copy())
#         im_array = np.asarray([im], dtype=np.float32)
        
#         # 提取特征
#         global_features, local_features_list, local_scores_list, local_locations_list = \
#             extract_vit_features(torch.from_numpy(im_array).cuda(), model)
        
#         # 处理结果
#         final_global = global_features.squeeze()  # [512]
        
#         if local_features_list and len(local_features_list[0]) > 0:
#             # 合并第一个batch的结果
#             final_local_features = local_features_list[0]  # [K, 512]
#             final_local_scores = local_scores_list[0]  # [K]
#             final_local_locations = local_locations_list[0]  # [K, 2]
            
#             # 转换为图像像素坐标
#             h, w = img.shape[:2]
#             final_local_locations[:, 0] *= w  # x坐标
#             final_local_locations[:, 1] *= h  # y坐标
            
#             local_data = {
#                 'locations': final_local_locations,
#                 'descriptors': final_local_features,
#                 'scores': final_local_scores
#             }
            
#             print(f"提取到 {len(final_local_features)} 个局部特征")
#         else:
#             print("未提取到有效的局部特征")
#             local_data = {
#                 'locations': np.zeros((0, 2), dtype=np.float32),
#                 'descriptors': np.zeros((0, 512), dtype=np.float32),
#                 'scores': np.zeros(0, dtype=np.float32)
#             }
        
#         return final_global, local_data
        
#     except Exception as e:
#         print(f"特征提取失败: {e}")
#         # 返回默认值
#         final_global = np.zeros(512, dtype=np.float32)
#         local_data = {
#             'locations': np.zeros((0, 2), dtype=np.float32),
#             'descriptors': np.zeros((0, 512), dtype=np.float32),
#             'scores': np.zeros(0, dtype=np.float32)
#         }
#         return final_global, local_data

def vit_delg_extract(img, model):
    """ViT版本的DELG特征提取 - 增强错误处理"""
    print(f"输入图像尺寸: {img.shape}")
    
    try:
        # ViT只处理标准尺寸，不做多尺度
        im = preprocess(img.copy())
        im_array = np.asarray([im], dtype=np.float32)
        
        # 提取特征
        result = extract_vit_features(torch.from_numpy(im_array).cuda(), model)
        
        # 🔥 检查返回值
        if result[0] is None:
            print("特征提取返回None，使用默认值")
            raise Exception("特征提取失败")
            
        global_features, local_features_list, local_scores_list, local_locations_list = result
        
        # 处理结果
        final_global = global_features.squeeze()  # [512]
        
        if (local_features_list and len(local_features_list) > 0 and 
            len(local_features_list[0]) > 0):
            # 合并第一个batch的结果
            final_local_features = local_features_list[0]  # [K, 512]
            final_local_scores = local_scores_list[0]  # [K]
            final_local_locations = local_locations_list[0]  # [K, 2]
            
            # 转换为图像像素坐标
            h, w = img.shape[:2]
            if len(final_local_locations) > 0:
                final_local_locations[:, 0] *= w  # x坐标
                final_local_locations[:, 1] *= h  # y坐标
            
            local_data = {
                'locations': final_local_locations,
                'descriptors': final_local_features,
                'scores': final_local_scores
            }
            
            print(f"成功提取到 {len(final_local_features)} 个局部特征")
        else:
            print("未提取到有效的局部特征，使用空值")
            local_data = {
                'locations': np.zeros((0, 2), dtype=np.float32),
                'descriptors': np.zeros((0, 512), dtype=np.float32),
                'scores': np.zeros(0, dtype=np.float32)
            }
        
        return final_global, local_data
        
    except Exception as e:
        print(f"特征提取失败: {e}")
        # 返回默认值
        final_global = np.zeros(512, dtype=np.float32)
        local_data = {
            'locations': np.zeros((0, 2), dtype=np.float32),
            'descriptors': np.zeros((0, 512), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32)
        }
        return final_global, local_data

def preprocess(im):
    """ViT预处理：固定尺寸224x224"""
    # 直接resize到标准尺寸
    im = cv2.resize(im, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # 转换为CHW格式并归一化
    im = im.transpose([2, 0, 1])
    im = im / 255.0
    im = color_norm(im, _MEAN, _SD)
    return im

def color_norm(im, mean, std):
    """颜色归一化"""
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im

def find_image_with_any_extension(img_path, directory):
    """查找具有支持扩展名的图像文件"""
    full_path = os.path.join(directory, img_path)
    if os.path.exists(full_path):
        return full_path
    
    name_wo_ext = os.path.splitext(img_path)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.pgm']:
        path = os.path.join(directory, name_wo_ext + ext)
        if os.path.exists(path):
            return path
    return None

def load_checkpoint(checkpoint_file, model, optimizer=None):
    """加载检查点"""
    err_str = f"Checkpoint '{checkpoint_file}' not found"
    assert os.path.exists(checkpoint_file), err_str
    print(f"Loading checkpoint from: {checkpoint_file}")
    
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    
    try:
        state_dict = checkpoint["model_state"]
    except (KeyError, TypeError):
        state_dict = checkpoint
    
    # 处理module前缀
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model_dict = model.state_dict()
    
    # 直接匹配
    pretrained_dict = {}
    direct_matches = 0
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].size() == v.size():
            pretrained_dict[k] = v
            direct_matches += 1
    
    print(f'模型总参数: {len(model_dict)}, checkpoint总参数: {len(state_dict)}')
    print(f'成功加载参数: {len(pretrained_dict)}')
    
    # 检查关键组件
    vit_loaded = sum(1 for k in pretrained_dict.keys() if 'vit' in k)
    local_loaded = sum(1 for k in pretrained_dict.keys() if 'localmodel' in k)
    print(f"ViT相关权重: {vit_loaded}, 局部模型权重: {local_loaded}")
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Checkpoint loaded successfully.")
    return checkpoint

def main():
    """主函数"""
    # 设置环境变量以优化内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("=== 开始ViT-DELG特征提取 ===")
    print(f"模型权重: {MODEL_WEIGHTS}")
    print(f"图像目录: {INFER_DIR}")
    print(f"PKL文件: {GND_FILE}")
    
    model = setup_model()
    Q = []  # 查询图像全局特征
    X = []  # 数据库图像全局特征
    local_features = {}
    failed_images = []
    
    # 读取 PKL 文件
    print("加载PKL文件...")
    with open(GND_FILE, 'rb') as f:
        cfg_dataset = pickle.load(f)
    qimlist = cfg_dataset.get('qimlist', [])
    imlist = cfg_dataset.get('imlist', [])
    
    print(f"查询图像数量: {len(qimlist)}")
    print(f"数据库图像数量: {len(imlist)}")
    
    # 处理查询图像
    print("\n=== 处理查询图像 ===")
    for i, qname in enumerate(tqdm(qimlist, desc="Processing query images")):
        img_full_path = os.path.join(INFER_DIR, qname)
        
        if not os.path.exists(img_full_path):
            img_full_path = find_image_with_any_extension(qname, INFER_DIR)
        
        if img_full_path is None or not os.path.exists(img_full_path):
            print(f"警告: 图像文件不存在 - {qname}")
            failed_images.append(qname)
            # 添加零特征以保持索引一致
            Q.append(np.zeros(512, dtype=np.float32))
            qname_key = os.path.basename(qname)
            local_features[qname_key] = {
                'locations': np.zeros((0, 2), dtype=np.float32),
                'descriptors': np.zeros((0, 512), dtype=np.float32),
                'scores': np.zeros(0, dtype=np.float32)
            }
            continue
        
        # 检查文件格式
        ext = os.path.splitext(img_full_path)[-1].lower()
        if ext not in ['.jpg', '.jpeg', '.bmp', '.png', '.pgm']:
            print(f"警告: 不支持的格式 {ext} - {img_full_path}")
            failed_images.append(qname)
            continue
        
        # 读取图像
        im = cv2.imread(img_full_path)
        if im is None:
            print(f"警告: 无法读取图像 - {img_full_path}")
            failed_images.append(qname)
            continue
        
        im = im.astype(np.float32, copy=False)
        
        try:
            # 提取特征
            global_feature, local_data = vit_delg_extract(im, model)
            Q.append(global_feature)
            
            # 使用文件名作为键
            qname_key = os.path.basename(qname)
            local_features[qname_key] = local_data
            
        except Exception as e:
            print(f"错误: 处理查询图像 {qname} 失败 - {str(e)}")
            failed_images.append(qname)
            # 添加零特征
            Q.append(np.zeros(512, dtype=np.float32))
            qname_key = os.path.basename(qname)
            local_features[qname_key] = {
                'locations': np.zeros((0, 2), dtype=np.float32),
            'descriptors': np.zeros((0, 512), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32)
        }
        finally:
            del im
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 处理数据库图像
    print("\n=== 处理数据库图像 ===")
    for i, dbname in enumerate(tqdm(imlist, desc="Processing database images")):
        img_full_path = os.path.join(INFER_DIR, dbname)
        
        if not os.path.exists(img_full_path):
            img_full_path = find_image_with_any_extension(dbname, INFER_DIR)
        
        if img_full_path is None or not os.path.exists(img_full_path):
            print(f"警告: 图像文件不存在 - {dbname}")
            failed_images.append(dbname)
            # 添加零特征以保持索引一致
            X.append(np.zeros(512, dtype=np.float32))
            dbname_key = os.path.basename(dbname)
            local_features[dbname_key] = {
                'locations': np.zeros((0, 2), dtype=np.float32),
                'descriptors': np.zeros((0, 512), dtype=np.float32),
                'scores': np.zeros(0, dtype=np.float32)
            }
            continue
        
        # 检查文件格式
        ext = os.path.splitext(img_full_path)[-1].lower()
        if ext not in ['.jpg', '.jpeg', '.bmp', '.png', '.pgm']:
            print(f"警告: 不支持的格式 {ext} - {img_full_path}")
            failed_images.append(dbname)
            continue
        
        # 读取图像
        im = cv2.imread(img_full_path)
        if im is None:
            print(f"警告: 无法读取图像 - {img_full_path}")
            failed_images.append(dbname)
            continue
        
        im = im.astype(np.float32, copy=False)
        
        try:
            # 提取特征
            global_feature, local_data = vit_delg_extract(im, model)
            X.append(global_feature)
            
            # 使用文件名作为键
            dbname_key = os.path.basename(dbname)
            local_features[dbname_key] = local_data
            
        except Exception as e:
            print(f"错误: 处理数据库图像 {dbname} 失败 - {str(e)}")
            failed_images.append(dbname)
            # 添加零特征
            X.append(np.zeros(512, dtype=np.float32))
            dbname_key = os.path.basename(dbname)
            local_features[dbname_key] = {
                'locations': np.zeros((0, 2), dtype=np.float32),
                'descriptors': np.zeros((0, 512), dtype=np.float32),
                'scores': np.zeros(0, dtype=np.float32)
            }
        finally:
            del im
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 保存特征
    print("\n=== 保存特征 ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存全局特征
    Q = np.vstack(Q) if Q else np.zeros((0, 512), dtype=np.float32)
    X = np.vstack(X) if X else np.zeros((0, 512), dtype=np.float32)
    savemat(GLOBAL_FEATURE_PATH, {'Q': Q, 'X': X})
    print(f"全局特征已保存至: {GLOBAL_FEATURE_PATH}")
    print(f"查询特征矩阵形状: {Q.shape}")
    print(f"数据库特征矩阵形状: {X.shape}")
    
    # 保存局部特征
    with open(LOCAL_FEATURE_PATH, 'wb') as f:
        pickle.dump(local_features, f, protocol=2)
    print(f"局部特征已保存至: {LOCAL_FEATURE_PATH}")
    print(f"局部特征数量: {len(local_features)}")
    
    # 统计局部特征情况
    valid_local_count = 0
    total_local_features = 0
    for key, data in local_features.items():
        if len(data['descriptors']) > 0:
            valid_local_count += 1
            total_local_features += len(data['descriptors'])
    
    print(f"有效局部特征的图像数量: {valid_local_count}")
    print(f"总局部特征点数量: {total_local_features}")
    if valid_local_count > 0:
        print(f"平均每张图的局部特征数量: {total_local_features / valid_local_count:.1f}")
    
    # 保存失败的图像列表
    if failed_images:
        failed_log_path = os.path.join(OUTPUT_DIR, "failed_images.txt")
        with open(failed_log_path, "w") as f:
            f.write("\n".join(failed_images))
        print(f"失败图像列表已保存至: {failed_log_path}")
        print(f"失败图像数量: {len(failed_images)}")
    
    print("\n=== ViT-DELG特征提取完成 ===")

if __name__ == '__main__':
    main()