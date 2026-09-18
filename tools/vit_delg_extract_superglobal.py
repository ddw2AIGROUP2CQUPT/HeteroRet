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
import torch.nn as nn

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
MODEL_WEIGHTS = '/home/ubuntu/san/hxl/delg-pytoch/DELG_VIT/output/8gpu_newzzdata300w_vit_20251121_CNN_V1/checkpoints/model_epoch_0010.pyth'
INFER_DIR = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2'  # 修改为newtest目录
GND_FILE = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2/newtest2.pkl'
OUTPUT_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG_VIT/output/features/8gpu_newzzdata300w_vit_20251121_CNN_V1'
GLOBAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_global_epoch10_spV1_gemp.mat')
# LOCAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_local_global_fea_epoch10_spV1.pickle')

# ViT模型参数
TARGET_SIZE = 224  # ViT标准输入尺寸
TOP_K_PATCHES = 100  # 每张图最多提取的关键patch数量

# 在文件开头添加配置
USE_GEM_ENHANCEMENT = True  # 是否使用GEM增强
GEM_SCALE_LIST = [0.25, 0.5, 1.0]  # 多尺度列表
# GEM_SCALE_LIST = [0.35, 0.5, 0.7071, 1.0, 1.4142, 2.0]
# GEM_SCALE_LIST = [1.0]
USE_RGEM = True
USE_GEMP = True  
USE_SGEM = True

# =========================
# 在import部分后添加GEM模块
class rgem(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, pr=2.5, size=5):
        super(rgem, self).__init__()
        self.pr = pr
        self.size = size
        self.lppool = nn.LPPool2d(self.pr, int(self.size), stride=1)
        self.pad = nn.ReflectionPad2d(int((self.size-1)//2.))
    def forward(self, x):
        nominater = (self.size**2) **(1./self.pr)
        x = 0.5*self.lppool(self.pad(x/nominater)) + 0.5*x
        return x

class sgem(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, ps=10., infinity=True):
        super(sgem, self).__init__()
        self.ps = ps
        self.infinity = infinity
    def forward(self, x):
        x = torch.stack(x, 0)
        if self.infinity:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
            x = torch.max(x, 0)[0] 
        else:
            gamma = x.min()
            x = (x - gamma).pow(self.ps).mean(0).pow(1./self.ps) + gamma
        return x

class gemp(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, p=1.0, eps=1e-8):
        super(gemp, self).__init__()
        self.p = p
        self.eps = eps
    def forward(self, x):
        x = x.clamp(self.eps).pow(self.p)
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1).pow(1. / self.p)
        return x

def setup_model():
    """设置ViT-DELG模型"""
    from model.delg_model import Delg
    model = Delg()
    load_checkpoint(MODEL_WEIGHTS, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model
        
def vit_cnn_delg_extract_with_gem(im, model, use_rgem=True, use_gemp=True, use_sgem=True, scale_list=[1.0]):
    """ViT+CNN版本的GEM增强特征提取"""
    # 预处理
    im = preprocess(im)
    input_data = torch.FloatTensor(im).unsqueeze(0)
    if torch.cuda.is_available():
        input_data = input_data.cuda()

    # print(f"输入图像维度: {input_data.shape}")
    
    # 初始化GEM模块
    gem_modules = {
        'rgem': rgem() if use_rgem else None,
        'gemp': gemp() if use_gemp else None,
        'sgem': sgem() if use_sgem else None
    }
    
    # 将GEM模块移到GPU
    for module in gem_modules.values():
        if module is not None and torch.cuda.is_available():
            module.cuda()
    
    with torch.no_grad():
        feature_list = []
        
        for scale in scale_list:
            # 缩放输入
            if scale != 1.0:
                h, w = input_data.shape[2], input_data.shape[3]
                new_h, new_w = int(h * scale), int(w * scale)
                # print(f"尺度{scale} - 计算的新尺寸: {new_h} x {new_w}")

                scaled_input = torch.nn.functional.interpolate(
                    input_data, size=(new_h, new_w), mode='bilinear', align_corners=False
                )
                # print(f"尺度{scale} - 第一次缩放后: {scaled_input.shape}")

                scaled_input = torch.nn.functional.interpolate(
                    scaled_input, size=(TARGET_SIZE, TARGET_SIZE), mode='bilinear', align_corners=False
                )
                # print(f"尺度{scale} - 第二次缩放后: {scaled_input.shape}")
            else:
                scaled_input = input_data
                # print(f"尺度{scale} - 缩放后输入维度: {scaled_input.shape}")
            
            

            # 🔥 使用 ViT+CNN 版本的调用方式
            global_feature, feamap = model.globalmodel(scaled_input)
            # print(f"模型输出 - global_feature维度: {global_feature.shape}, feamap维度: {feamap.shape}")

            
            
            # 对 feamap 应用 GEM 模块
            if gem_modules['rgem'] is not None:
                # print(f"RGEM前 feamap维度: {feamap.shape}")
                feamap = gem_modules['rgem'](feamap)
                # print(f"RGEM后 feamap维度: {feamap.shape}")

            
            # 如果需要进一步处理 feamap，可以添加池化等操作
            # 这里直接使用 global_feature
            # feature_list.append(global_feature)

            # 使用GEMp对特征图进行全局池化
            if gem_modules['gemp'] is not None:
                B, C, H, W = feamap.shape
                # 将特征图展平为1D，然后使用GEMp池化
                flattened_feamap = feamap.view(B, C, H*W)  # [B, C, H*W]
                gem_global_feature = gem_modules['gemp'](flattened_feamap).squeeze(-1)  # [B, C]
                feature_list.append(gem_global_feature)
            else:
                feature_list.append(global_feature)
        
        # 多尺度融合
        if gem_modules['sgem'] is not None and len(feature_list) > 1:
            final_features = gem_modules['sgem'](feature_list)
        else:
            final_features = torch.stack(feature_list, 0).mean(0)
        
        return final_features.cpu().numpy().squeeze()

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


def debug_feature_quality(features, name=""):
    """调试特征质量"""
    if len(features) > 0:
        features_array = np.vstack(features) if isinstance(features, list) else features
        norms = np.linalg.norm(features_array, axis=1)
        print(f"=== {name} 特征调试 ===")
        print(f"形状: {features_array.shape}")
        print(f"范数均值: {norms.mean():.6f}, 标准差: {norms.std():.6f}")
        print(f"范数范围: [{norms.min():.6f}, {norms.max():.6f}]")
        print(f"零特征数量: {np.sum(norms < 1e-6)}")

def process_single_image(img_name, model, image_dir, failed_images):
    """处理单张图像的通用函数"""
    img_full_path = os.path.join(image_dir, img_name)
    
    # 检查文件是否存在
    if not os.path.exists(img_full_path):
        img_full_path = find_image_with_any_extension(img_name, image_dir)
    
    if img_full_path is None or not os.path.exists(img_full_path):
        print(f"警告: 图像文件不存在 - {img_name}")
        failed_images.append(img_name)
        return np.zeros(512, dtype=np.float32)
    
    # 检查文件格式
    ext = os.path.splitext(img_full_path)[-1].lower()
    if ext not in ['.jpg', '.jpeg', '.bmp', '.png', '.pgm']:
        print(f"警告: 不支持的格式 {ext} - {img_full_path}")
        failed_images.append(img_name)
        return np.zeros(512, dtype=np.float32)
    
    # 读取图像
    im = cv2.imread(img_full_path)
    if im is None:
        print(f"警告: 无法读取图像 - {img_full_path}")
        failed_images.append(img_name)
        return np.zeros(512, dtype=np.float32)
    
    im = im.astype(np.float32, copy=False)
    
    try:
        if USE_GEM_ENHANCEMENT:
            global_feature = vit_cnn_delg_extract_with_gem(
                im, model, USE_RGEM, USE_GEMP, USE_SGEM, GEM_SCALE_LIST
            )
        else:
            # 🔥 改成 ViT+CNN 版本的简单特征提取
            global_feature, _ = vit_cnn_delg_extract_simple(im, model)
        
        return global_feature
        
    except Exception as e:
        print(f"错误: 处理图像 {img_name} 失败 - {str(e)}")
        failed_images.append(img_name)
        return np.zeros(512, dtype=np.float32)
    finally:
        del im
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def vit_cnn_delg_extract_simple(im, model):
    """ViT+CNN版本的简单特征提取"""
    # 预处理
    im = preprocess(im)
    input_data = torch.FloatTensor(im).unsqueeze(0)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    
    with torch.no_grad():
        # 🔥 直接调用 ViT+CNN 模型
        global_feature, feamap = model.globalmodel(input_data)
        return global_feature.cpu().numpy().squeeze(), None

def main():
    """主函数 - GEM增强版本 (重构后)"""
    # 设置环境变量以优化内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("=== 开始ViT-DELG GEM增强特征提取 ===")
    print(f"模型权重: {MODEL_WEIGHTS}")
    print(f"图像目录: {INFER_DIR}")
    print(f"PKL文件: {GND_FILE}")
    print(f"GEM配置: RGEM={USE_RGEM}, GEMP={USE_GEMP}, SGEM={USE_SGEM}")
    print(f"多尺度列表: {GEM_SCALE_LIST}")
    
    model = setup_model()
    Q = []  # 查询图像全局特征
    X = []  # 数据库图像全局特征
    failed_images = []
    
    # 读取 PKL 文件
    print("加载PKL文件...")
    with open(GND_FILE, 'rb') as f:
        cfg_dataset = pickle.load(f)
    qimlist = cfg_dataset.get('qimlist', [])
    imlist = cfg_dataset.get('imlist', [])
    
    print(f"查询图像数量: {len(qimlist)}")
    print(f"数据库图像数量: {len(imlist)}")
    
    # 🔥 处理查询图像 - 重构后的简洁版本
    print("\n=== 处理查询图像 (GEM增强) ===")
    for i, qname in enumerate(tqdm(qimlist, desc="Processing query images")):
        global_feature = process_single_image(qname, model, INFER_DIR, failed_images)
        Q.append(global_feature)
        
        if len(Q) == 10:  # 处理完前10张查询图像后调试
            debug_feature_quality(Q, "前10个查询特征")
        
        # 每处理100张图像显示一次进度
        if (i + 1) % 100 == 0:
            print(f"已处理查询图像: {i + 1}/{len(qimlist)}")
    
    # 🔥 处理数据库图像 - 重构后的简洁版本
    print("\n=== 处理数据库图像 (GEM增强) ===")
    for i, dbname in enumerate(tqdm(imlist, desc="Processing database images")):
        global_feature = process_single_image(dbname, model, INFER_DIR, failed_images)
        X.append(global_feature)
        
        # 每处理1000张图像显示一次进度
        if (i + 1) % 1000 == 0:
            print(f"已处理数据库图像: {i + 1}/{len(imlist)}")
    
    # 保存GEM增强的全局特征
    print("\n=== 保存GEM增强的全局特征 ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存全局特征
    Q = np.vstack(Q) if Q else np.zeros((0, 512), dtype=np.float32)
    X = np.vstack(X) if X else np.zeros((0, 512), dtype=np.float32)
    savemat(GLOBAL_FEATURE_PATH, {'Q': Q, 'X': X})
    print(f"GEM增强的全局特征已保存至: {GLOBAL_FEATURE_PATH}")
    print(f"查询特征矩阵形状: {Q.shape}")
    print(f"数据库特征矩阵形状: {X.shape}")
    
    # 保存详细的配置和统计信息
    config_info = {
        'USE_GEM_ENHANCEMENT': USE_GEM_ENHANCEMENT,
        'USE_RGEM': USE_RGEM,
        'USE_GEMP': USE_GEMP,
        'USE_SGEM': USE_SGEM,
        'GEM_SCALE_LIST': GEM_SCALE_LIST,
        'TARGET_SIZE': TARGET_SIZE,
        'total_query_images': len(qimlist),
        'total_database_images': len(imlist),
        'failed_images_count': len(failed_images),
        'success_rate': (len(qimlist) + len(imlist) - len(failed_images)) / (len(qimlist) + len(imlist)) * 100
    }
    
    config_path = os.path.join(OUTPUT_DIR, "gem_extraction_report.txt")
    with open(config_path, "w") as f:
        f.write("=== ViT-DELG GEM增强特征提取报告 ===\n\n")
        f.write("配置参数:\n")
        for key, value in config_info.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\n特征文件路径:\n")
        f.write(f"  全局特征: {GLOBAL_FEATURE_PATH}\n")
        f.write(f"\n处理结果:\n")
        f.write(f"  成功处理: {len(qimlist) + len(imlist) - len(failed_images)} 张图像\n")
        f.write(f"  处理失败: {len(failed_images)} 张图像\n")
        f.write(f"  成功率: {config_info['success_rate']:.2f}%\n")
        
        if failed_images:
            f.write(f"\n失败图像列表:\n")
            for img_name in failed_images:
                f.write(f"  {img_name}\n")
    
    print(f"GEM配置和统计信息已保存至: {config_path}")
    
    # 如果有失败的图像，单独保存失败列表
    if failed_images:
        failed_log_path = os.path.join(OUTPUT_DIR, "failed_images_gem.txt")
        with open(failed_log_path, "w") as f:
            f.write("# 使用GEM增强时处理失败的图像列表\n")
            f.write(f"# 总失败数量: {len(failed_images)}\n")
            f.write(f"# 生成时间: {str(torch.datetime.now())}\n\n")  # 🔥 修复datetime问题
            for img_name in failed_images:
                f.write(f"{img_name}\n")
        print(f"失败图像列表已保存至: {failed_log_path}")
    
    # 验证特征质量
    print("\n=== 特征质量检查 ===")
    if Q.size > 0 and X.size > 0:
        # 检查是否有全零特征
        zero_queries = np.sum(np.all(Q == 0, axis=1))
        zero_database = np.sum(np.all(X == 0, axis=1))
        
        # 检查特征的范数分布
        query_norms = np.linalg.norm(Q, axis=1)
        db_norms = np.linalg.norm(X, axis=1)
        
        print(f"零特征统计:")
        print(f"  查询图像中的零特征: {zero_queries}/{Q.shape[0]}")
        print(f"  数据库图像中的零特征: {zero_database}/{X.shape[0]}")
        
        print(f"特征范数统计:")
        print(f"  查询特征范数 - 均值: {query_norms.mean():.4f}, 标准差: {query_norms.std():.4f}")
        print(f"  数据库特征范数 - 均值: {db_norms.mean():.4f}, 标准差: {db_norms.std():.4f}")
        
        # 保存统计信息
        stats_path = os.path.join(OUTPUT_DIR, "feature_statistics_gem.txt")
        with open(stats_path, "w") as f:
            f.write("=== GEM增强特征统计信息 ===\n\n")
            f.write(f"特征维度: {Q.shape[1]}\n")
            f.write(f"查询图像数量: {Q.shape[0]}\n")
            f.write(f"数据库图像数量: {X.shape[0]}\n\n")
            f.write(f"零特征统计:\n")
            f.write(f"  查询图像零特征: {zero_queries}/{Q.shape[0]} ({zero_queries/Q.shape[0]*100:.2f}%)\n")
            f.write(f"  数据库图像零特征: {zero_database}/{X.shape[0]} ({zero_database/X.shape[0]*100:.2f}%)\n\n")
            f.write(f"特征范数统计:\n")
            f.write(f"  查询特征范数 - 均值: {query_norms.mean():.6f}, 标准差: {query_norms.std():.6f}\n")
            f.write(f"  查询特征范数 - 最小值: {query_norms.min():.6f}, 最大值: {query_norms.max():.6f}\n")
            f.write(f"  数据库特征范数 - 均值: {db_norms.mean():.6f}, 标准差: {db_norms.std():.6f}\n")
            f.write(f"  数据库特征范数 - 最小值: {db_norms.min():.6f}, 最大值: {db_norms.max():.6f}\n")
        
        print(f"特征统计信息已保存至: {stats_path}")
    else:
        print("警告: 没有成功提取到任何特征！")
    
    print(f"\n=== ViT-DELG GEM增强特征提取完成 ===")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"全局特征文件: {os.path.basename(GLOBAL_FEATURE_PATH)}")
    if failed_images:
        print(f"注意: {len(failed_images)} 张图像处理失败，详见失败日志")
    else:
        print("所有图像均成功处理！")


if __name__ == '__main__':
    main()