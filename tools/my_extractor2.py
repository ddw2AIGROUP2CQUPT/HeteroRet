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

from util import walkfile
import delg_utils

""" 全局设置 """
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

# 配置路径
# MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG_VIT/output/8gpu_newzzdata300w_vit_20250924_V1/checkpoints/model_epoch_0050.pyth'
# INFER_DIR = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2'  # 修改为newtest目录
# GND_FILE = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2/new_zz_200wtest.pkl'
# OUTPUT_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG_VIT/output/features'
# GLOBAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_global_20240924V1.mat')
# LOCAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_local_global_fea_20240924V1.pickle')

# MODEL_WEIGHTS = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/output/8gpu_newzzdata300w_vit_20251019_V5/checkpoints/model_epoch_0060.pyth'
# INFER_DIR = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2'  # 修改为newtest目录
# GND_FILE = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2/newtest2.pkl'
# OUTPUT_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG_VIT/output/features/8gpu_newzzdata300w_vit_20251019V5'
# GLOBAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_global_20251019V5.mat')
# LOCAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_local_global_fea_20251019V5.pickle')

MODEL_WEIGHTS = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/output/8gpu_newzzdata300w_vit_20251121_CNN/checkpoints/model_epoch_0030.pyth'
INFER_DIR = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2'  # 修改为newtest目录
GND_FILE = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2/newtest2.pkl'
OUTPUT_DIR = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/features/8gpu_newzzdata300w_vit_251121_CNN'
GLOBAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_global_epoch30.mat')
LOCAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_vit_test_local_global_fea_epoch30.pickle')

# 模型参数
SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142]
IOU_THRES = 0.98
ATTN_THRES = 260.0
TOP_K = 1000
RF = 291.0
STRIDE = 16.0
PADDING = 145.0

def setup_model():
    """设置模型"""
    model = delg_utils.DelgExtraction()
    # print(model)
    load_checkpoint(MODEL_WEIGHTS, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

# def extract(im_array, model):
#     """执行模型前向传播"""
#     input_data = torch.from_numpy(im_array)
#     if torch.cuda.is_available():
#         input_data = input_data.cuda()
#     # 提取全局和局部特征
#     global_feature, delg_features, delg_scores = model(input_data, targets=None)
#     return global_feature, delg_features, delg_scores

def extract(im_array, model):
    """执行模型前向传播"""
    input_data = torch.from_numpy(im_array)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    
    # 修改：不传递targets参数，因为推理时不需要
    global_feature, delg_features, delg_scores = model(input_data)
    return global_feature, delg_features, delg_scores

def delg_extract(img, model):
    """多尺度处理，提取局部和全局特征"""
    print(f"输入图像尺寸: {img.shape}")  # 调试信息
    
    output_boxes = []
    output_features = []
    output_scores = []
    output_scales = []
    output_original_scale_attn = None
    output_global_feature = None
    
    for scale_factor in SCALE_LIST:
        # 🔥 在循环开始时初始化变量
        im = None
        im_array = None
        global_feature = None
        delg_features = None
        delg_scores = None
        
        try:
            im = preprocess(img.copy(), scale_factor)
            im_array = np.asarray([im], dtype=np.float32)
            global_feature, delg_features, delg_scores = extract(im_array, model)
            
            # 选择 scale_factor=1.0 的全局特征
            if scale_factor == 1.0:
                output_global_feature = global_feature.squeeze()
            
            # 添加张量检查
            if delg_features is None or delg_scores is None:
                print(f"[警告] scale {scale_factor}: 特征为空，跳过")
                continue
                
            if delg_features.numel() == 0 or delg_scores.numel() == 0:
                print(f"[警告] scale {scale_factor}: 特征张量为空，跳过")
                continue
            
            selected_boxes, selected_features, \
            selected_scales, selected_scores, \
            selected_original_scale_attn = \
                        delg_utils.GetDelgFeature(delg_features, 
                                                delg_scores,
                                                scale_factor,
                                                RF,
                                                STRIDE,
                                                PADDING,
                                                ATTN_THRES)
            
            if selected_boxes is not None:
                output_boxes.append(selected_boxes)
            if selected_features is not None:
                output_features.append(selected_features)
            if selected_scales is not None:
                output_scales.append(selected_scales)
            if selected_scores is not None:
                output_scores.append(selected_scores)
            if selected_original_scale_attn is not None:
                output_original_scale_attn = selected_original_scale_attn
                
        except Exception as e:
            print(f"[错误] scale {scale_factor} 处理失败: {e}")
            continue
        finally:
            # 🔥 修复：只删除已定义的变量
            try:
                if im is not None:
                    del im
                if im_array is not None:
                    del im_array
                if global_feature is not None:
                    del global_feature
                if delg_features is not None:
                    del delg_features
                if delg_scores is not None:
                    del delg_scores
            except:
                pass  # 忽略删除时的任何错误
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if not output_boxes:
        # 确保全局特征是 NumPy 数组
        global_feature_np = to_numpy(output_global_feature) if output_global_feature is not None else np.zeros(512, dtype=np.float32)
        if global_feature_np.size == 0:
            global_feature_np = np.zeros(512, dtype=np.float32)
        elif global_feature_np.ndim == 0:
            global_feature_np = np.array([global_feature_np.item()], dtype=np.float32)
        return global_feature_np, {
            'locations': np.zeros((0, 2), dtype=np.float32),
            'descriptors': np.zeros((0, 128), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32)
        }
    
    if output_original_scale_attn is None:
        output_original_scale_attn = torch.zeros(1).uniform_()
    
    # 合并多尺度局部特征
    output_boxes = delg_utils.concat_tensors_in_list(output_boxes, dim=0)
    output_features = delg_utils.concat_tensors_in_list(output_features, dim=0)
    output_scales = delg_utils.concat_tensors_in_list(output_scales, dim=0)
    output_scores = delg_utils.concat_tensors_in_list(output_scores, dim=0)
    
    # 非极大值抑制 (NMS)
    keep_indices, count = delg_utils.nms(boxes=output_boxes,
                                        scores=output_scores,
                                        overlap=IOU_THRES,
                                        top_k=TOP_K)
    keep_indices = keep_indices[:TOP_K]
    output_boxes = torch.index_select(output_boxes, dim=0, index=keep_indices)
    output_features = torch.index_select(output_features, dim=0, index=keep_indices)
    output_scales = torch.index_select(output_scales, dim=0, index=keep_indices)
    output_scores = torch.index_select(output_scores, dim=0, index=keep_indices)
    output_locations = delg_utils.CalculateKeypointCenters(output_boxes)
    
    local_data = {
        'locations': to_numpy(output_locations),
        'descriptors': to_numpy(output_features),
        'scores': to_numpy(output_scores)
    }
    
    # 确保全局特征是 NumPy 数组
    global_feature_np = to_numpy(output_global_feature) if output_global_feature is not None else np.zeros(512, dtype=np.float32)
    if global_feature_np.size == 0:
        global_feature_np = np.zeros(512, dtype=np.float32)
    elif global_feature_np.ndim == 0:
        global_feature_np = np.array([global_feature_np.item()], dtype=np.float32)
    
    # 释放中间张量
    try:
        del output_boxes, output_features, output_scales, output_scores, output_locations
    except:
        pass
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return global_feature_np, local_data

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

# def preprocess(im, scale_factor):
#     im = im_scale(im, scale_factor)
#     im = im.transpose([2, 0, 1])
#     im = im / 255.0
#     im = color_norm(im, _MEAN, _SD)
#     return im

def preprocess(im, scale_factor):
    # 先按比例缩放
    im = im_scale(im, scale_factor)
    
    # 🔥 关键修改：确保输入尺寸符合ViT要求
    target_size = 224  # ViT标准输入尺寸
    h, w = im.shape[:2]
    
    # 如果图像太小，先padding到合适尺寸
    if h < target_size or w < target_size:
        # 计算padding
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        
        # 对称padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        im = cv2.copyMakeBorder(im, pad_top, pad_bottom, pad_left, pad_right, 
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # 然后resize到标准尺寸
    im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # 转换为CHW格式并归一化
    im = im.transpose([2, 0, 1])
    im = im / 255.0
    im = color_norm(im, _MEAN, _SD)
    return im

def im_scale(im, scale_factor):
    h, w = im.shape[:2]
    h_new = int(round(h * scale_factor))
    w_new = int(round(w * scale_factor))
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float32)

def color_norm(im, mean, std):
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# def load_checkpoint(checkpoint_file, model, optimizer=None):
#     """加载检查点"""
#     err_str = f"Checkpoint '{checkpoint_file}' not found"
#     assert os.path.exists(checkpoint_file), err_str
#     print(f"Loading checkpoint from: {checkpoint_file}")
#     checkpoint = torch.load(checkpoint_file, map_location="cpu")
#     try:
#         state_dict = checkpoint["model_state"]
#     except (KeyError, TypeError):
#         state_dict = checkpoint
#     if list(state_dict.keys())[0].startswith('module.'):
#         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#     ms = model
#     model_dict = ms.state_dict()
#     pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
#     print(f'construct model total {len(model_dict)} keys and pretrain model total {len(state_dict)} keys.')
#     print(f'{len(pretrained_dict)} pretrain keys load successfully.')
#     not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
#     if not_loaded_keys:
#         print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
#     model_dict.update(pretrained_dict)
#     ms.load_state_dict(model_dict)
#     print("Checkpoint loaded successfully.")
#     return checkpoint

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
    
    # 🔥 关键修改：调试键名匹配
    print("=== 调试权重加载 ===")
    print("模型结构中的前几个键:")
    for i, key in enumerate(list(model_dict.keys())[:5]):
        print(f"  model[{i}]: {key}")
    
    print("checkpoint中的前几个键:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"  checkpoint[{i}]: {key}")
    
    # 更智能的键名映射
    pretrained_dict = {}
    
    # 先尝试直接匹配
    direct_matches = 0
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].size() == v.size():
            pretrained_dict[k] = v
            direct_matches += 1
    
    print(f"直接匹配成功: {direct_matches}")
    
    # 如果直接匹配很少，尝试键名转换
    if direct_matches < len(state_dict) * 0.5:  # 如果直接匹配少于50%
        print("尝试键名转换...")
        
        # 创建键名映射表
        model_keys = list(model_dict.keys())
        checkpoint_keys = list(state_dict.keys())
        
        # 尝试不同的映射策略
        for ckpt_key, ckpt_value in state_dict.items():
            if ckpt_key in pretrained_dict:
                continue  # 已经匹配过了
                
            # 策略1: 移除不同的前缀
            for prefix_remove in ['globalmodel.', 'localmodel.', 'model.']:
                if ckpt_key.startswith(prefix_remove):
                    new_key = ckpt_key[len(prefix_remove):]
                    target_key = f"model.{new_key}"
                    if target_key in model_dict and model_dict[target_key].size() == ckpt_value.size():
                        pretrained_dict[target_key] = ckpt_value
                        break
            
            # 策略2: 添加model.前缀
            if ckpt_key not in [k.replace('model.', '') for k in pretrained_dict.keys()]:
                target_key = f"model.{ckpt_key}"
                if target_key in model_dict and model_dict[target_key].size() == ckpt_value.size():
                    pretrained_dict[target_key] = ckpt_value
    
    print(f'construct model total {len(model_dict)} keys and pretrain model total {len(state_dict)} keys.')
    print(f'{len(pretrained_dict)} pretrain keys load successfully.')
    
    # 检查关键的ViT组件是否加载成功
    vit_loaded = sum(1 for k in pretrained_dict.keys() if 'vit' in k)
    print(f"ViT相关权重加载数量: {vit_loaded}")
    
    if vit_loaded == 0:
        print("❌ 警告：没有ViT权重被加载！检查键名映射...")
        # 显示一些具体的映射示例
        print("checkpoint中的ViT键示例:")
        vit_keys = [k for k in state_dict.keys() if 'vit' in k.lower()][:3]
        for vk in vit_keys:
            print(f"  {vk}")
        print("模型中的ViT键示例:")
        model_vit_keys = [k for k in model_dict.keys() if 'vit' in k.lower()][:3]
        for mk in model_vit_keys:
            print(f"  {mk}")
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Checkpoint loaded successfully.")
    return checkpoint

def main():
    # 设置环境变量以优化内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("=== 开始特征提取 ===")
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
    
    # 调试信息
    if qimlist:
        print(f"示例查询路径: {qimlist[0]}")
        sample_query_path = os.path.join(INFER_DIR, qimlist[0])
        print(f"示例完整路径: {sample_query_path}")
        print(f"示例文件存在: {os.path.exists(sample_query_path)}")
    
    # 处理查询图像
    print("\n=== 处理查询图像 ===")
    for i, qname in enumerate(tqdm(qimlist, desc="Processing query images")):
        # 构建完整路径
        img_full_path = os.path.join(INFER_DIR, qname)
        
        # 如果直接路径不存在，尝试查找其他扩展名
        if not os.path.exists(img_full_path):
            img_full_path = find_image_with_any_extension(qname, INFER_DIR)
        
        if img_full_path is None or not os.path.exists(img_full_path):
            print(f"警告: 图像文件不存在 - {qname}")
            failed_images.append(qname)
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
            global_feature, local_data = delg_extract(im, model)
            Q.append(global_feature)
            
            # 使用文件名作为键
            qname_key = os.path.basename(qname)
            local_features[qname_key] = local_data
            
        except Exception as e:
            print(f"错误: 处理查询图像 {qname} 失败 - {str(e)}")
            failed_images.append(qname)
            continue
        finally:
            del im
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 处理数据库图像
    print("\n=== 处理数据库图像 ===")
    for i, dbname in enumerate(tqdm(imlist, desc="Processing database images")):
        # 构建完整路径
        img_full_path = os.path.join(INFER_DIR, dbname)
        
        # 如果直接路径不存在，尝试查找其他扩展名
        if not os.path.exists(img_full_path):
            img_full_path = find_image_with_any_extension(dbname, INFER_DIR)
        
        if img_full_path is None or not os.path.exists(img_full_path):
            print(f"警告: 图像文件不存在 - {dbname}")
            failed_images.append(dbname)
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
            global_feature, local_data = delg_extract(im, model)
            X.append(global_feature)
            
            # 使用文件名作为键
            dbname_key = os.path.basename(dbname)
            local_features[dbname_key] = local_data
            
        except Exception as e:
            print(f"错误: 处理数据库图像 {dbname} 失败 - {str(e)}")
            failed_images.append(dbname)
            continue
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
    print(f"全局特征保存至: {GLOBAL_FEATURE_PATH}")
    print(f"查询特征矩阵形状: {Q.shape}")
    print(f"数据库特征矩阵形状: {X.shape}")
    
    # 保存局部特征
    with open(LOCAL_FEATURE_PATH, 'wb') as f:
        pickle.dump(local_features, f, protocol=2)
    print(f"局部特征保存至: {LOCAL_FEATURE_PATH}")
    print(f"局部特征数量: {len(local_features)}")
    
    # 保存失败的图像列表
    if failed_images:
        failed_log_path = os.path.join(OUTPUT_DIR, "./output/failed_images.txt")
        with open(failed_log_path, "w") as f:
            f.write("\n".join(failed_images))
        print(f"失败图像列表保存至: {failed_log_path}")
        print(f"失败图像数量: {len(failed_images)}")
    
    print("\n=== 特征提取完成 ===")

if __name__ == '__main__':
    # config.load_cfg_fom_args("Extract feature.")
    # config.assert_and_infer_cfg()
    # cfg.freeze()
    main()