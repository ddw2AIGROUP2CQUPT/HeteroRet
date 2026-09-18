
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
from util import walkfile
import delg_utils

""" 全局设置 """
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
# MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/node1_gpu8_res101_2025_7_20/checkpoints/model_epoch_0100.pyth'
# #MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/8gpu_zzdata_20250729/checkpoints/model_epoch_0050.pyth'
# #MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/pretrained/r50_delg_s512.pyth'
# #MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/node1_gpu8_res101_2025_7_20/checkpoints/model_epoch_0100.pyth'
# INFER_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/datasets/data/newzzdata/augmented_dataset_1'
# #INFER_DIR = '/home/ubuntu/san/hxl/models/research/delf/delf/python/data/oxford5k_images'
# TEST_LIST = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/datasets/data/newzzdata/augmented_dataset_1/newtest.txt'
# #TEST_LIST = '/home/ubuntu/san/hxl/models/research/delf/delf/python/data/roxford5k.txt'
# GND_FILE = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/datasets/data/newzzdata/augmented_dataset_1/gnd_newzz_50.pkl'
# #GND_FILE = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/gnd_roxford5k.pkl'
# OUTPUT_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/features'
# GLOBAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'zz_100_global.mat')
# LOCAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'zz100_test_local_global_fea.pickle')



MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/8gpu_newzzdata300w_20250812/checkpoints/model_epoch_0050.pyth'
INFER_DIR = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest/newtrain'
TEST_LIST = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest/basetest.txt'
GND_FILE = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest/new_zz_200wtest.pkl'
OUTPUT_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/features'
GLOBAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_res101_global.mat')
LOCAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'newzz300w_res101_test_local_global_fea.pickle')

SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142]  # 移除 2.0 以减少内存占用
IOU_THRES = 0.98
ATTN_THRES = 260.0
TOP_K = 1000
RF = 291.0
STRIDE = 16.0
PADDING = 145.0

# 临时添加调试代码
with open(GND_FILE, 'rb') as f:
    cfg_dataset = pickle.load(f)
qimlist = cfg_dataset.get('qimlist', [])
imlist = cfg_dataset.get('imlist', [])

print("PKL文件中的前5个查询图像路径:")
for i in range(min(5, len(qimlist))):
    print(f"  {qimlist[i]}")

print("PKL文件中的前5个数据库图像路径:")
for i in range(min(5, len(imlist))):
    print(f"  {imlist[i]}")

def setup_model():
    """设置模型"""
    model = delg_utils.DelgExtraction()
    print(model)
    load_checkpoint(MODEL_WEIGHTS, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

def extract(im_array, model):
    """执行模型前向传播"""
    input_data = torch.from_numpy(im_array)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    # 提取全局和局部特征
    global_feature, delg_features, delg_scores = model(input_data, targets=None)
    return global_feature, delg_features, delg_scores

def delg_extract(img, model):
    """多尺度处理，提取局部和全局特征"""
    output_boxes = []
    output_features = []
    output_scores = []
    output_scales = []
    output_original_scale_attn = None
    output_global_feature = None

    for scale_factor in SCALE_LIST:
        im = preprocess(img.copy(), scale_factor)
        im_array = np.asarray([im], dtype=np.float32)
        global_feature, delg_features, delg_scores = extract(im_array, model)

        # 选择 scale_factor=1.0 的全局特征
        if scale_factor == 1.0:
            output_global_feature = global_feature.squeeze()

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

        output_boxes.append(selected_boxes) if selected_boxes is not None else output_boxes
        output_features.append(selected_features) if selected_features is not None else output_features
        output_scales.append(selected_scales) if selected_scales is not None else output_scales
        output_scores.append(selected_scores) if selected_scores is not None else output_scores
        if selected_original_scale_attn is not None:
            output_original_scale_attn = selected_original_scale_attn
        
        # 释放中间张量
        del im, im_array, global_feature, delg_features, delg_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if not output_boxes:
        # 确保全局特征是 NumPy 数组
        global_feature_np = to_numpy(output_global_feature)
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
    global_feature_np = to_numpy(output_global_feature)
    if global_feature_np.size == 0:
        global_feature_np = np.zeros(512, dtype=np.float32)
    elif global_feature_np.ndim == 0:
        global_feature_np = np.array([global_feature_np.item()], dtype=np.float32)
    
    # 释放中间张量
    del output_boxes, output_features, output_scales, output_scores, output_locations
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

def preprocess(im, scale_factor):
    im = im_scale(im, scale_factor)
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

def load_checkpoint(checkpoint_file, model, optimizer=None):
    """加载检查点"""
    err_str = f"Checkpoint '{checkpoint_file}' not found"
    assert os.path.exists(checkpoint_file), err_str
    print(f"Loading checkpoint from: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        state_dict = checkpoint["model_state"]
    except (KeyError, TypeError):
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    ms = model
    model_dict = ms.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    print(f'construct model total {len(model_dict)} keys and pretrain model total {len(state_dict)} keys.')
    print(f'{len(pretrained_dict)} pretrain keys load successfully.')
    not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
    if not_loaded_keys:
        print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
    model_dict.update(pretrained_dict)
    ms.load_state_dict(model_dict)
    print("Checkpoint loaded successfully.")
    return checkpoint

def main():
    # 设置环境变量以优化内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    model = setup_model()
    Q = []  # 查询图像全局特征
    X = []  # 数据库图像全局特征
    local_features = {}
    failed_images = []
    
    # 读取 gnd_newzz_50.pkl
    with open(GND_FILE, 'rb') as f:
        cfg_dataset = pickle.load(f)
    qimlist = cfg_dataset.get('qimlist', [])
    imlist = cfg_dataset.get('imlist', [])
    print(f"Length of qimlist: {len(qimlist)}")
    print(f"Length of imlist: {len(imlist)}")
    
    # 读取 newtest.txt
    with open(TEST_LIST, 'r') as f:
        lines = f.readlines()
    
    # 构建图像路径字典
    image_paths = {}
    for line in lines:
        img_path, _ = line.strip().split()
        name = os.path.basename(img_path)
        image_paths[name] = img_path
    
    # 处理查询图像
    for i, qname in enumerate(tqdm(qimlist, desc="Processing query images")):
        qname = os.path.basename(qname)
        img_path = image_paths.get(qname)
        if not img_path:
            print(f"警告: 查询图像 {qname} 不在 newtest.txt 中")
            failed_images.append(qname)
            continue
        
        img_full_path = find_image_with_any_extension(img_path, INFER_DIR)
        if img_full_path is None:
            print(f"警告: 图像文件不存在 {img_path}")
            failed_images.append(img_path)
            continue
        
        ext = os.path.splitext(img_full_path)[-1].lower()
        if ext not in ['.jpg', '.jpeg', '.bmp', '.png', '.pgm']:
            print(f"警告: 跳过不支持的格式 {ext} - {img_full_path}")
            failed_images.append(img_path)
            continue
        
        im = cv2.imread(img_full_path)
        if im is None:
            print(f"警告: 无法读取图像 {img_full_path}")
            failed_images.append(img_path)
            continue
        
        im = im.astype(np.float32, copy=False)
        try:
            global_feature, local_data = delg_extract(im, model)
            Q.append(global_feature)
            local_features[qname] = local_data
        except Exception as e:
            print(f"错误: 处理 {qname} 时失败 - {str(e)}")
            failed_images.append(img_path)
            continue
        finally:
            del im
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 处理数据库图像
    for i, dbname in enumerate(tqdm(imlist, desc="Processing database images")):
        dbname = os.path.basename(dbname)
        img_path = image_paths.get(dbname)
        if not img_path:
            print(f"警告: 数据库图像 {dbname} 不在 newtest.txt 中")
            failed_images.append(dbname)
            continue
        
        img_full_path = find_image_with_any_extension(img_path, INFER_DIR)
        if img_full_path is None:
            print(f"警告: 图像文件不存在 {img_path}")
            failed_images.append(img_path)
            continue
        
        ext = os.path.splitext(img_full_path)[-1].lower()
        if ext not in ['.jpg', '.jpeg', '.bmp', '.png', '.pgm']:
            print(f"警告: 跳过不支持的格式 {ext} - {img_full_path}")
            failed_images.append(img_path)
            continue
        
        im = cv2.imread(img_full_path)
        if im is None:
            print(f"警告: 无法读取图像 {img_full_path}")
            failed_images.append(img_path)
            continue
        
        im = im.astype(np.float32, copy=False)
        try:
            global_feature, local_data = delg_extract(im, model)
            X.append(global_feature)
            local_features[dbname] = local_data
        except Exception as e:
            print(f"错误: 处理 {dbname} 时失败 - {str(e)}")
            failed_images.append(img_path)
            continue
        finally:
            del im
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 保存全局特征
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Q = np.vstack(Q) if Q else np.zeros((0, 512), dtype=np.float32)
    X = np.vstack(X) if X else np.zeros((0, 512), dtype=np.float32)
    savemat(GLOBAL_FEATURE_PATH, {'Q': Q, 'X': X})
    print(f"Global features saved to {GLOBAL_FEATURE_PATH}")
    
    # 保存局部特征
    with open(LOCAL_FEATURE_PATH, 'wb') as f:
        pickle.dump(local_features, f, protocol=2)
    print(f"Local features saved to {LOCAL_FEATURE_PATH}")
    
    # 保存无法读取的图像列表
    if failed_images:
        failed_log_path = os.path.join(OUTPUT_DIR, "failed_images.txt")
        with open(failed_log_path, "w") as f:
            f.write("\n".join(failed_images))
        print(f"无法读取的图像列表保存至 {failed_log_path}")

if __name__ == '__main__':
    config.load_cfg_fom_args("Extract feature.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    main()
