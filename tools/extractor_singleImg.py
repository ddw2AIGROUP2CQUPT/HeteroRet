# extractor_singleImg.py
import sys
import os
import numpy as np
import cv2
import pickle
import torch
from scipy.io import savemat

# 添加项目根目录到路径
data_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(data_root)

# 导入必要模块
import core.config as config
from core.config import cfg
import delg_utils

# 设置输出路径
QUERY_FEATURE_DIR = os.path.join(data_root, './features/query_features')
GLOBAL_FEATURE_DIR = os.path.join(QUERY_FEATURE_DIR, 'global_features')
LOCAL_FEATURE_DIR = os.path.join(QUERY_FEATURE_DIR, 'local_features')
os.makedirs(GLOBAL_FEATURE_DIR, exist_ok=True)
os.makedirs(LOCAL_FEATURE_DIR, exist_ok=True)

# 参数设定（复制原始）
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0]
IOU_THRES = 0.98
ATTN_THRES = 260.0
TOP_K = 1000
RF = 291.0
STRIDE = 16.0
PADDING = 145.0
MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/node01_gpu8_20250716/checkpoints/model_epoch_0100.pyth'

def setup_model():
    model = delg_utils.DelgExtraction()
    load_checkpoint(MODEL_WEIGHTS, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

def load_checkpoint(checkpoint_file, model, optimizer=None):
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return checkpoint

def im_scale(im, scale_factor):
    h, w = im.shape[:2]
    h_new, w_new = int(round(h * scale_factor)), int(round(w * scale_factor))
    return cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR).astype(np.float32)

def color_norm(im, mean, std):
    for i in range(im.shape[0]):
        im[i] = (im[i] - mean[i]) / std[i]
    return im

def preprocess(im, scale_factor):
    im = im_scale(im, scale_factor)
    im = im.transpose([2, 0, 1])
    im = im / 255.0
    im = color_norm(im, _MEAN, _SD)
    return im

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def extract(im_array, model):
    input_data = torch.from_numpy(im_array).cuda() if torch.cuda.is_available() else torch.from_numpy(im_array)
    return model(input_data, targets=None)

def delg_extract(img, model):
    output_boxes, output_features, output_scores, output_scales = [], [], [], []
    output_original_scale_attn = None
    output_global_feature = None

    for scale_factor in SCALE_LIST:
        im = preprocess(img.copy(), scale_factor)
        im_array = np.asarray([im], dtype=np.float32)
        global_feature, delg_features, delg_scores = extract(im_array, model)
        if scale_factor == 1.0:
            output_global_feature = global_feature.squeeze()

        selected_boxes, selected_features, selected_scales, selected_scores, selected_original_scale_attn = \
            delg_utils.GetDelgFeature(delg_features, delg_scores, scale_factor,
                                      RF, STRIDE, PADDING, ATTN_THRES)
        if selected_boxes is not None:
            output_boxes.append(selected_boxes)
            output_features.append(selected_features)
            output_scales.append(selected_scales)
            output_scores.append(selected_scores)
            if selected_original_scale_attn is not None:
                output_original_scale_attn = selected_original_scale_attn

    if not output_boxes:
        return to_numpy(output_global_feature), {
            'locations': np.zeros((0, 2), dtype=np.float32),
            'descriptors': np.zeros((0, 128), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32)
        }

    output_boxes = delg_utils.concat_tensors_in_list(output_boxes, dim=0)
    output_features = delg_utils.concat_tensors_in_list(output_features, dim=0)
    output_scales = delg_utils.concat_tensors_in_list(output_scales, dim=0)
    output_scores = delg_utils.concat_tensors_in_list(output_scores, dim=0)

    keep_indices, count = delg_utils.nms(output_boxes, output_scores, IOU_THRES, TOP_K)
    keep_indices = keep_indices[:TOP_K]
    output_boxes = torch.index_select(output_boxes, 0, keep_indices)
    output_features = torch.index_select(output_features, 0, keep_indices)
    output_scores = torch.index_select(output_scores, 0, keep_indices)
    output_locations = delg_utils.CalculateKeypointCenters(output_boxes)

    local_data = {
        'locations': to_numpy(output_locations),
        'descriptors': to_numpy(output_features),
        'scores': to_numpy(output_scores)
    }

    return to_numpy(output_global_feature), local_data

def save_single_feature(filename, global_feature, local_data):
    global_path = os.path.join(GLOBAL_FEATURE_DIR, f"{filename}.mat")
    savemat(global_path, {'global_feature': global_feature})
    local_path = os.path.join(LOCAL_FEATURE_DIR, f"{filename}.pickle")
    with open(local_path, 'wb') as f:
        pickle.dump(local_data, f, protocol=2)

def extract_single_image(image_path):
    config.load_cfg_fom_args("Extract feature.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    model = setup_model()

    filename = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = img.astype(np.float32, copy=False)
    global_feature, local_data = delg_extract(img, model)
    save_single_feature(filename, global_feature, local_data)
    return filename

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help='Path to a single image.')
    args = parser.parse_args()
    extract_single_image(args.image_path)
