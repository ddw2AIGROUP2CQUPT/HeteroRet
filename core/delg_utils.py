import os
import numpy as np
import cv2
import pickle
from scipy.io import loadmat
from scipy import spatial
import pydegensac
import torch
from flask import Flask, request, jsonify
import json
from delg_utils import DelgExtraction
from util import walkfile

app = Flask(__name__)

# 配置路径
DATA_ROOT = '/home/ubuntu/workplace01/hxl/delg-pytorch/revisitop/data'
IMAGE_DIR = os.path.join(DATA_ROOT, 'datasets/roxford5k/jpg')
SINGLE_IMG_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/extraction/singleImg'
SINGLE_FEATURES_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/extraction/singleFeatures'
GLOBAL_FEATURE_PATH = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/features/roxford5k512gem_delg_res50_3global.mat'
LOCAL_FEATURE_PATH = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/features/roxford5k__s512_res50_localfea.pickle'

# 模型参数
MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/pretrained/r50_delg_s512.pyth'
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0]
IOU_THRES = 0.98
ATTN_THRES = 260.0
TOP_K = 1000
RF = 291.0
STRIDE = 16.0
PADDING = 145.0
NUM_RERANK = 10
MAX_REPROJECTION_ERROR = 20.0
MAX_RANSAC_ITERATIONS = 1000
HOMOGRAPHY_CONFIDENCE = 1.0
MATCHING_THRESHOLD = 1.0
MAX_DISTANCE = 0.99
USE_RATIO_TEST = False

# 加载模型
def setup_model():
    model = DelgExtraction()
    checkpoint = torch.load(MODEL_WEIGHTS, map_location="cpu")
    
    try:
        state_dict = checkpoint["model_state"]
    except (KeyError, TypeError):
        state_dict = checkpoint

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

model = setup_model()

# 图像预处理
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

# 提取DELG特征
def delg_extract(img, model):
    output_boxes = []
    output_features = []
    output_scores = []
    output_scales = []
    output_original_scale_attn = None
    output_global_feature = None

    for scale_factor in SCALE_LIST:
        im = preprocess(img.copy(), scale_factor)
        im_array = np.asarray([im], dtype=np.float32)
        
        input_data = torch.from_numpy(im_array)
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        
        global_feature, delg_features, delg_scores = model(input_data, targets=None)
        
        if scale_factor == 1.0:
            output_global_feature = global_feature.squeeze()

        selected_boxes, selected_features, selected_scales, selected_scores, selected_original_scale_attn = \
            GetDelgFeature(delg_features, delg_scores, scale_factor, RF, STRIDE, PADDING, ATTN_THRES)

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
    
    if not output_boxes:
        return to_numpy(output_global_feature), {
            'locations': np.zeros((0, 2), dtype=np.float32),
            'descriptors': np.zeros((0, 128), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32)
        }
    
    if output_original_scale_attn is None:
        output_original_scale_attn = torch.zeros(1).uniform_()
    
    output_boxes = concat_tensors_in_list(output_boxes, dim=0)
    output_features = concat_tensors_in_list(output_features, dim=0)
    output_scales = concat_tensors_in_list(output_scales, dim=0)
    output_scores = concat_tensors_in_list(output_scores, dim=0)
    
    keep_indices, count = nms(boxes=output_boxes, scores=output_scores, overlap=IOU_THRES, top_k=TOP_K)
    keep_indices = keep_indices[:TOP_K]

    output_boxes = torch.index_select(output_boxes, dim=0, index=keep_indices)
    output_features = torch.index_select(output_features, dim=0, index=keep_indices)
    output_scales = torch.index_select(output_scales, dim=0, index=keep_indices)
    output_scores = torch.index_select(output_scores, dim=0, index=keep_indices)
    output_locations = CalculateKeypointCenters(output_boxes)
    
    local_data = {
        'locations': to_numpy(output_locations),
        'descriptors': to_numpy(output_features),
        'scores': to_numpy(output_scores)
    }
    
    return to_numpy(output_global_feature), local_data

# 辅助函数
def concat_tensors_in_list(tensor_list, dim=0):
    return torch.cat(tensor_list, dim=dim)

def CalculateKeypointCenters(boxes):
    return (boxes[:, :2] + boxes[:, 2:]) / 2.0

def nms(boxes, scores, overlap, top_k):
    if boxes.numel() == 0:
        return torch.zeros(0), 0
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    
    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        
        if idx.size(0) == 1:
            break
        
        idx = idx[:-1]
        
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        
        inter = w * h
        
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union
        
        idx = idx[IoU.le(overlap)]
    
    return keep, count

def GetDelgFeature(delg_features, delg_scores, scale_factor, rf, stride, padding, attn_thres):
    if delg_features is None or delg_scores is None:
        return None, None, None, None, None
    
    batch_size, channel, height, width = delg_features.size()
    spatial_dim = height * width
    
    delg_features = delg_features.view(batch_size, channel, spatial_dim)
    delg_scores = delg_scores.view(batch_size, spatial_dim)
    
    attn_scores = delg_scores.sigmoid().squeeze(0)
    valid_indices = (attn_scores > attn_thres).nonzero().squeeze(1)
    
    if valid_indices.numel() == 0:
        return None, None, None, None, None
    
    selected_features = delg_features.index_select(2, valid_indices).squeeze(0).t()
    selected_scores = attn_scores.index_select(0, valid_indices)
    
    # Calculate boxes
    grid_x = valid_indices % width
    grid_y = valid_indices // width
    
    x1 = (grid_x.float() * stride + padding - rf / 2) / scale_factor
    y1 = (grid_y.float() * stride + padding - rf / 2) / scale_factor
    x2 = x1 + rf / scale_factor
    y2 = y1 + rf / scale_factor
    
    selected_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    selected_scales = torch.ones(valid_indices.size(0)) * scale_factor
    
    original_scale_mask = (scale_factor == 1.0)
    original_scale_attn = selected_scores[original_scale_mask] if original_scale_mask.any() else None
    
    return selected_boxes, selected_features, selected_scales, selected_scores, original_scale_attn

# 计算匹配关键点
def compute_putative_matching_keypoints(test_keypoints, test_descriptors, train_keypoints, train_descriptors):
    train_descriptor_tree = spatial.cKDTree(train_descriptors)
    
    if USE_RATIO_TEST:
        distances, matches = train_descriptor_tree.query(test_descriptors, k=2, n_jobs=-1)
        test_kp_count = test_keypoints.shape[0]
        test_matching_keypoints = np.array([
            test_keypoints[i,] 
            for i in range(test_kp_count) 
            if distances[i][0] < MATCHING_THRESHOLD * distances[i][1]
        ])
        train_matching_keypoints = np.array([
            train_keypoints[matches[i][0],] 
            for i in range(test_kp_count) 
            if distances[i][0] < MATCHING_THRESHOLD * distances[i][1]
        ])
    else:
        _, matches = train_descriptor_tree.query(
            test_descriptors, distance_upper_bound=MAX_DISTANCE)
        
        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]
        
        test_matching_keypoints = np.array([
            test_keypoints[i,] 
            for i in range(test_kp_count) 
            if matches[i] != train_kp_count
        ])
        train_matching_keypoints = np.array([
            train_keypoints[matches[i],] 
            for i in range(test_kp_count) 
            if matches[i] != train_kp_count
        ])
    return test_matching_keypoints, train_matching_keypoints

# 计算内点数量
def compute_num_inliers(test_keypoints, test_descriptors, train_keypoints, train_descriptors):
    test_match_kp, train_match_kp = compute_putative_matching_keypoints(
        test_keypoints, test_descriptors, train_keypoints, train_descriptors)
    
    if test_match_kp.shape[0] <= 4:
        return 0
    
    try:
        _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,
                                           MAX_REPROJECTION_ERROR,
                                           HOMOGRAPHY_CONFIDENCE,
                                           MAX_RANSAC_ITERATIONS)
    except np.linalg.LinAlgError:
        return 0
    
    return int(copy.deepcopy(mask).astype(np.float32).sum())

# 重排序
def rerank(query_local_feature, ranks_before_gv, local_features, train_ids):
    ranks_after_gv = ranks_before_gv.copy()
    query_locations = query_local_feature['locations']
    query_descriptors = query_local_feature['descriptors']
    
    inliers_numrerank = np.zeros(NUM_RERANK)
    
    for j in range(NUM_RERANK):
        index_img = train_ids[ranks_before_gv[j]]
        index_features = local_features[index_img]
        
        try:
            num_inliers = compute_num_inliers(
                query_locations, query_descriptors,
                index_features['locations'], index_features['descriptors'])
            inliers_numrerank[j] = num_inliers
        except:
            continue
    
    ranks_after_gv[:NUM_RERANK] = ranks_before_gv[np.argsort(-1 * inliers_numrerank)]
    return ranks_after_gv

# API端点
@app.route('/search', methods=['POST'])
def search():
    # 获取请求数据
    data = request.get_json()
    image_name = data.get('image_name')
    
    if not image_name:
        return jsonify({'error': 'No image name provided'}), 400
    
    # 检查图片是否存在
    image_path = os.path.join(SINGLE_IMG_DIR, image_name)
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({'error': 'Failed to read image'}), 500
    
    img = img.astype(np.float32, copy=False)
    
    # 提取特征
    try:
        global_feature, local_feature = delg_extract(img, model)
    except Exception as e:
        return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500
    
    # 保存特征
    os.makedirs(SINGLE_FEATURES_DIR, exist_ok=True)
    feature_path = os.path.join(SINGLE_FEATURES_DIR, f'{os.path.splitext(image_name)[0]}_features.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump({'global': global_feature, 'local': local_feature}, f)
    
    # 加载ROxford特征
    try:
        oxford_features = loadmat(GLOBAL_FEATURE_PATH)
        Q = oxford_features['Q']
        X = oxford_features['X']
        
        with open(LOCAL_FEATURE_PATH, 'rb') as f:
            local_features = pickle.load(f)
    except Exception as e:
        return jsonify({'error': f'Failed to load Oxford features: {str(e)}'}), 500
    
    # 全局搜索
    sim = np.dot(X, global_feature.reshape(-1, 1))
    ranks = np.argsort(-sim, axis=0).squeeze()
    
    # 重排序
    train_ids = [f'{i}.jpg' for i in range(X.shape[0])]  # 假设图片名称为0.jpg, 1.jpg等
    ranks_rerank = rerank(local_feature, ranks[:NUM_RERANK], local_features, train_ids)
    
    # 获取前10个结果
    results = []
    for i, rank in enumerate(ranks_rerank[:10]):
        results.append({
            'rank': i + 1,
            'image_name': train_ids[rank],
            'score': float(sim[rank])
        })
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)