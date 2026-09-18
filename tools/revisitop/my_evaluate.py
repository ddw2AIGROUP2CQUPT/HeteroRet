
import os
import pickle
import numpy as np
from scipy.io import loadmat
from scipy import spatial
from concurrent import futures
import pydegensac
from skimage import io as skio

from dataset import configdataset, find_image_with_any_extension
from compute import compute_map

# data_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
data_root = '/home/ubuntu/public-Datasets/hxl'  # 直接指定到你的数据根目录
test_dataset = 'newzzdata_300w'
#test_dataset = 'roxford5k'
IMAGE_PATH = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest/newtrain'
#IMAGE_PATH = '/home/ubuntu/san/hxl/models/research/delf/delf/python/data/oxford5k_images'
GLOBAL_FEATURE_PATH = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/features/newzz300w_res101_global.mat'
LOCAL_FEATURE_PATH = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/features/newzz300w_res101_test_local_global_fea.pickle'
OUTPUT_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/eval'

NUM_RERANK = 50
MAX_REPROJECTION_ERROR = 20.0
MAX_RANSAC_ITERATIONS = 1000
HOMOGRAPHY_CONFIDENCE = 1.0
MATCHING_THRESHOLD = 1.0
MAX_DISTANCE = 0.99
USE_RATIO_TEST = False
DRAW_MATCHES = False

def global_search(global_feature_path):
    """基于全局特征进行排序"""
    features = loadmat(global_feature_path)
    Q = features['Q']
    X = features['X']
    sim = np.dot(X, Q.T)
    ranks = np.argsort(-sim, axis=0)
    return ranks

def compute_putative_matching_keypoints(test_keypoints, test_descriptors, train_keypoints, train_descriptors,
                                       use_ratio_test=USE_RATIO_TEST, matching_threshold=MATCHING_THRESHOLD,
                                       max_distance=MAX_DISTANCE):
    """查找测试描述子和训练描述子的匹配"""
    train_descriptor_tree = spatial.cKDTree(train_descriptors)
    if use_ratio_test:
        distances, matches = train_descriptor_tree.query(test_descriptors, k=2, n_jobs=-1)
        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]
        test_matching_keypoints = np.array([
            test_keypoints[i] for i in range(test_kp_count)
            if distances[i][0] < matching_threshold * distances[i][1]
        ])
        train_matching_keypoints = np.array([
            train_keypoints[matches[i][0]] for i in range(test_kp_count)
            if distances[i][0] < matching_threshold * distances[i][1]
        ])
    else:
        _, matches = train_descriptor_tree.query(test_descriptors, distance_upper_bound=max_distance)
        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]
        test_matching_keypoints = np.array([
            test_keypoints[i] for i in range(test_kp_count) if matches[i] != train_kp_count
        ])
        train_matching_keypoints = np.array([
            train_keypoints[matches[i]] for i in range(test_kp_count) if matches[i] != train_kp_count
        ])
    return test_matching_keypoints, train_matching_keypoints

def compute_num_inliers(test_keypoints, test_descriptors, train_keypoints, train_descriptors,
                       use_ratio_test=False):
    """计算 RANSAC 内点数"""
    test_match_kp, train_match_kp = compute_putative_matching_keypoints(
        test_keypoints, test_descriptors, train_keypoints, train_descriptors, use_ratio_test)
    if test_match_kp.shape[0] <= 4:  # pydegensac.findHomography 所需的最小关键点数
        return 0
    try:
        _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,
                                            MAX_REPROJECTION_ERROR, HOMOGRAPHY_CONFIDENCE, MAX_RANSAC_ITERATIONS)
    except np.linalg.LinAlgError:
        return 0
    return int(mask.astype(np.float32).sum()) if mask is not None else 0

def rerankGV(cfg, local_feature_path, ranks_before_gv):
    """基于局部特征进行几何验证重排序"""
    print('>> Reranking with geometric verification...')
    ranks_after_gv = ranks_before_gv.copy()
    train_ids = [os.path.basename(x) for x in cfg['imlist']]
    test_ids = [os.path.basename(x) for x in cfg['qimlist']]
    
    with open(local_feature_path, 'rb') as fin:
        local_features = pickle.load(fin)
    
    for i in range(len(test_ids)):
        test_img = test_ids[i]
        if test_img not in local_features:
            print(f"警告: 查询图像 {test_img} 无特征，跳过")
            continue
        if i % 10 == 0:
            print(f">> Rerank {i}: {test_img}")
        test_img_path = find_image_with_any_extension(os.path.join('newtest', cfg['qimlist'][i]), IMAGE_PATH)
        if test_img_path is None:
            print(f"警告: 无法找到查询图像 {cfg['qimlist'][i]}，尝试路径: {os.path.join(IMAGE_PATH, cfg['qimlist'][i])}")
            continue
        test_array = skio.imread(test_img_path)
        locations = local_features[test_img]['locations']
        descriptors = local_features[test_img]['descriptors']
        
        inliers_numrerank = np.zeros(NUM_RERANK)
        for j in range(NUM_RERANK):
            if ranks_before_gv[j, i] in cfg['gnd'][i]['junk']:
                continue
            index_img = train_ids[ranks_before_gv[j, i]]
            if index_img not in local_features:
                print(f"警告: 数据库图像 {index_img} 无特征，跳过")
                continue
            index_img_path = find_image_with_any_extension(os.path.join('newtest', cfg['imlist'][ranks_before_gv[j, i]]), IMAGE_PATH)
            if index_img_path is None:
                print(f"警告: 无法找到数据库图像 {cfg['imlist'][ranks_before_gv[j, i]]}，尝试路径: {os.path.join(IMAGE_PATH, cfg['imlist'][ranks_before_gv[j, i]])}")
                continue
            index_array = skio.imread(index_img_path)
            tlocations = local_features[index_img]['locations']
            tdescriptors = local_features[index_img]['descriptors']
            try:
                num_inliers = compute_num_inliers(locations, descriptors, tlocations, tdescriptors,
                                                 use_ratio_test=USE_RATIO_TEST)
                inliers_numrerank[j] = num_inliers
            except Exception as e:
                print(f"警告: 处理 {test_img} vs {index_img} 时失败 - {str(e)}")
                continue
        ranks_after_gv[:NUM_RERANK, i] = ranks_before_gv[np.argsort(-inliers_numrerank), i]
    
    return ranks_before_gv, ranks_after_gv

def reportMAP(test_dataset, cfg, ranks):
    """计算并报告 mAP 和 mP@k"""
    gnd = cfg['gnd']
    ks = [1, 5, 10]
    
    # Easy
    gnd_t = [{'ok': np.concatenate([g['easy']]), 'junk': np.concatenate([g['junk'], g['hard']])} for g in gnd]
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)
    
    # Medium
    gnd_t = [{'ok': np.concatenate([g['easy'], g['hard']]), 'junk': np.concatenate([g['junk']])} for g in gnd]
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)
    
    # Hard
    gnd_t = [{'ok': np.concatenate([g['hard']]), 'junk': np.concatenate([g['junk'], g['easy']])} for g in gnd]
    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)
    
    print(f">> {test_dataset}: mAP E: {np.around(mapE*100, decimals=2)}, "
          f"M: {np.around(mapM*100, decimals=2)}, H: {np.around(mapH*100, decimals=2)}")
    print(f">> {test_dataset}: mP@k{ks} E: {np.around(mprE*100, decimals=2)}, "
          f"M: {np.around(mprM*100, decimals=2)}, H: {np.around(mprH*100, decimals=2)}")
    
    return mapE, mapM, mapH, mprE, mprM, mprH

def main():
    print(f">> {test_dataset}: Evaluating test dataset...")
    # cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets', 'data'))
    # 修改这一行 - 传递正确的数据根目录
    cfg = configdataset(test_dataset, data_root)
    
    # 加载局部特征以确定有效查询图像
    with open(LOCAL_FEATURE_PATH, 'rb') as fin:
        local_features = pickle.load(fin)
    valid_qimlist = [q for q in cfg['qimlist'] if os.path.basename(q) in local_features]
    valid_indices = [i for i, q in enumerate(cfg['qimlist']) if os.path.basename(q) in local_features]
    
    # 更新 cfg
    cfg['qimlist'] = valid_qimlist
    cfg['gnd'] = [cfg['gnd'][i] for i in valid_indices]
    cfg['nq'] = len(cfg['qimlist'])
    print(f"有效查询图像数: {cfg['nq']}")
    
    # 全局搜索
    ranks = global_search(GLOBAL_FEATURE_PATH)
    print(">> Evaluating global search...")
    mapE, mapM, mapH, mprE, mprM, mprH = reportMAP(test_dataset, cfg, ranks)
    
    # 几何验证重排序
    ranks_before_gv, ranks_after_gv = rerankGV(cfg, LOCAL_FEATURE_PATH, ranks)
    print(">> Evaluating after geometric verification...")
    mapE_gv, mapM_gv, mapH_gv, mprE_gv, mprM_gv, mprH_gv = reportMAP(test_dataset, cfg, ranks_after_gv)
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "100_eval_results.txt"), "w") as f:
        f.write(f"Global Search mAP E: {np.around(mapE*100, decimals=2)}, M: {np.around(mapM*100, decimals=2)}, H: {np.around(mapH*100, decimals=2)}\n")
        f.write(f"Global Search mP@k[1, 5, 10] E: {np.around(mprE*100, decimals=2)}, M: {np.around(mprM*100, decimals=2)}, H: {np.around(mprH*100, decimals=2)}\n")
        f.write(f"After GV mAP E: {np.around(mapE_gv*100, decimals=2)}, M: {np.around(mapM_gv*100, decimals=2)}, H: {np.around(mapH_gv*100, decimals=2)}\n")
        f.write(f"After GV mP@k[1, 5, 10] E: {np.around(mprE_gv*100, decimals=2)}, M: {np.around(mprM_gv*100, decimals=2)}, H: {np.around(mprH_gv*100, decimals=2)}\n")
    
    print("Done!")

if __name__ == '__main__':
    main()
