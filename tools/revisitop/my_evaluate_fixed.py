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

# 直接配置所有路径
# PKL_FILE = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_no_chart3_test200/no_chart3_test20.pkl'
# IMAGE_PATH = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_no_chart3_test200'
# GLOBAL_FEATURE_PATH = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/features/8gpu_newzzdata300w_vit_casdata9w_V2/newzz300w_vit_test_global_casdata200.mat'
# LOCAL_FEATURE_PATH = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/features/8gpu_newzzdata300w_vit_casdata9w_V2/newzz300w_vit_test_local_global_fea_casdata200.pickle'
# OUTPUT_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG_VIT/output/eval/8gpu_newzzdata300w_vit_casdata9w_V2'

PKL_FILE = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2/newtest2.pkl'
IMAGE_PATH = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2/'
GLOBAL_FEATURE_PATH = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/features/8gpu_newzzdata300w_vit_251121_CNN/newzz300w_vit_test_global_epoch08.mat'
LOCAL_FEATURE_PATH = '/home/ubuntu/san/hxl/delg-pytoch/DELG-VIT/features/8gpu_newzzdata300w_vit_251121_CNN/newzz300w_vit_test_local_global_fea_epoch08.pickle'
OUTPUT_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG_VIT/output/eval/8gpu_newzzdata300w_vit_251121_CNN'

test_dataset = 'newzzdata_300w'
NUM_RERANK = 50
MAX_REPROJECTION_ERROR = 20.0
MAX_RANSAC_ITERATIONS = 1000
HOMOGRAPHY_CONFIDENCE = 1.0
MATCHING_THRESHOLD = 1.0
MAX_DISTANCE = 0.99
USE_RATIO_TEST = False
DRAW_MATCHES = False

def load_config_directly():
    """直接加载配置，不依赖 configdataset 函数"""
    print(f"加载PKL文件: {PKL_FILE}")
    
    # 检查文件是否存在
    if not os.path.exists(PKL_FILE):
        raise FileNotFoundError(f"PKL文件不存在: {PKL_FILE}")
    
    with open(PKL_FILE, 'rb') as f:
        cfg = pickle.load(f)
    
    # 添加必要的配置
    cfg['gnd_fname'] = PKL_FILE
    cfg['dir_images'] = IMAGE_PATH
    cfg['ext'] = ''
    cfg['qext'] = ''
    cfg['nq'] = len(cfg['qimlist'])
    cfg['n'] = len(cfg['imlist'])
    
    print(f"查询图像数量: {cfg['nq']}")
    print(f"数据库图像数量: {cfg['n']}")
    
    return cfg

def global_search(global_feature_path):
    """基于全局特征进行排序"""
    print(f"加载全局特征: {global_feature_path}")
    features = loadmat(global_feature_path)
    Q = features['Q']
    X = features['X']
    print(f"查询特征形状: {Q.shape}")
    print(f"数据库特征形状: {X.shape}")
    
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
    if test_match_kp.shape[0] <= 4:
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
        
        # 修改路径构建
        test_img_path = os.path.join(IMAGE_PATH, cfg['qimlist'][i])
        if not os.path.exists(test_img_path):
            test_img_path = find_image_with_any_extension(cfg['qimlist'][i], IMAGE_PATH)
        
        if test_img_path is None:
            print(f"警告: 无法找到查询图像 {cfg['qimlist'][i]}")
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
                continue
                
            # 修改路径构建
            index_img_path = os.path.join(IMAGE_PATH, cfg['imlist'][ranks_before_gv[j, i]])
            if not os.path.exists(index_img_path):
                index_img_path = find_image_with_any_extension(cfg['imlist'][ranks_before_gv[j, i]], IMAGE_PATH)
            
            if index_img_path is None:
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
    
    # 直接加载配置，避免路径问题
    cfg = load_config_directly()
    
    # 检查特征文件是否存在
    if not os.path.exists(GLOBAL_FEATURE_PATH):
        raise FileNotFoundError(f"全局特征文件不存在: {GLOBAL_FEATURE_PATH}")
    if not os.path.exists(LOCAL_FEATURE_PATH):
        raise FileNotFoundError(f"局部特征文件不存在: {LOCAL_FEATURE_PATH}")
    
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
    with open(os.path.join(OUTPUT_DIR, "newzzdata300w_vit_eval_results_251121_CNN.txt"), "w") as f:
        f.write(f"Global Search mAP E: {np.around(mapE*100, decimals=2)}, M: {np.around(mapM*100, decimals=2)}, H: {np.around(mapH*100, decimals=2)}\n")
        f.write(f"Global Search mP@k[1, 5, 10] E: {np.around(mprE*100, decimals=2)}, M: {np.around(mprM*100, decimals=2)}, H: {np.around(mprH*100, decimals=2)}\n")
        f.write(f"After GV mAP E: {np.around(mapE_gv*100, decimals=2)}, M: {np.around(mapM_gv*100, decimals=2)}, H: {np.around(mapH_gv*100, decimals=2)}\n")
        f.write(f"After GV mP@k[1, 5, 10] E: {np.around(mprE_gv*100, decimals=2)}, M: {np.around(mprM_gv*100, decimals=2)}, H: {np.around(mprH_gv*100, decimals=2)}\n")
    
    print("Done!")

if __name__ == '__main__':
    main()