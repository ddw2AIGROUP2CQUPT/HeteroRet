# 版本2==============
import os
import pickle
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from dataset import configdataset, find_image_with_any_extension
from compute import compute_map

# 直接配置所有路径
PKL_FILE = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2/newtest2.pkl'
IMAGE_PATH = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest2'
GLOBAL_FEATURE_PATH = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG_VIT/output/features/8gpu_newzzdata300w_vit_20251121_CNN_V1/newzz300w_vit_test_global_epoch10_spV1_gemp.mat'
OUTPUT_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG_VIT/output/eval/8gpu_newzzdata300w_vit_20251121_CNN_V1'
test_dataset = 'newzzdata_300w'

# 新增参数
# NUM_RERANK_M = 400  # Top-M 重排序
# NUM_RERANK_K = 9    # K近邻
# BETA = 0.15         # 权重参数

NUM_RERANK_M = 400  # Top-M 重排序
NUM_RERANK_K = 9    # K近邻
BETA = 0.15         # 权重参数

class MDescAug(nn.Module):
    """ Top-M Descriptor Augmentation"""
    def __init__(self, M=400, K=9, beta=0.15):
        super(MDescAug, self).__init__()
        self.M = M
        self.K = K + 1  # including oneself
        self.beta = beta
    
    def forward(self, X, Q, ranks):
        # X: [num_db, dim], Q: [num_query, dim], ranks: [num_db, num_query]
        num_query = Q.shape[0]
        
        ranks_trans_1000 = ranks[:self.M, :].T  # [num_query, M]
        
        # 修复tensor构造警告
        X_tensor1 = X[ranks_trans_1000].clone().detach()  # [num_query, M, dim]
        
        res_ie = torch.einsum('abc,adc->abd', X_tensor1, X_tensor1)  # [num_query, M, M]
        res_ie_ranks = torch.argsort(-res_ie.clone(), dim=-1)[:, :, :self.K]  # [num_query, M, K]
        res_ie_ranks_value = -torch.sort(-res_ie.clone(), dim=-1)[0][:, :, :self.K]  # [num_query, M, K]
        
        # 调整权重
        res_ie_ranks_value_adj = res_ie_ranks_value.clone()
        res_ie_ranks_value_adj[:, :, 1:] *= self.beta
        res_ie_ranks_value_adj[:, :, 0] = 1.0
        
        x_dba = X[ranks_trans_1000]  # [num_query, M, dim]
        
        # 使用torch.gather替代for循环
        x_dba_expanded = x_dba.unsqueeze(2).expand(-1, -1, self.K, -1)  # [num_query, M, K, dim]
        res_ie_ranks_expanded = res_ie_ranks.unsqueeze(-1).expand(-1, -1, -1, x_dba.shape[-1])  # [num_query, M, K, dim]
        x_dba_gathered = torch.gather(x_dba_expanded, 1, res_ie_ranks_expanded)  # [num_query, M, K, dim]
        
        # 加权平均
        res_ie_ranks_value_expanded = res_ie_ranks_value_adj.unsqueeze(-1)  # [num_query, M, K, 1]
        x_dba = torch.sum(x_dba_gathered * res_ie_ranks_value_expanded, dim=2) / torch.sum(res_ie_ranks_value_expanded, dim=2)  # [num_query, M, dim]
        
        # 计算新的相似度
        res_top1000_dba = torch.einsum('ac,abc->ab', Q, x_dba)  # [num_query, M]
        
        ranks_trans_1000_pre = torch.argsort(-res_top1000_dba, dim=-1)  # [num_query, M]
        
        # 重新排序
        rerank_dba_final = []
        for i in range(num_query):
            temp_concat = ranks_trans_1000[i][ranks_trans_1000_pre[i]]
            rerank_dba_final.append(temp_concat)
        
        return rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba

class RerankwMDA(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, M=400, K=9, beta=0.15):
        super(RerankwMDA, self).__init__()
        self.M = M 
        self.K = K + 1  # including oneself
        self.beta = beta
    
    def forward(self, ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba):
        num_query = len(rerank_dba_final)
        num_db = ranks.shape[0]
        
        ranks_trans_1000 = torch.stack(rerank_dba_final, dim=0)  # [num_query, M]
        ranks_value_trans_1000 = -torch.sort(-res_top1000_dba, dim=-1)[0]  # [num_query, M]
        
        ranks_trans = ranks_trans_1000_pre[:, :self.K]  # [num_query, K]
        ranks_value_trans = ranks_value_trans_1000[:, :self.K].clone()  # [num_query, K]
        ranks_value_trans *= self.beta
        
        # 使用torch.gather获取特征
        ranks_trans_expanded = ranks_trans.unsqueeze(-1).expand(-1, -1, x_dba.shape[-1])  # [num_query, K, dim]
        X1 = torch.gather(x_dba, 1, ranks_trans_expanded)  # [num_query, K, dim]
        X1 = torch.max(X1, dim=1, keepdim=True)[0]  # [num_query, 1, dim]

        # # 平均
        # weights = ranks_value_trans.unsqueeze(-1)  # [num_query, K, 1]
        # weights = torch.softmax(weights / 0.1, dim=1)  # 温度缩放的softmax
        # X1 = torch.sum(X1 * weights, dim=1, keepdim=True)  # [num_query, 1, dim]
        
        # 计算重排序分数
        # res_rerank = torch.sum(torch.einsum('abc,abc->ab', X1.expand(-1, self.M, -1), x_dba), dim=-1)  # [num_query, M]

        res_rerank = torch.einsum('abc,abc->ab', X1.expand(-1, self.M, -1), x_dba)  # [num_query, M]
        
        res_rerank = (ranks_value_trans_1000 + res_rerank) / 2.0  # [num_query, M]

        # # 新增
        # cosine_sim = torch.einsum('abc,abc->ab', X1.expand(-1, self.M, -1), x_dba)
        # # 使用不同的融合权重
        # alpha = 0.7  # 给原始相似度更大权重
        # res_rerank = alpha * ranks_value_trans_1000 + (1 - alpha) * cosine_sim

        res_rerank_ranks = torch.argsort(-res_rerank, dim=-1)  # [num_query, M]
        
        # 构建最终排序
        rerank_qe_final = []
        ranks_remaining = ranks[self.M:, :].T  # [num_query, num_db-M]
        
        for i in range(num_query):
            reranked_top_m = ranks_trans_1000[i][res_rerank_ranks[i]]
            temp_concat = torch.cat([reranked_top_m, ranks_remaining[i]], dim=0)
            rerank_qe_final.append(temp_concat)
        
        final_ranks = torch.stack(rerank_qe_final, dim=0).T  # [num_db, num_query]
        
        return final_ranks

    # # 在RerankwMDA的forward方法中，完全改变策略：
    # def forward(self, ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba, Q_tensor):
    #     num_query = len(rerank_dba_final)
    #     ranks_trans_1000 = torch.stack(rerank_dba_final, dim=0)
    #     ranks_value_trans_1000 = -torch.sort(-res_top1000_dba, dim=-1)[0]
        
    #     # 不再做特征聚合，直接用原查询特征但加强其权重
    #     enhanced_Q = Q_tensor.unsqueeze(1)  # [num_query, 1, dim]
        
    #     # 计算增强的相似度 - 结合原始相似度和排序置信度
    #     confidence_weights = torch.exp(-torch.arange(self.M, dtype=torch.float32, device=x_dba.device) * 0.01)
    #     confidence_weights = confidence_weights.unsqueeze(0).expand(num_query, -1)  # [num_query, M]
        
    #     # 重新计算相似度，但加入位置偏置
    #     res_rerank = torch.einsum('qd,qmd->qm', Q_tensor, x_dba)  # [num_query, M]
    #     res_rerank = res_rerank * confidence_weights + ranks_value_trans_1000 * 0.1
        
    #     res_rerank_ranks = torch.argsort(-res_rerank, dim=-1)
        
    #     # 构建最终排序
    #     rerank_qe_final = []
    #     ranks_remaining = ranks[self.M:, :].T
    #     for i in range(num_query):
    #         reranked_top_m = ranks_trans_1000[i][res_rerank_ranks[i]]
    #         temp_concat = torch.cat([reranked_top_m, ranks_remaining[i]], dim=0)
    #         rerank_qe_final.append(temp_concat)
        
    #     final_ranks = torch.stack(rerank_qe_final, dim=0).T
    #     return final_ranks

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
    Q = features['Q']  # 查询特征
    X = features['X']  # 数据库特征
    
    print(f"原始查询特征形状: {Q.shape}")
    print(f"原始数据库特征形状: {X.shape}")
    
    # 让我们检查实际的维度情况
    # 通常在MATLAB中，特征存储为 [dim, num_samples]
    # 但有时也可能是 [num_samples, dim]
    
    # 判断特征的存储格式
    if Q.shape[0] == X.shape[0]:  # 如果第一个维度相同，说明都是特征维度
        # 格式是 [dim, num_samples]
        print("检测到特征格式为 [dim, num_samples]")
        Q_processed = Q.T  # 转为 [num_query, dim]
        X_processed = X.T  # 转为 [num_db, dim]
        
        # 计算相似度矩阵
        sim = np.dot(X_processed, Q_processed.T)  # [num_db, num_query]
        
    else:
        # 格式可能是混合的，需要具体分析
        print("检测到特征格式需要特殊处理")
        # 一般来说，特征维度应该相同
        if Q.shape[1] == X.shape[1]:  # 第二个维度相同，说明是特征维度
            # Q: [num_query, dim], X: [num_db, dim]
            Q_processed = Q
            X_processed = X
        elif Q.shape[0] == X.shape[1]:  # Q的第一维 = X的第二维 (特征维度)
            # Q: [dim, num_query], X: [num_db, dim]
            Q_processed = Q.T  # 转为 [num_query, dim]
            X_processed = X     # 保持 [num_db, dim]
        elif Q.shape[1] == X.shape[0]:  # Q的第二维 = X的第一维 (特征维度)
            # Q: [num_query, dim], X: [dim, num_db]
            Q_processed = Q     # 保持 [num_query, dim]
            X_processed = X.T   # 转为 [num_db, dim]
        else:
            raise ValueError(f"无法匹配特征维度: Q shape {Q.shape}, X shape {X.shape}")
        
        # 计算相似度矩阵
        sim = np.dot(X_processed, Q_processed.T)  # [num_db, num_query]
    
    ranks = np.argsort(-sim, axis=0)  # [num_db, num_query]
    
    print(f"处理后查询特征形状: {Q_processed.shape}")
    print(f"处理后数据库特征形状: {X_processed.shape}")
    print(f"相似度矩阵形状: {sim.shape}")
    print(f"排序结果形状: {ranks.shape}")
    
    return ranks, Q_processed, X_processed

def global_rerank_with_mda(Q, X, ranks):
    """使用MDA方法进行全局重排序"""
    print('>> 使用MDA方法进行全局重排序...')
    
    # 转换为tensor并移到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    Q_tensor = torch.tensor(Q, dtype=torch.float32).to(device)  # [num_query, dim]
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # [num_db, dim]
    ranks_tensor = torch.tensor(ranks, dtype=torch.long).to(device)  # [num_db, num_query]
    
    print(f"Q_tensor形状: {Q_tensor.shape}")
    print(f"X_tensor形状: {X_tensor.shape}")
    print(f"ranks_tensor形状: {ranks_tensor.shape}")
    
    # 初始化MDA模块
    mdesc_aug = MDescAug(M=NUM_RERANK_M, K=NUM_RERANK_K, beta=BETA).to(device)
    rerank_mda = RerankwMDA(M=NUM_RERANK_M, K=NUM_RERANK_K, beta=BETA).to(device)
    
    with torch.no_grad():
        # 第一阶段：Top-M描述符增强 (MDA重排序)
        rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = mdesc_aug(
            X_tensor, Q_tensor, ranks_tensor)
        
        # 构建MDA重排序结果 - 只重排前M个结果
        ranks_mda = ranks_tensor.clone()
        rerank_dba_stacked = torch.stack(rerank_dba_final, dim=0)  # [num_query, M]
        ranks_mda[:NUM_RERANK_M, :] = rerank_dba_stacked.T  # 转置为 [M, num_query]
        
        # 第二阶段：使用最大描述符聚合的重排序 (RMDA重排序)
        ranks_rmda = rerank_mda(ranks_tensor, rerank_dba_final, res_top1000_dba, 
                               ranks_trans_1000_pre, x_dba)

        # 阶段2
        # ranks_rmda = rerank_mda(ranks_tensor, rerank_dba_final, res_top1000_dba,
        #                ranks_trans_1000_pre, x_dba, Q_tensor)
        
        # 转换回numpy
        ranks_mda_np = ranks_mda.cpu().numpy()
        ranks_rmda_np = ranks_rmda.cpu().numpy()
    
    return ranks_mda_np, ranks_rmda_np

def reportMAP(test_dataset, cfg, ranks, method_name=""):
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
    
    print(f">> {test_dataset} {method_name}: mAP E: {np.around(mapE*100, decimals=2)}, "
          f"M: {np.around(mapM*100, decimals=2)}, H: {np.around(mapH*100, decimals=2)}")
    print(f">> {test_dataset} {method_name}: mP@k{ks} E: {np.around(mprE*100, decimals=2)}, "
          f"M: {np.around(mprM*100, decimals=2)}, H: {np.around(mprH*100, decimals=2)}")
    
    return mapE, mapM, mapH, mprE, mprM, mprH

def main():
    print(f">> {test_dataset}: 开始评估测试数据集...")
    
    # 直接加载配置
    cfg = load_config_directly()
    
    # 检查全局特征文件是否存在
    if not os.path.exists(GLOBAL_FEATURE_PATH):
        raise FileNotFoundError(f"全局特征文件不存在: {GLOBAL_FEATURE_PATH}")
    
    print(f"有效查询图像数: {cfg['nq']}")
    
    # 1. 原始全局搜索
    ranks, Q, X = global_search(GLOBAL_FEATURE_PATH)
    print("\n>> 评估原始全局搜索...")
    mapE_orig, mapM_orig, mapH_orig, mprE_orig, mprM_orig, mprH_orig = reportMAP(
        test_dataset, cfg, ranks, "原始全局搜索")
    
    # 2. 使用MDA方法的全局重排序
    print(f"\n>> 使用MDA方法进行重排序 (M={NUM_RERANK_M}, K={NUM_RERANK_K}, beta={BETA})...")
    ranks_mda, ranks_rmda = global_rerank_with_mda(Q, X, ranks)

    print(">> 评估MDA重排序后的结果...")
    mapE_mda, mapM_mda, mapH_mda, mprE_mda, mprM_mda, mprH_mda = reportMAP(
        test_dataset, cfg, ranks_mda, "MDA重排序")
    print(">> 评估RMDA重排序后的结果...")
    mapE_rmda, mapM_rmda, mapH_rmda, mprE_rmda, mprM_rmda, mprH_rmda = reportMAP(
        test_dataset, cfg, ranks_rmda, "RMDA重排序")
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_file = os.path.join(OUTPUT_DIR, "newzzdata300w_vit_eval_results_epoch10_V2.txt")
    with open(results_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"实例级图像检索评估结果 - {test_dataset}\n")
        f.write("=" * 80 + "\n")
        f.write(f"配置参数: M={NUM_RERANK_M}, K={NUM_RERANK_K}, beta={BETA}\n")
        f.write(f"查询图像数: {cfg['nq']}, 数据库图像数: {cfg['n']}\n\n")
        
        f.write("原始全局搜索结果:\n")
        f.write(f"mAP E: {np.around(mapE_orig*100, decimals=2)}, M: {np.around(mapM_orig*100, decimals=2)}, H: {np.around(mapH_orig*100, decimals=2)}\n")
        f.write(f"mP@k[1,5,10] E: {np.around(mprE_orig*100, decimals=2)}, M: {np.around(mprM_orig*100, decimals=2)}, H: {np.around(mprH_orig*100, decimals=2)}\n\n")
        
        f.write("MDA全局排序结果:\n")
        f.write(f"mAP E: {np.around(mapE_mda*100, decimals=2)}, M: {np.around(mapM_mda*100, decimals=2)}, H: {np.around(mapH_mda*100, decimals=2)}\n")
        f.write(f"mP@k[1,5,10] E: {np.around(mprE_mda*100, decimals=2)}, M: {np.around(mprM_mda*100, decimals=2)}, H: {np.around(mprH_mda*100, decimals=2)}\n\n")
        
        f.write("RMDA重排序结果:\n")
        f.write(f"mAP E: {np.around(mapE_rmda*100, decimals=2)}, M: {np.around(mapM_rmda*100, decimals=2)}, H: {np.around(mapH_rmda*100, decimals=2)}\n")
        f.write(f"mP@k[1,5,10] E: {np.around(mprE_rmda*100, decimals=2)}, M: {np.around(mprM_rmda*100, decimals=2)}, H: {np.around(mprH_rmda*100, decimals=2)}\n\n")
        
        f.write("性能提升:\n")
        f.write(f"mAP提升 E: {np.around((mapE_mda-mapE_orig)*100, decimals=2)}, M: {np.around((mapM_mda-mapM_orig)*100, decimals=2)}, H: {np.around((mapH_mda-mapH_orig)*100, decimals=2)}\n")
        f.write(f"mP@1提升 E: {np.around((mprE_mda[0]-mprE_orig[0])*100, decimals=2)}, M: {np.around((mprM_mda[0]-mprM_orig[0])*100, decimals=2)}, H: {np.around((mprH_mda[0]-mprH_orig[0])*100, decimals=2)}\n")
    
        # 打印对比结果
        print("\n" + "="*80)
        print("性能对比总结:")
        print("="*80)
        print(f"原始方法   - mAP E: {np.around(mapE_orig*100, decimals=2)}, M: {np.around(mapM_orig*100, decimals=2)}, H: {np.around(mapH_orig*100, decimals=2)}")
        print(f"MDA全局排序 - mAP E: {np.around(mapE_mda*100, decimals=2)}, M: {np.around(mapM_mda*100, decimals=2)}, H: {np.around(mapH_mda*100, decimals=2)}")
        print(f"RMDA重排序 - mAP E: {np.around(mapE_rmda*100, decimals=2)}, M: {np.around(mapM_rmda*100, decimals=2)}, H: {np.around(mapH_rmda*100, decimals=2)}")

if __name__ == '__main__':
    main()