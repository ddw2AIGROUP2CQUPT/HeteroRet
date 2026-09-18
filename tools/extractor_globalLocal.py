import sys
import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from scipy.io import savemat

# 添加项目根目录到路径
data_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(data_root)

# 导入必要的模块
import core.config as config
from core.config import cfg
import delg_utils
from util import walkfile

# 设置路径
IMAGE_DIR = '/home/ubuntu/workplace01/hxl/delg-pytorch/revisitop/data/datasets/roxford5k/jpg'

GND_FILE = '/home/ubuntu/workplace01/hxl/delg-pytorch/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl'

OUTPUT_DIR = os.path.join(data_root, './features')
# MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/node01_gpu8_20250716/checkpoints/model_epoch_0100.pyth'
# MODEL_WEIGHTS ='/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/node1_gpu8_res101_2025_7_20/checkpoints/model_epoch_0100.pyth'
# MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/pretrained/r50_delg_s512.pyth'
#MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/8gpu_zzdata_20250724/checkpoints/model_epoch_0100.pyth'
MODEL_WEIGHTS = '/home/ubuntu/workplace01/hxl/delg-pytorch/DELG/output/8gpu_zzdata_20250729/checkpoints/model_epoch_0050.pyth'
# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 全局特征文件路径
GLOBAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'zzdata_chongfan_global.mat')
# 局部特征文件路径
LOCAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'zzdata_chongfan_localfea.pickle')

# 全局特征文件路径
# GLOBAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'roxford5k512gem_delg_res101_3global.mat')
# # 局部特征文件路径
# LOCAL_FEATURE_PATH = os.path.join(OUTPUT_DIR, 'roxford5k__s512_res101_localfea.pickle')

# 参数设置
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0]
IOU_THRES = 0.98
ATTN_THRES = 260.0
TOP_K = 1000
RF = 291.0
STRIDE = 16.0
PADDING = 145.0

# 设置模型
def setup_model():
    """Sets up the model for extraction."""
    model = delg_utils.DelgExtraction()
    # print(model)
    load_checkpoint(MODEL_WEIGHTS, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

# 提取特征
def extract(im_array, model):
    """Performs a forward pass on the model."""
    input_data = torch.from_numpy(im_array)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    
    # Capture the global feature (first return value)
    global_feature, delg_features, delg_scores = model(input_data, targets=None)
    
    return global_feature, delg_features, delg_scores

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

# 加载检查点
def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = f"Checkpoint '{checkpoint_file}' not found"
    assert os.path.exists(checkpoint_file), err_str
    
    print(f"Loading checkpoint from: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    
    try:
        state_dict = checkpoint["model_state"]
    except (KeyError, TypeError):
        state_dict = checkpoint

    # Handle DDP-wrapped models by removing 'module.' prefix if it exists
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Account for the DDP wrapper in the multi-gpu setting
    ms = model
    model_dict = ms.state_dict()

    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    if len(pretrained_dict) == len(state_dict):
        print('All params loaded')
    else:
        print('construct model total {} keys and pretrin model total {} keys.'.format(len(model_dict), len(state_dict)))
        print('{} pretrain keys load successfully.'.format(len(pretrained_dict)))
        not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
        print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
    model_dict.update(pretrained_dict)
    ms.load_state_dict(model_dict)
    
    print("Checkpoint loaded successfully.")
    return checkpoint

# 提取DELG特征
def delg_extract(img, model):
    """Processes an image at multiple scales to extract features."""
    output_boxes = []
    output_features = []
    output_scores = []
    output_scales = []
    output_original_scale_attn = None
    
    # Variable to store the global feature from the original scale (1.0)
    output_global_feature = None

    for scale_factor in SCALE_LIST:
        im = preprocess(img.copy(), scale_factor)
        im_array = np.asarray([im], dtype=np.float32)
        
        # Receive the global feature
        global_feature, delg_features, delg_scores = extract(im_array, model)
        
        # Store the global feature if it's from the original scale
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
    
    if not output_boxes:  # 如果没有找到特征点
        # Handle case where no features are detected
        return to_numpy(output_global_feature), {
            'locations': np.zeros((0, 2), dtype=np.float32),
            'descriptors': np.zeros((0, 128), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32)
        }
    
    if output_original_scale_attn is None:
        output_original_scale_attn = torch.zeros(1).uniform_()
    
    # concat tensors processed from different scales.
    output_boxes = delg_utils.concat_tensors_in_list(output_boxes, dim=0)
    output_features = delg_utils.concat_tensors_in_list(output_features, dim=0)
    output_scales = delg_utils.concat_tensors_in_list(output_scales, dim=0)
    output_scores = delg_utils.concat_tensors_in_list(output_scores, dim=0)
    
    # perform Non Max Suppression(NMS) to select top-k bboxes according to the attn_score.
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
    
    return to_numpy(output_global_feature), local_data

def find_image_with_any_extension(img_name, directory):
    # 如果本身已经是一个完整路径（含扩展名）
    full_path = os.path.join(directory, img_name)
    if os.path.exists(full_path):
        return full_path

    # 如果没有扩展名则尝试常见扩展
    name_wo_ext = os.path.splitext(img_name)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        path = os.path.join(directory, name_wo_ext + ext)
        if os.path.exists(path):
            return path
    return None

# 主函数
def main():
    # 加载配置
    config.load_cfg_fom_args("Extract feature.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    
    # 设置模型
    model = setup_model()
    
    # 读取数据集配置
    print(f"Loading dataset configuration from: {GND_FILE}")
    with open(GND_FILE, 'rb') as f:
        cfg_dataset = pickle.load(f)
    
    # 打印配置文件的内容，以便了解其结构
    print("Dataset configuration keys:", cfg_dataset.keys())
    
    # 检查qimlist的长度
    qimlist = cfg_dataset.get('qimlist', [])
    imlist = cfg_dataset.get('imlist', [])
    
    print(f"Length of qimlist: {len(qimlist)}")
    print(f"Length of imlist: {len(imlist)}")
    
    # 打印前几个查询图像名称，以便检查
    print("First 5 query image names:", qimlist[:5])
    
    # 获取查询和数据库图像路径
    query_list = []
    db_list = []
    
    # 默认扩展名为.jpg
    query_ext = '.jpg'
    db_ext = '.jpg'
    
    # 获取查询和数据库图像路径
    #query_list = [os.path.join(IMAGE_DIR, q + query_ext) for q in qimlist]
    #db_list = [os.path.join(IMAGE_DIR, db + db_ext) for db in imlist]
    
    query_list = [find_image_with_any_extension(q, IMAGE_DIR) for q in qimlist]
    db_list = [find_image_with_any_extension(db, IMAGE_DIR) for db in imlist]

    # 检查文件是否存在
    query_exists = [os.path.exists(path) for path in query_list]
    db_exists = [os.path.exists(path) for path in db_list]
    
    print(f"Query images exist: {sum(query_exists)}/{len(query_list)}")
    print(f"Database images exist: {sum(db_exists)}/{len(db_list)}")
    
    # 如果有文件不存在，打印前几个不存在的文件
    if sum(query_exists) < len(query_list):
        missing_queries = [path for path, exists in zip(query_list, query_exists) if not exists]
        print(f"First 5 missing query images: {missing_queries[:5]}")
    
    if sum(db_exists) < len(db_list):
        missing_dbs = [path for path, exists in zip(db_list, db_exists) if not exists]
        print(f"First 5 missing database images: {missing_dbs[:5]}")
    
    # 只使用存在的文件
    query_list = [path for path, exists in zip(query_list, query_exists) if exists]
    db_list = [path for path, exists in zip(db_list, db_exists) if exists]
    
    print(f"Found {len(query_list)} query images and {len(db_list)} database images")
    
    # 提取特征
    Q = []  # 查询图像全局特征
    X = []  # 数据库图像全局特征
    local_features = {}  # 所有图像的局部特征
    
    # 处理查询图像
    for i, path in enumerate(tqdm(query_list, desc="Processing query images")):
        # 读取图像
        img = cv2.imread(path)
        if img is None:
            print(f"Error loading image: {path}")
            continue
        
        img = img.astype(np.float32, copy=False)
        
        # 提取特征
        global_feature, local_data = delg_extract(img, model)
        
        # 保存特征
        Q.append(global_feature)
        local_features[os.path.basename(path)] = local_data
    
    # 处理数据库图像
    for i, path in enumerate(tqdm(db_list, desc="Processing database images")):
        # 读取图像
        img = cv2.imread(path)
        if img is None:
            print(f"Error loading image: {path}")
            continue
        
        img = img.astype(np.float32, copy=False)
        
        # 提取特征
        global_feature, local_data = delg_extract(img, model)
        
        # 保存特征
        X.append(global_feature)
        local_features[os.path.basename(path)] = local_data
    
    # 将特征列表转换为numpy数组
    Q = np.vstack(Q)
    X = np.vstack(X)
    
    # 保存全局特征
    savemat(GLOBAL_FEATURE_PATH, {'Q': Q, 'X': X})
    print(f"Global features saved to {GLOBAL_FEATURE_PATH}")
    
    # 保存局部特征
    with open(LOCAL_FEATURE_PATH, 'wb') as f:
        pickle.dump(local_features, f, protocol=2)
    print(f"Local features saved to {LOCAL_FEATURE_PATH}")

if __name__ == '__main__':
    import torch  # 确保torch被导入
    main()