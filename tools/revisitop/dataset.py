
import os
import pickle
import glob

DATASETS = ['roxford5k', 'rparis6k', 'revisitop1m', 'newzzdata','newzzdata_300w']

def find_image_with_any_extension(img_name, directory):
    """查找具有支持扩展名的图像文件，包括子目录"""
    # 尝试原始名称（包含子目录路径）
    full_path = os.path.join(directory, img_name)
    if os.path.exists(full_path):
        return full_path
    # 尝试去掉扩展名后添加支持的扩展名
    name_wo_ext = os.path.splitext(img_name)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.pgm']:
        # 搜索目录及其子目录
        pattern = os.path.join(directory, '**', f"{os.path.basename(name_wo_ext)}{ext}")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    # 调试：列出尝试的路径
    print(f"调试: 尝试查找 {img_name} 在 {directory}，无匹配文件")
    return None

def configdataset(dataset, dir_main):
    dataset = dataset.lower()

    if dataset not in DATASETS:    
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    if dataset == 'roxford5k' or dataset == 'rparis6k':
        gnd_fname = '/home/ubuntu/san/hxl/models/research/delf/delf/python/data/gnd_{}.pkl'.format(dataset)
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['dir_images'] = '/home/ubuntu/san/hxl/models/research/delf/delf/python/data/oxford5k_images'
        cfg['ext'] = ''  # 动态检测扩展名
        cfg['qext'] = ''  # 动态检测扩展名
        # 保留原始路径
        cfg['imlist'] = [x for x in cfg['imlist']]
        cfg['qimlist'] = [x for x in cfg['qimlist']]

    elif dataset == 'revisitop1m':
        cfg = {}
        cfg['imlist_fname'] = os.path.join(dir_main, dataset, '{}.txt'.format(dataset))
        cfg['imlist'] = read_imlist(cfg['imlist_fname'])
        cfg['qimlist'] = []
        cfg['ext'] = ''
        cfg['qext'] = ''
        cfg['dir_images'] = os.path.join(dir_main, dataset, 'jpg')

    elif dataset == 'newzzdata':
        gnd_fname = os.path.join(dir_main, 'newzzdata', 'augmented_dataset_1', 'gnd_newzz_50.pkl')
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['dir_images'] = os.path.join(dir_main, 'newzzdata', 'augmented_dataset_1', 'newtest')
        cfg['ext'] = ''  # 动态检测扩展名
        cfg['qext'] = ''  # 动态检测扩展名
        # 保留原始路径
        cfg['imlist'] = [x for x in cfg['imlist']]
        cfg['qimlist'] = [x for x in cfg['qimlist']]

    # elif dataset == 'newzzdata_300w':
    #     gnd_fname = os.path.join(dir_main, 'newzzdata_300w', 'augmented_basedata', 'newtest', 'new_zz_200wtest.pkl')
    #     with open(gnd_fname, 'rb') as f:
    #         cfg = pickle.load(f)
    #     cfg['gnd_fname'] = gnd_fname
    #     cfg['dir_images'] = os.path.join(dir_main, 'newzzdata_300w', 'augmented_basedata', 'newtest')
    #     cfg['ext'] = ''  # 动态检测扩展名
    #     cfg['qext'] = ''  # 动态检测扩展名
    #     # 保留原始路径
    #     cfg['imlist'] = [x for x in cfg['imlist']]
    #     cfg['qimlist'] = [x for x in cfg['qimlist']]

    elif dataset == 'newzzdata_300w':
        # 修改路径构建 - 使用绝对路径
        gnd_fname = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest/new_zz_200wtest.pkl'
        
        # 检查文件是否存在
        if not os.path.exists(gnd_fname):
            raise FileNotFoundError(f"PKL文件不存在: {gnd_fname}")
        
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['dir_images'] = '/home/ubuntu/public-Datasets/hxl/newzzdata_300w/augmented_basedata/newtest'
        cfg['ext'] = ''  # 动态检测扩展名
        cfg['qext'] = ''  # 动态检测扩展名
        # 保留原始路径
        cfg['imlist'] = [x for x in cfg['imlist']]
        cfg['qimlist'] = [x for x in cfg['qimlist']]


    cfg['dir_data'] = os.path.join(dir_main, dataset) if dataset != 'roxford5k' else '/home/ubuntu/san/hxl/models/research/delf/delf/python/data'
    
    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg

def config_imname(cfg, i):
    """为数据库图像生成文件路径，动态检测扩展名"""
    img_name = cfg['imlist'][i]
    path = find_image_with_any_extension(img_name, cfg['dir_images'])
    if path is None:
        raise FileNotFoundError(f"Image {img_name} not found in {cfg['dir_images']}")
    return path

def config_qimname(cfg, i):
    """为查询图像生成文件路径，动态检测扩展名"""
    img_name = cfg['qimlist'][i]
    path = find_image_with_any_extension(img_name, cfg['dir_images'])
    if path is None:
        raise FileNotFoundError(f"Query image {img_name} not found in {cfg['dir_images']}")
    return path

def read_imlist(imlist_fn):
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return [x for x in imlist]
