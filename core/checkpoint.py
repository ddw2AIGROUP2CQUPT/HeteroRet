#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions that handle saving and loading of checkpoints."""

import os
import copy

import core.distributed as dist
import torch
from core.config import cfg


# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"
# Checkpoints directory name
_DIR_NAME = "checkpoints"


def get_checkpoint_dir():
    """Retrieves the location for storing checkpoints."""
    return os.path.join(cfg.OUT_DIR, _DIR_NAME)


def get_checkpoint(epoch):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(), name)


def get_last_checkpoint():
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir()
    # Checkpoint file names are in lexicographic order
    checkpoints = [f for f in os.listdir(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint():
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir()
    if not os.path.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in os.listdir(checkpoint_dir))


def save_checkpoint(model, optimizer, epoch):
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    if not dist.is_master_proc():
        return
    # Ensure that the checkpoint dir exists
    os.makedirs(get_checkpoint_dir(), exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    checkpoint_file = get_checkpoint(epoch + 1)
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file

# #旧版本
# def load_checkpoint(checkpoint_file, model, optimizer=None):
#     """Loads the checkpoint from the given file."""
#     err_str = "Checkpoint '{}' not found"
#     assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
#     # Load the checkpoint on CPU to avoid GPU mem spike
#     checkpoint = torch.load(checkpoint_file, map_location="cpu")
#     try:
#         state_dict = checkpoint["model_state"]
#     except KeyError:
#         state_dict = checkpoint
#     # Account for the DDP wrapper in the multi-gpu setting
#     ms = model.module if cfg.NUM_GPUS > 1 else model
#     model_dict = ms.state_dict()
    
#     state_dict = {'globalmodel.'+k : v for k, v in state_dict.items()}
#     pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
#     if len(pretrained_dict) == len(state_dict):
#         print('All params loaded')
#     else:
#         print('construct model total {} keys and pretrin model total {} keys.'.format(len(model_dict), len(state_dict)))
#         print('{} pretrain keys load successfully.'.format(len(pretrained_dict)))
#         not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
#         print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
#     model_dict.update(pretrained_dict)
#     ms.load_state_dict(model_dict)
#     #ms.load_state_dict(checkpoint["model_state"])
#     # Load the optimizer state (commonly not done when fine-tuning)
#     if optimizer:
#         optimizer.load_state_dict(checkpoint["optimizer_state"])
#     #return checkpoint["epoch"]
#     return checkpoint


# # 新版本
# def load_checkpoint(checkpoint_file, model, optimizer=None):
#     """Loads the checkpoint from the given file."""
#     err_str = "Checkpoint '{}' not found"
#     assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
#     # Load the checkpoint on CPU to avoid GPU mem spike
#     checkpoint = torch.load(checkpoint_file, map_location="cpu")
    
#     try:
#         state_dict = checkpoint["model_state"]
#     except KeyError:
#         state_dict = checkpoint
    
#     # Account for the DDP wrapper in the multi-gpu setting
#     ms = model.module if cfg.NUM_GPUS > 1 else model
#     model_dict = ms.state_dict()
    
#     # 尝试不同的键名映射方式
#     pretrained_dict = {}
    
#     # 1. 直接匹配
#     direct_match = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
#     pretrained_dict.update(direct_match)
    
#     # 2. 尝试添加globalmodel前缀
#     if len(direct_match) < len(state_dict) * 0.5:  # 如果直接匹配的比例低于50%
#         prefix_match = {'globalmodel.'+k: v for k, v in state_dict.items() 
#                        if 'globalmodel.'+k in model_dict and model_dict['globalmodel.'+k].size() == v.size()}
#         pretrained_dict.update(prefix_match)
    
#     # 3. 尝试去除可能的前缀
#     if len(pretrained_dict) < len(state_dict) * 0.5:  # 如果匹配的比例仍然低于50%
#         for k, v in state_dict.items():
#             parts = k.split('.')
#             if len(parts) > 1:
#                 # 尝试移除第一级前缀
#                 new_key = '.'.join(parts[1:])
#                 if new_key in model_dict and model_dict[new_key].size() == v.size():
#                     pretrained_dict[new_key] = v
    
#     if len(pretrained_dict) == 0:
#         print('No params loaded - trying to match by parameter shape')
#         # 4. 如果上述方法都失败，尝试根据参数形状匹配
#         model_shapes = {k: v.shape for k, v in model_dict.items()}
#         state_shapes = {k: v.shape for k, v in state_dict.items()}
        
#         for model_key, model_shape in model_shapes.items():
#             for state_key, state_shape in state_shapes.items():
#                 if model_shape == state_shape:
#                     print(f"Shape match: {model_key} <- {state_key}")
#                     pretrained_dict[model_key] = state_dict[state_key]
#                     break
    
#     print(f'{len(pretrained_dict)} pretrain keys load successfully.')
#     if len(pretrained_dict) < len(state_dict):
#         print(f'construct model total {len(model_dict)} keys and pretrain model total {len(state_dict)} keys.')
#         not_loaded_keys = [k for k in state_dict.keys() if k not in [key.replace('globalmodel.', '') for key in pretrained_dict.keys()]]
#         if len(not_loaded_keys) > 10:
#             print(f'First 10 not loaded keys: {not_loaded_keys[:10]}')
#         else:
#             print(f'Not loaded keys: {not_loaded_keys}')
    
#     model_dict.update(pretrained_dict)
#     ms.load_state_dict(model_dict, strict=False)  # 使用strict=False允许部分加载
    
#     # Load the optimizer state (commonly not done when fine-tuning)
#     if optimizer and "optimizer_state" in checkpoint:
#         optimizer.load_state_dict(checkpoint["optimizer_state"])
    
#     return checkpoint

# 新版本-新增正确加载学习率
def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    
    # ... 你现有的模型权重加载代码保持不变 ...
    try:
        state_dict = checkpoint["model_state"]
    except KeyError:
        state_dict = checkpoint
    
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model.module if cfg.NUM_GPUS > 1 else model
    model_dict = ms.state_dict()
    
    # 尝试不同的键名映射方式
    pretrained_dict = {}
    
    # 1. 直接匹配
    direct_match = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    pretrained_dict.update(direct_match)
    
    # 2. 尝试添加globalmodel前缀
    if len(direct_match) < len(state_dict) * 0.5:  # 如果直接匹配的比例低于50%
        prefix_match = {'globalmodel.'+k: v for k, v in state_dict.items() 
                       if 'globalmodel.'+k in model_dict and model_dict['globalmodel.'+k].size() == v.size()}
        pretrained_dict.update(prefix_match)
    
    # 3. 尝试去除可能的前缀
    if len(pretrained_dict) < len(state_dict) * 0.5:  # 如果匹配的比例仍然低于50%
        for k, v in state_dict.items():
            parts = k.split('.')
            if len(parts) > 1:
                # 尝试移除第一级前缀
                new_key = '.'.join(parts[1:])
                if new_key in model_dict and model_dict[new_key].size() == v.size():
                    pretrained_dict[new_key] = v
    
    if len(pretrained_dict) == 0:
        print('No params loaded - trying to match by parameter shape')
        # 4. 如果上述方法都失败，尝试根据参数形状匹配
        model_shapes = {k: v.shape for k, v in model_dict.items()}
        state_shapes = {k: v.shape for k, v in state_dict.items()}
        
        for model_key, model_shape in model_shapes.items():
            for state_key, state_shape in state_shapes.items():
                if model_shape == state_shape:
                    print(f"Shape match: {model_key} <- {state_key}")
                    pretrained_dict[model_key] = state_dict[state_key]
                    break
    
    print(f'{len(pretrained_dict)} pretrain keys load successfully.')
    if len(pretrained_dict) < len(state_dict):
        print(f'construct model total {len(model_dict)} keys and pretrain model total {len(state_dict)} keys.')
        not_loaded_keys = [k for k in state_dict.keys() if k not in [key.replace('globalmodel.', '') for key in pretrained_dict.keys()]]
        if len(not_loaded_keys) > 10:
            print(f'First 10 not loaded keys: {not_loaded_keys[:10]}')
        else:
            print(f'Not loaded keys: {not_loaded_keys}')
    
    model_dict.update(pretrained_dict)
    ms.load_state_dict(model_dict, strict=False)  # 使用strict=False允许部分加载
    
    # Load the optimizer state
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"✅ Optimizer state loaded")
    
    # 🔥 新增：返回epoch和其他训练状态信息
    epoch = checkpoint.get("epoch", 0)
    print(f"✅ Checkpoint epoch: {epoch}")
    
    # 🔥 新增：检查并打印学习率信息
    if "optimizer_state" in checkpoint:
        current_lr = checkpoint["optimizer_state"]["param_groups"][0]["lr"]
        print(f"✅ Restored learning rate: {current_lr}")
    
    return checkpoint

