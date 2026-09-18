#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Written by yangmin09

"""Tools for training and testing a model."""

import os

import numpy as np
import core.benchmark as benchmark
import core.builders as builders
import core.checkpoint as checkpoint
import core.config as config
import core.distributed as dist
import core.logging as logging
import core.meters as meters
import core.net as net
import core.optimizer as optim
import datasets.loader as loader
import torch
import torch.nn.functional as F
from core.config import cfg

from model.delg_model import Delg

logger = logging.get_logger(__name__)


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = Delg()

    logger.info("Model:\n{}".format(model))
    # Log model complexity
    # logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
        )
        # Set complexity function to be module's complexity function
        # model.complexity = model.module.complexity
    return model

def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    # 动态调整提醒
    if cur_epoch in [10, 30, 60]:
        print(f"\n🔥 [EPOCH {cur_epoch}] 进入新的训练阶段，权重策略已调整")

        # 修改：设置epoch给localmodel
    if hasattr(model, 'module'):
        # 分布式训练情况
        model.module.localmodel.set_epoch(cur_epoch)
    else:
        # 单GPU训练情况
        model.localmodel.set_epoch(cur_epoch)

    """联合训练版本"""
    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    print(f"[LR DEBUG] Epoch {cur_epoch}: lr={lr:.8f}")
    print(f"[LR DEBUG] Config BASE_LR: {cfg.OPTIM.BASE_LR}")
    print(f"[LR DEBUG] Warmup epochs: {cfg.OPTIM.WARMUP_EPOCHS}")
    optim.set_lr(optimizer, lr)
    
    model.train()
    train_meter.iter_tic()
    
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        
        # 🔥 联合训练：一次前向传播，同时训练两个分支
        optimizer.zero_grad()
        
        # 前向传播
        _, global_logits, _, local_logits, _ = model(inputs, labels)
        
        # 计算两个分支的损失
        desc_loss = loss_fun(global_logits, labels)
        att_loss = loss_fun(local_logits, labels)

        # total_loss = att_loss * 1.0

        if cur_epoch <= 5:
            total_loss = desc_loss * 10.0 + att_loss * 2.0  # 全局分支3倍权重！
        elif cur_epoch <= 15:
            alpha = (cur_epoch - 5) / 10
            desc_weight = 10.0 - alpha * 4.0
            att_weight = 2.0 + alpha * 3.0
            total_loss = desc_loss * desc_weight + att_loss * att_weight
        elif cur_epoch <= 35:
            total_loss = desc_loss * 6.0 + att_loss * 6.0  # 保持全局优势
        else:
            total_loss = desc_loss * 5.0 + att_loss * 4.0  # 仍然偏向全局

        # 添加调试（前几个iteration）
        if cur_iter < 10:
            print(f"\n[JOINT TRAIN DEBUG - Iter {cur_iter}]")
            print(f"  Desc loss: {desc_loss.item():.4f}")
            print(f"  Att loss: {att_loss.item():.4f}")
            print(f"  Total loss: {total_loss.item():.4f}")
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度（前几个iteration）
        if cur_iter < 10:
            att_grad_norm = 0
            desc_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if 'localmodel' in name or 'att_cls' in name:
                        att_grad_norm += grad_norm ** 2
                    elif 'globalmodel' in name or 'desc_cls' in name:
                        desc_grad_norm += grad_norm ** 2
            
            att_grad_norm = att_grad_norm ** 0.5
            desc_grad_norm = desc_grad_norm ** 0.5
            print(f"  Local branch grad norm: {att_grad_norm:.6f}")
            print(f"  Global branch grad norm: {desc_grad_norm:.6f}")

        if cur_epoch <= 10:
            clip_norm = 1.0   # 从0.1增加到0.5，给空间注意力足够学习空间
        elif cur_epoch <= 30:
            clip_norm = 2.0   # 从0.3增加到1.0
        else:
            clip_norm = 3.0   # 从0.5增加到2.0

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        
        # 参数更新
        optimizer.step()
        
        # 保存用于评估的logits
        global_logits_eval = global_logits.detach()
        local_logits_eval = local_logits.detach()
        
        # 计算评估指标
        desc_top1_err, desc_top5_err = meters.topk_errors(global_logits_eval, labels, [1, 5])
        att_top1_err, att_top5_err = meters.topk_errors(local_logits_eval, labels, [1, 5])
        
        # 处理分布式训练
        desc_loss_value = desc_loss.item()
        att_loss_value = att_loss.item()
        desc_loss_tensor = torch.tensor(desc_loss_value).cuda()
        att_loss_tensor = torch.tensor(att_loss_value).cuda()
        
        desc_loss, desc_top1_err, desc_top5_err = dist.scaled_all_reduce(
            [desc_loss_tensor, desc_top1_err, desc_top5_err])
        att_loss, att_top1_err, att_top5_err = dist.scaled_all_reduce(
            [att_loss_tensor, att_top1_err, att_top5_err])
        
        desc_loss, desc_top1_err, desc_top5_err = desc_loss.item(), desc_top1_err.item(), desc_top5_err.item()
        att_loss, att_top1_err, att_top5_err = att_loss.item(), att_top1_err.item(), att_top5_err.item()
        
        # 更新统计信息
        train_meter.iter_toc()
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        train_meter.update_stats(desc_top1_err, desc_top5_err, att_top1_err, att_top5_err, 
                               desc_loss, att_loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        _, global_logits, _, local_logits, _ = model(inputs, labels)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(global_logits, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()


#=======新增特征图检测=========#
def diagnose_attention_quality(model, val_loader, num_samples=100):
    """诊断attention质量和局部特征有效性"""
    model.eval()
    attention_stats = []
    local_feature_stats = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            if i >= num_samples // inputs.size(0):
                break
                
            inputs = inputs.cuda()
            # 获取attention权重和局部特征
            global_desc, _, attention_weights, local_logits, local_features = model(inputs)
            
            # 统计attention分布
            att_weights = attention_weights.cpu().numpy()
            for batch_idx in range(att_weights.shape[0]):
                att_map = att_weights[batch_idx]
                attention_stats.append({
                    'max_attention': att_map.max(),
                    'min_attention': att_map.min(),
                    'attention_std': att_map.std(),
                    'attention_mean': att_map.mean(),
                    'high_attention_ratio': (att_map > att_map.mean() + att_map.std()).sum() / att_map.size
                })
            
            # 统计局部特征数量
            if local_features is not None:
                local_feature_stats.extend([len(feat) for feat in local_features])
    
    # 打印诊断结果
    import numpy as np
    print("\n=== ATTENTION QUALITY DIAGNOSIS ===")
    att_max = [s['max_attention'] for s in attention_stats]
    att_std = [s['attention_std'] for s in attention_stats]
    att_ratio = [s['high_attention_ratio'] for s in attention_stats]
    
    print(f"Attention Max: mean={np.mean(att_max):.4f}, std={np.std(att_max):.4f}")
    print(f"Attention Std: mean={np.mean(att_std):.4f}, std={np.std(att_std):.4f}")
    print(f"High Attention Ratio: mean={np.mean(att_ratio):.4f}, std={np.std(att_ratio):.4f}")
    
    if local_feature_stats:
        print(f"Local Features Count: mean={np.mean(local_feature_stats):.2f}, std={np.std(local_feature_stats):.2f}")
        print(f"Zero Local Features: {local_feature_stats.count(0)}/{len(local_feature_stats)} samples")
    
    # 判断attention质量
    if np.mean(att_std) < 0.1:
        print("WARNING: Attention权重分布太平滑，可能无法定位关键区域")
    if np.mean(att_ratio) < 0.1:
        print("WARNING: 高attention区域比例过低")
    if local_feature_stats and np.mean(local_feature_stats) < 50:
        print("WARNING: 局部特征数量过少，可能影响reranking效果")
    
    return attention_stats, local_feature_stats


def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    optimizer = optim.construct_optimizer(model)

    # # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(last_checkpoint, model, optimizer)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = int(checkpoint_epoch['epoch']) + 1
    elif cfg.TRAIN.WEIGHTS:
        checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    # Compute model and loader timings
    # if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
    # benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            test_epoch(test_loader, model, test_meter, cur_epoch)


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)


def time_model():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    # Compute model and loader timings
    # benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
