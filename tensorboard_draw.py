import re
import json
import os
from torch.utils.tensorboard import SummaryWriter

def extract_train_iter_logs(file_path):
    """
    读取日志文件并提取train_iter相关的信息
    """
    train_iter_logs = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 查找包含train_iter的行
                if '"_type": "train_iter"' in line:
                    # 使用正则表达式提取json_stats后面的JSON部分
                    match = re.search(r'json_stats: ({.*})', line)
                    if match:
                        json_str = match.group(1)
                        try:
                            # 解析JSON
                            json_data = json.loads(json_str)
                            train_iter_logs.append(json_data)
                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误: {e}")
                            continue
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return train_iter_logs
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return train_iter_logs
    
    return train_iter_logs

def parse_epoch_from_string(epoch_str):
    """
    从epoch字符串中提取当前epoch数字
    例如: "14/50" -> 14
    """
    try:
        current_epoch = int(epoch_str.split('/')[0])
        return current_epoch
    except:
        return None

def parse_iter_from_string(iter_str):
    """
    从iter字符串中提取当前iter数字
    例如: "7580/7812" -> 7580
    """
    try:
        current_iter = int(iter_str.split('/')[0])
        return current_iter
    except:
        return None

def process_lr_value(lr_str):
    """
    处理学习率值，支持科学计数法
    """
    try:
        return float(lr_str)
    except:
        return None

def calculate_cumulative_iter(train_logs):
    """
    计算累加的iter数
    """
    # 按epoch分组，记录每个epoch的最大iter
    epoch_max_iter = {}
    processed_logs = []
    
    # 首先遍历所有数据，找出每个epoch的最大iter数
    for log in train_logs:
        if 'epoch' in log and 'iter' in log:
            epoch_num = parse_epoch_from_string(log['epoch'])
            iter_num = parse_iter_from_string(log['iter'])
            
            if epoch_num is not None and iter_num is not None:
                if epoch_num not in epoch_max_iter:
                    epoch_max_iter[epoch_num] = iter_num
                else:
                    epoch_max_iter[epoch_num] = max(epoch_max_iter[epoch_num], iter_num)
    
    # 计算每个epoch的累加基数
    sorted_epochs = sorted(epoch_max_iter.keys())
    epoch_cumulative_base = {}
    cumulative_base = 0
    
    for epoch in sorted_epochs:
        epoch_cumulative_base[epoch] = cumulative_base
        cumulative_base += epoch_max_iter[epoch]
    
    print(f"Epoch累加基数: {epoch_cumulative_base}")
    
    # 为每条记录计算累加iter
    for log in train_logs:
        if 'epoch' in log and 'iter' in log:
            epoch_num = parse_epoch_from_string(log['epoch'])
            iter_num = parse_iter_from_string(log['iter'])
            
            if epoch_num is not None and iter_num is not None:
                cumulative_iter = epoch_cumulative_base.get(epoch_num, 0) + iter_num
                processed_log = log.copy()
                processed_log['cumulative_iter'] = cumulative_iter
                processed_logs.append(processed_log)
    
    return processed_logs

def plot_tensorboard_charts(processed_logs, save_dir="./tensorboard_logs/8gpu_newzzdata300w_vit_20251121_CNN_V1"):
    """
    使用TensorBoard绘制所有图表
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化TensorBoard writer
    writer = SummaryWriter(log_dir=save_dir)
    
    # 用于学习率的epoch数据
    epoch_lr_dict = {}
    
    # 用于各种指标的累加iter数据
    metrics_data = {
        'att_loss': [],
        'att_top1_err': [],
        'att_top5_err': [],
        'desc_loss': [],
        'desc_top1_err': [],
        'desc_top5_err': []
    }
    
    for log in processed_logs:
        # 处理学习率数据（按epoch）
        if 'epoch' in log and 'lr' in log:
            epoch_num = parse_epoch_from_string(log['epoch'])
            lr_value = process_lr_value(log['lr'])
            
            if epoch_num is not None and lr_value is not None:
                epoch_lr_dict[epoch_num] = lr_value
        
        # 处理各种指标数据（按累加iter）
        if 'cumulative_iter' in log:
            cumulative_iter = log['cumulative_iter']
            
            for metric_name in metrics_data.keys():
                if metric_name in log:
                    try:
                        metric_value = float(log[metric_name])
                        metrics_data[metric_name].append((cumulative_iter, metric_value))
                    except:
                        continue
    
    # 绘制学习率图（按epoch）
    sorted_epochs = sorted(epoch_lr_dict.keys())
    print(f"找到 {len(sorted_epochs)} 个epoch的学习率数据")
    print("前3个epoch的学习率:")
    
    for i, epoch in enumerate(sorted_epochs):
        lr_value = epoch_lr_dict[epoch]
        writer.add_scalar('Learning_Rate/LR', lr_value, epoch)
        
        if i < 3:
            print(f"  Epoch {epoch}: {lr_value:.2e}")
    
    # 绘制各种指标图（按累加iter）
    print(f"\n各指标数据统计:")
    
    for metric_name, data_points in metrics_data.items():
        if data_points:
            # 按累加iter排序
            data_points.sort(key=lambda x: x[0])
            
            # 确定TensorBoard中的分组和名称
            if 'loss' in metric_name:
                group_name = 'Loss'
                if metric_name == 'att_loss':
                    scalar_name = 'Attention_Loss'
                elif metric_name == 'desc_loss':
                    scalar_name = 'Description_Loss'
            else:  # error metrics
                group_name = 'Error'
                if metric_name == 'att_top1_err':
                    scalar_name = 'Attention_Top1_Error'
                elif metric_name == 'att_top5_err':
                    scalar_name = 'Attention_Top5_Error'
                elif metric_name == 'desc_top1_err':
                    scalar_name = 'Description_Top1_Error'
                elif metric_name == 'desc_top5_err':
                    scalar_name = 'Description_Top5_Error'
            
            # 写入TensorBoard
            for cumulative_iter, metric_value in data_points:
                writer.add_scalar(f'{group_name}/{scalar_name}', metric_value, cumulative_iter)
            
            # 统计信息
            metric_values = [x[1] for x in data_points]
            print(f"  {metric_name}: {len(data_points)} 个数据点, 范围: {min(metric_values):.4f} ~ {max(metric_values):.4f}")
            
            # 显示前3个数据点
            print(f"    前3个数据点:")
            for i, (cumulative_iter, metric_value) in enumerate(data_points[:3]):
                print(f"      Iter {cumulative_iter}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: 未找到数据")
    
    print(f"\nTensorBoard日志已保存到: {save_dir}")
    print(f"运行以下命令查看图表:")
    print(f"tensorboard --logdir {save_dir}")
    
    # 关闭writer
    writer.close()
    
    # 总结
    if epoch_lr_dict:
        lr_values = list(epoch_lr_dict.values())
        print(f"\n学习率范围: {min(lr_values):.2e} ~ {max(lr_values):.2e}")

def main():
    # 文件路径
    file_path = "/home/ubuntu/san/hxl/delg-pytoch/DELG_VIT/output/8gpu_newzzdata300w_vit_20251121_CNN_V1/stdout.log"
    
    # 提取train_iter日志
    print("正在读取日志文件...")
    train_logs = extract_train_iter_logs(file_path)
    print(f"总共找到 {len(train_logs)} 条train_iter记录")
    
    # 计算累加iter
    print("\n正在计算累加iter...")
    processed_logs = calculate_cumulative_iter(train_logs)
    print(f"处理了 {len(processed_logs)} 条有效记录")
    
    # 绘制图表
    print("\n正在生成TensorBoard图表...")
    plot_tensorboard_charts(processed_logs)
    
    print("\nTensorBoard图表生成完成！")
    print("您将看到以下图表:")
    print("- Learning_Rate/LR (按epoch)")
    print("- Loss/Attention_Loss (按累加iter)")
    print("- Loss/Description_Loss (按累加iter)")
    print("- Error/Attention_Top1_Error (按累加iter)")
    print("- Error/Attention_Top5_Error (按累加iter)")
    print("- Error/Description_Top1_Error (按累加iter)")
    print("- Error/Description_Top5_Error (按累加iter)")

if __name__ == "__main__":
    main()