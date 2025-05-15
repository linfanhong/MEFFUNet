import argparse
import os
from collections import OrderedDict
from glob import glob
import datetime
import logging
import json
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

import albumentations as albu
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import cv2

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from metrics import multi_class_iou_score
from utils import AverageMeter, str2bool, generate_ds_weights, downsample_tensors

# 添加可视化相关的库
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，这样不需要GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

# 随机seed
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果使用多个GPU）
    # cudnn.deterministic = True            # 确保每次卷积操作结果一致
    # cudnn.benchmark = False               # 禁用CUDNN的自动优化


set_seed(3407)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--memo', default=None, help='a flag of different model')

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)  # For NestedUNet
    parser.add_argument('--deep_supervision_unet', default=False, type=str2bool)  # For UNetwDS
    parser.add_argument('--ds_unetmnx', default=False, type=str2bool)  # For UNetMNX
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')
    parser.add_argument('--mnx_kernel_size', default=3, type=int,
                        help='number of classes')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
    parser.add_argument('--valsize', type=float, default=0.2,
                        help='The proportion of the dataset to include in the validation split (default: 0.2).')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD', 'AdamW'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD', 'AdamW']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR', 'PolynomialLR',
                                 'OneCycleLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate in CosineAnnealingLR and ReduceLROnPlateau')
    parser.add_argument('--factor', default=0.7, type=float)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--power', default=1.0, type=float,
                        help='The power of the polynomial.')

    parser.add_argument('--num_workers', default=4, type=int)

    # 新增: 恢复训练参数
    parser.add_argument('--resume', default=False, type=str2bool,
                        help='resume training from saved checkpoint')
    parser.add_argument('--resume_path', default=None, type=str,
                        help='specific checkpoint path to resume from (default: use latest checkpoint)')
    
    # 新增: 可视化参数
    parser.add_argument('--vis_interval', default=10, type=int,
                        help='visualization interval (epochs)')
    parser.add_argument('--save_interval', default=5, type=int,
                        help='interval for saving intermediate results (epochs)')
    parser.add_argument('--vis_backend', default='png', choices=['png', 'svg'], 
                        help='backend for visualization (png or svg)')
    parser.add_argument('--gpu', type=str, default='0',
                        help="Specify which GPU to use (default: '0'). Use '-1' for CPU.")

    config = parser.parse_args()
    return config


# deep_supervision_unet
# 在计算损失函数时，lower weight at lower resolution
# 取output5的loss*(1/1),
# 取output4的loss*(1/2),
# 取output3的loss*(1/4),
# 取output2的loss*(1/8),
# 取output1的loss*(1/16),
# 最后把这几个loss相加

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        # target = target.squeeze(1).long()
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print(input.size())
        # print(target.size())
        # print(model(input)[0])
        # torch.Size([6, 3, 256, 256]) 三通道原始数据
        # torch.Size([6, 1, 256, 256]) 二分类目标结果
        # torch.Size([6, 1, 256, 256]) 二分类预测结果

        # compute output
        if config['deep_supervision']:  # For NestedUNet
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        elif config['deep_supervision_unet']:  # For UNet
            outputs = model(input)
            loss = 0
            weights = generate_ds_weights(len(outputs))  # 五个就是[1/16, 1/8, 1/4, 1/2, 1]
            for i, output in enumerate(outputs):
                _loss = criterion(output, target)
                loss += weights[i] * _loss
            iou = iou_score(outputs[-1], target)
        elif config['ds_unetmnx']:  # For UNetMNX
            outputs = model(input)
            # print(outputs[0].size())
            loss = 0
            scales = [1, 0.5, 0.25, 0.125, 0.0625]
            ds_weights = [1, 0.5, 0.25, 0.125, 0.0625]
            ds_targets = downsample_tensors(target, scales)
            for output, ds_target, ds_weight in zip(outputs, ds_targets, ds_weights):
                # 如果使用的是深度监督模型，outouts会包含多个尺度的不同输出，分别取出不同尺度的输出计算loss
                # print('*************')
                # print(output.size())
                # print('*************')
                # print(ds_target.size())
                _loss = criterion(output, ds_target)
                loss += ds_weight * _loss
            iou = iou_score(outputs[0], target)  # 顺序和上面不一样啦
        else:
            # print(input.shape)
            output = model(input)

            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            # target = target.squeeze(1).long()
            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            elif config['deep_supervision_unet']:
                outputs = model(input)
                loss = 0
                weights = generate_ds_weights(len(outputs))  # 五个就是[1/16, 1/8, 1/4, 1/2, 1]
                for i, output in enumerate(outputs):
                    _loss = criterion(output, target)
                    loss += weights[i] * _loss
                iou = iou_score(outputs[-1], target)
            elif config['ds_unetmnx']:
                outputs = model(input)

                loss = 0
                scales = [1, 0.5, 0.25, 0.125, 0.0625]
                ds_weights = [1, 0.5, 0.25, 0.125, 0.0625]
                ds_targets = downsample_tensors(target, scales)
                for output, ds_target, ds_weight in zip(outputs, ds_targets, ds_weights):
                    _loss = criterion(output, ds_target)
                    loss += ds_weight * _loss
                iou = iou_score(outputs[0], target)  # 顺序和上面不一样啦
            else:

                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def check_deep_supervision_options(config):
    options = [config['deep_supervision'], config['deep_supervision_unet'], config['ds_unetmnx']]
    if sum(options) > 1:
        raise ValueError("Only one of deep_supervision, deep_supervision_unet, or ds_unetmnx can be True.")


def save_checkpoint(model, optimizer, scheduler, epoch, config, log, best_iou):
    """保存检查点以便于恢复训练"""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': config,
        'log': log,
        'best_iou': best_iou
    }

    # 如果使用了调度器且不是None，保存调度器状态
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()

    checkpoint_path = os.path.join('models', config['arch'], config['name'], 'checkpoint.pth')
    torch.save(state, checkpoint_path)
    logging.info(f"Checkpoint saved at epoch {epoch}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """加载检查点恢复训练"""
    logging.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    # print('&&&&&&&&&&&&&&&&&&&&&&&&')
    # print(checkpoint.keys())
    # print('&&&&&&&&&&&&&&&&&&&&&&&&')
    # 加载模型权重
    model.load_state_dict(checkpoint['model'])

    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer'])

    # 加载调度器状态（如果存在）
    if 'scheduler' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    # 返回训练状态
    return (
        checkpoint['epoch'],
        checkpoint['log'],
        checkpoint['best_iou']
    )


def find_latest_checkpoint(model_dir):
    """查找最新的检查点文件"""
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return None


def setup_visualization_dirs(model_dir):
    """设置可视化相关的目录"""
    # 创建可视化目录
    vis_dir = os.path.join(model_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 为不同类型的可视化创建子目录
    plots_dir = os.path.join(vis_dir, 'plots')  # 最终图表
    os.makedirs(plots_dir, exist_ok=True)
    
    progress_dir = os.path.join(vis_dir, 'progress')  # 训练进度图
    os.makedirs(progress_dir, exist_ok=True)
    
    data_dir = os.path.join(vis_dir, 'data')  # 保存原始数据用于Origin
    os.makedirs(data_dir, exist_ok=True)
    
    return {'vis': vis_dir, 'plots': plots_dir, 'progress': progress_dir, 'data': data_dir}


def save_training_data_for_origin(log, epoch, vis_dirs):
    """保存训练数据为Origin可用的格式"""
    # 将数据保存为CSV格式
    #data_path = os.path.join(vis_dirs['data'], f'training_data_epoch_{epoch}.csv')
    data_path = os.path.join(vis_dirs['data'], f'training_data.csv')

    # 准备要保存的数据
    df = pd.DataFrame({
        'epoch': log['epoch'][:epoch + 1],
        'lr': log['lr'][:epoch + 1],
        'train_loss': log['loss'][:epoch + 1],
        'train_iou': log['iou'][:epoch + 1],
        'val_loss': log['val_loss'][:epoch + 1],
        'val_iou': log['val_iou'][:epoch + 1]
    })

    # 保存为带有标题的CSV文件
    df.to_csv(data_path, index=False)
    logging.info(f"Training data for Origin saved to {data_path}")

    # 另存为适合Origin的格式的文本文件
    #origin_path = os.path.join(vis_dirs['data'], f'origin_data_epoch_{epoch}.txt')
    origin_path = os.path.join(vis_dirs['data'], f'origin_data.txt')
    with open(origin_path, 'w') as f:
        f.write('Epoch\tLearning Rate\tTrain Loss\tTrain IoU\tVal Loss\tVal IoU\n')
        for i in range(len(log['epoch'][:epoch + 1])):
            f.write(f"{log['epoch'][i]}\t{log['lr'][i]:.8f}\t{log['loss'][i]:.6f}\t"
                    f"{log['iou'][i]:.6f}\t{log['val_loss'][i]:.6f}\t{log['val_iou'][i]:.6f}\n")
    logging.info(f"Training data formatted for Origin saved to {origin_path}")


def create_training_progress_plot(log, epoch, vis_dirs, config):
    """创建训练进度可视化图"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 准备数据
    epochs = log['epoch'][:epoch+1]
    
    # 1. Loss曲线 (左上)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, log['loss'][:epoch+1], 'o-', label='Train Loss', color='#FF5722', linewidth=2)
    ax1.plot(epochs, log['val_loss'][:epoch+1], 's-', label='Val Loss', color='#2196F3', linewidth=2)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 添加最佳值标注
    best_val_loss_idx = np.argmin(log['val_loss'][:epoch+1])
    best_val_loss = log['val_loss'][best_val_loss_idx]
    ax1.annotate(f'Best: {best_val_loss:.4f}, epoch: {best_val_loss_idx}',
                xy=(epochs[best_val_loss_idx], best_val_loss),
                xytext=(epochs[best_val_loss_idx], best_val_loss*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    # 2. IoU曲线 (右上)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, log['iou'][:epoch+1], 'o-', label='Train IoU', color='#FF5722', linewidth=2)
    ax2.plot(epochs, log['val_iou'][:epoch+1], 's-', label='Val IoU', color='#2196F3', linewidth=2)
    ax2.set_title('IoU Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('IoU Score', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 添加最佳值标注
    best_val_iou_idx = np.argmax(log['val_iou'][:epoch+1])
    best_val_iou = log['val_iou'][best_val_iou_idx]
    ax2.annotate(f'Best: {best_val_iou:.4f}, epoch: {best_val_iou_idx}',
                xy=(epochs[best_val_iou_idx], best_val_iou),
                xytext=(epochs[best_val_iou_idx], best_val_iou*0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    # 3. 学习率变化曲线 (左下)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, log['lr'][:epoch+1], '.-', color='#4CAF50', linewidth=2)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epochs', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 使用对数刻度更好地显示学习率变化
    ax3.set_yscale('log')
    
    # 4. Train vs Val Gap (右下)
    ax4 = fig.add_subplot(gs[1, 1])
    train_val_loss_gap = np.array(log['loss'][:epoch+1]) - np.array(log['val_loss'][:epoch+1])
    train_val_iou_gap = np.array(log['iou'][:epoch+1]) - np.array(log['val_iou'][:epoch+1])
    
    ax4.plot(epochs, train_val_loss_gap, 'o-', label='Train-Val Loss Gap', color='#9C27B0', linewidth=2)
    ax4.plot(epochs, train_val_iou_gap, 's-', label='Train-Val IoU Gap', color='#FFC107', linewidth=2)
    ax4.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax4.set_title('Train-Validation Gaps', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epochs', fontsize=12)
    ax4.set_ylabel('Gap (Train - Val)', fontsize=12)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # 添加训练配置信息
    plt.figtext(0.5, 0.01, 
                f"Model: {config['arch']} | Optimizer: {config['optimizer']} | LR: {config['lr']} | BS: {config['batch_size']} | Scheduler: {config['scheduler']}",
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.suptitle(f"Training Progress - Epoch {epoch+1}/{config['epochs']}", fontsize=16, y=0.99)
    
    # 保存进度图
    # progress_path = os.path.join(vis_dirs['progress'], f'progress_epoch_{epoch+1}.{config["vis_backend"]}')
    # plt.savefig(progress_path, dpi=300, bbox_inches='tight')
    # plt.close(fig)
    # logging.info(f"Training progress plot saved to {progress_path}")


def create_final_visualization(log, vis_dirs, config):
    """创建最终的可视化图表，适合用于论文"""
    # 设置绘图风格
    sns.set_style("whitegrid")

    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })

    # 创建图表
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1])
    
    # 1. 损失曲线和IoU曲线
    ax1 = fig.add_subplot(gs[0])
    epochs = log['epoch']
    
    # 主坐标轴：损失
    color_train_loss = '#E41A1C'  # 红色
    color_val_loss = '#377EB8'    # 蓝色
    line1, = ax1.plot(epochs, log['loss'], '-', color=color_train_loss, linewidth=2, 
                     marker='o', markersize=4, label='Training Loss')
    line2, = ax1.plot(epochs, log['val_loss'], '-', color=color_val_loss, linewidth=2, 
                     marker='s', markersize=4, label='Validation Loss')
    ax1.set_xlabel('')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_train_loss)
    
    # 次坐标轴：IoU
    ax1_twin = ax1.twinx()
    color_train_iou = '#4DAF4A'  # 绿色
    color_val_iou = '#FF7F00'    # 橙色
    line3, = ax1_twin.plot(epochs, log['iou'], '--', color=color_train_iou, linewidth=2, 
                          marker='^', markersize=4, label='Training IoU')
    line4, = ax1_twin.plot(epochs, log['val_iou'], '--', color=color_val_iou, linewidth=2, 
                          marker='d', markersize=4, label='Validation IoU')
    ax1_twin.set_ylabel('IoU Score', fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=color_val_iou)
    
    # 组合图例
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), 
              frameon=True, fontsize=10)
    
    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 学习率和Train-Val Gap
    ax2 = fig.add_subplot(gs[1])
    
    # 主坐标轴：学习率
    color_lr = '#984EA3'  # 紫色
    line5, = ax2.semilogy(epochs, log['lr'], '-', color=color_lr, linewidth=2, 
                         marker='*', markersize=4, label='Learning Rate')
    ax2.set_xlabel('Epochs', fontweight='bold')
    ax2.set_ylabel('Learning Rate (log scale)', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_lr)
    
    # 次坐标轴：Train-Val Gap
    ax2_twin = ax2.twinx()
    color_loss_gap = '#F781BF'  # 粉色
    color_iou_gap = '#A65628'   # 棕色
    
    train_val_loss_gap = np.array(log['loss']) - np.array(log['val_loss'])
    train_val_iou_gap = np.array(log['iou']) - np.array(log['val_iou'])
    
    line6, = ax2_twin.plot(epochs, train_val_loss_gap, '--', color=color_loss_gap, linewidth=1.5, 
                           marker='o', markersize=3, label='Train-Val Loss Gap')
    line7, = ax2_twin.plot(epochs, train_val_iou_gap, '--', color=color_iou_gap, linewidth=1.5,
                           marker='s', markersize=3, label='Train-Val IoU Gap')
    
    ax2_twin.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2_twin.set_ylabel('Train-Val Gap', fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor='black')
    
    # 组合图例
    lines2 = [line5, line6, line7]
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc='center right', bbox_to_anchor=(1.15, 0.5), 
              frameon=True, fontsize=10)
    
    # 添加网格
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 添加标题和模型信息
    plt.suptitle(f"Training Metrics - {config['arch']}", fontsize=16, fontweight='bold', y=0.98)
    plt.figtext(0.5, 0.01, 
                f"Optimizer: {config['optimizer']} | Learning Rate: {config['lr']} | Batch Size: {config['batch_size']} | Loss: {config['loss']}",
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    
    # 保存高分辨率图片，适合论文使用
    final_path = os.path.join(vis_dirs['plots'], f'final_training_metrics.{config["vis_backend"]}')
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    
    # 额外保存为PDF格式，便于出版物使用
    pdf_path = os.path.join(vis_dirs['plots'], 'final_training_metrics.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close(fig)
    logging.info(f"Final visualization saved to {final_path} and {pdf_path}")
    
    # 创建辅助图表：单独的指标比较
    create_comparison_plots(log, vis_dirs, config)


def create_comparison_plots(log, vis_dirs, config):
    """创建辅助的比较图表"""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 12,
    })

    # 1. 对比训练和验证损失
    plt.figure(figsize=(8, 6))
    epochs = log['epoch']
    
    plt.plot(epochs, log['loss'], 'o-', label='Training Loss', color='#E41A1C', linewidth=2)
    plt.plot(epochs, log['val_loss'], 's-', label='Validation Loss', color='#377EB8', linewidth=2)
    
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.title('Training vs. Validation Loss', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    loss_path = os.path.join(vis_dirs['plots'], f'loss_comparison.{config["vis_backend"]}')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 对比训练和验证IoU
    plt.figure(figsize=(8, 6))
    
    plt.plot(epochs, log['iou'], 'o-', label='Training IoU', color='#4DAF4A', linewidth=2)
    plt.plot(epochs, log['val_iou'], 's-', label='Validation IoU', color='#FF7F00', linewidth=2)
    
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('IoU Score', fontweight='bold')
    plt.title('Training vs. Validation IoU', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    iou_path = os.path.join(vis_dirs['plots'], f'iou_comparison.{config["vis_backend"]}')
    plt.savefig(iou_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 学习率曲线
    plt.figure(figsize=(8, 6))
    
    plt.semilogy(epochs, log['lr'], '.-', color='#984EA3', linewidth=2)
    
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Learning Rate (log scale)', fontweight='bold')
    plt.title(f'Learning Rate Schedule ({config["scheduler"]})', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    lr_path = os.path.join(vis_dirs['plots'], f'learning_rate.{config["vis_backend"]}')
    plt.savefig(lr_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Comparison plots saved to {vis_dirs['plots']}")


def main():
    config = vars(parse_args())
    check_deep_supervision_options(config)

    #os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']

    # 设置模型名称
    if config['name'] is None:
        if config['deep_supervision'] or config['deep_supervision_unet'] or config['ds_unetmnx']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

        # 只有在非恢复模式下才添加时间戳
        if not config['resume']:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            config['name'] = f"{config['name']}_{timestamp}"
        else:
            config['name'] = f"{config['name']}_20250325_121201"

    # 创建保存模型的目录
    model_dir = os.path.join('models', config['arch'], config['name'])
    os.makedirs(model_dir, exist_ok=True)

    # 设置可视化目录
    vis_dirs = setup_visualization_dirs(model_dir)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(model_dir, 'program.log'), mode="a"),  # 追加日志
            logging.StreamHandler()  # 在终端输出日志
        ]
    )
    logging.info("程序开始运行")
    logging.info(f"恢复训练模式: {'是' if config['resume'] else '否'}")

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    # 保存配置
    with open(os.path.join(model_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']](num_classes=config['num_classes']).cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])

    model = archs.__dict__[config['arch']](num_classes=config['num_classes'],
                                       input_channels=config['input_channels'],
                                       deep_supervision=config['deep_supervision'],
                                       deep_supervision_unet=config['deep_supervision_unet'],
                                       ds_unetmnx=config['ds_unetmnx'],
                                       kernel_size=config['mnx_kernel_size'])


    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    params = filter(lambda p: p.requires_grad, model.parameters())

    # optimizer
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = []
    if config['dataset'] == 'spine1k_without_pre_more':
        img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*_CT'))
    elif config['dataset'] in ['totalsegmentator_slices', 'totalsegmentator_slices_resize', 'totalsegmentator_slices_mul_T']:
        img_ids = sorted(glob(os.path.join('inputs', config['dataset'], 'images', 's*')))
        img_ids = img_ids[:200]  # 选数据集中的一部分
        # print(config['dataset'])
        # print(img_ids[:10])
        # print(len(img_ids))
        # print(img_ids[200])

    # 训练集验证集划分
    train_img_path, val_img_path = train_test_split(img_ids, test_size=config['valsize'], random_state=3407)  # 随机seed
    train_img_ids = []
    val_img_ids = []

    # 收集训练集和验证集的图像ID
    for patient in train_img_path:
        patient_path = glob(os.path.join(patient, '*' + config['img_ext']))
        for p in patient_path:
            train_img_ids.append(os.path.splitext(os.path.basename(p))[0])

    for patient in val_img_path:
        patient_path = glob(os.path.join(patient, '*' + config['img_ext']))
        for p in patient_path:
            val_img_ids.append(os.path.splitext(os.path.basename(p))[0])

    # 保存训练集ID列表
    with open(os.path.join(model_dir, "train_ids.txt"), "w") as f:
        for item in train_img_ids:
            f.write(str(item) + "\n")

    # 取样步长
    step = 3
    train_img_ids = train_img_ids[::step]
    val_img_ids = val_img_ids[::]

    logging.info('每隔%d张图片取一张图片,训练集总共%d张图片, 验证集总共%d张图片' % (step, len(train_img_ids), len(val_img_ids)))

    # 保存验证集ID列表
    with open(os.path.join(model_dir, "val_ids.txt"), "w") as f:
        for item in val_img_ids:
            f.write(str(item) + "\n")

    # 数据增强/预处理
    train_transform = Compose([
        albu.RandomRotate90(),
        OneOf([
            albu.HorizontalFlip(),
            albu.VerticalFlip()
        ], p=1),  # 让它随机选水平或垂直翻转
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightnessContrast(),
        ], p=1),
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # 数据集
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # 学习率调度器
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'PolynomialLR':
        scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=config['epochs'], power=config['power'])
    elif config['scheduler'] == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'],
                                                        steps_per_epoch=len(train_loader), epochs=config['epochs'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # 初始化训练日志和状态
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    # 初始化训练状态
    start_epoch = 0
    best_iou = 0
    trigger = 0

    # 恢复训练（如果需要）
    if config['resume']:
        checkpoint_path = config['resume_path']

        # 如果没有指定检查点路径，自动寻找最新的
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(model_dir)

        if checkpoint_path and os.path.exists(checkpoint_path):
            logging.info(f"恢复训练从检查点: {checkpoint_path}")
            start_epoch, log, best_iou = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
            # 更新起始轮次
            start_epoch += 1
            trigger = 0  # 重置early stopping计数器
            logging.info(f"恢复训练成功，从第 {start_epoch} 轮开始")
        else:
            logging.warning("找不到有效的检查点文件，将从头开始训练")

    # 主训练循环
    for epoch in range(start_epoch, config['epochs']):
        logging.info('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        # scheduler
        scheduler_name = config['scheduler']
        if scheduler_name in ['CosineAnnealingLR', 'PolynomialLR', 'MultiStepLR', 'OneCycleLR']:
            scheduler.step()
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        logging.info('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                     % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        # append the real-time learning rate
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        # 保存CSV日志
        #pd.DataFrame(log).to_csv(os.path.join(model_dir, 'log.csv'), index=False)

        # 保存当前检查点，以便可以随时恢复训练
        save_checkpoint(model, optimizer, scheduler, epoch, config, log, best_iou)
        
        # 创建可视化图表（按指定间隔）
        if epoch % config['vis_interval'] == 0 or epoch == config['epochs'] - 1:
            create_training_progress_plot(log, epoch, vis_dirs, config)
        
        # 保存中间训练数据（按指定间隔）
        if epoch % config['save_interval'] == 0 or epoch == config['epochs'] - 1:
            save_training_data_for_origin(log, epoch, vis_dirs)

        trigger += 1

        # 如果当前模型是最佳模型，保存模型
        if val_log['iou'] > best_iou:
            best_iou = val_log['iou']
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
            logging.info("=> 保存最佳模型, 验证集IoU: %.4f" % best_iou)
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            logging.info("=> 早停条件达成，训练结束")
            break

        torch.cuda.empty_cache()

    # 训练完成后创建最终可视化
    create_final_visualization(log, vis_dirs, config)
    
    print('-' * 20)
    logging.info("model %s 训练完成。" % config['name'])
    print("model %s training complete." % config['name'])
    logging.info("程序运行结束")


if __name__ == '__main__':
    main()