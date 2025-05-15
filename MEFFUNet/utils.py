import argparse

import torch


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def generate_ds_weights(num_outputs):
    """
    动态生成适用于 Deep Supervision U-Net 模型的权重列表。
    
    Deep Supervision U-Net 在不同分辨率下产生多个输出，
    这些输出在计算损失函数时需要不同的权重。
    该函数根据给定的输出数量 `num_outputs` 生成一个权重列表，
    其中最高分辨率的输出具有最大的权重（默认为 1），
    而较低分辨率的输出则按比例减少权重。
    
    参数:
        num_outputs (int): 模型产生的输出数量，从最低分辨率到最高分辨率排列。
        
    返回:
        list: 包含按指定规则生成的权重列表。
        
    示例:
        如果 num_outputs 是 5，则返回的权重列表将是 [1/16, 1/8, 1/4, 1/2, 1]。
        计算损失时，最底层的输出（较低分辨率）将被赋予较小的权重，
        而最高层的输出（最高分辨率）将被赋予最大的权重（默认为 1）。
    """
    max_weight = 1
    
    weights = [max_weight / (2 ** (num_outputs - i - 1)) for i in range(num_outputs)]
    return weights

def downsample_tensors(tensors, scales):
        """
        对给定的一批张量进行下采样处理。

        参数:
            tensors (torch.Tensor): 形状为 [N, 1, H, W] 的张量。
            scales (list of float): 每个元素表示下采样的比例。

        返回:
            list of torch.Tensor: 下采样后的张量列表。
        """
        # 初始化空列表来保存下采样后的张量
        downsampled_tensors = []
        
        # 遍历所有需要的下采样比例
        for scale in scales:
            if scale == 1.0:
                # 如果比例是1，则直接添加原始张量
                downsampled_tensors.append(tensors)
            else:
                # 使用最近邻插值进行下采样
                new_size = (int(tensors.shape[2] * scale), int(tensors.shape[3] * scale))
                downsampled_tensor = torch.nn.functional.interpolate(
                    tensors, size=new_size, mode='nearest', align_corners=None)  # 使用 nearest 插值
                downsampled_tensors.append(downsampled_tensor)
        
        return downsampled_tensors