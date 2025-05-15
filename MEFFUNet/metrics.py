import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    # print('**********************')
    # print(output.size())
    # print(target.size())
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    # print('**********************')
    # print(output.shape)
    # print(target.shape)
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def multi_class_iou_score(output, target):
    """
    计算多分类分割的IoU得分

    参数:
        output: 模型输出张量，形状为(B, C, H, W)
        target: 真实标签张量，形状为(B, H, W)，包含类别值

    返回:
        所有类别和批次样本的平均IoU
    """
    smooth = 1e-5
    if torch.is_tensor(output):
        output = F.softmax(output, dim=1).data.cpu().numpy()
        output = np.argmax(output, axis=1)  # 转换为类别索引，形状为(B, H, W)
        # print(output)
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
        # print(np.unique(target[0]))

    # if len(np.unique(target)) != 13: print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    batch_size = output.shape[0]
    batch_ious = []

    # 对批次中的每个样本计算
    for i in range(batch_size):
        # 确定该样本中存在的所有类别
        #classes = np.unique(np.concatenate([output[i].flatten(), target[i].flatten()]))
        classes = np.unique(target[i])
        # print(classes)
        # print("class=========")
        # print(np.unique(output[i]))
        # print(np.unique(target[i]))
        ious = []
        i = 0
        # 对每个存在的类别计算IoU
        for cls in classes:
            if cls == 0:
                continue
            output_ = (output[i] == cls)
            target_ = (target[i] == cls)
            intersection = (output_ & target_).sum()
            union = (output_ | target_).sum()
            i = i + 1

            if union > 0:  # 避免除以零
                ious.append((intersection + smooth) / (union + smooth))
        # print(i)
        batch_ious.append(np.mean(ious) if ious else 0.0)

    return np.mean(batch_ious)

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def recall_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output_ = output > 0.5
    target_ = target > 0.5
    
    true_positives = (output_ & target_).sum()
    total_positives = target_.sum()

    return (true_positives + smooth) / (total_positives + smooth)

def precision_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output_ = output > 0.5
    target_ = target > 0.5
    
    true_positives = (output_ & target_).sum()
    predicted_positives = output_.sum()

    return (true_positives + smooth) / (predicted_positives + smooth)


def hausdorff_distance_95(output, target, percentile=95):
    """
    计算二值分割的95%豪斯多夫距离

    参数:
        output: 模型输出张量，形状为(B,1,H,W)
        target: 真实标签张量，形状为(B,1,H,W)
        percentile: 豪斯多夫距离的百分位数（默认：95）

    返回:
        所有批次样本的平均95%豪斯多夫距离
    """
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    batch_size = output_.shape[0]
    hausdorff_distances = []

    # 对批次中的每个样本计算
    for i in range(batch_size):
        # 从[1,H,W]压缩为[H,W]
        pred = output_[i, 0]
        gt = target_[i, 0]

        # 如果预测或目标中不包含前景像素，则跳过
        if not np.any(pred) or not np.any(gt):
            continue

        # 获取前景像素的坐标
        pred_coords = np.array(np.where(pred)).T
        gt_coords = np.array(np.where(gt)).T

        if len(pred_coords) == 0 or len(gt_coords) == 0:
            continue

        # 计算所有点对之间的距离，用于百分位数计算

        # 方法1：计算每个预测点到最近目标点的距离
        all_dists_1 = []
        for pred_coord in pred_coords:
            # 计算该预测点到所有目标点的距离
            dists = np.sqrt(np.sum((pred_coord - gt_coords) ** 2, axis=1))
            # 取最小距离
            all_dists_1.append(np.min(dists))
        all_dists_1 = np.array(all_dists_1)

        # 方法2：计算每个目标点到最近预测点的距离
        all_dists_2 = []
        for gt_coord in gt_coords:
            # 计算该目标点到所有预测点的距离
            dists = np.sqrt(np.sum((gt_coord - pred_coords) ** 2, axis=1))
            # 取最小距离
            all_dists_2.append(np.min(dists))
        all_dists_2 = np.array(all_dists_2)

        # 计算95%豪斯多夫距离
        if len(all_dists_1) > 0:
            h1 = np.percentile(all_dists_1, percentile)
        else:
            h1 = 0

        if len(all_dists_2) > 0:
            h2 = np.percentile(all_dists_2, percentile)
        else:
            h2 = 0

        # 取两个方向的最大值作为最终的豪斯多夫距离
        hd = max(h1, h2)
        hausdorff_distances.append(hd)

    # 返回所有样本的平均豪斯多夫距离
    return np.mean(hausdorff_distances) if hausdorff_distances else 0.0

# 使用示例
# output = torch.randn(2, 1, 256, 256)  # 模型输出，形状为[B,1,H,W]
# target = torch.zeros(2, 1, 256, 256)  # 真实标签，形状为[B,1,H,W]
# target[:, :, 100:150, 100:150] = 1    # 在中间位置设置一个方形区域
# hd95 = hausdorff_distance_95(output, target)
# print(f"95% Hausdorff Distance: {hd95:.4f}")