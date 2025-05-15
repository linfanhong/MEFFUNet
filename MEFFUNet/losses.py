import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

try:
    from lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss','BCEDiceLoss2','LovaszHingeLoss',
           'BinaryDiceLoss','BinaryIoULoss',
           'BinaryBCEDiceLoss','BinaryBCEIoULoss',
           'BinaryFocalLoss','BinaryFocalDiceLoss',
           'LovaszHingeLoss','BinaryBCELovaszHingeLoss',
           'CEDiceLoss']


class CEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, num_classes=None, weight=None, reduction='mean'):
        """
        Combined Cross Entropy and Dice Loss for multi-class segmentation

        Args:
            alpha: weight for CE loss (alpha * CE + (1-alpha) * Dice)
            num_classes: number of classes (needed for one-hot encoding)
            weight: class weights for CE loss
            reduction: 'mean' or 'sum' for loss reduction
        """
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.weight = weight
        self.reduction = reduction
        self.smooth = 1e-5

    def forward(self, input, target):
        # input shape: (B, C, H, W)
        # target shape: (B, H, W) with class indices (0, 1, ..., C-1)

        # Calculate Cross Entropy Loss
        ce_loss = F.cross_entropy(
            input,
            target,
            weight=self.weight,
            reduction=self.reduction
        )

        # For Dice, convert target to one-hot encoding if not already
        if self.num_classes is not None and target.dim() != input.dim():
            # Convert target to one-hot encoding
            target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        else:
            # Assume target is already in one-hot format
            target_one_hot = target

        # Apply softmax to input
        input_softmax = F.softmax(input, dim=1)

        # Calculate Dice Loss
        # Reshape tensors for calculation
        batch_size = input.size(0)
        input_flat = input_softmax.view(batch_size, self.num_classes, -1)
        target_flat = target_one_hot.view(batch_size, self.num_classes, -1)

        # Calculate intersection and dice scores per class per batch
        intersection = (input_flat * target_flat).sum(dim=2)
        input_sum = input_flat.sum(dim=2)
        target_sum = target_flat.sum(dim=2)

        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (input_sum + target_sum + self.smooth)

        # Calculate Dice Loss
        if self.reduction == 'mean':
            dice_loss = 1 - dice.mean()
        else:  # 'sum'
            dice_loss = self.num_classes - dice.sum()

        # Combined loss
        combined_loss = self.alpha * ce_loss + (1 - self.alpha) * dice_loss

        return combined_loss


# Example usage:
# loss_fn = CEDiceLoss(alpha=0.5, num_classes=4)
# prediction = torch.randn(8, 4, 256, 256)  # batch_size=8, classes=4, height=256, width=256
# target = torch.randint(0, 4, (8, 256, 256))  # class indices from 0 to 3
# loss = loss_fn(prediction, target)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)

        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)

        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
    

class BCEDiceLoss2(nn.Module):
    def __init__(self, alpha=0.5, num_classes=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return self.alpha * bce + (1 - self.alpha) * dice


class CEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, num_classes=12, weight=None, reduction='mean'):
        """
        Combined Cross Entropy and Dice Loss for multi-class segmentation

        Args:
            alpha: weight for CE loss (alpha * CE + (1-alpha) * Dice)
            num_classes: number of classes (needed for one-hot encoding)
            weight: class weights for CE loss
            reduction: 'mean' or 'sum' for loss reduction
        """
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.weight = weight
        self.reduction = reduction
        self.smooth = 1e-5

    def forward(self, input, target):
        # input shape: (B, C, H, W)
        # target shape: (B, H, W) with class indices (0, 1, ..., C-1)

        # Calculate Cross Entropy Loss
        ce_loss = F.cross_entropy(
            input,
            target,
            weight=self.weight,
            reduction=self.reduction
        )

        # For Dice, convert target to one-hot encoding if not already
        if self.num_classes is not None and target.dim() != input.dim():
            # Convert target to one-hot encoding
            target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
            # print("dddddddddddd")
            # print(target_one_hot.shape)
            # valid_mask = target_one_hot > 0
            # input[~valid_mask] = 0.0

        else:
            # Assume target is already in one-hot format
            target_one_hot = target

        # Apply softmax to input
        input_softmax = F.softmax(input, dim=1)

        # Calculate Dice Loss
        # Reshape tensors for calculation
        batch_size = input.size(0)
        input_flat = input_softmax.view(batch_size, self.num_classes, -1)
        target_flat = target_one_hot.view(batch_size, self.num_classes, -1)

        # Calculate intersection and dice scores per class per batch
        intersection = (input_flat * target_flat).sum(dim=2)
        input_sum = input_flat.sum(dim=2)
        target_sum = target_flat.sum(dim=2)

        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (input_sum + target_sum + self.smooth)

        # Calculate Dice Loss
        if self.reduction == 'mean':
            dice_loss = 1 - dice.mean()
        else:  # 'sum'
            dice_loss = self.num_classes - dice.sum()

        # Combined loss
        combined_loss = self.alpha * ce_loss + (1 - self.alpha) * dice_loss

        return dice_loss


# Example usage:
# loss_fn = CEDiceLoss(alpha=0.5, num_classes=4)
# prediction = torch.randn(8, 4, 256, 256)  # batch_size=8, classes=4, height=256, width=256
# target = torch.randint(0, 4, (8, 256, 256))  # class indices from 0 to 3
# loss = loss_fn(prediction, target)

# 通用类
class BinarySegmentationLoss(torch.nn.Module):
    def __init__(self, smooth=1., activation='sigmoid'):
        super(BinarySegmentationLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def _flatten(self, tensor):
        return tensor.view(tensor.shape[0], -1)
    
    def forward(self, pred, target):
        raise NotImplementedError("Subclasses should implement this method.")

class BinaryDiceLoss(BinarySegmentationLoss):
    def __init__(self, smooth=1., activation='sigmoid'):
        super(BinaryDiceLoss, self).__init__(smooth, activation)

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0], "predictions and targets shapes don't match"
        
        if self.activation == 'sigmoid':
            pred = torch.sigmoid(pred)
        
        pred_flat = self._flatten(pred)
        target_flat = self._flatten(target)

        intersection = (pred_flat * target_flat).sum(dim=1)

        loss = 1 - ((2. * intersection + self.smooth) /
                    (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth))
        return loss.mean()
    
class BinaryIoULoss(BinarySegmentationLoss):
    def __init__(self, smooth=1., activation='sigmoid'):
        super(BinaryIoULoss, self).__init__(smooth, activation)

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0], "predictions and targets shapes don't match"
        
        if self.activation == 'sigmoid':
            pred = torch.sigmoid(pred)
        
        pred_flat = self._flatten(pred)
        target_flat = self._flatten(target)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = (pred_flat + target_flat).sum(dim=1) - intersection

        loss = 1 - ((intersection + self.smooth) / (union + self.smooth))
        return loss.mean()
    
class BinaryBCEDiceLoss(BinarySegmentationLoss):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1., activation='sigmoid'):
        super(BinaryBCEDiceLoss, self).__init__(smooth, activation)
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = torch.nn.BCEWithLogitsLoss()  # 适用于未激活的预测

    def forward(self, pred, target):
        bce_loss = self.bce_loss(pred, target.float())
        
        pred_sigmoid = torch.sigmoid(pred)
        
        dice_loss = BinaryDiceLoss(smooth=self.smooth, activation=None)(pred_sigmoid, target)
        
        return self.alpha * dice_loss + self.beta * bce_loss
    
class BinaryBCEIoULoss(BinarySegmentationLoss):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1., activation='sigmoid'):
        super(BinaryBCEIoULoss, self).__init__(smooth, activation)
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = torch.nn.BCEWithLogitsLoss()  # 适用于未激活的预测

    def forward(self, pred, target):
        bce_loss = self.bce_loss(pred, target.float())
        
        pred_sigmoid = torch.sigmoid(pred)
        
        iou_loss = BinaryIoULoss(smooth=self.smooth, activation=None)(pred_sigmoid, target)
        
        return self.alpha * iou_loss + self.beta * bce_loss
    
class BinaryFocalLoss(BinarySegmentationLoss):
    def __init__(self, gamma=2., alpha=0.25, smooth=1., activation='sigmoid'):
        super(BinaryFocalLoss, self).__init__(smooth, activation)
        self.gamma = gamma  # γ是一个聚焦参数，用于控制容易分类样本的权重降低的程度
        self.alpha = alpha  # α是一个平衡因子，用于调整正负样本的比例, 默认值为 0.25，用于处理类别不平衡问题

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0], "predictions and targets shapes don't match"
        
        if self.activation == 'sigmoid':
            pred = torch.sigmoid(pred)
        
        # Flatten the tensors for batch computation
        pred_flat = self._flatten(pred)
        target_flat = self._flatten(target)

        # Calculate focal loss
        pt = pred_flat * target_flat + (1 - pred_flat) * (1 - target_flat)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = -self.alpha * torch.log(pt) * focal_weight

        # Average loss over the batch
        return focal_loss.mean()

class BinaryFocalDiceLoss(BinarySegmentationLoss):
    def __init__(self, alpha_dice=0.5, beta_focal=0.5, gamma=2., alpha=0.25, smooth=1., activation='sigmoid'):
        super(BinaryFocalDiceLoss, self).__init__(smooth, activation)
        self.alpha_dice = alpha_dice
        self.beta_focal = beta_focal
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0], "predictions and targets shapes don't match"
        
        if self.activation == 'sigmoid':
            pred = torch.sigmoid(pred)
        
        # Flatten the tensors for batch computation
        pred_flat = self._flatten(pred)
        target_flat = self._flatten(target)

        # Compute Dice Loss
        dice_loss = BinaryDiceLoss(smooth=self.smooth, activation=None)(pred_flat, target_flat)
        # Compute Focal Loss
        focal_loss = BinaryFocalLoss(self.gamma, self.alpha, activation=None)(pred_flat, target_flat)
        # Combine losses
        combined_loss = self.alpha_dice * dice_loss + self.beta_focal * focal_loss

        return combined_loss
    
class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

class BinaryBCELovaszHingeLoss(BinarySegmentationLoss):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1., activation='sigmoid'):
        super(BinaryBCELovaszHingeLoss, self).__init__(smooth, activation)
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = torch.nn.BCEWithLogitsLoss()  # 适用于未激活的预测

    def forward(self, pred, target):
        bce_loss = self.bce_loss(pred, target.float())
        lovaszhinge_loss = LovaszHingeLoss()(pred, target)
        
        return self.alpha * lovaszhinge_loss + self.beta * bce_loss
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    pred = torch.randn((10, 1, 256, 256), requires_grad=True).to(device)
    target = torch.randint(2, (10, 1, 256, 256)).float().to(device)

    # BinaryDiceLoss,BinaryIoULoss,BinaryBCEDiceLoss,BinaryBCEIoULoss,BinaryFocalLoss,BinaryFocalDiceLoss,LovaszHingeLoss,BinaryBCELovaszHingeLoss
    criterion = LovaszHingeLoss()

    loss = criterion(pred, target)
    loss.backward()
    print(f"{type(criterion).__name__}:{loss}")

