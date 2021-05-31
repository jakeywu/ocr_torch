import torch.nn as nn
import torch
import torch.nn.functional as F


class MaskDiceLoss(nn.Module):
    """
    最终的thresh_binary采用DiceLoss
    """
    def __init__(self, eps=1e-6):
        super(MaskDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        :param pred: N * H * W
        :param gt:  N* H * W
        :param mask: N * H * W
        :return:
        """
        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        :param pred: N * H * W
        :param gt: N * H * W
        :param mask: N * H * W
        :return:
        """
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        :param pred: N * H * W
        :param gt:  N * H * W
        :param mask: N * H * W
        :return:
        """
        loss = F.binary_cross_entropy(pred, gt, reduction="none")

        positive = (gt * mask).byte()
        negative = ((1-gt) * mask).byte()

        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()

        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))

        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)

        return balance_loss
