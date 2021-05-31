import torch.nn as nn
from collections import OrderedDict
from losses.loss import MaskDiceLoss, MaskL1Loss, BalanceCrossEntropyLoss


class L1BalanceCELoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6, l1_scale=10, bce_scale=5):
        super(L1BalanceCELoss, self).__init__()
        self.dice_loss = MaskDiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=negative_ratio, eps=eps)

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, batch):
        pred_binary = pred[:, 0, :, :]
        pred_thresh = pred[:, 1, :, :]
        pred_binary_thresh = pred[:, 2, :, :]

        bce_loss = self.bce_loss(pred_binary, batch['prob_map'], batch['prob_mask'])
        l1_loss = self.l1_loss(pred_thresh, batch['thresh_map'], batch['thresh_mask'])
        dice_loss = self.dice_loss(pred_binary_thresh, batch['prob_map'], batch['prob_mask'])
        loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        loss_dict = OrderedDict()
        loss_dict["loss"] = loss
        loss_dict["prob_loss"] = bce_loss
        loss_dict["thresh_loss"] = l1_loss
        loss_dict["binary_loss"] = dice_loss
        return loss_dict


if __name__ == "__main__":
    import torch
    lbce = L1BalanceCELoss()
    input_x = torch.rand(4, 3, 160, 160)
    input_y = {
        "prob_map": torch.rand(4, 160, 160),
        "prob_mask": torch.rand(4, 160, 160),
        "thresh_map": torch.rand(4, 160, 160),
        "thresh_mask": torch.rand(4, 160, 160),
    }
    print(lbce(input_x, input_y))