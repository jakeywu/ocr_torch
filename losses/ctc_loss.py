import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    def __init__(self):
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=0,
            reduction="mean"
        )

    def forward(self, x, batch):
        """
        :param x: T * N * Classes
        :param batch:
        :return:
        """
        t, n, _ = x.shape
        loss = self.ctc_loss(
            log_probs=x,  # T * N * Classes
            targets=batch["label_idx"],  # 训练标签
            input_lengths=torch.tensor([t] * n),  # 固定的额输出序列长度
            target_lengths=batch["sequence_length"]  # 真实长度
        )
        return {"loss": loss}
