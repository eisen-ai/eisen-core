import torch

from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(DiceLoss, self).__init__()

        self.weight = weight

    def forward(self, predictions, labels):
        dice_loss = 1.0 - 2.0 * (labels * predictions) / (labels ** 2 + predictions ** 2)

        return self.weight * dice_loss.mean()
