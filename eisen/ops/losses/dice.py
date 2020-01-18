import torch

from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, weight=1.0, labels_field='labels', predictions_field='predictions'):
        super(DiceLoss, self).__init__()

        self.weight = weight

        self.labels_field = labels_field
        self.predictions_field = predictions_field

    def forward(self, predictions, labels):
        dice_loss = 1.0 - 2.0 * (labels * predictions) / (labels ** 2 + predictions ** 2)

        return self.weight * dice_loss.mean()