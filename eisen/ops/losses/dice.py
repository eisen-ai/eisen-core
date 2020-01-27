from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, weight=1.0, dim=None):
        """
        :param weight: absolute weight of this loss
        :type weight: float

        <json>
        [
            {"name": "input_names", "type": "list:string", "value": "['predictions', 'labels']"},
            {"name": "output_names", "type": "list:string", "value": "['dice_loss']"},
            {"name": "weight", "type": "float", "value": "1.0"},
            {"name": "dim", "type": "list:int", "value": [1, 2, 3, 4]}
        ]
        </json>
        """
        super(DiceLoss, self).__init__()

        if dim is None:
            dim = [1, 2, 3, 4]

        self.dim = dim

        self.weight = weight

    def forward(self, predictions, labels):
        dice_loss = 1.0 - \
                    2.0 * (labels * predictions).sum(dim=self.dim) / (labels ** 2 + predictions ** 2).sum(dim=self.dim)

        dice_loss = self.weight * dice_loss.mean()

        return dice_loss
