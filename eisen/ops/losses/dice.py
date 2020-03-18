from torch import nn


class DiceLoss(nn.Module):
    """
    Dice loss is often used in segmentation tasks to optimize the overlap between the ground truth contour
    and the prediction. Dice loss is robust to class imbalance and therefore suitable to segment small foreground
    regions in images or volumes.

    This version of the Dice loss supports multi-class segmentation (although in a naive manner).
    """
    def __init__(self, weight=1.0, dim=None):
        """
        :param weight: absolute weight of this loss
        :type weight: float

        <json>
        [
            {"name": "input_names", "type": "list:string", "value": "['predictions', 'labels']"},
            {"name": "output_names", "type": "list:string", "value": "['dice_loss']"},
            {"name": "weight", "type": "float", "value": "1.0"},
            {"name": "dim", "type": "list:int", "value": "[1, 2, 3, 4]"}
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
