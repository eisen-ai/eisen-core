from torch import nn


class DiceMetric(nn.Module):
    """
    The Dice coefficient is often used in segmentation tasks to evaluate the performance of algorithms by providing a
    scalar result expressing the amount of overlap between the ground truth contour and the prediction.
    The Dice coefficient is robust to class imbalance and therefore suitable to evaluate small foreground
    regions in images or volumes.

    This version of the Dice metrics supports multi-class segmentation (although in a naive manner).
    """
    def __init__(self, weight=1.0, dim=None):
        """
        :param weight: absolute weight of this metric
        :type weight: float

        <json>
        [
            {"name": "input_names", "type": "list:string", "value": "['predictions', 'labels']"},
            {"name": "output_names", "type": "list:string", "value": "['dice_metric']"},
            {"name": "weight", "type": "float", "value": "1.0"},
            {"name": "dim", "type": "list:int", "value": "[1, 2, 3, 4]"}
        ]
        </json>
        """
        super(DiceMetric, self).__init__()

        if dim is None:
            dim = [1, 2, 3, 4]

        self.dim = dim

        self.weight = weight

    def forward(self, predictions, labels):
        predictions = (predictions >= 0.5).float()

        dice = 2.0 * (labels * predictions).sum(dim=self.dim) / (labels ** 2 + predictions ** 2).sum(dim=self.dim)

        dice_metric = self.weight * dice.mean()

        return dice_metric