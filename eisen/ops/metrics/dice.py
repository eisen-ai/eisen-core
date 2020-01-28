from torch import nn


class DiceMetric(nn.Module):
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
        dice = 2.0 * (labels * predictions).sum(dim=self.dim) / (labels ** 2 + predictions ** 2).sum(dim=self.dim)

        dice_metric = self.weight * dice.mean()

        return dice_metric