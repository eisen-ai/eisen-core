from torch import nn


class DiceMetric(nn.Module):
    def __init__(self, weight=1.0):
        """
        :param weight: absolute weight of this metric
        :type weight: float

        <json>
        [
            {"name": "input_names", "type": "list:string", "value": "['predictions', 'labels']"},
            {"name": "output_names", "type": "list:string", "value": "['dice_metric']"},
            {"name": "weight", "type": "float", "value": "1.0"}
        ]
        </json>
        """
        super(DiceMetric, self).__init__()

        self.weight = weight

    def forward(self, predictions, labels):
        dice = 2.0 * (labels * predictions) / (labels ** 2 + predictions ** 2)

        dice_metric = self.weight * dice.mean()

        return dice_metric