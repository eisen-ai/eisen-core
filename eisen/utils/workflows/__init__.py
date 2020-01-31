from .testing import *
from .training import *


class GenericWorkflow:
    def compute_losses(self, arguments):
        results = []

        for loss in self.losses:
            loss_argument_dict = {key: arguments[key] for key in loss.input_names}

            loss_result = loss(**loss_argument_dict)

            results.append(loss_result)

        return results

    def compute_metrics(self, arguments):
        results = []

        for metric in self.metrics:
            metric_argument_dict = {key: arguments[key] for key in metric.input_names}

            metric_result = metric(**metric_argument_dict)

            results.append(metric_result)

        return results