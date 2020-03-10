import numpy as np

from eisen import EISEN_BEST_MODEL_LOSS, EISEN_BEST_MODEL_METRIC
from pydispatch import dispatcher


def convert_output_dict_to_cpu(output_dict):
    for typ in ['losses', 'metrics']:
        for i in range(len(output_dict[typ])):
            for key in output_dict[typ][i].keys():
                output_dict[typ][i][key] = output_dict[typ][i][key].cpu().data.numpy()

    for typ in ['inputs', 'outputs']:
        for key in output_dict[typ].keys():
            output_dict[typ][key] = output_dict[typ][key].cpu().data.numpy()

    return output_dict


class EpochDataAggregator:
    def __init__(self, workflow_id):
        self.best_avg_loss = 10 ** 10
        self.best_avg_metric = -10 ** 10
        self.workflow_id = workflow_id

    def __enter__(self):
        self.epoch_data = {}

        return self

    def __call__(self, output_dictionary):
        output_dictionary = convert_output_dict_to_cpu(output_dictionary)

        for typ in ['losses', 'metrics']:
            for i in range(len(output_dictionary[typ])):
                for key in output_dictionary[typ][i].keys():
                    output_dictionary[typ][i][key] = [np.mean(output_dictionary[typ][i][key])]

        if len(self.epoch_data.keys()) == 0:
            self.epoch_data = output_dictionary
            return

        for typ in ['losses', 'metrics']:
            for i in range(len(output_dictionary[typ])):
                for key in output_dictionary[typ][i].keys():
                    self.epoch_data[typ][i][key].append(output_dictionary[typ][i][key])

        for typ in ['inputs', 'outputs']:
            for key in output_dictionary[typ].keys():
                self.epoch_data[typ][key] = output_dictionary[typ][key]

    def __exit__(self, *args, **kwargs):
        for typ in ['losses', 'metrics']:
            for i in range(len(self.epoch_data[typ])):
                for key in self.epoch_data[typ][i].keys():
                    self.epoch_data[typ][i][key] = np.mean(self.epoch_data['losses'][i][key])

        all_losses = []
        for dct in self.epoch_data['losses']:
            for key in dct.keys():
                all_losses.append(dct[key])

        if len(all_losses) > 0:
            avg_all_losses = np.mean(all_losses)

            if avg_all_losses <= self.best_avg_loss:
                self.best_avg_loss = avg_all_losses
                dispatcher.send(message=self.epoch_data, signal=EISEN_BEST_MODEL_LOSS, sender=self.workflow_id)

        all_metrics = []
        for dct in self.epoch_data['metrics']:
            for key in dct.keys():
                all_metrics.append(dct[key])

        if len(all_metrics) > 0:
            avg_all_metrics = np.mean(all_metrics)

            if avg_all_metrics >= self.best_avg_metric:
                self.best_avg_metric = avg_all_metrics
                dispatcher.send(message=self.epoch_data, signal=EISEN_BEST_MODEL_METRIC, sender=self.workflow_id)


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
