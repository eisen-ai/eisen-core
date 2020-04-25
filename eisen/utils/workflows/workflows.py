import numpy as np
import torch
import uuid

from eisen import EISEN_BEST_MODEL_LOSS, EISEN_BEST_MODEL_METRIC
from eisen.utils import merge_two_dicts
from pydispatch import dispatcher


def convert_output_dict_to_cpu(output_dict):
    for typ in ['losses', 'metrics']:
        for i in range(len(output_dict[typ])):
            for key in list(output_dict[typ][i].keys()):
                if isinstance(output_dict[typ][i][key], torch.Tensor):
                    output_dict[typ][i][key] = output_dict[typ][i][key].cpu().data.numpy()
                elif isinstance(output_dict[typ][key], np.ndarray):
                    pass
                else:
                    output_dict[typ][i].pop(key, None)

    for typ in ['inputs', 'outputs']:
        for key in list(output_dict[typ].keys()):
            if isinstance(output_dict[typ][key], torch.Tensor):
                output_dict[typ][key] = output_dict[typ][key].cpu().data.numpy()
            elif isinstance(output_dict[typ][key], np.ndarray):
                pass
            else:
                output_dict[typ].pop(key, None)

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

        self.epoch_data['epoch'] = output_dictionary['epoch']
        self.epoch_data['model'] = output_dictionary['model']

        for typ in ['losses', 'metrics']:
            if typ not in self.epoch_data.keys():
                self.epoch_data[typ] = [{}] * len(output_dictionary[typ])

            for i in range(len(output_dictionary[typ])):
                for key in output_dictionary[typ][i].keys():
                    try:
                        data = output_dictionary[typ][i][key]
                        if isinstance(data, np.ndarray):
                            if key not in self.epoch_data[typ][i].keys():
                                self.epoch_data[typ][i][key] = [data]
                            else:
                                self.epoch_data[typ][i][key].append(data)
                    except KeyError:
                        pass

        for typ in ['inputs', 'outputs']:
            if typ not in self.epoch_data.keys():
                self.epoch_data[typ] = {}

            for key in output_dictionary[typ].keys():
                try:
                    data = output_dictionary[typ][key]

                    if isinstance(data, np.ndarray):
                        if key not in self.epoch_data[typ].keys():
                            self.epoch_data[typ][key] = data

                        else:
                            # if data is NOT high dimensional (Eg. it is a vector) we save all of it (throughout the epoch)
                            # the behaviour we want to have is that classification data (for example) can be saved for the
                            # whole epoch instead of only one batch
                            if output_dictionary[typ][key].ndim == 0:
                                output_dictionary[typ][key] = output_dictionary[typ][key][np.newaxis]

                            if output_dictionary[typ][key].ndim == 1:
                                self.epoch_data[typ][key] = \
                                    np.concatenate([self.epoch_data[typ][key], output_dictionary[typ][key]], axis=0)
                            else:
                                # we do not save high dimensional data throughout the epoch, we just save the last batch
                                # the behaviour in this case is to save images and volumes only for the last batch of the epoch
                                self.epoch_data[typ][key] = output_dictionary[typ][key]
                except KeyError:
                    pass

    def __exit__(self, *args, **kwargs):
        for typ in ['losses', 'metrics']:
            for i in range(len(self.epoch_data[typ])):
                for key in self.epoch_data[typ][i].keys():
                    self.epoch_data[typ][i][key] = np.asarray(self.epoch_data[typ][i][key])

        all_losses = []
        for dct in self.epoch_data['losses']:
            for key in dct.keys():
                all_losses.append(np.mean(dct[key]))

        if len(all_losses) > 0:
            avg_all_losses = np.mean(all_losses)

            if avg_all_losses <= self.best_avg_loss:
                self.best_avg_loss = avg_all_losses
                dispatcher.send(message=self.epoch_data, signal=EISEN_BEST_MODEL_LOSS, sender=self.workflow_id)

        all_metrics = []
        for dct in self.epoch_data['metrics']:
            for key in dct.keys():
                all_metrics.append(np.mean(dct[key]))

        if len(all_metrics) > 0:
            avg_all_metrics = np.mean(all_metrics)

            if avg_all_metrics >= self.best_avg_metric:
                self.best_avg_metric = avg_all_metrics
                dispatcher.send(message=self.epoch_data, signal=EISEN_BEST_MODEL_METRIC, sender=self.workflow_id)


class GenericWorkflow:
    """
    The generic workflow implements basic workflow functionality and serves as base class for more specific workflow
    such as those used for training, testing and validation.
    """
    def __init__(self, model, gpu=True):
        """
        :param model: The model to be used for training. This model instance will be optimized by the Training module.
        :type model: torch.nn.Module
        :param gpu: A flag indicating whether GPUs should be used during training
        :type gpu: bool

        """
        self.model = model

        self.losses = []
        self.metrics = []
        self.optimizer = None

        self.gpu = gpu

        if self.gpu and not next(self.model.parameters()).is_cuda:
            self.model.cuda()

        self.id = uuid.uuid4()

    def __call__(self, batch):
        """
        A workflow is callable on a batches of data. Eisen batches are dictionaries containing (at least)
        fields named after the input names contained in model.input_names

        :param batch: Data batch to be processed by the
        :type batch: dict

        :return: tuple containing outputs, losses and metrics

        """
        model_argument_dict = {key: batch[key] for key in self.model.input_names}

        self.optimizer.zero_grad()

        outputs = self.model(**model_argument_dict)

        losses = self.compute_losses(merge_two_dicts(batch, outputs))

        for loss in losses:
            for key in loss.keys():
                loss[key].backward(retain_graph=True)

        if self.optimizer is not None:
            self.optimizer.step()

        metrics = self.compute_metrics(merge_two_dicts(batch, outputs))

        return outputs, losses, metrics

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
