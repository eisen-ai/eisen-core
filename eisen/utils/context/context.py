import inspect

from torch.nn import Module
from torch.optim import Optimizer
from eisen.utils import merge_two_dicts, check_arg_type


class OptimizationContext:
    def __init__(self, losses, optimizer, metrics=None):
        """
        :param losses: Dictionary of losses that need to be optimized
        :type losses: list
        :param optimizer:
        :type optimizer: torch.nn.Optimizer
        :param metrics: Dictionary of metrics that need to be optmized
        :type metrics: list
        """
        if not metrics:
            metrics = []

        check_arg_type(losses, list, 'losses')
        check_arg_type(metrics, list, 'metrics')

        for loss in losses:
            check_arg_type(loss, Module, 'loss {}'.format(loss.__class__.__name__))

        for metric in metrics:
            check_arg_type(metric, Module, 'metric {}'.format(metric.__class__.__name__))

        check_arg_type(optimizer, Optimizer, 'optimizer')

        self.losses = losses
        self.metrics = metrics
        self.optimizer = optimizer

    def compute_losses(self, arguments):
        """
        :param arguments: Arguments to compute losses
        :type arguments: dict
        :return: dictionary of results
        """
        results = []

        for loss in self.losses:
            loss_argument_list = inspect.getfullargspec(loss.forward)[0]

            loss_argument_list.remove('self')

            loss_argument_dict = {key: arguments[key] for key in loss_argument_list}

            results.append(loss(**loss_argument_dict))

        return results

    def compute_metrics(self, arguments):
        """
        :param arguments: Arguments to compute metrics
        :type arguments: dict
        :return: dictionary of results
        """
        results = {}

        for key in self.metrics.keys():
            metric_argument_list = inspect.getfullargspec(self.metrics[key].__call__)[0]

            metric_argument_dict = {key: arguments[key] for key in metric_argument_list}

            results[key] = self.metrics[key](**metric_argument_dict)

        return results

    def __call__(self, model, batch):
        """
        :param model: Model to operate optimization on
        :type model: torch.nn.Module
        :param batch: A dictionary containing one batch of data
        :type batch: dict
        :return: dictionary of results
        """
        model_argument_list = inspect.getfullargspec(model.forward)[0]

        model_argument_list.remove('self')

        model_argument_dict = {key: batch[key] for key in model_argument_list}

        self.optimizer.zero_grad()

        outputs = model(**model_argument_dict)

        losses = self.compute_losses(merge_two_dicts(batch, outputs))

        for loss in losses:
            loss.backward(retain_graph=True)

        metrics = self.compute_metrics(merge_two_dicts(batch, outputs))

        self.optimizer.step()

        output_dictionary = {
            'inputs': batch,
            'outputs': outputs,
            'losses': losses,
            'metrics': metrics,
        }

        return output_dictionary


class InferenceContext:
    def __init__(self, model, metrics=None):
        pass
