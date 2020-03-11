import logging
import torch
import uuid

from eisen import (
    EISEN_END_EPOCH_EVENT,
    EISEN_END_BATCH_EVENT,
)
from eisen.utils import merge_two_dicts
from eisen.utils.workflows.workflows import GenericWorkflow, EpochDataAggregator

from torch import Tensor

from pydispatch import dispatcher


class Validation(GenericWorkflow):
    """
    This object implements a Validation workflow which allows to validate a model by running the forward pass on a
    specific dataset (passed as argument to init). This Validation workflow can take advantage of GPUs and implements
    data parallelism which allows workload to be distributed across multiple processors (CPU/GPU).
    """
    def __init__(self, model, data_loader, losses, metrics=None, gpu=False, data_parallel=False):
        """
        :param model: The model to be used for validation. This model instance will be used only for forward passes.
        :type model: torch.nn.Module
        :param data_loader: A DataLoader instance which handles the data loading and batching
        :type data_loader: torch.utils.data.DataLoader
        :param losses: A list of losses objects to be evaluated on the validation set
        :type losses: list of torch.nn.Modules
        :param metrics: A list of metrics objects to be evaluated during validation
        :type metrics: list
        :param gpu: A flag indicating whether GPUs should be used during validation
        :type gpu: bool
        :param data_parallel: A flag indicating whether the network should be data parallel (torch.nn.DataParallel)
        :type data_parallel: bool

        <json>
        [
            {"name": "gpu", "type": "bool", "value": "false"},
            {"name": "data_parallel", "type": "bool", "value": "false"}
        ]
        </json>
        """

        self.model = model
        self.data_loader = data_loader
        self.losses = losses
        self.metrics = metrics

        self.gpu = gpu
        self.data_parallel = data_parallel

        self.epoch = 0

        if self.gpu:  # todo check if already gpu
            self.model.cuda()

        if self.data_parallel:  # todo check if already data parallel
            self.model = torch.nn.DataParallel(self.model)

        self.id = uuid.uuid4()

        self.epoch_aggregator = EpochDataAggregator(self.id)

    def process_batch(self, batch):
        model_argument_dict = {key: batch[key] for key in self.model.input_names}

        outputs = self.model(**model_argument_dict)

        losses = self.compute_losses(merge_two_dicts(batch, outputs))

        metrics = self.compute_metrics(merge_two_dicts(batch, outputs))

        output_dictionary = {
            'inputs': model_argument_dict,
            'outputs': outputs,
            'losses': losses,
            'metrics': metrics,
            'model': self.model,
            'epoch': self.epoch,
        }

        return output_dictionary

    def run(self):
        logging.info('INFO: Validation epoch {}'.format(self.epoch))

        self.model.eval()

        with self.epoch_aggregator as ea:
            with torch.no_grad():
                for i, batch in enumerate(self.data_loader):
                    if self.gpu:
                        for key in batch.keys():
                            if isinstance(batch[key], Tensor):
                                batch[key] = batch[key].cuda()

                    logging.debug('DEBUG: Validation epoch {}, batch {}'.format(self.epoch, i))

                    output_dictionary = self.process_batch(batch)

                    ea(output_dictionary)

        dispatcher.send(message=ea.epoch_data, signal=EISEN_END_EPOCH_EVENT, sender=self.id)

        self.epoch += 1
