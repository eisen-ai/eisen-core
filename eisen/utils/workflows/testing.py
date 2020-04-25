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


class Testing(GenericWorkflow):
    """
    This object implements a testing workflow, which allows to test a model on a certain dataset by running only the
    forward pass of the model. The user is allowed to specify model, data loader and metrics to use for evaluation.
    This workflow supports GPUs and data parallelism across multiple processors.
    """
    def __init__(self, model, data_loader, metrics, gpu=False):
        """
        :param model: The model to be used for testing. This model instance will be used only for forward passes.
        :type model: torch.nn.Module
        :param data_loader: A DataLoader instance which handles the data loading and batching
        :type data_loader: torch.utils.data.DataLoader
        :param metrics: A list of metrics objects to be evaluated during test
        :type metrics: list
        :param gpu: A flag indicating whether GPUs should be used during test
        :type gpu: bool

        <json>
        [
            {"name": "gpu", "type": "bool", "value": "false"}
        ]
        </json>
        """

        super(Testing, self).__init__(model, gpu)

        self.data_loader = data_loader
        self.metrics = metrics

        self.epoch_aggregator = EpochDataAggregator(self.id)

    def get_output_dictionary(self, batch):
        """
        Calls the class on the batch and converts output tuple to an output dictionary.

        :param batch: a dictionary containing a batch of data (as per Eisen specifications)
        :type batch: dict

        :return: output dictionary

        """

        outputs, losses, metrics = super(Testing, self).__call__(batch)

        output_dictionary = {
            'inputs': batch,
            'losses': losses,
            'outputs': outputs,
            'metrics': metrics,
            'model': self.model,
            'epoch': 0,
        }

        return output_dictionary

    def run(self):
        logging.info('INFO: Running Testing')

        self.model.eval()

        with self.epoch_aggregator as ea:
            with torch.no_grad():
                for i, batch in enumerate(self.data_loader):
                    if self.gpu:
                        for key in batch.keys():
                            if isinstance(batch[key], Tensor):
                                batch[key] = batch[key].cuda()

                    logging.debug('DEBUG: Testing epoch batch {}'.format(i))

                    output_dictionary = self.get_output_dictionary(batch)

                    dispatcher.send(
                        message=output_dictionary,
                        signal=EISEN_END_BATCH_EVENT,
                        sender=self.id
                    )

                    ea(output_dictionary)

        dispatcher.send(message=ea.epoch_data, signal=EISEN_END_EPOCH_EVENT, sender=self.id)
