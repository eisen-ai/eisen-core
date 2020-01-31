import logging
import torch

from eisen import (
    EISEN_VALIDATION_SENDER,
    EISEN_END_EPOCH_EVENT,
    EISEN_END_BATCH_EVENT,
)
from eisen.utils import merge_two_dicts
from eisen.utils.workflows import GenericWorkflow

from torch import Tensor

from pydispatch import dispatcher


class Validation(GenericWorkflow):
    def __init__(self, model, data_loader, losses, metrics=None, gpu=False, data_parallel=False):
        """
        :param model:
        :type model: torch.nn.Module
        :param data_loader:
        :type data_loader: torch.utils.data.DataLoader
        :param losses:
        :type losses: list
        :param metrics:
        :type metrics: list
        :param gpu:
        :type gpu: bool
        :param data_parallel:
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

        if self.gpu:
            self.model.cuda()

        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def process_batch(self, batch):
        model_argument_dict = {key: batch[key] for key in self.model.input_names}

        outputs = self.model(**model_argument_dict)

        losses = self.compute_losses(merge_two_dicts(batch, outputs))

        metrics = self.compute_metrics(merge_two_dicts(batch, outputs))

        output_dictionary = {
            'inputs': batch,
            'outputs': outputs,
            'losses': losses,
            'metrics': metrics,
        }

        return output_dictionary

    def run(self):
        logging.info('INFO: Validation epoch {}'.format(self.epoch))

        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                if self.gpu:
                    for key in batch.keys():
                        if isinstance(batch[key], Tensor):
                            batch[key] = batch[key].cuda()

                logging.debug('DEBUG: Validation epoch {}, batch {}'.format(self.epoch, i))

                output_dictionary = self.process_batch(batch)

                dispatcher.send(message=output_dictionary, signal=EISEN_END_BATCH_EVENT, sender=EISEN_VALIDATION_SENDER)

            dispatcher.send(message=self.epoch, signal=EISEN_END_EPOCH_EVENT, sender=EISEN_VALIDATION_SENDER)

        self.epoch += 1
