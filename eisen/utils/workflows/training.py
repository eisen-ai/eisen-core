import logging

from eisen.utils.context import OptimizationContext
from eisen.utils import check_arg_type
from eisen import (
    EISEN_TRAINING_SENDER,
    EISEN_END_EPOCH_EVENT,
    EISEN_END_BATCH_EVENT,
)

from torch.utils.data import DataLoader
from torch.nn import Module

from pydispatch import dispatcher


class Training:
    def __init__(self, model, context, data_loader):
        """
        :param model:
        :type model: torch.nn.Module
        :param context:
        :type context: object
        :param data_loader:
        :type data_loader: torch.utils.data.DataLoader
        """
        check_arg_type(model, Module, 'model')
        check_arg_type(context, OptimizationContext, 'context')
        check_arg_type(data_loader, DataLoader, 'data_loader')

        self.model = model
        self.context = context
        self.data_loader = data_loader

        self.epoch = 0

    def run(self):
        logging.info('INFO: Training epoch {}'.format(self.epoch))

        self.model.train()

        for i, batch in enumerate(self.data_loader):
            logging.debug('DEBUG: Training epoch {}, batch {}'.format(self.epoch, i))

            output_dictionary = self.context(self.model, batch)

            dispatcher.send(message=output_dictionary, signal=EISEN_END_BATCH_EVENT, sender=EISEN_TRAINING_SENDER)

        dispatcher.send(message=self.epoch, signal=EISEN_END_EPOCH_EVENT, sender=EISEN_TRAINING_SENDER)

        self.epoch += 1


class DataParallelTraining:
    pass


class DistributedDataParallelTraining:
    pass