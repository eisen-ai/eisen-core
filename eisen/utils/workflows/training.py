import logging
import torch

from eisen import (
    EISEN_TRAINING_SENDER,
    EISEN_END_EPOCH_EVENT,
    EISEN_END_BATCH_EVENT,
)
from eisen.utils import merge_two_dicts
from eisen.utils.workflows.workflows import GenericWorkflow

from torch import Tensor

from pydispatch import dispatcher

try:
    from apex import amp
except ImportError:
    logging.info('Eisen could benefit from NVIDIA Apex (https://github.com/NVIDIA/apex), which is not installed.')


class Training(GenericWorkflow):
    """
    This training workflow implements training of a model on a specific dataset with a specific set of losses passed
    as an argument. There is support for both GPU computation and data parallelism (multi GPU/CPU training) via
    PyTorch. For what concerns GPU utilization, PyTorch must be installed with GPU support. For what concerns
    data parallelism, it is implemented using torch.nn.DataParallel which might not be the most efficient solution
    but it is definitely the easiest solution to use and implement.
    """
    def __init__(self, model, data_loader, losses, optimizer, metrics=None, gpu=False, data_parallel=False):
        """
        :param model: The model to be used for training. This model instance will be optimized by the Training module.
        :type model: torch.nn.Module
        :param data_loader: A DataLoader instance which handles the data loading and batching
        :type data_loader: torch.utils.data.DataLoader
        :param losses: A list of losses objects to be optimized
        :type losses: list of torch.nn.Modules
        :param optimizer: An optimizer object such as torch.optim.Adam
        :type optimizer: torch.optim.optimizer
        :param metrics: A list of metrics objects to be evaluated during training (similar to losses, but not optimized)
        :type metrics: list
        :param gpu: A flag indicating whether GPUs should be used during training
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
        self.optimizer = optimizer
        self.metrics = metrics

        self.gpu = gpu
        self.data_parallel = data_parallel

        self.epoch = 0

        if self.gpu:  # todo check if already gpu
            self.model.cuda()

        if self.data_parallel:  # todo check if already data parallel
            self.model = torch.nn.DataParallel(self.model)

    def process_batch(self, batch):
        model_argument_dict = {key: batch[key] for key in self.model.input_names}

        self.optimizer.zero_grad()

        outputs = self.model(**model_argument_dict)

        losses = self.compute_losses(merge_two_dicts(batch, outputs))

        for loss in losses:
            for key in loss.keys():
                loss[key].backward(retain_graph=True)

        self.optimizer.step()

        metrics = self.compute_metrics(merge_two_dicts(batch, outputs))

        output_dictionary = {
            'inputs': batch,
            'outputs': outputs,
            'losses': losses,
            'metrics': metrics,
        }

        return output_dictionary

    def run(self):
        logging.info('INFO: Training epoch {}'.format(self.epoch))

        self.model.train()

        for i, batch in enumerate(self.data_loader):
            if self.gpu:
                for key in batch.keys():
                    if isinstance(batch[key], Tensor):
                        batch[key] = batch[key].cuda()

            logging.debug('DEBUG: Training epoch {}, batch {}'.format(self.epoch, i))

            output_dictionary = self.process_batch(batch)

            dispatcher.send(message=output_dictionary, signal=EISEN_END_BATCH_EVENT, sender=EISEN_TRAINING_SENDER)

        dispatcher.send(message=self.epoch, signal=EISEN_END_EPOCH_EVENT, sender=EISEN_TRAINING_SENDER)

        self.epoch += 1


class TrainingAMP(Training):
    """
    This training workflow is able to take advantage of automatic mixed precision mechanism implemented in APEX
    (https://github.com/NVIDIA/apex). This offers significant training speed up thanks to the utilization of tensor
    cores on NVIDIA GPUs. When using TrainingAMP, GPU support must be present (GPU computation enabled by default),
    APEX must be installed in the system (APEX is not part of Eisen requirements) and the GPU must be capable of
    mixed precision computation.
    """
    def __init__(
            self,
            model,
            data_loader,
            losses,
            optimizer,
            metrics=None,
            data_parallel=False,
            opt_level='O1'
    ):
        """
        :param model: The model to be used for training. This model instance will be optimized by the Training module.
        :type model: torch.nn.Module
        :param data_loader: A DataLoader instance which handles the data loading and batching
        :type data_loader: torch.utils.data.DataLoader
        :param losses: A list of losses objects to be optimized
        :type losses: list of torch.nn.Modules
        :param optimizer: An optimizer object such as torch.optim.Adam
        :type optimizer: torch.optim.optimizer
        :param metrics: A list of metrics objects to be evaluated during training (similar to losses, but not optimized)
        :type metrics: list
        :param data_parallel: A flag indicating whether the network should be data parallel (torch.nn.DataParallel)
        :type data_parallel: bool
        :param opt_level: Level of optimization for Apex
        :type opt_level: str

        <json>
        [
            {"name": "data_parallel", "type": "bool", "value": "false"},
            {"name": "opt_level", "type": "string", "value": "O1"}
        ]
        </json>
        """
        super(TrainingAMP, self).__init__(model, data_loader, losses, optimizer, metrics, True, False)

        self.data_parallel = data_parallel
        self.opt_level = opt_level

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.opt_level)

        if self.data_parallel:  # todo check if already data parallel
            self.model = torch.nn.DataParallel(self.model)

    def process_batch(self, batch):
        model_argument_dict = {key: batch[key] for key in self.model.input_names}

        self.optimizer.zero_grad()

        outputs = self.model(**model_argument_dict)

        losses = self.compute_losses(merge_two_dicts(batch, outputs))

        for loss in losses:
            for key in loss.keys():
                with amp.scale_loss(loss[key], self.optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)

        self.optimizer.step()

        metrics = self.compute_metrics(merge_two_dicts(batch, outputs))

        output_dictionary = {
            'inputs': batch,
            'outputs': outputs,
            'losses': losses,
            'metrics': metrics,
        }

        return output_dictionary
