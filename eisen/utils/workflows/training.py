import logging
import torch

from eisen import (
    EISEN_END_EPOCH_EVENT,
    EISEN_END_BATCH_EVENT
)
from eisen.utils import merge_two_dicts
from eisen.utils.workflows.workflows import GenericWorkflow, EpochDataAggregator

from torch import Tensor
from torch.cuda.amp import GradScaler #, autocast

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
    def __init__(self, model, data_loader, losses, optimizer, metrics=None, gpu=True):
        """
        A training workflow is usually employed to train models. Takes as input, in addition to a model, instance of
        torch.nn.Module, a data loader instance of class torch.utils.data.DataLoader, a list of losses, a torch
        optimizer, a list of metrics and a flag indicating whether the GPU should be used for computation or not.

        An example is shown here:

        .. code-block:: python

            workflow = Training(model, data_loader, [loss1, loss2, loss3], torch.optim.Adam(lr=0.001), [], gpu=True)

        Once it is instantiated it can be run with .run() on the data provided by the data_loader (for an entire epoch)
        and it can be also called on data batches in order to obtain results.

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

        <json>
        [
            {"name": "gpu", "type": "bool", "value": "true"}
        ]
        </json>
        """

        super(Training, self).__init__(model, gpu)

        self.losses = losses
        self.metrics = metrics

        self.optimizer = optimizer

        self.data_loader = data_loader

        self.epoch = 0

        self.epoch_aggregator = EpochDataAggregator(self.id)

    def get_output_dictionary(self, batch):
        """
        Calls the class on the batch and converts output tuple to an output dictionary.

        :param batch: a dictionary containing a batch of data (as per Eisen specifications)
        :type batch: dict

        :return: output dictionary
        """
        outputs, losses, metrics = super(Training, self).__call__(batch)

        output_dictionary = {
            'inputs': batch,
            'outputs': outputs,
            'losses': losses,
            'metrics': metrics,
            'epoch': self.epoch,
            'model': self.model,
        }

        return output_dictionary

    def run(self):
        """
        Runs an entire Training epoch on the data provided by the DataLoader passed as argument during initialization.

        PyDispatch events will be generated as a result of the computation so that model saving, summaries and logs
        can be generated via associated hooks.

        :return: None
        """
        logging.info('INFO: Training epoch {}'.format(self.epoch))

        self.model.train()

        with self.epoch_aggregator as ea:
            for i, batch in enumerate(self.data_loader):
                if self.gpu:
                    for key in batch.keys():
                        if isinstance(batch[key], Tensor):
                            batch[key] = batch[key].cuda()

                logging.debug('DEBUG: Training epoch {}, batch {}'.format(self.epoch, i))

                output_dictionary = self.get_output_dictionary(batch)

                dispatcher.send(
                    message=output_dictionary,
                    signal=EISEN_END_BATCH_EVENT,
                    sender=self.id
                )

                ea(output_dictionary)

        dispatcher.send(
            message=ea.epoch_data,
            signal=EISEN_END_EPOCH_EVENT,
            sender=self.id
        )

        self.epoch += 1


class TrainingApexAMP(Training):
    """
    This training workflow is able to take advantage of automatic mixed precision mechanism implemented in APEX
    (https://github.com/NVIDIA/apex). This offers significant training speed up thanks to the utilization of tensor
    cores on NVIDIA GPUs. When using TrainingApexAMP, GPU support must be present (GPU computation enabled by default),
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
        :param opt_level: Level of optimization for Apex
        :type opt_level: str

        <json>
        [
            {"name": "opt_level", "type": "string", "value": "O1"}
        ]
        </json>
        """
        super(TrainingApexAMP, self).__init__(model, data_loader, losses, optimizer, metrics, gpu=True)

        self.opt_level = opt_level

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.opt_level)

    def get_output_dictionary(self, batch):
        """
        Calls the class on the batch and converts output tuple to an output dictionary.
        With Automatic Mixed Precision (AMP) as implemented by AMP.

        :param batch: Data batch to be processed by the
        :type batch: dict

        :return: output_dictionary
        """
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
            'epoch': self.epoch,
            'model': self.model,
        }

        return output_dictionary


class TrainingAMP(Training):
    """
    This training workflow is able to take advantage of automatic mixed precision mechanism implemented natively
    by PyTorch. This requirese PyTorch >= 1.5.0. AMP offers significant training speed up thanks to the utilization
    of tensor cores on NVIDIA GPUs. When using TrainingAMP, GPU support must be present
    (GPU computation enabled by default) and the GPU must be capable of mixed precision computation.
    """
    def __init__(
            self,
            model,
            data_loader,
            losses,
            optimizer,
            metrics=None
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

        <json>
        []
        </json>
        """
        super(TrainingAMP, self).__init__(model, data_loader, losses, optimizer, metrics, gpu=True)

        self.scaler = GradScaler()

    def get_output_dictionary(self, batch):
        """
        Calls the class on the batch and converts output tuple to an output dictionary.
        With Automatic Mixed Precision (AMP) as implemented by PyTorch.

        :param batch: Data batch to be processed by the
        :type batch: dict

        :return: output_dictionary
        """
        model_argument_dict = {key: batch[key] for key in self.model.input_names}

        self.optimizer.zero_grad()

        #with autocast():
        outputs = self.model(**model_argument_dict)
        losses = self.compute_losses(merge_two_dicts(batch, outputs))

        for loss in losses:
            for key in loss.keys():
                self.scaler.scale(loss[key]).backward(retain_graph=True)

        self.scaler.step(self.optimizer)

        self.scaler.update()

        metrics = self.compute_metrics(merge_two_dicts(batch, outputs))

        output_dictionary = {
            'inputs': batch,
            'outputs': outputs,
            'losses': losses,
            'metrics': metrics,
            'epoch': self.epoch,
            'model': self.model,
        }

        return output_dictionary




