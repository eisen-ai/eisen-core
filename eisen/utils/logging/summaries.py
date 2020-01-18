import numpy as np

from eisen import (
    EISEN_END_BATCH_EVENT,
    EISEN_END_EPOCH_EVENT,
    EISEN_TRAINING_SENDER,
    EISEN_VALIDATION_SENDER
)

from torch.utils.tensorboard.writer import SummaryWriter
from pydispatch import dispatcher


class AutoSummaryManager:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)


    def _write_volumetric_image(self, name, value, global_step):
        pass

    def _write_2D_image(self, name, value, global_step):
        self.writer.add_scalar('/mean/' + name, np.mean(value), global_step=global_step)
        self.writer.add_scalar('/std/' + name, np.std(value), global_step=global_step)
        self.writer.add_histogram('/histogram/' + name, value.flatten(), global_step=global_step)
        self.writer.add_images(name, value, global_step=global_step, dataformats='NCHW')

    def _write_embedding(self, name, value, global_step):
        pass

    def _write_class_probabilities(self, name, value, global_step):
        self.writer.add_image(name, value, global_step=global_step, dataformats='HW')
        self.writer.add_histogram('/distribution/' + name, np.argmax(value), global_step=global_step)
        # todo need to add confusion matrix

    def _write_vector(self, name, value, global_step):
        self.writer.add_histogram(name, value, global_step=global_step)
        self.writer.add_scalar('/mean/' + name, np.mean(value), global_step=global_step)
        self.writer.add_scalar('/std/' + name, np.std(value), global_step=global_step)
        pass

    def _write_scalar(self, name, value, global_step):
        self.writer.add_scalar(name, value, global_step=global_step)

    def write(self, data):
        if data.ndims == 5:
            # Volumetric image (N, C, W, H, D)
            pass

        if data.ndims == 4:
            # 2D image (N, C, W, H)
            pass

        if data.ndims == 3:
            # embedding (N, C, W)
            pass

        if data.ndims == 2:
            # classes probabilities (N, C)
            pass

        if data.ndims == 1:
            # vector of numbers (N)
            pass

        if data.ndims == 0:
            # scalar ()
            pass


class SummaryHook:
    def __init__(self, logs_base_dir):
        # training signals
        dispatcher.connect(self.end_training_batch, signal=EISEN_END_BATCH_EVENT, sender=EISEN_TRAINING_SENDER)
        dispatcher.connect(self.end_training_epoch, signal=EISEN_END_EPOCH_EVENT, sender=EISEN_TRAINING_SENDER)

        # validation signals
        dispatcher.connect(self.end_validation_batch, signal=EISEN_END_BATCH_EVENT, sender=EISEN_VALIDATION_SENDER)
        dispatcher.connect(self.end_validation_epoch, signal=EISEN_END_EPOCH_EVENT, sender=EISEN_VALIDATION_SENDER)

    def end_training_batch(self, message):
        pass

    def end_training_epoch(self, message):
        pass

    def end_validation_batch(self, message):
        pass

    def end_validation_epoch(self, message):
        pass

