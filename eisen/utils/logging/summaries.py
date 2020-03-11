import numpy as np
import os

from eisen import (
    EISEN_END_EPOCH_EVENT
)

from torch.utils.tensorboard.writer import SummaryWriter
from pydispatch import dispatcher


class TensorboardSummaryHook:
    def __init__(self, workflow_id, phase, artifacts_dir):
        self.workflow_id = workflow_id
        self.phase = phase

        if not os.path.exists(artifacts_dir):
            raise ValueError('The directory specified to save artifacts does not exist!')

        dispatcher.connect(self.end_epoch, signal=EISEN_END_EPOCH_EVENT, sender=workflow_id)

        self.artifacts_dir = os.path.join(artifacts_dir, 'summaries', phase)

        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)

        self.writer = SummaryWriter(log_dir=self.artifacts_dir)

    def end_epoch(self, message):
        epoch = message['epoch']

        for typ in ['losses', 'metrics']:
            for dct in message[typ]:
                for key in dct.keys():
                    self.write_vector(typ + '/{}'.format(key), dct[key], epoch)

        for typ in ['inputs', 'outputs']:
            for key in message[typ].keys():
                print(message[typ][key].ndim)

                if message[typ][key].ndim == 5:
                    # Volumetric image (N, C, W, H, D)
                    pass

                if message[typ][key].ndim == 4:
                    self.write_2D_image(typ + '/{}'.format(key), message[typ][key], epoch)

                if message[typ][key].ndim == 3:
                    self.write_embedding(typ + '/{}'.format(key), message[typ][key], epoch)

                if message[typ][key].ndim == 2:
                    self.write_class_probabilities(typ + '/{}'.format(key), message[typ][key], epoch)

                if message[typ][key].ndim == 1:
                    self.write_vector(typ + '/{}'.format(key), message[typ][key], epoch)

                if message[typ][key].ndim == 0:
                    self.write_scalar(typ + '/{}'.format(key), message[typ][key], epoch)

    def write_volumetric_image(self, name, value, global_step):
        pass

    def write_2D_image(self, name, value, global_step):
        self.writer.add_scalar(name + '/mean', np.mean(value), global_step=global_step)
        self.writer.add_scalar(name + '/std', np.std(value), global_step=global_step)
        self.writer.add_histogram(name + '/histogram', value.flatten(), global_step=global_step)
        self.writer.add_images(name, value, global_step=global_step, dataformats='NCHW')

    def write_embedding(self, name, value, global_step):
        pass

    def write_class_probabilities(self, name, value, global_step):
        self.writer.add_image(name, value, global_step=global_step, dataformats='HW')
        self.writer.add_histogram(name + '/distribution', np.argmax(value), global_step=global_step)
        # todo need to add confusion matrix

    def write_vector(self, name, value, global_step):
        self.writer.add_histogram(name, value, global_step=global_step)
        self.writer.add_scalar(name + '/mean', np.mean(value), global_step=global_step)
        self.writer.add_scalar(name + '/std', np.std(value), global_step=global_step)

    def write_scalar(self, name, value, global_step):
        self.writer.add_scalar(name, value, global_step=global_step)