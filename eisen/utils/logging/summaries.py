import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import torch

from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard.writer import SummaryWriter
from pydispatch import dispatcher

from eisen import (
    EISEN_END_EPOCH_EVENT
)


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, np.around(cm[i, j], decimals=2),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=15
        )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig = plt.gcf()

    fig.set_dpi(72)
    fig.set_size_inches(int(len(classes) * 1.0) + 3, int(len(classes) * 1.0))

    fig.canvas.draw()

    image = np.array(fig.canvas.renderer._renderer)

    plt.close()

    return image


class TensorboardSummaryHook:
    """
    Logging object allowing Tensorboard summaries to be automatically exported to the tensorboard. Much of its
    functionality is automated. This means that the hook will export as much information as possible to the
    tensorboard.

    Losses, Metrics, Inputs and Outputs are all interpreted and exported according to their dimensionality. Vectors
    results in mean and standard deviation estimates as well as histograms; Pictures results in image summaries and
    histograms; etc.

    There is also the possibily of comparing inputs and outputs pair. This needs to be specified during object
    instantiation.

    Once the user instantiates this object, the workflow corresponding to the ID passes as argument will be
    tracked and the results of the workflow will be exported to the tensorboard.

    .. code-block:: python

            from eisen.utils.logging import TensorboardSummaryHook

            workflow = # Eg. An instance of Training workflow

            logger = TensorboardSummaryHook(workflow.id, 'Training', '/artifacts/dir')
    """
    def __init__(self, workflow_id, phase, artifacts_dir, comparison_pairs=None):
        """
        This method instantiates an object of type TensorboardSummaryHook. The signature of this method is similar to
        that of every other hook. There is one additional parameter called `comparison_pairs` which is meant to
        hold a list of lists each containing a pair of input/output names that share the same dimensionality and can be
        compared to each other.

        A typical use of `comparison_pairs` is when users want to plot a pr_curve or a confusion matrix by comparing
        some input with some output. Eg. by comparing the labels with the predictions.

        .. code-block:: python

            from eisen.utils.logging import TensorboardSummaryHook

            workflow = # Eg. An instance of Training workflow

            logger = TensorboardSummaryHook(
                workflow_id=workflow.id,
                phase='Training',
                artifacts_dir='/artifacts/dir'
                comparison_pairs=[['labels', 'predictions']]
            )

        :param workflow_id: string containing the workflow id of the workflow being monitored (workflow_instance.id)
        :type workflow_id: UUID
        :param phase: string containing the name of the phase (training, testing, ...) of the workflow monitored
        :type phase: str
        :param artifacts_dir: whether the history of all models that were at a certain point the best should be saved
        :type artifacts_dir: bool
        :param comparison_pairs: list of lists of pairs, which are names of inputs and outputs to be compared directly
        :type comparison_pairs: list of lists of strings

        <json>
        [
            {"name": "comparison_pairs", "type": "list:list:string", "value": ""}
        ]
        </json>
        """
        self.workflow_id = workflow_id
        self.phase = phase

        self.comparison_pairs = comparison_pairs

        if not os.path.exists(artifacts_dir):
            raise ValueError('The directory specified to save artifacts does not exist!')

        dispatcher.connect(self.end_epoch, signal=EISEN_END_EPOCH_EVENT, sender=workflow_id)

        self.artifacts_dir = os.path.join(artifacts_dir, 'summaries', phase)

        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)

        self.writer = SummaryWriter(log_dir=self.artifacts_dir)

    def end_epoch(self, message):
        epoch = message['epoch']

        # if epoch == 0:
        #     self.writer.add_graph(message['model'], ...)

        for typ in ['losses', 'metrics']:
            for dct in message[typ]:
                for key in dct.keys():
                    self.write_vector(typ + '/{}'.format(key), dct[key], epoch)

        for typ in ['inputs', 'outputs']:
            for key in message[typ].keys():
                if message[typ][key].ndim == 5:
                    # Volumetric image (N, C, W, H, D)
                    self.write_volumetric_image(typ + '/{}'.format(key), message[typ][key], epoch)

                if message[typ][key].ndim == 4:
                    self.write_image(typ + '/{}'.format(key), message[typ][key], epoch)

                if message[typ][key].ndim == 3:
                    self.write_embedding(typ + '/{}'.format(key), message[typ][key], epoch)

                if message[typ][key].ndim == 2:
                    self.write_class_probabilities(typ + '/{}'.format(key), message[typ][key], epoch)

                if message[typ][key].ndim == 1:
                    self.write_vector(typ + '/{}'.format(key), message[typ][key], epoch)

                if message[typ][key].ndim == 0:
                    self.write_scalar(typ + '/{}'.format(key), message[typ][key], epoch)

        if self.comparison_pairs:
            for inp, out in self.comparison_pairs:
                assert message['inputs'][inp].ndim == message['outputs'][out].ndim

                if message['inputs'][inp].ndim == 1:
                    # in case of binary classification >> PR curve
                    if np.max(message['inputs'][inp]) <= 1 and np.max(message['outputs'][out]) <= 1:
                        self.write_pr_curve(
                            '{}_Vs_{}/pr_curve'.format(inp, out),
                            message['inputs'][inp],
                            message['outputs'][out],
                            epoch
                        )

                    # in any case for classification >> Confusion Matrix
                    self.write_confusion_matrix(
                        '{}_Vs_{}/confusion_matrix'.format(inp, out),
                        message['inputs'][inp],
                        message['outputs'][out],
                        epoch
                    )

    def write_volumetric_image(self, name, value, global_step):
        value = np.transpose(value, [0, 2, 1, 3, 4])

        if value.shape[2] != 3 and value.shape[2] != 1:
            value = np.sum(value, axis=2, keepdims=True)

        torch_value = torch.tensor(value).float()

        self.writer.add_video(name, torch_value, fps=10, global_step=global_step)
        self.writer.add_scalar(name + '/mean', np.mean(value), global_step=global_step)
        self.writer.add_scalar(name + '/std', np.std(value), global_step=global_step)
        self.writer.add_histogram(name + '/histogram', value.flatten(), global_step=global_step)

    def write_image(self, name, value, global_step):
        self.writer.add_scalar(name + '/mean', np.mean(value), global_step=global_step)
        self.writer.add_scalar(name + '/std', np.std(value), global_step=global_step)
        self.writer.add_histogram(name + '/histogram', value.flatten(), global_step=global_step)
        self.writer.add_images(name, value, global_step=global_step, dataformats='NCHW')

    def write_embedding(self, name, value, global_step):
        pass

    def write_pr_curve(self, name, labels, predictions, global_step):
        self.writer.add_pr_curve(name + '/pr_curve', labels, predictions, global_step)

    def write_confusion_matrix(self, name, labels, predictions, global_step):
        cnf_matrix = confusion_matrix(labels, predictions)
        image = plot_confusion_matrix(cnf_matrix, range(np.max(labels) + 1), normalize=True, title=name)[:, :, 0:3]
        self.writer.add_image(name, image.astype(float)/255.0, global_step=global_step, dataformats='HWC')

    def write_class_probabilities(self, name, value, global_step):
        self.writer.add_image(name, value, global_step=global_step, dataformats='HW')
        self.writer.add_histogram(name + '/distribution', np.argmax(value), global_step=global_step)

    def write_vector(self, name, value, global_step):
        self.writer.add_histogram(name, value, global_step=global_step)
        self.writer.add_scalar(name + '/mean', np.mean(value), global_step=global_step)
        self.writer.add_scalar(name + '/std', np.std(value), global_step=global_step)

    def write_scalar(self, name, value, global_step):
        self.writer.add_scalar(name, value, global_step=global_step)
