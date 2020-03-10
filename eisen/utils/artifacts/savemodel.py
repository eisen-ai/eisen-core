import torch
import os
import time

from pydispatch import dispatcher
from eisen import EISEN_BEST_MODEL_METRIC, EISEN_BEST_MODEL_LOSS


class SaveTorchModel:
    """
    This object saves a snapshot of the current model when either the best average of all the losses or the best
    average of all metrics is achieved.

    It can be very useful to attach this "hook" to a validation workflow, for example, and save the parameters of
    the model every time the validation phase terminates with a loss better than any previous loss.

    It is also possible to save the whole history of the model, so that the snapshot created throughout training are
    maintained as opposed as overwritten

    .. code-block:: python

        from eisen.utils.artifacts import SaveTorchModel

        workflow = # Eg. An instance of Validation workflow

        saver = SaveTorchModel(workflow.id, 'Validation', '/my/artifacts')
    """
    def __init__(self, workflow_id, phase, artifacts_dir, select_best_loss=True, save_history=False):
        """
        :param workflow_id: the ID of the workflow that should be tracked by this hook
        :type workflow_id: UUID
        :param phase: the phase where this hook is being used (Training, Testing, etc.)
        :type phase: str
        :param artifacts_dir: the path of the artifacts where the results of this hook should be stored
        :type artifacts_dir: str
        :param select_best_loss: whether the criterion for saving the model should be best loss or best metric
        :type select_best_loss: bool
        :param artifacts_dir: whether the history of all models that were at a certain point the best should be saved
        :type artifacts_dir: bool

        .. code-block:: python

            from eisen.utils.artifacts import SaveTorchModel

            workflow = # Eg. An instance of Validation workflow

            saver = SaveTorchModel(
                workflow_id=workflow.id,
                phase='Validation',
                artifacts_dir='/my/artifacts',
                select_best_loss=True,
                save_history=False
            )


        <json>
        [
            {"name": "select_best_loss", "type": "bool", "value": "True"},
            {"name": "save_history", "type": "bool", "value": "False"}
        ]
        </json>

        """
        if select_best_loss:
            dispatcher.connect(self.save_model, signal=EISEN_BEST_MODEL_LOSS, sender=workflow_id)
        else:
            dispatcher.connect(self.save_model, signal=EISEN_BEST_MODEL_METRIC, sender=workflow_id)

        self.artifacts_dir = os.path.join(artifacts_dir, 'models')

        self.save_history = save_history

        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)

    def save_model(self, message):
        statedict = message['model'].state_dict()

        torch.save(statedict, os.path.join(self.artifacts_dir, 'model.pt'))

        if self.save_history:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            torch.save(statedict, os.path.join(self.artifacts_dir, 'model_{}.pt'.format(timestr)))
