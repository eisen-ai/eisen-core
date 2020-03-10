import torch
import os
import time

from pydispatch import dispatcher
from eisen import EISEN_BEST_MODEL_METRIC, EISEN_BEST_MODEL_LOSS


class SaveTorchModel:
    """
    Saves a Torch model snapshot of the current best model. The best model can be selected based using the best
    average loss or the best average metric. It is possible to save the whole history of best models seen throughout
    the workflow.

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


class SaveONNXModel:
    """
    Saves a ONNX model snapshot of the current best model. The best model can be selected based using the best
    average loss or the best average metric. It is possible to save the whole history of best models seen throughout
    the workflow.

    .. code-block:: python

        from eisen.utils.artifacts import SaveONNXModel

        workflow = # Eg. An instance of Validation workflow

        saver = SaveONNXModel(workflow.id, 'Validation', '/my/artifacts', [1, 1, 224, 224])

    """
    def __init__(self, workflow_id, phase, artifacts_dir, input_size, select_best_loss=True, save_history=False):
        """
        :param workflow_id: the ID of the workflow that should be tracked by this hook
        :type workflow_id: UUID
        :param phase: the phase where this hook is being used (Training, Testing, etc.)
        :type phase: str
        :param artifacts_dir: the path of the artifacts where the results of this hook should be stored
        :type artifacts_dir: str
        :param input_size: a list of integers expressing the input size that the saved model will process
        :type input_size: list of int
        :param select_best_loss: whether the criterion for saving the model should be best loss or best metric
        :type select_best_loss: bool
        :param artifacts_dir: whether the history of all models that were at a certain point the best should be saved
        :type artifacts_dir: bool

        .. code-block:: python

            from eisen.utils.artifacts import SaveONNXModel

            workflow = # Eg. An instance of Validation workflow

            saver = SaveONNXModel(
                workflow_id=workflow.id,
                phase='Validation',
                artifacts_dir='/my/artifacts',
                input_size=[1, 1, 224, 224],
                select_best_loss=True,
                save_history=False
            )


        <json>
        [
            {"name": "input_size", "type": "list:int", "value": ""},
            {"name": "select_best_loss", "type": "bool", "value": "True"},
            {"name": "save_history", "type": "bool", "value": "False"}
        ]
        </json>
        """
        if select_best_loss:
            dispatcher.connect(self.save_model, signal=EISEN_BEST_MODEL_LOSS, sender=workflow_id)
        else:
            dispatcher.connect(self.save_model, signal=EISEN_BEST_MODEL_METRIC, sender=workflow_id)

        self.artifacts_dir = os.path.join(artifacts_dir, 'onnx_models')

        self.save_history = save_history

        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)

        self.input_size = input_size

    def save_model(self, message):
        dummy_input = torch.randn(*self.input_size)

        torch.onnx.export(message['model'], dummy_input, "model.onnx", verbose=True)

        if self.save_history:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            torch.onnx.export(message['model'], dummy_input, "model_{}.onnx".format(timestr), verbose=True)
