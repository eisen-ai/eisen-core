import torch
import os
import time

from pydispatch import dispatcher
from eisen import EISEN_BEST_MODEL_METRIC, EISEN_BEST_MODEL_LOSS


class SaveTorchModel:
    """
    This object implements model saving for pytorch models. Once instantiated with a parameter consisting of a string
    representing the path of the directory where the model shall be saved, it can be called on a model in order to save
    it.

    No information about optimizer and training is saved in the process.

    .. code-block:: python

        from eisen.utils.artifacts import SaveTorchModel

        my_model = # Eg. A torch.nn.Module instance

        saver = SaveTorchModel('/my/artifacts')

        saver(my_model)

    """
    def __init__(self, artifacts_dir):
        """
        Initializes a SaveTorchModel object.

        :param artifacts_dir: The path of the directory where the model shall be stored after serialization
        :type artifacts_dir: str
        """
        self.artifacts_dir = artifacts_dir

        if not os.path.exists(self.artifacts_dir):
            raise ValueError('The artifacts directory passed as parameter to the SaveTorchModel object does not exist')

    def __call__(self, model, filename='model.pt'):
        """
        Saves a model passed as argument. The model will be saved in Torch (statedict) format.

        :param model: Model to be saved (refrain from using wrapped modules, see EisenModuleWrapper)
        :type model: torch.nn.Module
        :param filename: The filename that shall be used to save the model
        :type filename: str
        :return: None
        """
        statedict = model.state_dict()

        torch.save(statedict, os.path.join(self.artifacts_dir, filename))


class SaveONNXModel:
    """
    This object exports a torch.nn.Module in ONNX format. The user is asked to supply two parameters for initialization.
    The first parameter is the artifact directory, a string representing the path where the model is supposed to be
    stored after serialization. The second parameter is the input size, a list of integers containing the size
    of the inputs to be processed by the network.

    .. code-block:: python

        from eisen.utils.artifacts import SaveONNXModel

        my_model = # Eg. A torch.nn.Module instance

        saver = SaveONNXModel('/my/artifacts', [1, 1, 224, 224])

        saver(my_model)

    """
    def __init__(self, artifacts_dir, input_size):
        """
        Initializes a SaveONNXModel object.

        :param artifacts_dir: The path of the directory where the model shall be stored after serialization
        :type artifacts_dir: str
        :param input_size: The size of the input the network will be processing after serialization
        :type input_size: list of int
        """
        self.artifacts_dir = artifacts_dir

        self.input_size = input_size

        if not os.path.exists(self.artifacts_dir):
            raise ValueError('The artifacts directory passed as parameter to the SaveTorchModel object does not exist')

    def __call__(self, model, filename='model.onnx'):
        """
        Saves a model passed as argument. The model will be saved in ONNX format.

        :param model: Model to be saved (refrain from using wrapped modules, see EisenModuleWrapper)
        :type model: torch.nn.Module
        :param filename: The filename that shall be used to save the model
        :type filename: str
        :return: None
        """

        dummy_input = torch.randn(*self.input_size)

        torch.onnx.export(
            model,
            dummy_input,
            os.path.join(self.artifacts_dir, filename),
            export_params=True,
            opset_version=10,
            verbose=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )


class SaveTorchModelHook:
    """
    Saves a Torch model snapshot of the current best model. The best model can be selected based using the best
    average loss or the best average metric. It is possible to save the whole history of best models seen throughout
    the workflow.

    .. code-block:: python

        from eisen.utils.artifacts import SaveTorchModelHook

        workflow = # Eg. An instance of Validation workflow

        saver = SaveTorchModelHook(workflow.id, 'Validation', '/my/artifacts')
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

        if not os.path.exists(artifacts_dir):
            raise ValueError('The directory specified to save artifacts does not exist!')

        self.artifacts_dir = os.path.join(artifacts_dir, 'models')

        self.save_history = save_history

        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)

        self.saver = SaveTorchModel(self.artifacts_dir)

    def save_model(self, message):
        # we save an unwrapped version of the model!! (see eisen.utils.EisenModuleWrapper)
        self.saver(message['model'].module, 'model.pt')

        if self.save_history:
            timestr = time.strftime("%Y%m%d-%H%M%S")

            # we save an unwrapped version of the model!! (see eisen.utils.EisenModuleWrapper)
            self.saver(message['model'].module, 'model_{}.pt'.format(timestr))


class SaveONNXModelHook:
    """
    Saves a ONNX model snapshot of the current best model. The best model can be selected based using the best
    average loss or the best average metric. It is possible to save the whole history of best models seen throughout
    the workflow.

    .. code-block:: python

        from eisen.utils.artifacts import SaveONNXModelHook

        workflow = # Eg. An instance of Validation workflow

        saver = SaveONNXModelHook(workflow.id, 'Validation', '/my/artifacts', [1, 3, 224, 224])

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

            from eisen.utils.artifacts import SaveONNXModelHook

            workflow = # Eg. An instance of Validation workflow

            saver = SaveONNXModelHook(
                workflow_id=workflow.id,
                phase='Validation',
                artifacts_dir='/my/artifacts',
                input_size=[1, 3, 224, 224],
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

        if not os.path.exists(artifacts_dir):
            raise ValueError('The directory specified to save artifacts does not exist!')

        self.artifacts_dir = os.path.join(artifacts_dir, 'onnx_models')

        self.save_history = save_history

        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)

        self.input_size = input_size

        self.saver = SaveONNXModel(self.artifacts_dir, self.input_size)

    def save_model(self, message):
        # we save an unwrapped version of the model!! (see eisen.utils.EisenModuleWrapper)
        self.saver(message['model'].module, 'model.onnx')

        if self.save_history:
            timestr = time.strftime("%Y%m%d-%H%M%S")

            # we save an unwrapped version of the model!! (see eisen.utils.EisenModuleWrapper)
            self.saver(message['model'].module, 'model_{}.onnx'.format(timestr))
