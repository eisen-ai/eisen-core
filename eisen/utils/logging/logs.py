import numpy as np

from eisen import (
    EISEN_END_EPOCH_EVENT,
)

from pydispatch import dispatcher
from prettytable import PrettyTable


class LoggingHook:
    """
    Logging object aiming at printing on the console the progress of model training/validation/testing.
    This logger uses an event based system. The training, validation and test workflows emit events such as
    EISEN_END_BATCH_EVENT and EISEN_END_EPOCH_EVENT which are picked up by this object and handled.

    Once the user instantiates such object, the workflow corresponding to the ID passes as argument will be
    tracked and the results of the workflow in terms of losses and metrics will be printed on the console

    .. code-block:: python

            from eisen.utils.logging import LoggingHook

            workflow = # Eg. An instance of Training workflow

            logger = LoggingHook(workflow.id, 'Training', '/artifacts/dir')

    """
    def __init__(self, workflow_id, phase, artifacts_dir):
        """
        :param workflow_id: string containing the workflow id of the workflow being monitored (workflow_instance.id)
        :type workflow_id: UUID
        :param phase: string containing the name of the phase (training, testing, ...) of the workflow monitored
        :type phase: str
        :param artifacts_dir: The path of the directory where the artifacts of the workflow are stored
        :type artifacts_dir: str

        .. code-block:: python

            from eisen.utils.logging import LoggingHook

            workflow = # Eg. An instance of Training workflow

            logger = LoggingHook(
                workflow_id=workflow.id,
                phase='Training',
                artifacts_dir='/artifacts/dir'
            )

        <json>
        []
        </json>
        """

        dispatcher.connect(self.end_epoch, signal=EISEN_END_EPOCH_EVENT, sender=workflow_id)

        self.table = PrettyTable()
        self.phase = phase
        self.workflow_id = workflow_id
        self.artifacts_dir = artifacts_dir

    def end_epoch(self, message):
        all_losses = []
        all_losses_names = []
        for dct in message['losses']:
            for key in dct.keys():
                all_losses_names.append(key)
                all_losses.append(np.mean(dct[key]))

        all_metrics = []
        all_metrics_names = []
        for dct in message['metrics']:
            for key in dct.keys():
                all_metrics_names.append(key)
                all_metrics.append(np.mean(dct[key]))

        self.table.field_names = \
            ["Phase"] + \
            [str(k) + ' (L)' for k in all_losses_names] + \
            [str(k) + ' (M)' for k in all_metrics_names]

        self.table.add_row(
            ["{} - Epoch {}".format(self.phase, message['epoch'])] +
            [str(loss) for loss in all_losses] +
            [str(metric) for metric in all_metrics]
        )

        print(self.table)
