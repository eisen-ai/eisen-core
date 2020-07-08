from pydispatch import dispatcher

from eisen import EISEN_END_EPOCH_EVENT, EISEN_BEST_MODEL_LOSS, EISEN_BEST_MODEL_METRIC


class PatienceHook:
    """
    Keeps track of the number of epochs after the best loss or metric is reached in the workflow,
    allowing for early stopping based on patience.  In the example below, early stopping is attached
    to the validation workflow, such that when validation loss does not improve for 6 epochs, the
    training loop will terminate.

    .. code-block:: python

        from eisen.utils.hooks import PatienceHook

        train_workflow = Training(model, train_data_loader, [loss])
        val_workflow = Validation(model, val_data_loader, [loss])

        early_stopping = PatienceHook(val_workflow.id, patience=6, select_best_loss=True)

        i = 0
        while not early_stopping.is_stop():
            train_workflow.run()
            val_workflow.run()
            i += 1
        print("stopped after %s epochs" % i)

    """

    def __init__(self, workflow_id, patience, select_best_loss=True):
        """
        :param workflow_id: the ID of the workflow that should be tracked by this hook
        :type workflow_id: UUID
        :param patience: number of epochs to run without reaching a new best loss or metric
        :type patience: int
        :param select_best_loss: whether the criterion for saving the model should be best loss or best metric
        :type select_best_loss: bool

        """
        if select_best_loss:
            dispatcher.connect(self._reset_on_best, signal=EISEN_BEST_MODEL_LOSS, sender=workflow_id)
        else:
            dispatcher.connect(self._reset_on_best, signal=EISEN_BEST_MODEL_METRIC, sender=workflow_id)
        dispatcher.connect(self.increment_patience, signal=EISEN_END_EPOCH_EVENT, sender=workflow_id)

        self.patience = patience

        self.patience_counter = 0

    def reset(self):
        """resets the patience counter manually, for example for using a learning rate scheduler"""
        self.patience_counter = 0

    def _reset_on_best(self, message):
        self.patience_counter = -1  # end epoch triggers after best loss/metric

    def increment_patience(self, message):
        self.patience_counter += 1

    def is_stop(self):
        """returns true after loss/metric has not improved for N epochs.

        :return: workflow should halt
        :rtype: bool
        """
        return self.patience_counter >= self.patience
