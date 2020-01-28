import numpy as np

from eisen import (
    EISEN_END_BATCH_EVENT,
    EISEN_END_EPOCH_EVENT,
    EISEN_TRAINING_SENDER,
    EISEN_VALIDATION_SENDER
)

from pydispatch import dispatcher
from prettytable import PrettyTable

class LoggingHook:
    def __init__(self):
        """
        <json>
        []
        </json>
        """
        # training signals
        dispatcher.connect(self.end_training_batch, signal=EISEN_END_BATCH_EVENT, sender=EISEN_TRAINING_SENDER)
        dispatcher.connect(self.end_training_epoch, signal=EISEN_END_EPOCH_EVENT, sender=EISEN_TRAINING_SENDER)

        # validation signals
        dispatcher.connect(self.end_validation_batch, signal=EISEN_END_BATCH_EVENT, sender=EISEN_VALIDATION_SENDER)
        dispatcher.connect(self.end_validation_epoch, signal=EISEN_END_EPOCH_EVENT, sender=EISEN_VALIDATION_SENDER)

        self.epoch_data = {}

        self.table_training = PrettyTable()
        self.table_validation = PrettyTable()

    def end_training_batch(self, message):
        for typ in ['losses', 'metrics']:
            if type not in self.epoch_data.keys():
                self.epoch_data[typ] = {}

            for dta in message[typ]:
                for key in dta.keys():
                    scalar_loss = np.mean(dta[key].cpu().data.numpy())

                    if key not in self.epoch_data[typ].keys():
                        self.epoch_data[typ][key] = []

                    self.epoch_data[typ][key].append(scalar_loss)

    def end_training_epoch(self, message):
        self.table_training.field_names = \
            ["Phase"] + \
            [str(k) + ' (L)' for k in self.epoch_data['losses'].keys()] + \
            [str(k) + ' (M)' for k in self.epoch_data['metrics'].keys()]

        self.table_training.add_row(
            ["Training Epoch {}".format(message)] +
            [str(np.mean(np.asarray(self.epoch_data['losses'][key]))) for key in self.epoch_data['losses'].keys()] +
            [str(np.mean(np.asarray(self.epoch_data['metrics'][key])))for key in self.epoch_data['metrics'].keys()]
        )

        self.epoch_data = {}

        print(self.table_training)

    def end_validation_batch(self, message):
        print(message)

    def end_validation_epoch(self, message):
        print(message)
