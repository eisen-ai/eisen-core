import numpy as np

from eisen import (
    EISEN_END_BATCH_EVENT,
    EISEN_END_EPOCH_EVENT,
    EISEN_TRAINING_SENDER,
    EISEN_VALIDATION_SENDER,
    EISEN_TESTING_SENDER
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

        # testing signals
        dispatcher.connect(self.end_testing_batch, signal=EISEN_END_BATCH_EVENT, sender=EISEN_TESTING_SENDER)
        dispatcher.connect(self.end_testing_epoch, signal=EISEN_END_EPOCH_EVENT, sender=EISEN_TESTING_SENDER)

        self.training_epoch_data = {}
        self.validation_epoch_data = {}
        self.testing_epoch_data = {}

        self.table_training = PrettyTable()
        self.table_validation = PrettyTable()
        self.table_testing = PrettyTable()

    def end_training_batch(self, message):
        for typ in ['losses', 'metrics']:
            if type not in self.training_epoch_data.keys():
                self.training_epoch_data[typ] = {}

            for dta in message[typ]:
                for key in dta.keys():
                    scalar_loss = np.mean(dta[key].cpu().data.numpy())

                    if key not in self.training_epoch_data[typ].keys():
                        self.training_epoch_data[typ][key] = []

                    self.training_epoch_data[typ][key].append(scalar_loss)

    def end_training_epoch(self, message):
        self.table_training.field_names = \
            ["Phase"] + \
            [str(k) + ' (L)' for k in self.training_epoch_data['losses'].keys()] + \
            [str(k) + ' (M)' for k in self.training_epoch_data['metrics'].keys()]

        self.table_training.add_row(
            ["Training Epoch {}".format(message)] +
            [
                str(np.mean(np.asarray(self.training_epoch_data['losses'][key])))
                for key in self.training_epoch_data['losses'].keys()
            ] +
            [
                str(np.mean(np.asarray(self.training_epoch_data['metrics'][key])))
                for key in self.training_epoch_data['metrics'].keys()
            ]
        )

        self.training_epoch_data = {}

        print(self.table_training)

    def end_validation_batch(self, message):
        for typ in ['losses', 'metrics']:
            if type not in self.validation_epoch_data.keys():
                self.validation_epoch_data[typ] = {}

            for dta in message[typ]:
                for key in dta.keys():
                    scalar_loss = np.mean(dta[key].cpu().data.numpy())

                    if key not in self.validation_epoch_data[typ].keys():
                        self.validation_epoch_data[typ][key] = []

                    self.validation_epoch_data[typ][key].append(scalar_loss)

    def end_validation_epoch(self, message):
        self.table_validation.field_names = \
            ["Phase"] + \
            [str(k) + ' (L)' for k in self.validation_epoch_data['losses'].keys()] + \
            [str(k) + ' (M)' for k in self.validation_epoch_data['metrics'].keys()]

        self.table_validation.add_row(
            ["Validation Epoch {}".format(message)] +
            [
                str(np.mean(np.asarray(self.validation_epoch_data['losses'][key])))
                for key in self.validation_epoch_data['losses'].keys()
            ] +
            [
                str(np.mean(np.asarray(self.validation_epoch_data['metrics'][key])))
                for key in self.validation_epoch_data['metrics'].keys()
            ]
        )

        self.validation_epoch_data = {}

        print(self.table_validation)

    def end_testing_batch(self, message):
        for typ in ['metrics']:
            if type not in self.testing_epoch_data.keys():
                self.testing_epoch_data[typ] = {}

            for dta in message[typ]:
                for key in dta.keys():
                    scalar_loss = np.mean(dta[key].cpu().data.numpy())

                    if key not in self.testing_epoch_data[typ].keys():
                        self.testing_epoch_data[typ][key] = []

                    self.testing_epoch_data[typ][key].append(scalar_loss)

    def end_testing_epoch(self, message):
        self.table_testing.field_names = \
            ["Phase"] + \
            [str(k) + ' (M)' for k in self.testing_epoch_data['metrics'].keys()]

        self.table_testing.add_row(
            ["Testing Epoch {}".format(message)] +
            [
                str(np.mean(np.asarray(self.testing_epoch_data['metrics'][key])))
                for key in self.testing_epoch_data['metrics'].keys()
            ]
        )

        self.testing_epoch_data = {}

        print(self.table_testing)
