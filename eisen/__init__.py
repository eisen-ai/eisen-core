__version__ = '0.0.3'

EPS = 0.00001


# SIGNALS AND SENDERS (PyDispatch)

EISEN_TRAINING_SENDER = 'eisen::training'
EISEN_TESTING_SENDER = 'eisen::testing'
EISEN_VALIDATION_SENDER = 'eisen::validation'

EISEN_END_BATCH_EVENT = 'eisen::end_batch'
EISEN_END_EPOCH_EVENT = 'eisen::end_epoch'