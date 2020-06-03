__version__ = '0.0.7'

EPS = 0.00001


# SIGNALS AND SENDERS (PyDispatch)

EISEN_END_BATCH_EVENT = 'eisen::end_batch'
EISEN_END_EPOCH_EVENT = 'eisen::end_epoch'
EISEN_BEST_MODEL_LOSS = 'eisen::best_model_loss'
EISEN_BEST_MODEL_METRIC = 'eisen::best_model_metric'