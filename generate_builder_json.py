import eisen
import json

from inspect import getmembers, isclass

from eisen import transforms
from eisen import io
from eisen import models
from eisen.ops import losses
from eisen.ops import metrics
from eisen.utils import workflows
from eisen import datasets


json_modules = {}

# transforms

class_list = [o for o in getmembers(transforms) if isclass(o[1])]

json_transforms = []

for class_spec in class_list:
    _, obj = class_spec

    if obj.__init__.__doc__ is None:
        continue

    if '<json>' not in obj.__init__.__doc__:
        continue

    string = obj.__init__.__doc__.split('<json>')[1].split('</json>')[0].replace('\n', ' ').replace('\r', '')
    params = json.loads(string)
    type = obj.__module__ + '.' + obj.__name__

    json_obj = {
        "type": type,
        "params": params
    }

    json_transforms.append(json_obj)


json_modules['transforms'] = json_transforms

# readers

class_list = [o for o in getmembers(io) if isclass(o[1])]

json_readers = []

for class_spec in class_list:
    _, obj = class_spec

    if obj.__init__.__doc__ is None:
        continue

    if '<json>' not in obj.__init__.__doc__:
        continue

    string = obj.__init__.__doc__.split('<json>')[1].split('</json>')[0].replace('\n', ' ').replace('\r', '')
    params = json.loads(string)
    type = obj.__module__ + '.' + obj.__name__

    json_obj = {
        "type": type,
        "params": params
    }

    json_readers.append(json_obj)


json_modules['readers'] = json_readers

# models

class_list = [o for o in getmembers(models) if isclass(o[1])]

json_models = []

for class_spec in class_list:
    _, obj = class_spec

    if obj.__init__.__doc__ is None:
        continue

    if '<json>' not in obj.__init__.__doc__:
        continue

    string = obj.__init__.__doc__.split('<json>')[1].split('</json>')[0].replace('\n', ' ').replace('\r', '')
    params = json.loads(string)
    type = obj.__module__ + '.' + obj.__name__

    json_obj = {
        "type": type,
        "params": params
    }

    json_models.append(json_obj)


json_modules['models'] = json_models

# losses

class_list = [o for o in getmembers(losses) if isclass(o[1])]

json_losses = []

for class_spec in class_list:
    _, obj = class_spec

    if obj.__init__.__doc__ is None:
        continue

    if '<json>' not in obj.__init__.__doc__:
        continue

    string = obj.__init__.__doc__.split('<json>')[1].split('</json>')[0].replace('\n', ' ').replace('\r', '')
    params = json.loads(string)
    type = obj.__module__ + '.' + obj.__name__

    json_obj = {
        "type": type,
        "params": params
    }

    json_losses.append(json_obj)


json_modules['losses'] = json_losses

# metrics

class_list = [o for o in getmembers(metrics) if isclass(o[1])]

json_metrics = []

for class_spec in class_list:
    _, obj = class_spec

    if obj.__init__.__doc__ is None:
        continue

    if '<json>' not in obj.__init__.__doc__:
        continue

    string = obj.__init__.__doc__.split('<json>')[1].split('</json>')[0].replace('\n', ' ').replace('\r', '')
    params = json.loads(string)
    type = obj.__module__ + '.' + obj.__name__

    json_obj = {
        "type": type,
        "params": params
    }

    json_metrics.append(json_obj)


json_modules['metrics'] = json_metrics

# workflow

class_list = [o for o in getmembers(workflows) if isclass(o[1])]

json_workflow = []

for class_spec in class_list:
    _, obj = class_spec

    if obj.__init__.__doc__ is None:
        continue

    if '<json>' not in obj.__init__.__doc__:
        continue

    string = obj.__init__.__doc__.split('<json>')[1].split('</json>')[0].replace('\n', ' ').replace('\r', '')
    params = json.loads(string)
    type = obj.__module__ + '.' + obj.__name__

    json_obj = {
        "type": type,
        "params": params
    }

    json_workflow.append(json_obj)


json_modules['workflow'] = json_workflow

# datasets

class_list = [o for o in getmembers(datasets) if isclass(o[1])]

json_datasets = []

for class_spec in class_list:
    _, obj = class_spec

    if obj.__init__.__doc__ is None:
        continue

    if '<json>' not in obj.__init__.__doc__:
        continue

    string = obj.__init__.__doc__.split('<json>')[1].split('</json>')[0].replace('\n', ' ').replace('\r', '')
    params = json.loads(string)
    type = obj.__module__ + '.' + obj.__name__

    json_obj = {
        "type": type,
        "params": params
    }

    json_datasets.append(json_obj)


json_modules['datasets'] = json_datasets

json_modules['hooks'] = []

json_modules['optimizers'] = []

json_modules['hyperparameters'] = []

with open('./eisen_modules_v{}.json'.format(eisen.__version__), 'w', encoding='utf-8') as file:
    json.dump(json_modules, file, ensure_ascii=False, indent=4)

