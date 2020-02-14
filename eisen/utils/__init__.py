import numpy as np
import os
import json
import inspect
import torch

from torch.nn import Module
from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data_list[idx]

        if self.transform:
            item = self.transform(item)

        return item


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def read_json_from_file(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError('The JSON file {} cannot be read'.format(json_file))

    with open(json_file) as json_file:
        dictionary = json.load(json_file)

    return dictionary


class EisenModuleWrapper(Module):
    """
    This object implements a wrapper allowing standard PyTorch Modules (Eg. those implemented in torchvision)
    to be used within Eisen.

    Modules in Eisen accept positional and named arguments in the forward() method. They return values or a tuple of
    values.

    Eisen workflows make use of dictionaries. That is, data batches are represented as dictionaries and directly fed
    into modules using the **kwargs mechanism provided by Python.

    This wrapper causes standard Modules to behave as prescribed by Eisen. Wrapped modules accept as input a dictionary
    of keyword arguments with arbitrary (user defined) keys. They return as output a dictionary of keyword values
    with arbitrary (user defined) keys.

    .. code-block:: python

        # We import the Module we want to wrap. In this case we import from torchvision

        from torchvision.models import resnet18

        # We can then instantiate an object of class EisenModuleWrapper and specify the Module we want to
        # wrap as well as the fields of the data dictionary that will interpreted as input, and the fields
        # that we desire the output to be stored at. Additional arguments for the Module itself can
        # be passed as named arguments.

        adapted_module = EisenModuleWrapper(resnet18, ['image'], ['prediction'], pretrained=False)

    """
    def __init__(self, module, input_names, output_names, *args, **kwargs):
        super(EisenModuleWrapper, self).__init__()

        self.input_names = input_names
        self.output_names = output_names

        self.module = module(*args, **kwargs)

        module_argument_list = inspect.getfullargspec(self.module.forward)[0]

        module_argument_list.remove('self')

        self.module_argument_list = module_argument_list

    def forward(self, **kwargs):
        input_dict = {}

        for dst_arg, src_arg in zip(self.module_argument_list, self.input_names):
            input_dict[dst_arg] = kwargs[src_arg]

        outputs = self.module(**input_dict)

        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)

        ret_dict = {}

        for output, output_name in zip(outputs, self.output_names):
            ret_dict[output_name] = output

        return ret_dict


class EisenTransformWrapper:
    """
    This object implements a wrapper allowing standard PyTorch Transform (Eg. those implemented in torchvision)
    to be used within Eisen.

    Transforms in Eisen operate on dictionaries. They are in fact always called on a dictionary containing multiple
    keys that store data.

    This wrapper causes standard Transforms to behave as prescribed by Eisen.

    .. code-block:: python

        # We import the transform we want to wrap. In this case we import from torchvision

        from torchvision.transforms import CenterCrop

        # We can then instantiate an object of class EisenTransformWrapper and specify the Transformation we want to
        # wrap as well as the field of the data dictionary that should be affected by such Transformation.
        # Additional arguments for the Transformation itself can be passed as named arguments.

        adapted_transform = EisenTransformWrapper(CenterCrop, ['image'], (224, 224))

    """
    def __init__(self, module, fields, *args, **kwargs):
        super(EisenTransformWrapper, self).__init__()
        self.fields = fields

        self.module = module(*args, **kwargs)

    def __call__(self, data):
        for field in self.fields:
           data[field] = self.module(data[field])

        return data


class EisenDatasetWrapper(Dataset):
    """
    This object implements a wrapper allowing standard PyTorch Datasets (Eg. those implemented in torchvision)
    to be used within Eisen.

    Datasets in Eisen return items that are always dictionaries. Each key of the dictionary contains information
    from the dataset.

    This wrapper causes standard Datasets to behave as prescribed by Eisen.

    .. code-block:: python

        # We import the dataset we want to wrap. In this case we import from torchvision

        from torchvision.datasets import MNIST

        # We can then instantiate an object of class EisenDatasetWrapper and specify the Dataset we want to
        # wrap as well as the fields of the data dictionary that will be returned by the adapted __getitem__ method.
        # Additional arguments for the Dataset itself can be passed as named arguments.

        adapted_dataset = EisenDatasetWrapper(MNIST, ['image', 'label'], './', download=True)

    """

    def __init__(self, module, field_names, transform=None, *args, **kwargs):
        super(EisenDatasetWrapper, self).__init__()
        self.field_names = field_names

        self.module = module(*args, **kwargs)

        self.transform = transform

    def __getitem__(self, item):
        items = self.module[item]

        assert len(self.field_names) == len(items)

        ret_arg = {}

        for item, name in zip(items, self.field_names):
            ret_arg[name] = item

        ret_arg = self.transform(ret_arg)

        return ret_arg

    def __len__(self):
        return len(self.module)


class EisenDatasetSplitter:
    def __init__(
            self,
            fraction_train,
            fraction_valid=None,
            fraction_test=None,
            transform_train=None,
            transform_valid=None,
            transform_test=None
    ):
        if fraction_train <= 1.0 and fraction_valid is None and fraction_test is None:
            self.fraction_train = fraction_train
            self.fraction_valid = 1.0 - self.fraction_train
            self.fraction_test = 0

        elif fraction_train <= 1.0 and fraction_valid <= 1.0 and fraction_test is None:
            self.fraction_train = fraction_train
            self.fraction_valid = fraction_valid

            assert (self.fraction_train + self.fraction_valid) <= 1.0

            self.fraction_test = 1.0 - (self.fraction_train + self.fraction_valid)

        else:
            self.fraction_train = fraction_train
            self.fraction_valid = fraction_valid
            self.fraction_test = fraction_test

        assert (self.fraction_test + self.fraction_train + self.fraction_valid) == 1.0

        self.transform_train = transform_train
        self.transform_valid = transform_valid
        self.transform_test = transform_test

    def __call__(self, data):
        perm = np.random.permutation(len(data))

        limit_test = int(self.fraction_test * len(perm))

        limit_validation = int(self.fraction_valid * len(perm)) + limit_test

        list_test = perm[0:limit_test].tolist()

        if len(list_test) > 0:
            data_test = [data[k] for k in list_test]
        else:
            data_test = None

        list_valid = perm[limit_test: limit_validation].tolist()

        if len(list_valid) > 0:
            data_validation = [data[k] for k in list_valid]
        else:
            data_validation = None

        list_train = perm[limit_validation:].tolist()

        if len(list_train) > 0:
            data_training = [data[k] for k in list_train]
        else:
            data_training = None

        dataset_train = ListDataset(data_training, self.transform_train)

        dataset_valid = ListDataset(data_validation, self.transform_valid)

        dataset_test = ListDataset(data_test, self.transform_test)

        return dataset_train, dataset_valid, dataset_test
