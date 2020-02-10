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
    This object implements a wrapper allowing standard PyTorch Modules to be used in a Eisen Workflow/context.

    In Eisen, when we iterate through a dataset using a torch.utils.data.DataLoader each batch is represented as a
    dictionary of entries. Thanks to the ability of Python to receive function arguments via a dictionary
    (kwargs) it is possible to input the batch directly into the nn.Module objects (the models) by just feeding
    the batch dictionary as an **argument. We would like also to receive the results of the nn.Module into a
    dictionary, so they can be fed to other nn.Module in the same way or stored by keyword for further processing
    (for example by a Hook).

    EisenModuleWrapper allows that. It wraps a nn.Module into an object that still behaves like an nn.Module but
    has additional data stored into it in order to translate the keys of the batch dictionary, to keys that
    can be used by the nn.Module. It also packs the results of the forward pass of the wrapped
    nn.Module into a dictionary.

    The fields of the input and output dictionaries are defined by the user during object instantiation.

    Let us suppose we have a torch.utils.data.DataLoader iterating the dataset
    this DataLoader returns batches (dictionaries) containing the keys 'picture' and 'ground_truth'.

    Our Module belongs to class UNet3D which accepts 'image' as input key for the forward() method
    This means that its forward method has signature def forward(image).

    Additionally we want that the outputs of the forward() method of our UNet3D object get
    packed in a dictionary with keys ['prediction'].

    .. code-block:: python

        # We import the module we want to wrap

        from eisen.models.segmentation import UNet3D

        # We can then instantiate an object of class EisenModuleWrapper
        # We want to feed our 'picture' into the 'image' field of the network. The second argument is therefore
        # ['picture']. We want to store the predictions using the key 'prediction' the third argumet is ['prediction']

        new_module = EisenModuleWrapper(UNet3D, ['picture'], ['prediction'])

        # ...

        for batch in batch_iterator:

            print(batch.keys())

            # should print ['picture', 'ground_truth']

            result_dictionary = new_module(**batch)

            print(result_dictionary.keys())

            # should print ['prediction']
    """
    def __init__(self, module, input_names, output_names, **kwargs):
        super(EisenModuleWrapper, self).__init__()

        self.input_names = input_names
        self.output_names = output_names

        self.module = module(**kwargs)

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
    pass


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
        """
        :param data: the dataset object. Needs to have a data attribute which is a list of dictionaries.
        :type data: list of dict
        :return: Random dataset split into training, validation and test set.
        :rtype: tuple
        """

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
