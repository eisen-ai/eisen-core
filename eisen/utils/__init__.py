import numpy as np
import os
import json
import inspect
import torch

from torch.nn import Module
from torch.utils.data import Dataset

from collections import OrderedDict


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
        """
        :param module: This is a Module type, not a Module instance. Aka a class name, not a class instance.
        :type module: type
        :param input_names: list of strings corresponding to batch keys, to be supplied to the module positionally
        :type input_names: list of str
        :param output_names: list of strings corresponding to output dictionary keys to hold module outputs
        :type output_names: list of str
        :param args: positional argument to instantiate the module
        :type args: list
        :param kwargs: keyword arguments to instantiate the module
        :type kwargs: dict
        """
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


def estimate_modulesize(model, input_size, type_size=4):
    """
    Not at all a precise estimate. Only serves the purprose of fair model splitting across GPUs
    """
    para = sum([np.prod(list(p.size())) for p in model.parameters()])

    input_ = torch.FloatTensor(*input_size).cuda()

    total_size = []

    def hook(module, input, output):
        total_size.append(np.prod(torch.tensor(output.size()).numpy()))

        return

    mods = list(model.modules())

    for i in range(1, len(mods)):
        sub_mod = list(mods[i].modules())
        if len(sub_mod) == 1:
            sub_mod[0].register_forward_hook(hook)

    out = model(input_)

    out_size = torch.tensor(out.size()).numpy()

    for i in range(1, len(mods)):
        sub_mod = list(mods[i].modules())
        if len(sub_mod) == 1:
            sub_mod[0]._forward_hooks = OrderedDict()

    total_size.append(np.prod(torch.tensor(out.size()).numpy()))

    total_tensorsize_fw = np.sum(np.asarray(total_size)) * type_size / 1024 / 1024
    total_paramsize = para * type_size / 1024 / 1024

    total_size = total_tensorsize_fw + total_paramsize * 2

    return total_size, out_size


def partition_idx_weight_list(weights, idx=None):
    weights = np.asarray(weights)

    total = np.sum(weights)

    half = total / 2 + 1

    for i in range(len(weights)):
        if np.sum(weights[0:i]) >= half:
            break

    if idx is None:
        return [np.asarray(range(0, i, 1)), np.asarray(range(i, len(weights), 1))]
    else:
        return [np.asarray(idx[range(0, i, 1)]), np.asarray(idx[range(i, len(weights), 1)])]


def get_n_idx_partitions(weights, n):
    assert n > 1 and ((n & (n - 1)) == 0), "the parameters n has to be power of 2. you supplied {}".format(n)

    weights = np.asarray(weights)

    partitioning = partition_idx_weight_list(weights, idx=None)

    while len(partitioning) is not n:
        result = []
        for entry in partitioning:
            result = result + list(partition_idx_weight_list(weights[entry], idx=entry))

        partitioning = result

    return partitioning


class InputArgumentPlacementChanger:
    def __init__(self, gpu_name):
        self.gpu_name = gpu_name

    def __call__(self, module, argument):
        if hasattr(argument, "__iter__"):
            tensor_list = []

            for element in argument:
                tensor_list.append(element.to(self.gpu_name))

            return tuple(tensor_list)
        else:
            argument = argument.to(self.gpu_name)

            return argument


class PipelineExecutionStreamer(torch.nn.Module):
    """
    This execution streamer takes a sequence of operations (torch.nn.Module) and executes them in a pipeline.
    Clearly this is only useful when each operation is executed on a different device. In this way,
    the execution can be asynchronously kicked off on each device separately, therefore maximizing the GPU usage.
    More details about this idea can be found here:
    https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html#speed-up-by-pipelining-inputs
    """
    def __init__(self, operations_sequence, split_size):
        """
        :param operations_sequence: A list containing operations that should be done in sequence
        :type operations_sequence: list of torch.nn.Module
        :param split_size: Split size in order to obtain chunks of each batch to fill the pipeline
        :type split_size: int
        """
        super(PipelineExecutionStreamer, self).__init__()
        self.operations_sequence = operations_sequence
        self.split_size = split_size

    def forward(self, x):
        splits = iter(list(x.split(self.split_size, dim=0)) + [None] * (len(self.operations_sequence) - 1))

        pipeline = [None] * (len(self.operations_sequence) - 1)

        outputs = []

        def make_tuple(argument):
            if not isinstance(outputs, tuple):
                argument = tuple([argument])

            return argument

        fresh_data = make_tuple(next(splits))

        pipeline[0] = self.operations_sequence[0](*fresh_data)

        # loops until element before the last
        reverse_loop_counter = list(range(len(pipeline)))[::-1]

        for split in splits:
            for idx in reverse_loop_counter:
                # from the last stage to the first
                if pipeline[idx] is not None:
                    if idx == (len(pipeline) - 1):
                        dta = make_tuple(pipeline[idx])
                        outputs.append(self.operations_sequence[idx + 1](*dta))
                    else:
                        dta = make_tuple(pipeline[idx])
                        pipeline[idx + 1] = self.operations_sequence[idx + 1](*dta)

                if idx == 0:
                    if split is not None:
                        dta = make_tuple(split)
                        pipeline[idx] = self.operations_sequence[idx](*dta)

                    if split is None:
                        pipeline[idx] = None

        return torch.cat(outputs)


class EisenAutoModelParallelModuleWrapper(EisenModuleWrapper):
    """
    Eisen Module Wrapper having the capability of distributing the model across different GPUs as specified by the user.
    The module being wrapped should have >= number_gpus children (sub modules) in order for this approach to be
    effective. In case this is not true, the wrapper still works (but won't make use/occupy some of the GPUs). Usually
    modules have multiple children, therefore for normal use this condition is verified.

    This object, estimates the size of each child module and decides a distribution that places consequent operations
    in each GPU so that the memory usage of each GPUs is balanced across all processors.
    """
    def __init__(self, module, number_gpus, split_size, *args, **kwargs):
        """
        :param module: This is a Module type, not a Module instance. Aka a class name, not a class instance.
        :type module: type
        :param number_gpus: how many GPUs should be used
        :type number_gpus: int
        :param split_size: split size for pipeline execution
        :type split_size: int
        :param args: positional arguments for superclass EisenModuleWrapper and the Module itself
        :type args: list
        :param kwargs: keyword arguments for superclass EisenModuleWrapper and the Module itself
        :type kwargs: dict
        """
        super(EisenAutoModelParallelModuleWrapper, self).__init__(module, *args, **kwargs)

        self.number_gpus = number_gpus

        self.gpu_device_ids = ['cuda:{}'.format(idx) for idx in range(self.number_gpus)]

        self.first_run = True

        self.split_size = split_size

    def forward(self, **kwargs):
        input_dict = {}

        execution_list = []

        for dst_arg, src_arg in zip(self.module_argument_list, self.input_names):
            input_dict[dst_arg] = kwargs[src_arg]

        if self.first_run:
            self.first_run = False

            print('Automatically distributing the model across {} GPUs...'.format(self.number_gpus))

            input_size = input_dict[list(input_dict.keys())[0]].size()

            # estimate module children size

            children = list(self.module.children())

            children_size = []

            for child in children:
                size, out_size = estimate_modulesize(child, input_size)

                children_size.append(size)

                input_size = out_size

            partitions_idx = get_n_idx_partitions(children_size, self.number_gpus)

            for curr_gpu, indices in enumerate(partitions_idx):
                curr_gpu_name = self.gpu_device_ids[curr_gpu]

                seq_list = []

                for idx in indices:
                    children[idx] = children[idx].to(curr_gpu_name)

                    seq_list.append(children[idx])

                sequential_block = torch.nn.Sequential(*seq_list)

                argument_placement_changer = InputArgumentPlacementChanger(curr_gpu_name)

                sequential_block.register_forward_pre_hook(argument_placement_changer)

                execution_list.append(sequential_block)

                self.module = PipelineExecutionStreamer(execution_list, split_size=self.split_size)

        input_tensor = input_dict[list(input_dict.keys())[0]]

        outputs = self.module(input_tensor).to(self.gpu_device_ids[0])

        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)

        ret_dict = {}

        for output, output_name in zip(outputs, self.output_names):
            ret_dict[output_name] = output

        return ret_dict
