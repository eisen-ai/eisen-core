import numpy as np
import os
import json
import inspect
import torch

from torch.nn import Module
from torch.utils.data import Dataset
from torch.cuda._utils import _get_device_index

from collections import OrderedDict


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


def _estimate_modulesize(model, input_size, type_size=4):
    """
    Not at all a precise estimate. Only serves the purpose of fair model splitting across GPUs
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


def _partition_idx_weight_list(weights, idx=None):
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


def _get_n_idx_partitions(weights, n):
    assert n > 1 and ((n & (n - 1)) == 0), "the parameters n has to be power of 2. you supplied {}".format(n)

    weights = np.asarray(weights)

    partitioning = _partition_idx_weight_list(weights, idx=None)

    while len(partitioning) is not n:
        result = []
        for entry in partitioning:
            result = result + list(_partition_idx_weight_list(weights[entry], idx=entry))

        partitioning = result

    return partitioning


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

        # We can then instantiate an object of class EisenModuleWrapper and instantiate the Module we want to
        # wrap as well as the fields of the data dictionary that will interpreted as input, and the fields
        # that we desire the output to be stored at. Additional arguments for the Module itself can
        # be passed as named arguments.

        module = resnet18(pretrained=False)

        adapted_module = EisenModuleWrapper(module, ['image'], ['prediction'])

    """
    def __init__(self, module, input_names, output_names):
        """
        :param module: This is a Module instance
        :type module: torch.nn.Module
        :param input_names: list of names for positional arguments of module. Must match field names in data batches
        :type input_names: list of str
        :param output_names: list of names for the outputs of the module
        :type output_names: list of str
        """
        super(EisenModuleWrapper, self).__init__()

        self.input_names = input_names
        self.output_names = output_names

        self.module = module

    def forward(self, *args, **kwargs):
        input_list = list(args)
        n_args = len(input_list)

        for key in kwargs.keys():
            if key in self.input_names[n_args:]:
                input_list.append(kwargs[key])

        outputs = self.module(*input_list)

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

        transform = CenterCrop((224, 224))

        adapted_transform = EisenTransformWrapper(transform, ['image'])

    """
    def __init__(self, transform, fields):
        super(EisenTransformWrapper, self).__init__()
        self.fields = fields

        self.transform = transform

    def __call__(self, data):
        for field in self.fields:
           data[field] = self.transform(data[field])

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

        dataset = MNIST('./', download=True)

        adapted_dataset = EisenDatasetWrapper(dataset, ['image', 'label'])

    """

    def __init__(self, dataset, field_names, transform=None):
        super(EisenDatasetWrapper, self).__init__()
        self.field_names = field_names

        self.dataset = dataset

        self.transform = transform

    def __getitem__(self, item):
        items = self.dataset[item]

        assert len(self.field_names) == len(items)

        ret_arg = {}

        for item, name in zip(items, self.field_names):
            ret_arg[name] = item

        ret_arg = self.transform(ret_arg)

        return ret_arg

    def __len__(self):
        return len(self.dataset)


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


class ModelParallel(Module):
    """
    This object implements model parallelism for PyTorch models. Model parallelism refers to the practice of using
    multiple GPUs for training by splitting layers across different GPUs. In this way huge models can be stored
    and trained. This module offers pipelined execution for model parallelism as shown in the PyTorch documentation:
    https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html#speed-up-by-pipelining-inputs

    Additionally, this module works in a completely automatic manner and it behaves similarly to
    torch.nn.DataParallel. The interface implemented here will look familiar to anyone using torch.nn.DataParallel

    .. warning::

        Only single input models can be parallelized via the current version of ModelParallel implemented here.
        Most models such as Resnet, VNet, Unet etc have a single input (for example a batch of images) therefore
        we trust that most use cases are covered by the current implementation.

    .. code-block:: python

        from eisen.utils import ModelParallel
        from eisen.models.segmentation import UNet

        # Transforming a model instance in a model parallel model instance

        model = ModelParallel(UNet(input_channels=1, output_channels=1), split_size=2)

        # model is ModelParallel and will execute on multiple GPUs

        """
    def __init__(self, module, split_size, device_ids=None, output_device=None):
        """
        This method instantiates a ModelParallel Module from a module instance passed by the user. The
        model must have a single input (forward(x) type of signature for the forward method) otherwise an error is
        returned.

        An example is here:

        .. code-block:: python

            from eisen.utils import ModelParallel
            from eisen.models.segmentation import UNet


            model = ModelParallel(
                module=UNet(input_channels=1, output_channels=1),
                split_size=2,
                device_ids=[0, 1, 2, 3],
                output_device=0
            )


        :param module: an instance of the model that should be parallelized
        :type module: torch.nn.Module
        :param split_size: split size for pipelined execution
        :type split_size: int
        :param device_ids: list of int or torch devices indicating GPUs to use
        :type device_ids: list
        :param output_device: int or torch device indicating output devices
        :type output_device: int or torch device
        """
        super(ModelParallel, self).__init__()

        module_argument_list = inspect.getfullargspec(module.forward)[0]

        if len(module_argument_list) > 2:
            raise NotImplementedError('Support for modules with more than one input is not yet implemented.')

        self.first_run = True
        self.split_size = split_size

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            self.first_run = False
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]

        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)

        if len(self.device_ids) == 1:
            self.first_run = False
            self.module.cuda(device_ids[0])

    def forward(self, x):
        execution_list = []

        if self.first_run:
            self.first_run = False

            input_size = x.size()

            # estimate module children size

            children = list(self.module.children())

            children_size = []

            for child in children:
                size, out_size = _estimate_modulesize(child, input_size)

                children_size.append(size)

                input_size = out_size

            partitions_idx = _get_n_idx_partitions(children_size, len(self.device_ids))

            for curr_gpu, indices in enumerate(partitions_idx):
                curr_gpu_name = torch.device("cuda:{}".format(self.device_ids[curr_gpu]))

                seq_list = []

                for idx in indices:
                    children[idx] = children[idx].to(curr_gpu_name)

                    seq_list.append(children[idx])

                sequential_block = torch.nn.Sequential(*seq_list)

                argument_placement_changer = InputArgumentPlacementChanger(curr_gpu_name)

                sequential_block.register_forward_pre_hook(argument_placement_changer)

                execution_list.append(sequential_block)

                self.module = PipelineExecutionStreamer(execution_list, split_size=self.split_size)

        outputs = self.module(x).to(self.output_device)

        return outputs

