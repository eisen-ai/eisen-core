import inspect
import os
import json
import inspect

from torch.nn import Module


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def check_arg_type(arg, type, arg_name=''):
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    if not isinstance(arg, type):
        raise TypeError('The {} passed to function {} has erroneous type {} vs. torch.nn.Module'.format(
            arg_name,
            calframe[1][3],
            arg.__class__.__name__
        ))


def read_json_from_file(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError('The JSON file {} cannot be read'.format(json_file))

    with open(json_file) as json_file:
        dictionary = json.load(json_file)

    return dictionary


class EisenModuleWrapper(Module):
    """
    This object implements a wrapper allowing models to be used in a Eisen Workflow/context
    The problem we attempt to solve with this wrapper is to manage all the data and arguments passed to training
    as dictionaries. In this way we can wrap torch.nn.Modules in a way allowing them to get fed and return dictionaries.
    This implementation is unfortunately not particularly elegant or clear from the code. Unfortunately it is needed
    """
    def __init__(self, module, input_names, output_names):
        super(EisenModuleWrapper, self).__init__()

        self.input_names = input_names
        self.output_names = output_names

        self.module = module

        module_argument_list = inspect.getfullargspec(self.module.forward)[0]

        module_argument_list.remove('self')

        self.module_argument_list = module_argument_list

    def forward(self, **kwargs):
        input_dict = {}

        for dst_arg, src_arg in zip(self.module_argument_list, self.input_names):
            input_dict[dst_arg] = kwargs[src_arg]

        outputs = self.module(**input_dict)

        outputs = (outputs,)

        ret_dict = {}

        for output, output_name in zip(outputs, self.output_names):
            ret_dict[output_name] = output

        return ret_dict