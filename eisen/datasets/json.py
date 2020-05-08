import os
import torch

from torch.utils.data import Dataset
from eisen.utils import read_json_from_file


class JsonDataset(Dataset):
    """
    This object implements the capability of reading arbitrary data contained in properly structured JSON file
    into Eisen. The expected JSON file structure is a list of dictionaries. Each entry of the list contains
    one element of the dataset. Each key of the dictionary stores different information about that data point.

    Example of JSON structure:

    .. code-block:: json

        [
            {'image': 'image_file1.png', 'label': 'label_file1.png'},
            {'image': 'image_file2.png', 'label': 'label_file2.png'}
        ]

    .. note::

        This dataset will generate data entries with fields corresponding to what is stored in each entry of the json
        dataset list.

    .. code-block:: python

        from eisen.datasets import JsonDataset
        dset = JsonDataset('/abs/path/to/data', '/abs/path/to/file.json', transform)

    """
    def __init__(self, data_dir, json_file, transform=None):
        """
        :param data_dir: the base directory where the data is located
        :type data_dir: str
        :param json_file: the name of the json file containing the data
        :type json_file: str
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: callable

        .. code-block:: python

            from eisen.datasets import JsonDataset
            dset = JsonDataset(
                data_dir='/abs/path/to/data',
                json_file='/abs/path/to/file.json',
                transform=transform
            )

        <json>
        [
            {"name": "json_file", "type": "string", "value": ""}
        ]
        </json>
        """
        json_file = os.path.join(data_dir, json_file)

        self.json_dataset = read_json_from_file(json_file)

        self.transform = transform

    def __len__(self):
        return len(self.json_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.json_dataset[idx]

        if self.transform:
            item = self.transform(item)

        return item
