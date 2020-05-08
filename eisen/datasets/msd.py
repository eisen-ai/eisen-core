import os
import torch

from torch.utils.data import Dataset
from eisen.utils import read_json_from_file


class MSDDataset(Dataset):
    """
    This object allows Medical Segmentation Decathlon data to be easily impoted in Eisen.
    More information about the data can be found here http://medicaldecathlon.com

    Through this module, users are able to make use of the challenge data by simply specifying the directory where
    the data is locally stored. Therefore it is necessary to first download the data, store or unpack it in a specific
    directory and then instantiate an object of type MSDDataset which will make use of the directory structure and
    the descriptive json file included in it and make the data available to Eisen.

    .. note::

        This dataset will return data items with fields: 'image' and, optionally, 'label'.

    .. code-block:: python

        from eisen.datasets import MSDDataset

        dataset = MSDDataset(
            '/abs/path/to/data',
            '/path/to/dataset.json',
            'training',
            transform,
        )

    """
    def __init__(self, data_dir, json_file, phase, transform=None):
        """
        :param data_dir: the base directory where the data is located (dataset location after unzipping)
        :type data_dir: str
        :param json_file: the name of the json file containing for the MSD dataset
        :type json_file: str
        :param phase: training or test phase as per MSD dataset convention (look at MSD json file)
        :type phase: string
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: callable

        .. code-block:: python

            from eisen.datasets import MSDDataset

            dataset = MSDDataset(
                data_dir='/abs/path/to/data',
                json_file='/path/to/dataset.json',
                phase='training',
                transform=transform,
            )

        <json>
        [
            {"name": "json_file", "type": "string", "value": ""},
            {"name": "phase", "type": "string", "value": ["training", "test"]}
        ]
        </json>
        """
        json_file = os.path.join(data_dir, json_file)

        msd_dataset = read_json_from_file(json_file)

        self.json_dataset = msd_dataset[phase]

        msd_dataset.pop('training', None)
        msd_dataset.pop('test', None)

        if phase == 'test':
            # test images are stored as list of filenames instead of dictionaries. Need to convert that.
            dset = []
            for elem in self.json_dataset:
                dset.append({'image': elem})

            self.json_dataset = dset

        self.attributes = msd_dataset

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
