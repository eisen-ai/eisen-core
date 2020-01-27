import os
import torch

from torch.utils.data import Dataset
from eisen.utils import read_json_from_file, check_arg_type


class MSDDataset(Dataset):
    def __init__(self, data_dir, json_file, phase, transform=None):
        """
        :param data_dir: the base directory where the data is located
        :type data_dir: str
        :param json_file: the name of the json file containing the MSD data
        :type json_file: str
        :param phase: training or test phase
        :type phase: string
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: object

        <json>
        [
            {"name": "data_dir", "type": "string", "value": ""},
            {"name": "json_file", "type": "string", "value": ""},
            {"name": "phase", "type": "string", "value": ["training", "test"]}
        ]
        </json>
        """
        json_file = os.path.join(data_dir, json_file)

        json_dataset = read_json_from_file(json_file)

        self.json_dataset = json_dataset[phase]

        json_dataset.pop('training', None)
        json_dataset.pop('test', None)

        self.attributes = json_dataset

        check_arg_type(self.json_dataset, list, 'json content')

        self.transform = transform

    def __len__(self):
        return len(self.json_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.json_dataset[idx]

        if self.transform:
            sample = self.transform(item)

        return sample
