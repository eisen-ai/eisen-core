import os
import torch

from torch.utils.data import Dataset


class EMIDEC(Dataset):
    """
    This object allows Data from the EMIDEC challenge (2020) data to be easily impoted in Eisen.
    More information about the data and challenge can be found here http://emidec.com

    Through this module, users are able to make use of the challenge data by simply specifying the directory where
    the data is locally stored. Therefore it is necessary to first download the data, store or unpack it in a specific
    directory and then instantiate an object of type ABCDataset which will make use of the directory structure and
    the descriptive json file included in it and make the data available to Eisen.

    For what concerns labels and data structure refer to the official website http://emidec.com

    .. note::

        This dataset returns the following fields: image, metadata and - during training - pathological and label.

    .. code-block:: python

        from eisen.datasets import EMIDEC

        dataset = EMIDEC(
            '/abs/path/to/data',
            True,
            False,
            transform,
        )

    """
    def __init__(self, data_dir, training, transform=None):
        """
        :param data_dir: the base directory where the data is located (dataset location after unzipping)
        :type data_dir: str
        :param training: whether data relative to the training phase should be loaded
        :type training: bool
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: callable

        .. code-block:: python

            from eisen.datasets import EMIDEC

            dataset = EMIDEC(
                data_dir='/abs/path/to/data',
                training=True,
                transform=transform,
            )

        <json>
        [
            {"name": "training", "type": "bool", "value": "True"}
        ]
        </json>
        """

        self.data_dir = data_dir
        self.training = training
        self.transform = transform

        self.dataset = []

        directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and ('.' not in d)]

        for directory in directories:
            image_rel_path = os.path.join(directory, 'Images', directory + '.nii.gz')

            with open(
                    os.path.join(data_dir, directory.replace('_', ' ') + '.txt'), 'r',
                    encoding='utf-8', errors='ignore'
            ) as file:
                metadata = file.read().replace('\n', '')

            entry = {
                'image': image_rel_path,
                'metadata': metadata,
            }

            if self.training:
                label_rel_path = os.path.join(directory, 'Contours', directory + '.nii.gz')
                entry['label'] = label_rel_path

                if 'P' in directory:
                    entry['pathological'] = True
                else:
                    entry['pathological'] = False

            self.dataset.append(entry)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.dataset[idx]

        if self.transform:
            item = self.transform(item)

        return item
