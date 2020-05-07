import csv
import os

from torch.utils.data import Dataset


class PANDA(Dataset):
    """
    This object implements the capability of reading PANDA dataset. Further information about this dataset
    can be found on the official website https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview

    Through this module, users are able to make use of the challenge data by simply specifying the directory where
    the data is locally stored. Therefore it is necessary to first download the data, store or unpack it in a specific
    directory and then instantiate an object of type PANDA which will make use of the data in the directory as
    well as the csv file that are part of the dataset and make it available to Eisen.

    .. note::

        This dataset will return data points in form of a dictionary with fields: 'image', 'provider' and optionally
        (during training) 'mask', 'isup', 'gleason'.

    .. code-block:: python

        from eisen.datasets import PANDA

        dset = PANDA(
            '/data/root/path',
            'train.csv',
            True
        )

    """
    def __init__(self, data_dir, csv_file, training, transform=None):
        """
        :param data_dir: the base directory where the data is located
        :type data_dir: str
        :param csv_file: the relative path of the csv file relative to current task
        :type csv_file: str
        :param training: whether the dataset is a training dataset or not
        :type training: bool
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: callable

        .. code-block:: python

            from eisen.datasets import PANDA

            dset = PANDA(
                data_dir='/data/root/path',
                csv_file='train.csv',
                training=True,
                transform=transform
            )

        <json>
        [
            {"name": "csv_file", "type": "string", "value": ""},
            {"name": "training", "type": "bool", "value": ""}
        ]
        </json>
        """

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.training = training

        self.dataset = []

        with open(os.path.join(self.data_dir, self.csv_file), 'r') as f:
            reader = csv.reader(f)

            for i, row in enumerate(reader):
                if i == 0:
                    continue

                if self.training:
                    entry = {
                        'image': os.path.join('train_images', row[0] + '.tiff'),
                        'mask': os.path.join('train_label_masks', row[0] + '_mask.tiff'),
                        'provider': row[1],
                        'isup': int(row[2]),
                        'gleason': row[3]
                    }
                else:
                    entry = {
                        'image': os.path.join('test_images', row[0] + '.tiff'),
                        'provider': row[1]
                    }

                self.dataset.append(entry)

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.transform:
            item = self.transform(item)

        return item
