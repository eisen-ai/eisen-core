import csv
import os

from torch.utils.data import Dataset


class KaggleCovid19(Dataset):
    """
    This object implements the capability of reading Kaggle COVID 19 CT dataset. Further information about this dataset
    can be found on the official website https://www.kaggle.com/andrewmvd/covid19-ct-scans

    Through this module, users are able to make use of the challenge data by simply specifying the directory where
    the data is locally stored. Therefore it is necessary to first download the data, store or unpack it in a specific
    directory and then instantiate an object of type KaggleCovid19 which will make use of the data in the directory as
    well as the csv file that are part of the dataset and make it available to Eisen.

    .. note::
        This dataset will generate data entries with fields: 'image', 'lung_mask', 'infection_mask',
        'lung_infection_mask'. This data is returned in form of relative paths (to data_dir) of image and mask files.

    .. code-block:: python

        from eisen.datasets import KaggleCovid19

        dset = KaggleCovid19(
            '/data/root/path',
            'metadata.csv',
        )

    """
    def __init__(self, data_dir, csv_file, transform=None):
        """
        :param data_dir: the base directory where the data is located
        :type data_dir: str
        :param csv_file: the relative path of the csv file relative to current task
        :type csv_file: str
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: callable

        .. code-block:: python

            from eisen.datasets import KaggleCovid19

            dset = KaggleCovid19(
                data_dir='/data/root/path',
                csv_file='metadata.csv',
                transform=transform
            )

        <json>
        [
            {"name": "csv_file", "type": "string", "value": ""}
        ]
        </json>
        """

        self.data_dir = data_dir
        self.csv_file = csv_file

        self.dataset = []

        with open(os.path.join(self.data_dir, self.csv_file), 'r') as f:
            reader = csv.reader(f)

            for i, row in enumerate(reader):
                if i == 0:
                    continue

                entry = {
                    'image': os.path.join('ct_scans', os.path.basename(row[0])),
                    'lung_mask': os.path.join('lung_mask', os.path.basename(row[1])),
                    'infection_mask': os.path.join('infection_mask', os.path.basename(row[2])),
                    'lung_infection_mask': os.path.join('lung_and_infection_mask', os.path.basename(row[3])),
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
