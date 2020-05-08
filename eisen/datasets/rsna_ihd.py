import csv
import os
import copy
import torch
import numpy as np

from torch.utils.data import Dataset


CLASSES = [
    'epidural',
    'intraparenchymal',
    'intraventricular',
    'subarachnoid',
    'subdural',
    'any'
]


def read_rsna_kaggle_label_csv(file):
    data = {}

    with open(file) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            field = row[0].split('_')
            image_id = field[0] + '_' + field[1]
            cls = field[2]

            proba = float(row[1])

            if image_id not in data.keys():
                data[image_id] = {}

            data[image_id][cls] = proba

    return data


def label_dict_to_array(label_dictionary):
    label_array = []

    for cls in CLASSES:
        label_array.append(label_dictionary[cls])

    return np.asarray(label_array)


class RSNAIntracranialHemorrhageDetection(Dataset):
    """
    This object implements the capability of reading the Kaggle RSNA Intracranial Hemorrhage Detection dataset.
    Further information about this dataset can be found on the official website
    https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview

    Through this module, users are able to make use of the challenge data by simply specifying the directory where
    the data is locally stored. Therefore it is necessary to first download the data, store or unpack it in a specific
    directory and then instantiate an object of type RSNAIntracranialHemorrhageDetection which will parse said
    directory and make the data available to Eisen.

    .. note::

        This dataset will return data points in form of a dictionary having keys: 'image' and during training
        'label' as well.


    .. code-block:: python

        from eisen.datasets import RSNAIntracranialHemorrhageDetection

        dset = RSNAIntracranialHemorrhageDetection('/data/root/path', True)



    """
    def __init__(self, data_dir, training, transform=None):
        """
        :param data_dir: The dataset root path directory where the challenge dataset is stored
        :type data_dir: str
        :param training: Boolean indicating whether training or test data should be loaded
        :type training: bool
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: callable

        .. code-block:: python

            from eisen.datasets import RSNAIntracranialHemorrhageDetection

            dset = RSNAIntracranialHemorrhageDetection(
                data_dir='/data/root/path',
                training=True
            )

        <json>
        [
            {"name": "training", "type": "bool", "value": ""}
        ]
        </json>
        """
        self.data_dir = data_dir
        self.training = training
        self.transform = transform

        self.data = []

        if self.training:
            train_dir = os.path.join(self.data_dir, 'stage_2_train')
            labels = read_rsna_kaggle_label_csv(os.path.join(data_dir, 'stage_2_train.csv'))
            images = [o for o in os.listdir(train_dir) if 'dcm' in o]

            for image in images:
                image_id = os.path.splitext(image)[0]

                label_array = label_dict_to_array(labels[image_id])

                item = {
                    'image': os.path.join('stage_2_train', image),
                    'label': label_array
                }

                self.data.append(item)

        else:
            test_dir = os.path.join(self.data_dir, 'stage_2_test')
            images = [o for o in os.listdir(test_dir) if 'dcm' in o]

            for image in images:
                item = {
                    'image': os.path.join('stage_2_test', image),
                }

                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = copy.deepcopy(self.data[idx])

        if self.transform:
            item = self.transform(item)

        return item
