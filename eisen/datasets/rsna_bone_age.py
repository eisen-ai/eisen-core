import csv
import os
import copy
import torch

from torch.utils.data import Dataset


def read_bone_age_kaggle_label_csv(file, training):
    data = {}

    with open(file) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            image_id = row[0]

            if training:
                male = row[2]
                bone_age = row[1]

                data[image_id] = {
                    'male': male == 'True',
                    'label': int(bone_age)
                }

            else:
                male = row[1]

                data[image_id] = {
                    'male': male == 'M'
                }

    return data


class RSNABoneAgeChallenge(Dataset):
    """
    This object implements the capability of reading the Kaggle RSNA Bone Age Estimation challenge dataset.
    Further information about this dataset can be found on the official website
    https://www.kaggle.com/kmader/rsna-bone-age

    Through this module, users are able to make use of the challenge data by simply specifying the directory where
    the data is locally stored. Therefore it is necessary to first download the data, store or unpack it in a specific
    directory and then instantiate an object of type RSNABoneAgeChallenge which will parse said directory and make the
    data available to Eisen.

    .. note::

        This dataset will return data points as dictionaries having fields: 'image', 'male' (boolean)
        and during training 'label'.

    .. code-block:: python

        from eisen.datasets import RSNABoneAgeChallenge

        dset = RSNABoneAgeChallenge('/data/root/path', True)

    This dataset will return data points as dictionaries having fields: 'image', 'male' (boolean) and during training
    'label'.

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

            from eisen.datasets import RSNABoneAgeChallenge

            dset = RSNABoneAgeChallenge(
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
            train_dir = os.path.join(self.data_dir, 'boneage-training-dataset', 'boneage-training-dataset')
            labels = read_bone_age_kaggle_label_csv(os.path.join(data_dir, 'boneage-training-dataset.csv'), True)
            images = [o for o in os.listdir(train_dir) if 'png' in o]

            for image in images:
                image_id = os.path.splitext(image)[0]

                item = {
                    'image': os.path.join('boneage-training-dataset', 'boneage-training-dataset', image),
                    'label': labels[image_id]['label'],
                    'male': labels[image_id]['male']
                }

                self.data.append(item)

        else:
            test_dir = os.path.join(self.data_dir, 'boneage-test-dataset', 'boneage-test-dataset')
            metadata = read_bone_age_kaggle_label_csv(os.path.join(data_dir, 'boneage-test-dataset.csv'), False)
            images = [o for o in os.listdir(test_dir) if 'png' in o]

            for image in images:
                image_id = os.path.splitext(image)[0]

                item = {
                    'image': os.path.join('boneage-test-dataset', 'boneage-test-dataset', image),
                    'male': metadata[image_id]['male']
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
