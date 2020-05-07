import os
import torch

from torch.utils.data import Dataset


class UCSDCovid19(Dataset):
    """
    This object allows the UCSD Covid-19 2D dataset to be easily imported into Eisen. Find more
    information about this dataset here: https://github.com/UCSD-AI4H/COVID-CT. This dataset is meant to be used
    for classification tasks. It also contains metadata which are currently NOT supported in Eisen.

    When instantiating this module it is necessary to point it to two directory names: one containing
    cases of sick patients, and the other containing images from healthy people (not affected by Covid-19)

    The first argument is the data base directory. The second and third argument should be strings
    representing the name of the two directories relative to the base directory and the fourth argument is a
    transform (or composition of transforms).

    Each entry of this dataset after loading will be a dictionary with two keys.
    The `'image'` key stores a path to a png file containing images, you can use `LoadPILImageFromFilename` IO module
    to read it, and the `'label'` key contains an integer that is `0` for healthy scans and `1` for sick individuals.

    .. note::

        This dataset will return data entries in form of a dictionary having fields: 'image' and 'label'

    .. code-block:: python

        from eisen.datasets import UCSDCovid19

        dataset = UCSDCovid19(
            '/abs/path/to/data',
            'positive',
            'negative',
            transform,
        )

    """
    def __init__(self, data_dir, positive_dir, negative_dir, transform=None):
        """
        :param data_dir: the base directory where the data is located (dataset location after unzipping)
        :type data_dir: str
        :param positive_dir: relative path of directory containing positive cases
        :type positive_dir: str
        :param negative_dir: relative path of directory containing negative cases
        :type negative_dir: string
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: callable

        .. code-block:: python

            from eisen.datasets import MedSegCovid19

            dataset = MedSegCovid19(
                image_file='tr_im.nii',
                mask_file='tr_mask.nii',
                transform=transform,
            )

        <json>
        [
            {"name": "image_file", "type": "string", "value": ""},
            {"name": "mask_file", "type": "string", "value": ""}
        ]
        </json>
        """

        self.data = []

        positive_path = os.path.join(data_dir, positive_dir)
        negative_path = os.path.join(data_dir, negative_dir)

        positive_images = [os.path.join(positive_dir, img) for img in os.listdir(positive_path) if 'png' in img]
        negative_images = [os.path.join(negative_dir, img) for img in os.listdir(negative_path) if 'png' in img]

        for img_path in positive_images:
            self.data.append({'image': img_path, 'label': 1})

        for img_path in negative_images:
            self.data.append({'image': img_path, 'label': 0})

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        return item
