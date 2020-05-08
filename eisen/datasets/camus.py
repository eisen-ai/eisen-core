import os
import torch
import copy

from torch.utils.data import Dataset


class CAMUS(Dataset):
    """
    This object implements the capability of reading CAMUS data. The CAMUS dataset is a dataset of ultrasound
    images of the heart. Further information about this dataset can be found on the official website
    https://www.creatis.insa-lyon.fr/Challenge/camus/index.html

    Through this module, users are able to make use of the challenge data by simply specifying the directory where
    the data is locally stored. Therefore it is necessary to first download the data, store or unpack it in a specific
    directory and then instantiate an object of type CAMUS which will make use of the data in the directory
    and make it available to Eisen.

    .. note::

        This dataset will generate data entries with keys: 'type', 'image_2CH', 'label_2CH', 'sequence_2CH',
        'image_4CH', 'label_4CH', sequence_4CH depending on the selected input parameter configuration.
        The data generated consists of paths to images and type (string).

    .. code-block:: python

        from eisen.datasets import CAMUS

        dset = CAMUS('/data/root/path')

    """
    def __init__(
            self,
            data_dir,
            with_ground_truth,
            with_2CH=True,
            with_4CH=True,
            with_entire_sequences=False,
            transform=None
    ):
        """
        :param data_dir: the base directory where the data is located
        :type data_dir: str
        :param with_ground_truth: whether ground truth annotation should be included (won't work during testing)
        :type with_ground_truth: bool
        :param with_2CH: whether 2 chambers data should be included (default True)
        :type with_2CH: bool
        :param with_4CH: whether 4 chambers data should be included (default True)
        :type with_4CH: bool
        :param with_entire_sequences: whether the entire sequences for 4CH and 2CH data should be included (default False)
        :type with_entire_sequences: bool
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: callable

        .. code-block:: python

            from eisen.datasets import CAMUS

            dset = CAMUS(
                data_dir='/data/root/path',
                with_ground_truth=True,
                with_2CH=True,
                with_4CH=True,
                with_entire_sequences=False
                transform=None
            )

        <json>
        [
            {"name": "with_ground_truth", "type": "bool", "value": ""},
            {"name": "with_2CH", "type": "bool", "value": "true"},
            {"name": "with_4CH", "type": "bool", "value": "true"},
            {"name": "with_entire_sequences", "type": "bool", "value": "false"}
        ]
        </json>
        """
        self.data_dir = data_dir

        self.with_ground_truth = with_ground_truth
        self.with_2CH = with_2CH
        self.with_4CH = with_4CH
        self.with_entire_sequences = with_entire_sequences

        self.data = []

        all_subdirs = [o for o in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, o))]

        for dir_name in all_subdirs:

            dir = os.path.join(self.data_dir, dir_name)

            if len(os.listdir(dir)) == 0:
                print('WARNING: dataset directory {} appears empty'.format(dir))
                continue

            for typ in ['ED', 'ES']:
                item = dict()

                item['type'] = typ

                if self.with_2CH:
                    item['image_2CH'] = os.path.join(dir_name, '{}_2CH_{}.mhd'.format(dir_name, typ))

                    if self.with_ground_truth:
                        item['label_2CH'] = os.path.join(dir_name, '{}_2CH_{}_gt.mhd'.format(dir_name, typ))

                    if self.with_entire_sequences:
                        item['sequence_2CH'] = os.path.join(dir_name, '{}_2CH_sequence.mhd'.format(dir_name))

                if self.with_4CH:
                    item['image_4CH'] = os.path.join(dir_name, '{}_4CH_{}.mhd'.format(dir_name, typ))

                    if self.with_ground_truth:
                        item['label_4CH'] = os.path.join(dir_name, '{}_4CH_{}_gt.mhd'.format(dir_name, typ))

                    if self.with_entire_sequences:
                        item['sequence_4CH'] = os.path.join(dir_name, '{}_4CH_sequence.mhd'.format(dir_name))

                self.data.append(item)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = copy.deepcopy(self.data[idx])

        if self.transform:
            item = self.transform(item)

        return item
