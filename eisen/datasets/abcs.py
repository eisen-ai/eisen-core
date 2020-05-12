import os
import torch

from torch.utils.data import Dataset


class ABCsDataset(Dataset):
    """
    This object allows Data from the ABC challenge (2020) data to be easily impoted in Eisen.
    More information about the data and challenge can be found here https://abcs.mgh.harvard.edu

    Through this module, users are able to make use of the challenge data by simply specifying the directory where
    the data is locally stored. Therefore it is necessary to first download the data, store or unpack it in a specific
    directory and then instantiate an object of type ABCDataset which will make use of the directory structure and
    the descriptive json file included in it and make the data available to Eisen.

    For what concerns labels and data structure refer to:
    https://abcs.mgh.harvard.edu/index.php/data/download/s end/3-data-for-abcs/14-readme

    .. note::

        This dataset returns the following fields: 'ct', 't1', 't2' and 'label_task1', 'label_task2' when training.
        The content of these fields consists of paths relative to data_dir, to ct, MR and labels.

        Get started code can be found here: https://gist.github.com/faustomilletari/af430acfecf0841d71508455cdadcbbf

    .. code-block:: python

        from eisen.datasets import ABCsDataset

        dataset = ABCsDataset(
            '/abs/path/to/data',
            True,
            False,
            transform,
        )

    """
    def __init__(self, data_dir, training, flat_dir_structure=False, transform=None):
        """
        :param data_dir: the base directory where the data is located (dataset location after unzipping)
        :type data_dir: str
        :param training: whether data relative to the training phase should be loaded
        :type training: bool
        :param flat_dir_structure: whether data is stored in a directory containing all images (without sub-dirs)
        :type flat_dir_structure: bool
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: callable

        .. code-block:: python

            from eisen.datasets import ABCsDataset

            dataset = ABCsDataset(
                data_dir='/abs/path/to/data',
                training=True,
                flat_dir_structure=False,
                transform=transform,
            )

        <json>
        [
            {"name": "training", "type": "bool", "value": "True"},
            {"name": "flat_dir_structure", "type": "bool", "value": "False"}
        ]
        </json>
        """
        self.data_dir = data_dir
        self.flat_dir = flat_dir_structure
        self.training = training

        self.dataset = []

        if self.flat_dir:
            all_images = [
                f.split('_')[0] for f in os.listdir(self.data_dir) if
                ('mha' in f) and
                ('label' not in f) and
                ('ct' in f)
            ]

        else:
            all_subdirs = [
                d for d in os.listdir(self.data_dir) if
                os.path.isdir(os.path.join(self.data_dir, d))
            ]

            all_images = []

            for d in all_subdirs:
                all_images += [
                    os.path.join(d, f.split('_')[0]) for f in os.listdir(os.path.join(self.data_dir, d)) if
                    ('mha' in f) and
                    ('label' not in f) and
                    ('ct' in f)
                ]

        if self.training:
            all_task1_labels = []
            all_task2_labels = []
            ct_images = []
            t1_images = []
            t2_images = []

            for image in all_images:
                all_task1_labels.append(image + '_labelmap_task1.mha')
                all_task2_labels.append(image + '_labelmap_task2.mha')

                ct_images.append(image + '_ct.mha')
                t1_images.append(image + '_t1.mha')
                t2_images.append(image + '_t2.mha')

            for tsk1, tsk2, ct, t1, t2 in zip(all_task1_labels, all_task2_labels, ct_images, t1_images, t2_images):
                self.dataset.append({
                    'ct': ct,
                    't1': t1,
                    't2': t2,
                    'label_task1': tsk1,
                    'label_task2': tsk2
                })

        else:
            ct_images = []
            t1_images = []
            t2_images = []

            for image in all_images:
                ct_images.append(image + '_ct.mha')
                t1_images.append(image + '_t1.mha')
                t2_images.append(image + '_t2.mha')

            for ct, t1, t2 in zip(ct_images, t1_images, t2_images):
                self.dataset.append({
                    'ct': ct,
                    't1': t1,
                    't2': t2
                })

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.dataset[idx]

        if self.transform:
            item = self.transform(item)

        return item


class ABCDataset(ABCsDataset):
    def __init__(self, *args, **kwargs):
        print('Warning: ABCDataset is the old name for ABCsDataset and is deprecated. Use ABCsDataset instead')
        super(ABCDataset, self).__init__(*args, **kwargs)
