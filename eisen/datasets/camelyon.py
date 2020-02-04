import h5py
import torch
import os

from torch.utils.data import Dataset


class PatchCamelyon(Dataset):
    """
    This object implements the capability of reading PatchCamelyon data. Further information about this dataset
    can be found on the official website https://patchcamelyon.grand-challenge.org/Introduction/

    .. code-block:: python

        from eisen.datasets import PatchCamelyon

        dset = PatchCamelyon(
            '/data/root/path',
            'camelyon_patch_level_2_split_train_x.h5',
            'camelyon_patch_level_2_split_train_y.h5',
            'camelyon_patch_level_2_split_train_mask.h5'
        )

    """
    def __init__(self, data_dir, x_h5_file, y_h5_file, mask_h5_file=None, transform=None):
        """
        :param data_dir: the base directory where the data is located
        :type data_dir: str
        :param x_h5_file: the relative path of the H5 file containing x (the images)
        :type x_h5_file: str
        :param y_h5_file: the relative path of the H5 file containing y (the labels)
        :type y_h5_file: str
        :param mask_h5_file: the relative path of the H5 file containing masks
        :type mask_h5_file: str
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: object

        .. code-block:: python

            from eisen.datasets import PatchCamelyon

            dset = PatchCamelyon(
                data_dir='/data/root/path',
                x_h5_file='camelyon_patch_level_2_split_train_x.h5',
                y_h5_file='camelyon_patch_level_2_split_train_y.h5',
                mask_h5_file='camelyon_patch_level_2_split_train_mask.h5',
                transform=transform
            )

        <json>
        [
            {"name": "x_h5_file", "type": "string", "value": ""},
            {"name": "y_h5_file", "type": "string", "value": ""},
            {"name": "mask_h5_file", "type": "string", "value": ""}
        ]
        </json>
        """

        self.x = h5py.File(os.path.join(data_dir, x_h5_file), 'r')['x']
        self.y = h5py.File(os.path.join(data_dir, y_h5_file), 'r')['y']

        assert len(self.x) == len(self.y)

        if mask_h5_file:
            self.mask = h5py.File(os.path.join(data_dir, mask_h5_file), 'r')['mask']

            assert len(self.x) == len(self.mask)

        else:
            self.mask = None

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = {
            'x': self.x[idx],
            'y': self.y[idx]
        }

        if self.mask:
            item['mask'] = self.mask[idx]

        if self.transform:
            item = self.transform(item)

        return item
