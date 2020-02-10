import numpy as np
import tempfile
import h5py
import os

from eisen.datasets import PatchCamelyon
from eisen.datasets import JsonDataset
from eisen.datasets import ListDataset
from eisen.datasets import MSDDataset


class TestLoadPatchCamelyon:
    def setup_class(self):
        data_x = [
            np.zeros([32, 32, 3], dtype=np.float32),
            np.ones([32, 32, 3], dtype=np.float32),
        ]

        data_y = [
            np.asarray([[[0]]]),
            np.asarray([[[1]]]),
        ]

        self.base_path = tempfile.mkdtemp()

        self.file_name_x = 'data_x.h5'
        self.file_name_y = 'data_y.h5'

        h5f_x = h5py.File(os.path.join(self.base_path, self.file_name_x), 'w')
        h5f_x.create_dataset('x', data=data_x)

        h5f_y = h5py.File(os.path.join(self.base_path, self.file_name_y), 'w')
        h5f_y.create_dataset('y', data=data_y)

        self.camelyon_dset = PatchCamelyon(self.base_path, self.file_name_x, self.file_name_y)

    def test_getitem(self):
        self.item = self.camelyon_dset[0]

        assert np.all(self.item['image'] == 0)
        assert np.all(self.item['label'] == 0)

        assert self.item['image'].shape[0] == 3
        assert self.item['image'].shape[1] == 32
        assert self.item['image'].shape[2] == 32

        assert self.item['label'].shape[0] == 1

    def test_len(self):
        assert len(self.camelyon_dset) == 2
