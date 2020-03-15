import numpy as np
import tempfile
import h5py
import os
import SimpleITK as sitk

from eisen.datasets import PatchCamelyon
from eisen.datasets import JsonDataset
from eisen.datasets import MSDDataset
from eisen.datasets import CAMUS


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


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


class TestLoadCAMUS:
    def setup_class(self):
        self.base_path = tempfile.mkdtemp()

        os.makedirs(os.path.join(self.base_path, 'patient0001'), exist_ok=True)

        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ED.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ES.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ED.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ES.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ED_gt.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ES_gt.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ED_gt.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ES_gt.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_sequence.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_sequence.mhd'))

        self.camus_dataset = CAMUS(
            self.base_path,
            with_ground_truth=True,
            with_2CH=True,
            with_4CH=True,
            with_entire_sequences=True
        )

    def test_getitem(self):
        self.item = self.camus_dataset[0]

        assert self.item['image_2CH'] == str(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ED.mhd'))
        assert self.item['image_4CH'] == str(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ED.mhd'))

        assert self.item['label_2CH'] == str(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ED_gt.mhd'))
        assert self.item['label_4CH'] == str(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ED_gt.mhd'))

        assert self.item['sequence_2CH'] == \
            str(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_sequence.mhd'))
        assert self.item['sequence_4CH'] == \
            str(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_sequence.mhd'))

        self.item = self.camus_dataset[1]

        assert self.item['image_2CH'] == str(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ES.mhd'))
        assert self.item['image_4CH'] == str(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ES.mhd'))

        assert self.item['label_2CH'] == str(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ES_gt.mhd'))
        assert self.item['label_4CH'] == str(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ES_gt.mhd'))

        assert self.item['sequence_2CH'] == \
            str(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_sequence.mhd'))
        assert self.item['sequence_4CH'] == \
            str(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_sequence.mhd'))

    def test_len(self):
        assert len(self.camus_dataset) == 2
