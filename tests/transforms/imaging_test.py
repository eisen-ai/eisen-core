import numpy as np
import nibabel as nib
import copy

from eisen.transforms.imaging import CreateConstantFlags
from eisen.transforms.imaging import RenameFields
from eisen.transforms.imaging import FilterFields
from eisen.transforms.imaging import ResampleNiftiVolumes
from eisen.transforms.imaging import NiftiToNumpy
from eisen.transforms.imaging import NumpyToNifti
from eisen.transforms.imaging import CropCenteredSubVolumes
from eisen.transforms.imaging import MapValues


class TestCreateConstantFlags:
    def setup_class(self):
        self.data = {
            'image': np.random.rand(32, 32, 3),
            'label': 1
        }

        self.tform_one = CreateConstantFlags(['flag1', 'flag2'], [32.2, 42.0])
        self.tform_two = CreateConstantFlags(['flag3', 'flag4', 'flag5'], ['flag3', 42, False])

    def test_call(self):
        self.data = self.tform_one(self.data)

        assert 'flag1' in self.data.keys()
        assert 'flag2' in self.data.keys()

        assert self.data['flag1'] == 32.2
        assert self.data['flag2'] == 42.0

        self.data = self.tform_two(self.data)

        assert 'flag3' in self.data.keys()
        assert 'flag4' in self.data.keys()
        assert 'flag5' in self.data.keys()

        assert self.data['flag3'] == 'flag3'
        assert self.data['flag4'] == 42
        assert self.data['flag5'] is False


class TestRenameFields:
    def setup_class(self):
        self.data = {
            'image': np.ones([32, 32, 3], dtype=np.float32),
            'label': 0
        }

        self.tform_one = RenameFields(['image', 'label'], ['new_image', 'new_label'])
        self.tform_two = RenameFields(['new_image'], ['image'])

    def test_call(self):
        self.data = self.tform_one(self.data)

        assert 'new_image' in self.data.keys()
        assert 'new_label' in self.data.keys()

        assert 'image' not in self.data.keys()
        assert 'label' not in self.data.keys()

        assert np.all(self.data['new_image'] == 1)
        assert self.data['new_label'] == 0

        self.data = self.tform_two(self.data)

        assert 'new_image' not in self.data.keys()
        assert 'image' in self.data.keys()

        assert np.all(self.data['image'] == 1)


class TestFilterFields:
    def setup_class(self):
        self.data = {
            'image': np.ones([32, 32, 3], dtype=np.float32),
            'label': 0
        }

        self.tform_one = FilterFields(['image', 'label'])
        self.tform_two = FilterFields(['image'])

    def test_call(self):
        self.data = self.tform_one(self.data)

        assert 'image' in self.data.keys()
        assert 'label' in self.data.keys()

        assert np.all(self.data['image'] == 1)
        assert self.data['label'] == 0

        self.data = self.tform_two(self.data)

        assert 'label' not in self.data.keys()
        assert 'image' in self.data.keys()

        assert np.all(self.data['image'] == 1)


class TestResampleNiftiVolumes:
    def setup_class(self):
        data = np.ones([32, 32, 32]).astype(np.float32)

        data = data * np.asarray(range(32))

        img = nib.Nifti1Image(data, np.eye(4))

        self.data = {
            'image': img,
            'label': 0
        }

        self.tform_one = ResampleNiftiVolumes(['image'], [0.5, 0.5, 0.5], interpolation='linear')
        self.tform_two = ResampleNiftiVolumes(['image'], [1.0, 1.0, 1.0], interpolation='linear')
        self.tform_three = ResampleNiftiVolumes(['image'], [2.0, 2.0, 2.0], interpolation='linear')

    def test_call(self):
        self.data = self.tform_one(self.data)

        assert 'image' in self.data.keys()
        assert 'label' in self.data.keys()

        assert self.data['image'].shape[0] == 63
        assert self.data['image'].shape[1] == 63
        assert self.data['image'].shape[2] == 63

        dta = np.asanyarray(self.data['image'].dataobj)

        assert np.max(dta) == 31
        assert np.min(dta) == 0
        assert np.all(dta[0, 0, :] == np.arange(0, 31.5, step=0.5))

        self.data = self.tform_two(self.data)

        assert 'image' in self.data.keys()
        assert 'label' in self.data.keys()

        assert self.data['image'].shape[0] == 32
        assert self.data['image'].shape[1] == 32
        assert self.data['image'].shape[2] == 32

        dta = np.asanyarray(self.data['image'].dataobj)

        assert np.max(dta) == 31
        assert np.min(dta) == 0
        assert np.all(dta[0, 0, :] == np.arange(0, 32, step=1))

        self.data = self.tform_three(self.data)

        assert 'image' in self.data.keys()
        assert 'label' in self.data.keys()

        assert self.data['image'].shape[0] == 17
        assert self.data['image'].shape[1] == 17
        assert self.data['image'].shape[2] == 17

        dta = np.asanyarray(self.data['image'].dataobj)

        assert np.max(dta) == 30
        assert np.min(dta) == 0
        assert np.all(dta[0, 0, 0:16] == np.arange(0, 32, step=2))


class TestNiftiToNumpy:
    def setup_class(self):
        self.np_data = np.random.rand(32, 32, 32).astype(np.float32)

        self.np_label = np.random.rand(32, 32, 32, 3).astype(np.float32)

        img = nib.Nifti1Image(self.np_data, np.eye(4))

        lbl = nib.Nifti1Image(self.np_label, np.eye(4))

        self.data = {
            'image': img,
            'label': lbl
        }

        self.tform_one = NiftiToNumpy(['image'])
        self.tform_two = NiftiToNumpy(['label'], multichannel=True)

    def test_call(self):
        self.data = self.tform_one(self.data)

        assert isinstance(self.data['image'], np.ndarray)

        assert self.data['image'].dtype == np.float32
        assert np.all(self.np_data == self.data['image'])

        self.data = self.tform_two(self.data)

        assert isinstance(self.data['label'], np.ndarray)

        assert self.data['label'].dtype == np.float32
        assert np.all(self.np_label == self.data['label'].transpose([1, 2, 3, 0]))
        assert self.data['label'].shape[0] == 3


class TestNumpyToNifti:

    def setup_class(self):
        self.np_img = np.random.rand(32, 32, 32).astype(np.float32)
        self.np_lbl = np.random.rand(32, 32, 32).astype(np.float32)
        self.img = nib.Nifti1Image(self.np_img, np.eye(4))
        self.lbl = nib.Nifti1Image(self.np_lbl, np.eye(4))

        self.data = {'image': self.np_img, 'label': self.np_lbl}

        self.tform_one = NumpyToNifti(['image', 'label'])

    def test_call(self):
        self.data = self.tform_one(self.data)

        assert isinstance(self.data['image'], type(self.img))
        assert isinstance(self.data['label'], type(self.lbl))

        assert np.array_equal(self.data['image'].affine, np.eye(4))
        assert np.array_equal(self.data['label'].affine, np.eye(4))

        img = np.asanyarray(self.data['image'].dataobj).astype(np.float32)
        assert np.array_equal(img, self.np_img)
        lbl = np.asanyarray(self.data['label'].dataobj).astype(np.float32)
        assert np.array_equal(lbl, self.np_lbl)


class TestCropCenteredSubVolumes:
    def setup_class(self):
        self.data_one = np.random.rand(32, 32, 32).astype(np.float32)

        self.data_two = np.random.rand(3, 32, 32, 32).astype(np.float32)

        self.data = {
            'image': self.data_one,
            'other': self.data_two
        }

        self.tform_one = CropCenteredSubVolumes(['image', 'other'], [30, 30, 30])
        self.tform_two = CropCenteredSubVolumes(['other'], [10, 40, 60])

        self.tform_three = CropCenteredSubVolumes(['image'], [20, 10, 8])

    def test_call(self):
        self.data = self.tform_one(self.data)

        assert isinstance(self.data['image'], np.ndarray)
        assert isinstance(self.data['other'], np.ndarray)

        assert self.data['image'].shape[0] == 30
        assert self.data['image'].shape[1] == 30
        assert self.data['image'].shape[2] == 30

        assert np.all(self.data['image'] == self.data_one[1:31, 1:31, 1:31])

        assert self.data['other'].shape[0] == 3
        assert self.data['other'].shape[1] == 30
        assert self.data['other'].shape[2] == 30
        assert self.data['other'].shape[3] == 30

        assert np.all(self.data['other'] == self.data_two[:, 1:31, 1:31, 1:31])

        self.data = self.tform_two(self.data)

        assert isinstance(self.data['other'], np.ndarray)

        assert self.data['other'].shape[0] == 3
        assert self.data['other'].shape[1] == 10
        assert self.data['other'].shape[2] == 40
        assert self.data['other'].shape[3] == 60

        self.data = self.tform_three(self.data)

        assert self.data['image'].shape[0] == 20
        assert self.data['image'].shape[1] == 10
        assert self.data['image'].shape[2] == 8


class TestMapValues:
    def setup_class(self):
        self.data = {
            'image': np.random.rand(32, 32, 32).astype(np.float32),
            'other': np.random.rand(32, 32, 32).astype(np.float32)
        }

        self.tform_one = MapValues(['image'], 0, 1)
        self.tform_two = MapValues(['image'], 0, 100)

        self.tform_three = MapValues(['other'], 100, 1000)

    def test_call(self):
        self.data = self.tform_one(self.data)

        assert np.isclose(np.max(self.data['image']), 1, atol=1e-04)
        assert np.min(self.data['image']) == 0

        self.data = self.tform_two(self.data)

        assert np.isclose(np.max(self.data['image']), 100, atol=1e-04)
        assert np.min(self.data['image']) == 0

        self.data = self.tform_three(self.data)

        assert np.isclose(np.max(self.data['other']), 1000, atol=1e-04)
        assert np.isclose(np.min(self.data['other']), 100, atol=1e-04)



















