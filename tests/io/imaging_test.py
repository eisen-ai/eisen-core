import numpy as np
import nibabel as nib
import tempfile
import os
import copy

from eisen.io.imaging import LoadNiftyFromFilename


class TestLoadNiftyFromFilename:
    def setup_class(self):
        self.data = np.random.randn(32, 32, 15, 100).astype(np.float32)

        self.img = nib.Nifti1Image(self.data, np.eye(4))
        assert self.img.get_data_dtype() == np.dtype(np.float32)

        self.base_path = tempfile.mkdtemp()

        self.nii_name = 'test4d.nii'
        self.nii_gz_name = 'test4d.nii.gz'

        nib.save(self.img, os.path.join(self.base_path, self.nii_name))
        nib.save(self.img, os.path.join(self.base_path, self.nii_gz_name))

        self.dset = {
            'nii': self.nii_name,
            'nii_gz': self.nii_gz_name
        }

        self.nii_loader = LoadNiftyFromFilename(['nii'], self.base_path)
        self.nii_gz_loader = LoadNiftyFromFilename(['nii_gz'], self.base_path)
        self.all_loader = LoadNiftyFromFilename(['nii', 'nii_gz'], self.base_path)

    def test_call(self):
        self.dset_nii = self.nii_loader(copy.deepcopy(self.dset))
        self.dset_nii_gz = self.nii_gz_loader(copy.deepcopy(self.dset))
        self.dset_all = self.all_loader(copy.deepcopy(self.dset))

        assert isinstance(self.dset_nii['nii'], nib.Nifti1Image)
        assert isinstance(self.dset_nii_gz['nii_gz'], nib.Nifti1Image)
        assert isinstance(self.dset_all['nii'], nib.Nifti1Image)
        assert isinstance(self.dset_all['nii_gz'], nib.Nifti1Image)

        assert np.all(self.data == np.asanyarray(self.dset_nii['nii'].dataobj))
        assert np.all(self.data == np.asanyarray(self.dset_nii_gz['nii_gz'].dataobj))
        assert np.all(self.data == np.asanyarray(self.dset_all['nii'].dataobj))
        assert np.all(self.data == np.asanyarray(self.dset_all['nii_gz'].dataobj))





