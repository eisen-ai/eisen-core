import numpy as np
import nibabel as nib
import tempfile
import os
import copy
import SimpleITK as sitk
import PIL

from pydicom.dataset import FileDataset

from eisen.io.imaging import LoadNiftiFromFilename, LoadNiftyFromFilename
from eisen.io.imaging import LoadDICOMFromFilename
from eisen.io.imaging import LoadITKFromFilename
from eisen.io.imaging import LoadPILImageFromFilename
from eisen.io.imaging import WriteNiftiToFile

from eisen.transforms.imaging import NumpyToNifti


class TestLoadNiftiFromFilename:
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

        self.nii_loader = LoadNiftiFromFilename(['nii'], self.base_path)
        self.nii_gz_loader = LoadNiftiFromFilename(['nii_gz'], self.base_path)
        self.all_loader = LoadNiftiFromFilename(['nii', 'nii_gz'], self.base_path)

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


class TestLoadPILImageFromFilename:
    def setup_class(self):
        self.data = np.random.rand(32, 32).astype(np.float32)

        self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data)) * 255.0

        self.data = np.round(self.data)

        self.base_path = tempfile.mkdtemp()

        self.png_name = 'file.png'

        self.file_path = os.path.join(self.base_path, self.png_name)

        im = PIL.Image.fromarray(self.data)
        im = im.convert("L")

        im.save(self.file_path)

        self.data_entry = {
            'png': self.png_name
        }

        self.loader = LoadPILImageFromFilename(['png'], self.base_path)

    def test_call(self):
        data_entry_transformed = self.loader(copy.deepcopy(self.data_entry))

        assert isinstance(data_entry_transformed['png'], PIL.PngImagePlugin.PngImageFile)

        np_im = np.array(data_entry_transformed['png'])

        assert np.all(self.data == np_im)


class TestLoadITKImageFromFilename:
    def setup_class(self):
        self.data = np.random.randn(32, 32, 32).astype(np.int32)

        self.base_path = tempfile.mkdtemp()

        self.mhd_name = 'file.mhd'

        self.file_path = os.path.join(self.base_path, self.mhd_name)

        sitk_image = sitk.GetImageFromArray(self.data)

        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(sitk.sitkInt32)

        sitk_image = castImageFilter.Execute(sitk_image)

        sitk.WriteImage(sitk_image, self.file_path)

        self.data_entry = {
            'mhd': self.mhd_name
        }

        self.loader = LoadITKFromFilename(['mhd'], self.base_path)

    def test_call(self):
        data_entry_transformed = self.loader(copy.deepcopy(self.data_entry))

        assert isinstance(data_entry_transformed['mhd'], sitk.Image)

        assert np.all(self.data == sitk.GetArrayFromImage(data_entry_transformed['mhd']))


class TestLoadDICOMImageFromFilename:
    def setup_class(self):
        self.data = np.random.randn(32, 32).astype(np.int32)

        self.base_path = tempfile.mkdtemp()

        self.dcm_name = 'file.dcm'

        self.file_path = os.path.join(self.base_path, self.dcm_name)

        sitk_image = sitk.GetImageFromArray(self.data, )

        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(sitk.sitkInt32)

        sitk_image = castImageFilter.Execute(sitk_image)

        sitk.WriteImage(sitk_image, self.file_path)

        self.data_entry = {
            'dcm': self.dcm_name
        }

        self.loader = LoadDICOMFromFilename(['dcm'], self.base_path)
        self.loader_px_data = LoadDICOMFromFilename(['dcm'], self.base_path, store_data_array=True)

    def test_call(self):
        data_entry_transformed = self.loader(copy.deepcopy(self.data_entry))

        data_entry_transformed_with_data_array = self.loader_px_data(copy.deepcopy(self.data_entry))

        assert isinstance(data_entry_transformed['dcm'], FileDataset)
        assert isinstance(data_entry_transformed_with_data_array['dcm'], FileDataset)
        assert isinstance(data_entry_transformed_with_data_array['dcm_pixel_array'], np.ndarray)

        assert np.all(self.data == np.asanyarray(data_entry_transformed['dcm'].pixel_array))
        assert np.all(self.data == np.asanyarray(data_entry_transformed_with_data_array['dcm_pixel_array']))
        assert np.all(self.data == np.asanyarray(data_entry_transformed_with_data_array['dcm'].pixel_array))


class TestWriteNiftiToFile:

    def setup_class(self):
        self.np_img = np.random.randn(32, 32, 15).astype(np.float32)
        self.np_lbl = np.random.randn(32, 32, 15).astype(np.uint8)
        self.data = {'image': self.np_img, 'label': self.np_lbl}
        self.img = nib.Nifti1Image(self.np_img, np.eye(4))
        self.lbl = nib.Nifti1Image(self.np_lbl, np.eye(4))
        self.data_types = {'image': np.float32, 'label': np.uint8}
        self.affine = np.array([[1, 0, 0, -100], 
                                [0, 1, 0, -200],
                                [0, 0, 2.5, -300],
                                [0, 0, 0, 1]])
        self.base_path = tempfile.mkdtemp()

    def test_call(self):
        nifti_tform = NumpyToNifti(self.data, affine=self.affine, data_types=self.data_types)
        data = nifti_tform(self.data)
        
        data['filename_image'] = 'my-image'
        data['filename_label'] = 'my-label'

        nifti_writer = WriteNiftiToFile(
            fields=['image', 'label'],
            filename_prefix=os.path.join(self.base_path, 'test'),
            name_fields = ['filename_image', 'filename_label']
        )

        nifti_writer(data)

        test_img = nib.load(os.path.join(self.base_path, 'test_image_my-image.nii.gz'))
        assert np.array_equal(np.asanyarray(test_img.dataobj).astype(np.float32), self.np_img)

        test_lbl = nib.load(os.path.join(self.base_path, 'test_label_my-label.nii.gz'))
        assert np.array_equal(np.asanyarray(test_lbl.dataobj).astype(np.uint8), self.np_lbl)
