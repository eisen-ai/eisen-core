import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import pydicom
import PIL.Image


class LoadITKFromFilename:
    """
    This transform loads ITK data from filenames contained in specific field of the data dictionary.
    Although this transform follows the general structure of other transforms, such as those contained in
    eisen.transforms, it's kept separated from the others as it is responsible for I/O operations interacting
    with the disk.

    .. code-block:: python

        from eisen.io import LoadITKFromFilename
        tform = LoadITKFromFilename(['image', 'label'], '/abs/path/to/dataset')

    """
    def __init__(self, fields, data_dir):
        """
        LoadITKFromFilename loads ITK compatible files. The data is always read as float32.

        :param fields: fields of the dictionary containing ITK file paths that need to be read
        :type fields: list
        :param data_dir: source data directory where data is located. This directory will be joined with data paths
        :type data_dir: str

        .. code-block:: python

            from eisen.io import LoadITKFromFilename
            tform = LoadITKFromFilename(
                fields=['image', 'label'],
                data_dir='/abs/path/to/dataset'
            )

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""}
        ]

        """
        self.fields = fields
        self.data_dir = data_dir

        self.type_filter = sitk.CastImageFilter()
        self.type_filter.SetOutputPixelType(sitk.sitkFloat32)

    def __call__(self, data):
        for field in self.fields:
            volume = sitk.ReadImage(os.path.join(self.data_dir, data[field]))

            volume = self.type_filter.Execute(volume)

            data[field] = volume

        return data


class LoadNiftiFromFilename:
    """
    This transform loads Nifti data from filenames contained in a specific field of the data dictionary.
    Although this transform follows the general structure of other transforms, such as those contained in
    eisen.transforms, it's kept separated from the others as it is responsible for I/O operations interacting
    with the disk

    .. code-block:: python

        from eisen.io import LoadNiftiFromFilename
        tform = LoadNiftiFromFilename(['image', 'label'], '/abs/path/to/dataset')

    """
    def __init__(self, fields, data_dir, canonical=False):
        """
        :param fields: list of names of the field of data dictionary to work on. These fields should contain data paths
        :type fields: list
        :param data_dir: source data directory where data is located. This directory will be joined with data paths
        :type data_dir: str
        :param canonical: whether data should be reordered to be closest to canonical (see nibabel documentation)
        :type canonical: bool

        .. code-block:: python

            from eisen.io import LoadNiftiFromFilename
            tform = LoadNiftiFromFilename(
                fields=['image', 'label'],
                data_dir='/abs/path/to/dataset'
                canonical=False
            )

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "canonical", "type": "bool", "value": "false"}
        ]
        </json>
        """
        self.data_dir = data_dir
        self.fields = fields
        self.canonical = canonical

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """
        for field in self.fields:
            img = nib.load(os.path.normpath(os.path.join(self.data_dir, data[field])))

            if self.canonical:
                img = nib.as_closest_canonical(img)

            data[field] = img
            data[field + '_affines'] = img.affine
            data[field + '_orientations'] = nib.aff2axcodes(img.affine)

        return data


class LoadNiftyFromFilename(LoadNiftiFromFilename):
    def __init__(self, *args, **kwargs):
        print('LoadNiftyFromFilename has been renamed LoadNiftiFromFilename. The older class is deprecated')
        super(LoadNiftyFromFilename, self).__init__(*args, **kwargs)


class LoadDICOMFromFilename:
    """
    This transform loads DICOM data from filenames contained in a specific field of the data dictionary.
    Although this transform follows the general structure of other transforms, such as those contained in
    eisen.transforms, it's kept separated from the others as it is responsible for I/O operations interacting
    with the disk

    .. code-block:: python

        from eisen.io import LoadDICOMFromFilename
        tform = LoadDICOMFromFilename(['image', 'label'], '/abs/path/to/dataset')

    """
    def __init__(self, fields, data_dir, store_data_array=True):
        """
        :param fields: list of names of the field of data dictionary to work on. These fields should contain data paths
        :type fields: list
        :param data_dir: source data directory where data is located. This directory will be joined with data paths
        :type data_dir: str
        :param store_data_array: whether image data as numpy array should be stored (in "field" + "_pixel_array")
        :type store_data_array: bool

        .. code-block:: python

            from eisen.io import LoadDICOMFromFilename
            tform = LoadDICOMFromFilename(
                fields=['image'],
                data_dir='/abs/path/to/dataset'
                store_data_array=True
            )

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "store_data_array", "type": "bool", "value": "false"}
        ]
        </json>
        """
        self.data_dir = data_dir
        self.fields = fields
        self.store_data_array = store_data_array

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """
        for field in self.fields:
            dataset = pydicom.dcmread(os.path.join(self.data_dir, data[field]))

            if self.store_data_array:
                img = dataset.pixel_array

                data[field + '_pixel_array'] = img

            data[field] = dataset

        return data


class LoadPILImageFromFilename:
    """
    This transform loads Images from filenames contained in a specific field of the data dictionary. The images are
    loaded via Pillow, an imaging library for Python.
    Although this transform follows the general structure of other transforms, such as those contained in
    eisen.transforms, it's kept separated from the others as it is responsible for I/O operations interacting
    with the disk

    .. code-block:: python

        from eisen.io import LoadPILImageFromFilename
        tform = LoadPILImageFromFilename(['image', 'label'], '/abs/path/to/dataset')

    """
    def __init__(self, fields, data_dir):
        """
        :param fields: list of names of the field of data dictionary to work on. These fields should contain data paths
        :type fields: list
        :param data_dir: source data directory where data is located. This directory will be joined with data paths
        :type data_dir: str

        .. code-block:: python

            from eisen.io import LoadPILImageFromFilename
            tform = LoadPILImageFromFilename(
                fields=['image'],
                data_dir='/abs/path/to/dataset'
            )

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""}
        ]
        </json>
        """
        self.data_dir = data_dir
        self.fields = fields

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """
        for field in self.fields:
            image = PIL.Image.open(os.path.join(self.data_dir, data[field]))

            data[field] = image

        return data


class WriteNiftiToFile:
    """
    This transform writes NIFTI data to a file on disk. Although this transform follows the general structure
    of other transforms, such as those contained in eisen.transforms, it's kept separated from the others as
    it is responsible for I/O operations interacting with the disk.

    .. code-block:: python

        from eisen.io import WriteNiftiToFile
        tform = WriteNiftiToFile(['image', 'label'], '/abs/path/to/filename')

    """

    def __init__(self, fields, name_fields=None, filename_prefix='./'):
        """
        :param fields: list of names of the field of data dictionary to work on. These fields should contain data paths
        :type fields: list
        :param filename_prefix: absolute path plus file prefix of output file
        :type filename_prefix: str

        .. code-block:: python

            from eisen.io import WriteNiftiToFile
            tform = WriteNiftiToFile(
                fields=['image', 'label'],
                name_fields=['image_name', 'label_name'],
                filename_prefix='/abs/path/to/dataset'
            )

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "name_fields", "type": "list:string", "value": ""},
            {"name": "filename_prefix", "type": "string", "value": ""}
        ]
        </json>
        """
        self.filename_prefix = filename_prefix
        self.name_fields = name_fields
        self.fields = fields

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """           
        for i, field in enumerate(self.fields):
            if self.name_fields is None:
                filename = '{}_{}.nii.gz'.format(self.filename_prefix, field)
            else:
                filename = '{}_{}_{}.nii.gz'.format(self.filename_prefix, field, data[self.name_fields[i]])
                
            nib.save(data[field], filename)
            
        return data
