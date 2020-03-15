import os
import nibabel as nib
import SimpleITK as sitk


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
            volume = sitk.ReadImage(data[field])

            volume = self.type_filter.Execute(volume)

            data[field] = volume

        return data


class LoadNiftyFromFilename:
    """
    This transform loads Nifty data from filenames contained in a specific field of the data dictionary.
    Although this transform follows the general structure of other transforms, such as those contained in
    eisen.transforms, it's kept separated from the others as it is responsible for I/O operations interacting
    with the disk

    .. code-block:: python

        from eisen.io import LoadNiftyFromFilename
        tform = LoadNiftyFromFilename(['image', 'label'], '/abs/path/to/dataset')

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

            from eisen.io import LoadNiftyFromFilename
            tform = LoadNiftyFromFilename(
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
