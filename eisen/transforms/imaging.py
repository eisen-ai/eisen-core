import nibabel as nib
import numpy as np
import SimpleITK as sitk

from eisen import EPS
from nilearn.image import resample_img


def pad_to_minimal_size(image, size, pad_mode='constant'):
    pad = size - np.asarray(image.shape[-3:]) + 1
    pad[pad < 0] = 0

    pad_before = np.floor(pad / 2.).astype(int)
    pad_after = (pad - pad_before).astype(int)

    pad_vector = []
    j = 0
    for i in range(image.ndim):
        if i < (image.ndim - 3):
            pad_vector.append((0, 0))
        else:
            pad_vector.append((pad_before[j], pad_after[j]))
            j += 1

    image = np.pad(array=image, pad_width=pad_vector, mode=pad_mode)

    return image, pad_before, pad_after


class CreateConstantFlags:
    """
    Transform allowing to create new fields in the data dictionary containing constants of any type

    .. code-block:: python

        from eisen.transforms import CreateConstantFlags
        tform = CreateConstantFlags(['my_field', 'my_text'], [42.0, 'hello'])
        tform = tform(data)

    """
    def __init__(self, fields, values):
        """
        :param fields: names of the fields of data dictionary to work on
        :type fields: list of str
        :param values: list of values to add to data
        :type values: list of values

        .. code-block:: python

            from eisen.transforms import CreateConstantFlags

            tform = CreateConstantFlags(
                fields=['my_field', 'my_text'],
                values=[42.0, 'hello']
            )

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "values", "type": "list:float", "value": ""}
        ]
        </json>

        """
        self.fields = fields
        self.values = values

        assert len(fields) == len(values)

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """
        for field, value in zip(self.fields, self.values):
            data[field] = value

        return data


class RenameFields:
    """
    Transform allowing to rename fields in the data dictionary

    .. code-block:: python

        from eisen.transforms import RenameFields
        tform = RenameFields(['old_name1', 'old_name2'], ['new_name1', 'new_name2'])
        tform = tform(data)

    """
    def __init__(self, fields, new_fields):
        """
        :param fields: list of names of the fields of data dictionary to rename
        :type fields: list of str
        :param new_fields: new field names for the data dictionary
        :type new_fields: list of str

        .. code-block:: python

            from eisen.transforms import RenameFields

            tform = RenameFields(
                fields=['old_name1', 'old_name2'],
                new_fields=['new_name1', 'new_name2']
            )

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "new_fields", "type": "list:string", "value": ""}
        ]
        </json>
        """
        self.fields = fields
        self.new_fields = new_fields

        assert len(self.new_fields) == len(self.fields)

    def __call__(self, data):
        for field, new_field in zip(self.fields, self.new_fields):
            data[new_field] = data.pop(field)

        return data


class FilterFields:
    """
    Transform allowing to retain in the data dictionary only a list of fields specified as init argument

    .. code-block:: python

        from eisen.transforms import FilterFields
        tform = FilterFields(['field1', 'field2'])
        tform = tform(data)

    The resulting data dictionary will only have 'field1' and 'field2' as keys.
    """
    def __init__(self, fields):
        """
        :param fields: list of fields to KEEP after the transform
        :type fields: list of str

        .. code-block:: python

            from eisen.transforms import FilterFields
            tform = FilterFields(fields=['field1', 'field2'])
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""}
        ]
        </json>
        """
        self.fields = fields

    def __call__(self, data):
        new_data = {}

        for field in self.fields:
            new_data[field] = data[field]

        return new_data


class ResampleNiftiVolumes:
    """
    Transform resampling nifti volumes to a new resolution (expressed in millimeters).
    This transform can be only applied to fields of the data dictionary containing objects of type Nifti (nibabel)

    .. code-block:: python

        from eisen.transforms import ResampleNiftiVolumes
        tform = ResampleNiftiVolumes(['nifti_data'], [1.0, 1.0, 1.0], 'linear')
        tform = tform(data)

    """
    def __init__(self, fields, resolution, interpolation='linear'):
        """
        :param fields: list of names of the fields of data dictionary to work on
        :type fields: list of str
        :param resolution: vector of float values expressing desired resolution in mm
        :type resolution: list of float
        :param interpolation: interpolation strategy to use
        :type interpolation: string

        .. code-block:: python

            from eisen.transforms import ResampleNiftiVolumes
            tform = ResampleNiftiVolumes(
                fields=['nifti_data'],
                resolution=[1.0, 1.0, 1.0],
                interpolation='linear'
            )

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "resolution", "type": "list:float", "value": ""},
            {"name": "interpolation", "type": "string", "value":
                [
                    "linear",
                    "nearest"
                ]
            }
        ]
        </json>
        """
        self.interpolation = interpolation
        self.resolution = resolution
        self.fields = fields

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """
        for field in self.fields:
            original_spacing = data[field].header.get_zooms()
            original_shape = data[field].header.get_data_shape()

            image_t = resample_img(
                img=data[field],
                target_affine=np.diag([self.resolution[0], self.resolution[1], self.resolution[2]]),
                interpolation=self.interpolation
            )

            data[field] = image_t
            data[field + '_original_spacing'] = original_spacing
            data[field + '_original_shape'] = original_shape

        return data
    
    
class ResampleITKVolumes:
    """
    Transform resampling ITK volumes to a new resolution (expressed in millimeters).
    This transform can be only applied to fields of the data dictionary containing objects of type ITK (SimpleITK)

    .. code-block:: python

        from eisen.transforms import ResampleITKVolumes
        tform = ResampleITKVolumes(['itk_data'], [1.0, 1.0, 1.0], 'linear')
        tform = tform(data)

    """
    def __init__(self, fields, resolution, interpolation='linear'):
        """
        :param fields: list of names of the fields of data dictionary to work on
        :type fields: list of str
        :param resolution: vector of float values expressing desired resolution in mm
        :type resolution: list of float
        :param interpolation: interpolation strategy to use
        :type interpolation: string

        .. code-block:: python

            from eisen.transforms import ResampleITKVolumes
            tform = ResampleITKVolumes(
                fields=['itk_data'],
                resolution=[1.0, 1.0, 1.0],
                interpolation='linear'
            )

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "resolution", "type": "list:float", "value": ""},
            {"name": "interpolation", "type": "string", "value":
                [
                    "linear",
                    "nearest"
                ]
            }
        ]
        </json>
        """

        self.fields = fields
        self.resolution = resolution
        self.interpolation = interpolation

    def __call__(self, data):
        original_spacings = {}

        for field in self.fields:
            original_spacings[field] = []

            resolution = np.asarray(self.resolution, dtype=float)

            original_spacing = np.asarray(data[field].GetSpacing())
            original_shape = np.asarray(data[field].GetSize())

            factor = original_spacing / resolution
            new_size = np.asarray(original_shape * factor, dtype=int)

            if self.interpolation == 'nearest':
                interpolator = sitk.sitkNearestNeighbor
            else:
                interpolator = sitk.sitkLinear

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(data[field])
            resampler.SetOutputSpacing(resolution)
            resampler.SetSize(new_size.tolist())
            resampler.SetInterpolator(interpolator)

            data[field] = resampler.Execute(data[field])

            data[field + '_original_spacing'] = original_spacing
            data[field + '_original_shape'] = original_shape

        return data


class NiftiToNumpy:
    """
    This transform allows a Nifti volume to be converted to Numpy format. It is necessary to have this transform
    at a certain point of every transformation chain as PyTorch uses data in Numpy format before converting it
    to PyTorch Tensor.

    .. code-block:: python

        from eisen.transforms import NiftiToNumpy
        tform = NiftiToNumpy(['image', 'label'])
        tform = tform(data)


    """
    def __init__(self, fields, multichannel=False):
        """
        :param fields: list of names of the fields of data dictionary to convert from Nifti to Numpy
        :type fields: list of str
        :param multichannel: need to set this parameter to True if data is multichannel
        :type multichannel: bool

        .. code-block:: python

            from eisen.transforms import NiftiToNumpy
            tform = NiftiToNumpy(fields=['image', 'label'])
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "multichannel", "type": "bool", "value": "False"}
        ]
        </json>
        """
        self.fields = fields
        self.multichannel = multichannel

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """
        for field in self.fields:
            entry_t = np.asanyarray(data[field].dataobj).astype(np.float32)

            if self.multichannel:
                dims = list(range(entry_t.ndim))
                entry_t = np.transpose(entry_t, [dims[-1]] + dims[0:-1])  # channel first if image is multichannel

            data[field] = entry_t

        return data


class NumpyToNifti:
    """
    This transform allows a Numpy volume to be converted to Nifti image object (nibabel).
    This transformation may be useful when writing Numpy array to disk in Nifti format
    using the WriteNiftiToFilename I/O transform. Note: the transform currently does not
    explicitly handle multichannel data.

    .. code-block:: python

        from eisen.transforms import NumpyToNifti
        tform = NumpyToNifti(['image', 'label'])
        tform = tform(data)


    """

    def __init__(self, fields, affine=None, data_types=None):
        """
        :param fields: list of names of the fields of data dictionary to convert from Nifti to Numpy
        :type fields: list of str
        :param affine: affine transformation matrix (see nibabel) specifying spacing, origin, and orientation
        :type affine: np.ndarray
        :data_types: dictionary in which key is the field and value is the output data type
        :type data_types dict
        
        .. code-block:: python

            from eisen.transforms import NumpyToNifti
            tform = NumpyToNifti(fields=['image', 'label'], affine=np.eye(4),
                data_types=np.float32)
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "multichannel", "type": "bool", "value": "False"},
            {"name": "data_types", "type": "dict", "value": "None"}
        ]
        </json>
        """
        self.fields = fields
        self.affine = affine
        if data_types is None:
            self.data_types = dict()
            for field in fields:
                self.data_types[field] = np.float32
        else:
            self.data_types = data_types
        assert isinstance(self.data_types, type(dict()))

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """
        # Validate that all fields have the same dimensionality, otherwise 'affine' parameter is inconsistent
        dims = []
        for field in self.fields:
            dims.append(data[field].ndim)
        assert all(i == dims[0] for i in dims)

        for field in self.fields:
            if self.affine is None:
                tform_affine = np.eye(data[field].ndim + 1)
            else:
                tform_affine = self.affine
            data[field] = nib.Nifti1Image(
                data[field].astype(self.data_types[field]), affine=tform_affine)

        return data


class ITKToNumpy:
    """
    This transform allows a ITK volume to be converted to Numpy format. It is necessary to have this transform
    at a certain point of every transformation chain as PyTorch uses data in Numpy format before converting it
    to PyTorch Tensor.

    .. code-block:: python

        from eisen.transforms import ITKToNumpy
        tform = ITKToNumpy(['image', 'label'])
        tform = tform(data)

    """
    def __init__(self, fields, multichannel=False):
        """
        :param fields: list of names of the fields of data dictionary to convert from ITK to Numpy
        :type fields: list of str
        :param multichannel: need to set this parameter to True if data is multichannel
        :type multichannel: bool

        .. code-block:: python

            from eisen.transforms import ITKToNumpy
            tform = ITKToNumpy(fields=['image', 'label'], multichannel=False)
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "multichannel", "type": "bool", "value": "False"}
        ]
        </json>
        """
        self.fields = fields
        self.multichannel = multichannel

    def __call__(self, data):
        for field in self.fields:
            entry_t = sitk.GetArrayFromImage(data[field]).astype(dtype=np.float32)

            if self.multichannel:
                dims = list(range(entry_t.ndim))
                entry_t = np.transpose(entry_t, [dims[-1]] + dims[0:-1])

            data[field] = entry_t

        return data


class PilToNumpy:
    """
    This transform allows a PIL image to be converted to Numpy format. It is necessary to have this transform
    at a certain point of every transformation chain as PyTorch uses data in Numpy format before converting it
    to PyTorch Tensor.

    .. code-block:: python

        from eisen.transforms import PilToNumpy
        tform = PilToNumpy(['image', 'label'])
        tform = tform(data)


    """
    def __init__(self, fields, multichannel=False):
        """
        :param fields: list of names of the fields of data dictionary to convert from PIL to Numpy
        :type fields: list of str
        :param multichannel: need to set this parameter to True if data is multichannel
        :type multichannel: bool

        .. code-block:: python

            from eisen.transforms import PilToNumpy
            tform = PilToNumpy(fields=['image', 'label'])
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "multichannel", "type": "bool", "value": "False"}
        ]
        </json>
        """
        self.fields = fields
        self.multichannel = multichannel

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """
        for field in self.fields:
            entry_t = np.asarray(data[field]).astype(np.float32)

            if self.multichannel:
                dims = list(range(entry_t.ndim))
                entry_t = np.transpose(entry_t, [dims[-1]] + dims[0:-1])  # channel first if image is multichannel

            data[field] = entry_t

        return data


class CropCenteredSubVolumes:
    """
    Transform implementing padding/cropping of 3D volumes. A 3D volume processed with this transform will be cropped
    or padded so that its final size will be corresponding to what specified by the user during instantiation.

    .. code-block:: python

        from eisen.transforms import CropCenteredSubVolumes
        tform = CropCenteredSubVolumes(['image', 'label'], [128, 128, 128])
        tform = tform(data)

    Will crop the content of the data dictionary at keys 'image' and 'label' (which need to be 3+D numpy volumes) to
    a size of 128 cubic pixels.
    """
    def __init__(self, fields, size):
        """
        :param fields: field of the data dictionary to modify and replace with cropped volumes
        :type fields: list of str
        :param size: list of 3 integers expressing the desired size of the cropped volumes
        :type size: list of int

        .. code-block:: python

            from eisen.transforms import CropCenteredSubVolumes
            tform = CropCenteredSubVolumes(
                fields=['image', 'label'],
                size=[128, 128, 128]
            )
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "size", "type": "list:int", "value": ""}
        ]
        </json>
        """
        self.size = size
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            image_entry, pad_before, pad_after = pad_to_minimal_size(data[field], self.size, pad_mode='constant')

            h_size = np.floor(np.asarray(self.size) / 2.).astype(int)
            centr_pix = np.floor(np.asarray(image_entry.shape[-3:]) / 2.).astype(int)

            start_px = (centr_pix - h_size).astype(int)

            end_px = (start_px + self.size).astype(int)

            assert np.all(end_px <= np.asarray(image_entry.shape[-3:]))
            assert np.all(start_px >= 0)

            image_patch = image_entry[..., start_px[0]:end_px[0], start_px[1]:end_px[1], start_px[2]:end_px[2]]

            crop_before = start_px
            crop_after = image_entry.shape[-3:] - end_px - 1

            assert np.all(np.asarray(image_patch.shape[-3:]) == self.size)

            data[field] = image_patch

            data[field + '_start_px'] = (crop_before - pad_before).tolist()

            data[field + '_end_px'] = (crop_after - pad_after).tolist()

        return data


class MapValues:
    """
    Transform implementing normalization by standardizing the range of data to a known interval. The formula used here
    is to subtract the minimum value to each data tensor and divide by its maximum range. After that the tensor is
    multiplied by the max_value .

    .. code-block:: python

        from eisen.transforms import MapValues
        tform = MapValues(['image'], 0, 10)
        tform = tform(data)

    Is an usage examples where data is normalized to fit the range [0, 10].
    """
    def __init__(self, fields, min_value=0, max_value=1, channelwise=False):
        """
        :param fields: list of fields of the data dictionary that will be affected by this transform
        :type fields: list of str
        :param min_value: minimum desired data value
        :type min_value: float
        :param max_value: maximum desired data value
        :type max_value: float
        :param channelwise: whether the transformation should be applied to each channel separately
        :type channelwise: bool

        .. code-block:: python

            from eisen.transforms import MapValues
            tform = MapValues(
                fields=['image'],
                min_value=0,
                max_value=1,
                channelwise=False
            )
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "min_value", "type": "float", "value": "0"},
            {"name": "max_value", "type": "float", "value": "1"},
            {"name": "channelwise", "type": "bool", "value": "false"}
        ]
        </json>
        """
        self.fields = fields
        self.min_value = min_value
        self.max_value = max_value
        self.channelwise = channelwise

    def __call__(self, data):
        for field in self.fields:
            if self.channelwise:
                for i in range(data[field].shape[0]):
                    data[field][i] = \
                        (data[field][i] - np.min(data[field][i])) / \
                        (np.max(data[field][i]) - np.min(data[field][i]) + EPS)
            else:
                data[field] = \
                    (data[field] - np.min(data[field])) / \
                    (np.max(data[field]) - np.min(data[field]) + EPS)

            data[field] *= (self.max_value - self.min_value)
            data[field] += self.min_value

        return data


class ThresholdValues:
    """
    This transformation threshold the values contained in a tensor. Depending on a parameter supplied by the user,
    all the value greater, smaller, greater/equal, smaller/equal of a certain threshold are set to 1 while the
    others are set to zero.

    .. code-block:: python

        from eisen.transforms import ThresholdValues
        tform = ThresholdValues(['label'], 0.5, 'greater')
        tform = tform(data)

    This example thresholds the values of the tensor stored in correspondence of the key 'label' such that
    those below 0.5 are set to zero and those above 0.5 are set to one.
    """
    def __init__(self, fields, threshold, direction='greater'):
        """
        :param fields: list of fields of the data dictionary that will be affected by this transform
        :type fields: list of str
        :param threshold: threshold value for the transform
        :type threshold: float
        :param direction: direction of the comparison values and the threshold
            possible values are: `greater`, `smaller`, `greater/equal`, `smaller/equal`
        :type direction: string

        .. code-block:: python

            from eisen.transforms import ThresholdValues
            tform = ThresholdValues(
                fields=['image'],
                threshold=0,
                direction='greater/equal'
            )
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "threshold", "type": "float", "value": "0"},
            {"name": "direction", "type": "string", "value":
                [
                    "greater",
                    "smaller",
                    "greater/equal",
                    "smaller/equal"
                ]
            }
        ]
        </json>
        """
        self.fields = fields
        self.threshold = threshold
        self.direction = direction

    def __call__(self, data):
        for field in self.fields:
            if self.direction == 'greater':
                data[field] = (data[field] > self.threshold).astype(dtype=data[field].dtype)
            elif self.direction == 'smaller':
                data[field] = (data[field] < self.threshold).astype(dtype=data[field].dtype)
            elif self.direction == 'greater/equal':
                data[field] = (data[field] >= self.threshold).astype(dtype=data[field].dtype)
            elif self.direction == 'smaller/equal':
                data[field] = (data[field] <= self.threshold).astype(dtype=data[field].dtype)
            else:
                raise ValueError('the direction of inequality {} is not supported'.format(self.direction))

        return data


class AddChannelDimension:
    """
    This transformation adds a "channel dimension" to a tensor. Since we use a representation NCHWD for our data,
    with channels first, this transform creates a new axis in correspondence of the first dimension of the
    resulting data tensor.

    .. code-block:: python

        from eisen.transforms import AddChannelDimension
        tform = AddChannelDimension(['image', 'label'])
        tform = tform(data)

    Adds a singleton dimension to the data stored in correspondence of the keys 'image' and 'label' of data dictionary.
    """
    def __init__(self, fields):
        """
        :param fields: list of fields of the data dictionary that will be affected by this transform
        :type fields: list of str

        .. code-block:: python

            from eisen.transforms import AddChannelDimension
            tform = AddChannelDimension(
                fields=['image', 'label']
            )
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""}
        ]
        </json>
        """
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            data[field] = data[field][np.newaxis]

        return data


class LabelMapToOneHot:
    """
    This transformation converts labels having integer values to one-hot labels. In other words, a single channel
    tensor data containing integer values representing classes is converted to a corresponding multi-channel tensor data
    having one-hot entries channel-wise. Each channel corresponds to a class.

     .. code-block:: python

        from eisen.transforms import LabelMapToOneHot
        tform = LabelMapToOneHot(['label'], [1, 2, 25, 3])
        tform = tform(data)

    This example converts the single channel data['label'] tensor to a 4-channel tensor where each entry
    represents the corresponding entry of the original tensor in one-hot encoding.
    """
    def __init__(self, fields, classes):
        """
        :param fields: list of fields of the data dictionary that will be affected by this transform
        :type fields: list of str
        :param classes: list of class identifiers (integers) to be converted to one-hot representation
        :type classes: list of int

        .. code-block:: python

            from eisen.transforms import LabelMapToOneHot
            tform = LabelMapToOneHot(
                fields=['label'],
                classes=[1, 2, 25, 3]
            )
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "classes", "type": "list:int", "value": ""}
        ]
        </json>
        """
        self.fields = fields
        self.classes = classes
        self.num_channels = len(self.classes)

    def __call__(self, data):

        for field in self.fields:

            data[field] = np.squeeze(data[field])

            onehot = np.zeros([self.num_channels] + list(data[field].shape), dtype=np.float32)

            for c in range(self.num_channels):
                onehot[c, ...] = (data[field] == self.classes[c]).astype(np.float32)

            data[field] = onehot

        return data


class StackImagesChannelwise:
    """
    This transform allows stacking together different tensors of the same size stored at different fields of
    the data dictionary. The tensors are stacked along the channel dimension. The resulting tensor is therefore
    multi-channel and contains data from all the fields passed as argument by the user.

    .. code-block:: python

        from eisen.transforms import StackImagesChannelwise
        tform = StackImagesChannelwise(['modality1', 'modality2', 'modality3'], 'allmodalities')
        tform = tform(data)

    This example stacks together multiple modalities in one multi-channel tensor.

    """
    def __init__(self, fields, dst_field, create_new_dim=True):
        """
        :param fields: list of fields of the data dictionary that will be stacked together in the output tensor
        :type fields: list of str
        :param dst_field: string representing the destination field of the data dictionary where outputs will be stored.
        :type dst_field: str
        :param create_new_dim: whether a new dimension should be created as result of concat.
        :type create_new_dim: bool

        .. code-block:: python

            from eisen.transforms import StackImagesChannelwise
            tform = StackImagesChannelwise(
                fields=['modality1', 'modality2', 'modality3'],
                dst_field='allmodalities'
                create_new_dim=True
            )
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "dst_field", "type": "string", "value": ""},
            {"name": "create_new_dim", "type": "bool", "value": "True"}
        ]
        </json>
        """
        self.fields = fields
        self.dst_field = dst_field
        self.create_new_dim = create_new_dim

    def __call__(self, data):

        composite_image = []

        for key in self.fields:
            composite_image.append(data[key])

        if self.create_new_dim:
            data[self.dst_field] = np.stack(composite_image, axis=0)
        else:
            data[self.dst_field] = np.concatenate(composite_image, axis=0)

        return data


class FixedMeanStdNormalization:
    """
    This transform operates demeaning and division by standard deviation of data tensors. The values for mean
    and standard deviation need to be provided by the user.

    .. code-block:: python

        from eisen.transforms import FixedMeanStdNormalization
        tform = FixedMeanStdNormalization(['image'], 0.5, 1.2)
        tform = tform(data)

    This example manipulates the data stored in data['images'] by removing the mean (0.5) and the std (1.2).
    """

    def __init__(self, fields, mean, std):
        """
        :param fields: list of fields of the data dictionary that will be affected by this transform
        :type fields: list of str
        :param mean: float value representing the mean. This value will be subtracted from the data
        :type mean: float
        :param std: float value representing the standard deviation. The data will be divided by this value.
        :type std: float

        .. code-block:: python

            from eisen.transforms import FixedMeanStdNormalization
            tform = FixedMeanStdNormalization(
                fields=['image'],
                mean=0.5,
                std=1.2
            )
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "mean", "type": "float", "value": ""},
            {"name": "std", "type": "float", "value": ""}
        ]
        </json>
        """
        assert std != 0

        self.fields = fields
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for field in self.fields:
            data[field] = (data[field] - self.mean) / self.std

        return data


class RepeatTensor:
    """
    This transform repeats tensors "reps" times on each axis according to user parameters

    .. code-block:: python

        from eisen.transforms import RepeatTensor
        tform = RepeatTensor(['image'], (10, 1, 1))
        tform = tform(data)

    This example repeats the tensor 10 times along the first axis and zero times along the others
    """

    def __init__(self, fields, reps):
        """
        :param fields: list of fields of the data dictionary that will be affected by this transform
        :type fields: list of str
        :param reps: list of integers representing repetitions along each axis
        :type reps: list of int

        .. code-block:: python

            from eisen.transforms import RepeatTensor
            tform = RepeatTensor(
                fields=['image'],
                reps=(10, 1, 2)
            )
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "reps", "type": "list:int", "value": ""}
        ]
        </json>
        """
        assert np.all(np.asarray(reps) > 0)

        self.fields = fields
        self.reps = reps

    def __call__(self, data):
        for field in self.fields:
            data[field] = np.tile(data[field], self.reps)

        return data
