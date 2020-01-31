import numpy as np

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
        :param values: list of float value to add to data
        :type values: list of float

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
        assert len(resolution) == 3
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
    def __init__(self, fields):
        """
        :param fields: list of names of the fields of data dictionary to convert from Nifti to Numpy
        :type fields: list of str

        .. code-block:: python

            from eisen.transforms import NiftiToNumpy
            tform = NiftiToNumpy(fields=['image', 'label'])
            tform = tform(data)

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""}
        ]
        </json>
        """
        self.fields = fields

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """
        for field in self.fields:
            entry_t = data[field].get_data().astype(np.float32)

            if entry_t.ndim > 3:
                entry_t = np.transpose(entry_t, [3, 0, 1, 2])  # channel first if image is multichannel

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

            data[field + '_start_px'] = crop_before - pad_before

            data[field + '_end_px'] = crop_after - pad_after

        return data


class MapValues:
    def __init__(self, fields, min_value=0, max_value=1, channelwise=True):
        """
        :param fields:
        :type fields: list of str
        :param min_value:
        :type min_value: float
        :param max_value:
        :type max_value: float
        :param channelwise:
        :type channelwise: bool

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "min_value", "type": "float", "value": "0"},
            {"name": "max_value", "type": "float", "value": "1"},
            {"name": "channelwise", "type": "bool", "value": "true"}
        ]
        </json>
        """
        self.fields = fields
        self.min_value = min_value
        self.max_value = max_value
        self.channelwise = channelwise

    def __call__(self, data):
        for field in self.fields:
            if data[field].ndim > 3 and self.channelwise:
                for i in range(data[field].shape[0]):
                    data[field][i] = \
                        (data[field][i] - np.min(data[field][i])) / \
                        (np.max(data[field][i]) - np.min(data[field][i]) + EPS)
            else:
                data[field] = \
                    (data[field] - np.min(data[field])) / \
                    (np.max(data[field]) - np.min(data[field]) + EPS)

            data[field] *= self.max_value - self.min_value
            data[field] += self.min_value

        return data


class ThresholdValues:
    def __init__(self, fields, threshold, direction='greater'):
        """
        :param fields:
        :type fields: list of str
        :param threshold:
        :type threshold: float
        :param direction:
        :type direction: string

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
    def __init__(self, fields):
        """
        :param fields:
        :type fields: list of str

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
    def __init__(self, fields, classes):
        """
        :param fields:
        :type fields: list of str
        :param classes:
        :type classes: list of int

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

            data[field] = data[field].astype(np.int32)

            onehot = np.zeros([self.num_channels] + list(data[field].shape[-3:]), dtype=np.float32)

            for c in range(self.num_channels):
                onehot[c, ...] = (data[field] == self.classes[c]).astype(np.float32)

            data[field] = onehot

        return data


class StackImagesChannelwise:
    def __init__(self, fields, dst_field):
        """
        :param fields:
        :type fields: list of str
        :param dst_field:
        :type dst_field: str

        <json>
        [
            {"name": "fields", "type": "list:string", "value": ""},
            {"name": "dst_field", "type": "string", "value": ""}
        ]
        </json>
        """
        self.fields = fields
        self.dst_field = dst_field

    def __call__(self, data):

        composite_image = []

        for key in self.fields:
            composite_image.append(data[key])

        data[self.dst_field] = np.stack(composite_image, axis=-1)

        return data


class FixedMeanStdNormalization:
    def __init__(self, fields, mean, std):
        """
        :param fields:
        :type fields: list of str
        :param mean:
        :type mean: float
        :param std:
        :type std: float

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
