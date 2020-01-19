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
    def __init__(self, fields, values):
        """
        :param fields: names of the fields of data dictionary to work on
        :type fields: list
        :param values: list of float value to add to data
        :type values: list
        """
        self.fields = fields
        self.values = values

        assert len(fields) == len(values)

    def __call__(self, data):
        for field, value in zip(self.fields, self.values):
            data[field] = value

        return data


class RenameFields:
    def __init__(self, fields, new_fields):
        """
        :param fields: list of names of the fields of data dictionary to rename
        :type fields: list
        :param new_fields: new field names for the data dictionary
        :type new_fields: list
        """
        self.fields = fields
        self.new_fields = new_fields

        assert len(self.new_fields) == len(self.fields)

    def __call__(self, data):
        for field, new_field in zip(self.fields, self.new_fields):
            data[new_field] = data.pop(field)

        return data


class FilterFields:
    def __init__(self, fields):
        """
        :param fields: list of fields to KEEP after the transform
        :type fields: list
        """
        self.fields = fields

    def __call__(self, data):
        new_data = {}

        for field in self.fields:
            new_data[field] = data[field]

        return new_data


class ResampleNiftiVolumes:
    def __init__(self, fields, resolution, interpolation='continuous'):
        """
        :param fields: list of names of the fields of data dictionary to work on
        :type fields: list
        :param resolution: vector of float values expressing desired resolution in mm
        :type resolution: list
        :param interpolation: interpolation strategy to use
        :type interpolation: string
        """
        assert len(resolution) == 3
        self.interpolation = interpolation
        self.resolution = resolution
        self.fields = fields

    def __call__(self, data):
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
    def __init__(self, fields):
        """
        :param fields: list of names of the fields of data dictionary to convert from Nifti to Numpy
        :type fields: list
        """
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            entry_t = data[field].get_data().astype(np.float32)

            if entry_t.ndim > 3:
                entry_t = np.transpose(entry_t, [3, 0, 1, 2])  # channel first if image is multichannel

            data[field] = entry_t

        return data


class CropCenteredSubVolumes:
    def __init__(self, fields, size):
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
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            data[field] = data[field][np.newaxis]

        return data


class LabelMapToOneHot:
    def __init__(self, fields, classes):
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
        assert std != 0

        self.fields = fields
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for field in self.fields:
            data[field] = (data[field] - self.mean) / self.std

        return data
