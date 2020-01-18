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


class CreateConstantFlag:
    def __init__(self, field, value):
        """
        :param field: name of the field of data dictionary to work on
        :type field: string
        :param value: float value to add to data
        :type field: float
        """
        self.field = field
        self.value = value

    def __call__(self, data):
        data[self.field] = self.value

        return data


class RenameField:
    def __init__(self, field, new_field):
        """
        :param field: name of the field of data dictionary to rename
        :type field: string
        :param new_field: new field name for the data dictionary
        :type new_field: string
        """
        self.field = field
        self.new_field = new_field

    def __call__(self, data):
        data[self.new_field] = data.pop(self.field)

        return data


class FilterFields:
    def __init__(self, fields):
        """
        :param fields: list of fields to keep after the transform
        :type fields: list
        """
        self.fields = fields

    def __call__(self, data):
        new_data = {}

        for field in self.fields:
            new_data[field] = data[field]

        return new_data


class ResampleNiftiVolume:
    def __init__(self, field, resolution, interpolation='continuous'):
        """
        :param field: name of the field of data dictionary to work on
        :type field: string
        :param resolution: list of float values expressing desired resolution in mm
        :type resolution: list
        :param interpolation: interpolation strategy to use
        :type interpolation: string
        """
        assert len(resolution) == 3
        self.interpolation = interpolation
        self.resolution = resolution
        self.field = field

    def __call__(self, data):
        original_spacing = data[self.field].header.get_zooms()
        original_shape = data[self.field].header.get_data_shape()

        image_t = resample_img(
            img=data[self.field],
            target_affine=np.diag([self.resolution[0], self.resolution[1], self.resolution[2]]),
            interpolation=self.interpolation
        )

        data[self.field] = image_t
        data[self.field + '_original_spacing'] = original_spacing
        data[self.field + '_original_shape'] = original_shape

        return data


class NiftiToNumpy:
    def __init__(self, field):
        """
        :param field: name of the field of data dictionary to convert from Nifti to Numpy
        :type field: string
        """
        self.field = field

    def __call__(self, data):

        entry_t = data[self.field].get_data().astype(np.float32)

        if entry_t.ndim > 3:
            entry_t = np.transpose(entry_t, [3, 0, 1, 2])  # channel first if image is multichannel

        data[self.field] = entry_t

        return data


class CropSubVolumes:
    def __init__(self, size, fields):
        """
        :param size: list of integers expressing desired volume size
        :type size: list
        :param fields: list of fields to apply the transform to
        :type fields: list
        """
        self.size = size
        self.fields = fields

    def pad_to_minimal_size(self, image, pad_mode='constant'):
        pad = self.size - np.asarray(image.shape[-3:]) + 1
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


class CropRandomSubVolume(CropSubVolumes):
    def __init__(self, size, image_field, label_field=None, max_displacement=50):
        super(CropRandomSubVolume, self).__init__(size, image_field, label_field)
        self.max_displacement = max_displacement

    def __call__(self, data):
        image_entries = []
        label_entries = []

        image_field = self.image_field
        label_field = self.label_field
        if self.label_field is None:
            label_field = self.image_field

        for image_entry, label_entry in zip(data[image_field], data[label_field]):
            assert np.all(np.asarray(image_entry.shape[-3:]) == np.asarray(label_entry.shape[-3:]))

            image_entry, _, _ = self.pad_to_minimal_size(image_entry, pad_mode='constant')
            label_entry, _, _ = self.pad_to_minimal_size(label_entry, pad_mode='constant')

            displacement = np.asarray(image_entry.shape[-3:]) - self.size

            displacement[displacement > self.max_displacement] = self.max_displacement

            if displacement[0] <= 1:
                x_disp = 0
            else:
                x_disp = np.random.randint(-np.floor(displacement[0] / 2.), np.floor(displacement[0] / 2.))

            if displacement[1] <= 1:
                y_disp = 0
            else:
                y_disp = np.random.randint(-np.floor(displacement[1] / 2.), np.floor(displacement[1] / 2.))

            if displacement[2] <= 1:
                z_disp = 0
            else:
                z_disp = np.random.randint(-np.floor(displacement[2] / 2.), np.floor(displacement[2] / 2.))

            h_size = np.floor(np.asarray(self.size) / 2.).astype(int)
            centr_pix = np.floor(np.asarray(image_entry.shape[-3:]) / 2.).astype(int)

            start_px = centr_pix - h_size - np.asarray([x_disp, y_disp, z_disp]).astype(int)

            end_px = (start_px + self.size).astype(int)

            assert np.all(end_px <= np.asarray(image_entry.shape[-3:]))
            assert np.all(start_px >= 0)

            image_patch = image_entry[..., start_px[0]:end_px[0], start_px[1]:end_px[1], start_px[2]:end_px[2]]

            label_patch = label_entry[..., start_px[0]:end_px[0], start_px[1]:end_px[1], start_px[2]:end_px[2]]

            assert np.all(np.asarray(image_patch.shape[-3:]) == self.size)
            assert np.all(np.asarray(label_patch.shape[-3:]) == self.size)

            image_entries.append(image_patch)
            label_entries.append(label_patch)

        data[self.image_field] = image_entries

        if self.label_field is not None:
            data[self.label_field] = label_entries

        return data


class CropCenteredSubVolume:
    def __init__(self, size, field):
        self.size = size
        self.field = field

    def __call__(self, data):
        image_entry, pad_before, pad_after = pad_to_minimal_size(data[self.field], self.size, pad_mode='constant')

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

        data[self.field] = image_patch

        data[self.field + '_start_px'] = crop_before - pad_before

        data[self.field + '_end_px'] = crop_after - pad_after

        return data


class MapValues:
    def __init__(self, field, min_value=0, max_value=1, channelwise=True):
        self.field = field
        self.min_value = min_value
        self.max_value = max_value
        self.channelwise = channelwise

    def __call__(self, data):
        if data[self.field].ndim > 3 and self.channelwise:
            for i in range(data[self.field].shape[0]):
                data[self.field][i] = \
                    (data[self.field][i] - np.min(data[self.field][i])) / \
                    (np.max(data[self.field][i]) - np.min(data[self.field][i]) + EPS)
        else:
            data[self.field] = \
                (data[self.field] - np.min(data[self.field])) / \
                (np.max(data[self.field]) - np.min(data[self.field]) + EPS)

        data[self.field] *= self.max_value - self.min_value
        data[self.field] += self.min_value

        return data


class ThresholdValues:
    def __init__(self, field, threshold, direction='greater'):
        self.field = field
        self.threshold = threshold
        self.direction = direction

    def __call__(self, data):
        if self.direction == 'greater':
            data[self.field] = (data[self.field] > self.threshold).astype(dtype=entry.dtype)
        elif self.direction == 'smaller':
            data[self.field] = (data[self.field] < self.threshold).astype(dtype=entry.dtype)
        elif self.direction == 'greater/equal':
            data[self.field] = (data[self.field] >= self.threshold).astype(dtype=entry.dtype)
        elif self.direction == 'smaller/equal':
            data[self.field] = (data[self.field] <= self.threshold).astype(dtype=entry.dtype)
        else:
            raise ValueError('the direction of inequality {} is not supported'.format(self.direction))

        return data


class AddChannelDimension:
    def __init__(self, field):
        self.field = field

    def __call__(self, data):
        data[self.field] = data[self.field][np.newaxis]

        return data


class LabelMapToOneHot:
    def __init__(self, field, classes):
        self.field = field
        self.classes = classes
        self.num_channels = len(self.classes)

    def __call__(self, data):

        data[self.field] = data[self.field].astype(np.int32)

        onehot = np.zeros([self.num_channels] + list(data[self.field].shape[-3:]), dtype=np.float32)

        for c in range(self.num_channels):
            onehot[c, ...] = (data[self.field] == self.classes[c]).astype(np.float32)

        data[self.field] = onehot

        return data


class StackImagesChannelwise:
    def __init__(self, source_fields, destination_field):
        self.source_fields = source_fields
        self.destination_field = destination_field

    def __call__(self, data):

        composite_image = []

        for key in self.source_fields:
            composite_image.append(data[key])

        data[self.destination_field] = np.stack(composite_image, axis=-1)

        return data


class FixedMeanStdNormalization:
    def __init__(self, field, mean, std):
        assert std != 0

        self.field = field
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data[self.field] = (data[self.field] - self.mean) / self.std

        return data
