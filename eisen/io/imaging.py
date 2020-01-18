import os
import nibabel as nib


class LoadNiftyFromFilename:
    """
    This transform loads Nifty data from filenames contained in a specific field of the data dictionary.
    """
    def __init__(self, data_dir, field, canonical=False):
        """
        :param data_dir: source data directory where data is located
        :type data_dir: str
        :param field: name of the field of data dictionary to work on
        :type field: string
        :param canonical: whether data should be reordered to be closest to canonical (see nibabel documentation)
        :type canonical: bool

        # todo change this: we still don't know how to do it!!
        # -- Machine readable documentation --
        <eisen-obj name=data_dir type=string default='any value' choices='' />
        <eisen-obj name=field type=string default='any value' choices='' />
        <eisen-obj name=canonical type=bool default=False choices=['True', 'False'] />
        """
        self.data_dir = data_dir
        self.field = field
        self.canonical = canonical

    def __call__(self, data):
        """
        :param data: data dictionary
        :type data: dict
        :return: updated data dictionary
        :rtype: dict
        """
        img = nib.load(os.path.normpath(os.path.join(self.data_dir, data[self.field])))

        if self.canonical:
            img = nib.as_closest_canonical(img)

        data[self.field] = img
        data[self.field + '_affines'] = img.affine
        data[self.field + '_orientations'] = nib.aff2axcodes(img.affine)

        return data
