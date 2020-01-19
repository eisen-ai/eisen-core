import os
import nibabel as nib


class LoadNiftyFromFilename:
    """
    This transform loads Nifty data from filenames contained in a specific field of the data dictionary.
    """
    def __init__(self, data_dir, fields, canonical=False):
        """
        :param data_dir: source data directory where data is located
        :type data_dir: str
        :param fields: list of names of the field of data dictionary to work on
        :type fields: list
        :param canonical: whether data should be reordered to be closest to canonical (see nibabel documentation)
        :type canonical: bool

        # todo change this: we still don't know how to do it!!
        # -- Machine readable documentation --
        <eisen-obj name=data_dir type=string default='any value' choices='' />
        <eisen-obj name=field type=string default='any value' choices='' />
        <eisen-obj name=canonical type=bool default=False choices=['True', 'False'] />
        """
        self.data_dir = data_dir
        self.fields = fields
        self.canonical = canonical

    def __call__(self, data):
        for field in self.fields:
            img = nib.load(os.path.normpath(os.path.join(self.data_dir, data[field])))

            if self.canonical:
                img = nib.as_closest_canonical(img)

            data[field] = img
            data[field + '_affines'] = img.affine
            data[field + '_orientations'] = nib.aff2axcodes(img.affine)

        return data
