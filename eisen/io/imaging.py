import os
import nibabel as nib


class LoadNiftyFromFilename:
    """
    This transform loads Nifty data from filenames contained in a specific field of the data dictionary.
    """
    def __init__(self, fields, data_dir, canonical=False):
        """
        :param fields: list of names of the field of data dictionary to work on
        :type fields: list
        :param data_dir: source data directory where data is located
        :type data_dir: str
        :param canonical: whether data should be reordered to be closest to canonical (see nibabel documentation)
        :type canonical: bool

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
        for field in self.fields:
            img = nib.load(os.path.normpath(os.path.join(self.data_dir, data[field])))

            if self.canonical:
                img = nib.as_closest_canonical(img)

            data[field] = img
            data[field + '_affines'] = img.affine
            data[field + '_orientations'] = nib.aff2axcodes(img.affine)

        return data
