import os
import torch
import nibabel as nib
import numpy as np
import copy

from torch.utils.data import Dataset


class MedSegCovid19(Dataset):
    """
    This object allows the medical segmentation covid-19 dataset to be easily imported into Eisen. Find more
    information about this dataset here: http://medicalsegmentation.com/covid19/

    In summary, this dataset is a collection of 100 slices of CT images that have been annotated and were made
    available to the community.

    When instantiating this module it is necessary to point it to the nifti image file and, optionally,
    the segmentation. The first argument is the data base directory. The second and third argument should be strings
    representing the name of the nifti images of this dataset, and the fourth argument is a transform (or composition
    of transforms).

    Each entry of this dataset after loading will be a dictionary with one (in case only images are loaded) or two
    (in case both images and labels are loaded) keys. Each key stores a numpy array containing the 2D data relative to
    one image.

    .. note::

        This dataset will generate data entries with keys: 'image' and (optionally) 'label'.


    .. code-block:: python

        from eisen.datasets import MedSegCovid19

        dataset = MedSegCovid19(
            '/abs/path/to/data',
            'tr_im.nii',
            'tr_mask.nii',
            transform,
        )

    """
    def __init__(self, data_dir, image_file, mask_file=None, transform=None):
        """
        :param data_dir: the base directory where the data is located (results of download)
        :type data_dir: str
        :param image_file: the name of the nifti file containing the images
        :type image_file: str
        :param mask_file: the name of the nifti file containing the masks (optional)
        :type mask_file: string
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: callable

        .. code-block:: python

            from eisen.datasets import MedSegCovid19

            dataset = MedSegCovid19(
                image_file='tr_im.nii',
                mask_file='tr_mask.nii',
                transform=transform,
            )

        <json>
        [
            {"name": "image_file", "type": "string", "value": ""},
            {"name": "mask_file", "type": "string", "value": ""}
        ]
        </json>
        """

        self.data = []

        image_path = os.path.join(data_dir, image_file)
        img_nii = nib.load(os.path.normpath(image_path))
        img_numpy = np.asanyarray(img_nii.dataobj).astype(np.float32)

        for i in range(img_numpy.shape[-1]):
            self.data.append({'image': img_numpy[..., i]})

        if mask_file is not None and mask_file is not "":
            mask_path = os.path.join(data_dir, mask_file)
            mask_nii = nib.load(os.path.normpath(mask_path))
            mask_numpy = np.asanyarray(mask_nii.dataobj).astype(np.float32)

            for i in range(img_numpy.shape[-1]):
                self.data[i]['label'] = mask_numpy[..., i]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data[idx]

        if self.transform:
            item = self.transform(copy.deepcopy(item))

        return item
