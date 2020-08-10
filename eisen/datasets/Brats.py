import os
import csv

from torch.utils.data import Dataset


def read_csv(file):
    with open(file, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        lines = []

        for row in csv_reader:
            lines.append(dict(row))

    return lines


class Brats2020:
    """
    BraTS 2020 challenge dataset. This multi modal brain tumor segmentation and survival prediction dataset
    contains multi-center and multi-stage MRI images of brain tumors. It contains images obtained via
    't1', 't1c', 't2' and 'flair' MRI acquisition sequences, and annotations relative to
    the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2),
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1).

    Find more info here: https://www.med.upenn.edu/cbica/brats2020/data.html

    .. note::

        This dataset will generate data entries with keys: 't1', 't1c', 't2, 'flair' and 'name_mapping'.
        If the training flag is set during initialization it will also provide 'label' and 'survival_info'.
        The data in 'name_mapping' and 'survival_info' is also represented in form of dictionary and contains data
        obtained from the fields (columns) of name_mapping.csv and surivival_info.csv

    .. code-block:: python

        from eisen.datasets import Brats2020

        dset = Brats2020('/data/root/path', True, tform)
    """

    def __init__(self, data_dir, training, transform=None):
        """
        :param data_dir: the base directory where the data is located (after unzipping the archive)
        :type data_dir: str
        :param training: whether the labels and survival information should be loaded for training
        :type training: bool
        :param transform: a transform object (can be the result of a composition of transforms)
        :type transform: callable

        .. code-block:: python

            from eisen.datasets import Brats2020

            dset = Brats2020(
                data_dir='/data/root/path',
                training=True,
                transform=tform
            )

        <json>
        [
            {"name": "training", "type": "bool", "value": ""}
        ]
        </json>
        """
        self.data_dir = data_dir
        self.training = training
        self.transform = transform
        self.dataset = []

        patient_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]

        name_mapping = read_csv(os.path.join(data_dir, "name_mapping.csv"))

        if training:
            survival_info = read_csv(os.path.join(data_dir, "survival_info.csv"))

        for patient in patient_dirs:
            data = {
                "t1": os.path.join(patient, "{}_t1.nii.gz".format(patient)),
                "t1ce": os.path.join(patient, "{}_t1ce.nii.gz".format(patient)),
                "t2": os.path.join(patient, "{}_t2.nii.gz".format(patient)),
                "flair": os.path.join(patient, "{}_flair.nii.gz".format(patient)),
            }

            for row in name_mapping:
                if row["BraTS_2020_subject_ID"] == patient:
                    data["name_mapping"] = row

            if self.training:
                for row in survival_info:
                    if row["Brats20ID"] == patient:
                        data["survival_info"] = row

                data["label"] = os.path.join(patient, "{}_seg.nii.gz".format(patient))

            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = copy.deepcopy(self.dataset[idx])

        if self.transform:
            item = self.transform(item)

        return item
