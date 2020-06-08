import os
import csv

from torch.utils.data import Dataset


def read_csv(file):
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        lines = []

        for row in csv_reader:
            lines.append(dict(row))

    return lines


class Brats2020:
    def __init__(self, data_dir, training, transform=None):
        self.data_dir = data_dir
        self.training = training
        self.transform = transform
        self.dataset = []

        patient_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]

        if training:
            name_mapping = read_csv(os.path.join(data_dir, 'name_mapping.csv'))
            survival_info = read_csv(os.path.join(data_dir, 'survival_info.csv'))

        for patient in patient_dirs:
            data = {
                't1': os.path.join(patient, '{}_t1.nii.gz'.format(patient)),
                't1ce': os.path.join(patient, '{}_t1ce.nii.gz'.format(patient)),
                't2': os.path.join(patient, '{}_t2.nii.gz'.format(patient)),
                'flair': os.path.join(patient, '{}_flair.nii.gz'.format(patient)),
            }

            if self.training:
                for row in name_mapping:
                    if row['BraTS_2020_subject_ID'] == patient:
                        data['name_mapping'] = row

                for row in survival_info:
                    if row['Brats20ID'] == patient:
                        data['survival_info'] = row

                data['label'] = os.path.join(patient, '{}_seg.nii.gz'.format(patient))

            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.transform:
            item = self.transform(item)

        return item
