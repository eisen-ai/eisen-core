import numpy as np
import tempfile
import h5py
import os
import csv
import json

from eisen.datasets import PatchCamelyon
from eisen.datasets import JsonDataset
from eisen.datasets import MSDDataset
from eisen.datasets import CAMUS
from eisen.datasets import RSNABoneAgeChallenge
from eisen.datasets import RSNAIntracranialHemorrhageDetection


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


class TestPatchCamelyon:
    def setup_class(self):
        data_x = [
            np.zeros([32, 32, 3], dtype=np.float32),
            np.ones([32, 32, 3], dtype=np.float32),
        ]

        data_y = [
            np.asarray([[[0]]]),
            np.asarray([[[1]]]),
        ]

        self.base_path = tempfile.mkdtemp()

        self.file_name_x = 'data_x.h5'
        self.file_name_y = 'data_y.h5'

        h5f_x = h5py.File(os.path.join(self.base_path, self.file_name_x), 'w')
        h5f_x.create_dataset('x', data=data_x)

        h5f_y = h5py.File(os.path.join(self.base_path, self.file_name_y), 'w')
        h5f_y.create_dataset('y', data=data_y)

        self.camelyon_dset = PatchCamelyon(self.base_path, self.file_name_x, self.file_name_y)

    def test_getitem(self):
        item = self.camelyon_dset[0]

        assert np.all(item['image'] == 0)
        assert np.all(item['label'] == 0)

        assert item['image'].shape[0] == 3
        assert item['image'].shape[1] == 32
        assert item['image'].shape[2] == 32

        assert item['label'].shape[0] == 1

    def test_len(self):
        assert len(self.camelyon_dset) == 2


class TestCAMUS:
    def setup_class(self):
        self.base_path = tempfile.mkdtemp()

        os.makedirs(os.path.join(self.base_path, 'patient0001'), exist_ok=True)

        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ED.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ES.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ED.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ES.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ED_gt.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_ES_gt.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ED_gt.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_ES_gt.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_4CH_sequence.mhd'))
        touch(os.path.join(self.base_path, 'patient0001', 'patient0001_2CH_sequence.mhd'))

        self.camus_dataset = CAMUS(
            self.base_path,
            with_ground_truth=True,
            with_2CH=True,
            with_4CH=True,
            with_entire_sequences=True
        )

    def test_getitem(self):
        item = self.camus_dataset[0]

        assert item['type'] == 'ED'

        assert item['image_2CH'] == str(os.path.join('patient0001', 'patient0001_2CH_ED.mhd'))
        assert item['image_4CH'] == str(os.path.join('patient0001', 'patient0001_4CH_ED.mhd'))

        assert item['label_2CH'] == str(os.path.join('patient0001', 'patient0001_2CH_ED_gt.mhd'))
        assert item['label_4CH'] == str(os.path.join('patient0001', 'patient0001_4CH_ED_gt.mhd'))

        assert item['sequence_2CH'] == \
            str(os.path.join('patient0001', 'patient0001_2CH_sequence.mhd'))
        assert item['sequence_4CH'] == \
            str(os.path.join('patient0001', 'patient0001_4CH_sequence.mhd'))

        item = self.camus_dataset[1]

        assert item['type'] == 'ES'

        assert item['image_2CH'] == str(os.path.join('patient0001', 'patient0001_2CH_ES.mhd'))
        assert item['image_4CH'] == str(os.path.join('patient0001', 'patient0001_4CH_ES.mhd'))

        assert item['label_2CH'] == str(os.path.join('patient0001', 'patient0001_2CH_ES_gt.mhd'))
        assert item['label_4CH'] == str(os.path.join('patient0001', 'patient0001_4CH_ES_gt.mhd'))

        assert item['sequence_2CH'] == \
            str(os.path.join('patient0001', 'patient0001_2CH_sequence.mhd'))
        assert item['sequence_4CH'] == \
            str(os.path.join('patient0001', 'patient0001_4CH_sequence.mhd'))

    def test_len(self):
        assert len(self.camus_dataset) == 2


class TestRSNABoneAgeChallenge:
    def setup_class(self):
        self.base_path = tempfile.mkdtemp()

        training_img_dir = os.path.join(self.base_path, 'boneage-training-dataset', 'boneage-training-dataset')
        test_img_dir = os.path.join(self.base_path, 'boneage-test-dataset', 'boneage-test-dataset')

        os.makedirs(test_img_dir, exist_ok=True)
        os.makedirs(training_img_dir, exist_ok=True)

        touch(os.path.join(training_img_dir, '12346.png'))
        touch(os.path.join(training_img_dir, '12345.png'))

        touch(os.path.join(test_img_dir, '22222.png'))
        touch(os.path.join(test_img_dir, '11111.png'))

        with open(os.path.join(self.base_path, 'boneage-training-dataset.csv'), mode='w') as metadata:
            fieldnames = ['id', 'boneage', 'male']
            writer = csv.DictWriter(metadata, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({'id': '12345', 'boneage': 150, 'male': True})
            writer.writerow({'id': '12346', 'boneage': 156, 'male': False})

        with open(os.path.join(self.base_path, 'boneage-test-dataset.csv'), mode='w') as metadata:
            fieldnames = ['CaseID', 'Sex']
            writer = csv.DictWriter(metadata, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({'CaseID': '11111', 'Sex': True})
            writer.writerow({'CaseID': '22222', 'Sex': False})

        self.dataset_train = RSNABoneAgeChallenge(
            self.base_path,
            training=True
        )

        self.dataset_test = RSNABoneAgeChallenge(
            self.base_path,
            training=False
        )

    def test_getitem(self):
        for item in self.dataset_train:
            assert item['image'] == os.path.join('boneage-training-dataset', 'boneage-training-dataset', '12345.png') or \
                   item['image'] == os.path.join('boneage-training-dataset', 'boneage-training-dataset', '12346.png')

            if item['image'] == os.path.join('boneage-training-dataset', 'boneage-training-dataset', '12345.png'):
                assert item['label'] == 150
                assert item['male']

            if item['image'] == os.path.join('boneage-training-dataset', 'boneage-training-dataset', '12346.png'):
                assert item['label'] == 156
                assert not item['male']

        for item in self.dataset_test:
            assert item['image'] == os.path.join('boneage-test-dataset', 'boneage-test-dataset', '11111.png') or \
                   item['image'] == os.path.join('boneage-test-dataset', 'boneage-test-dataset', '22222.png')

            if item['image'] == os.path.join('boneage-training-dataset', 'boneage-training-dataset', '11111.png'):
                assert item['male']

            if item['image'] == os.path.join('boneage-training-dataset', 'boneage-training-dataset', '22222.png'):
                assert not item['male']

    def test_len(self):
        assert len(self.dataset_train) == 2
        assert len(self.dataset_test) == 2


class TestRSNAIntracranialHemorrhageDetection:
    def setup_class(self):
        self.base_path = tempfile.mkdtemp()

        training_img_dir = os.path.join(self.base_path, 'stage_2_train')
        test_img_dir = os.path.join(self.base_path, 'stage_2_test')

        os.makedirs(test_img_dir, exist_ok=True)
        os.makedirs(training_img_dir, exist_ok=True)

        touch(os.path.join(training_img_dir, 'ID_000012eaf.dcm'))
        touch(os.path.join(training_img_dir, 'ID_000039fa0.dcm'))

        touch(os.path.join(test_img_dir, 'ID_000000e27.dcm'))
        touch(os.path.join(test_img_dir, 'ID_000009146.dcm'))

        with open(os.path.join(self.base_path, 'stage_2_train.csv'), mode='w') as metadata:
            fieldnames = ['ID', 'Label']
            writer = csv.DictWriter(metadata, fieldnames=fieldnames)

            writer.writeheader()

            writer.writerow({'ID': 'ID_000012eaf_epidural', 'Label': 0.1})
            writer.writerow({'ID': 'ID_000012eaf_intraparenchymal', 'Label': 0.2})
            writer.writerow({'ID': 'ID_000012eaf_intraventricular', 'Label': 0.1})
            writer.writerow({'ID': 'ID_000012eaf_subarachnoid', 'Label': 0})
            writer.writerow({'ID': 'ID_000012eaf_subdural', 'Label': 0})
            writer.writerow({'ID': 'ID_000012eaf_any', 'Label': 0.6})

            writer.writerow({'ID': 'ID_000039fa0_epidural', 'Label': 0})
            writer.writerow({'ID': 'ID_000039fa0_intraparenchymal', 'Label': 0})
            writer.writerow({'ID': 'ID_000039fa0_intraventricular', 'Label': 0})
            writer.writerow({'ID': 'ID_000039fa0_subarachnoid', 'Label': 0})
            writer.writerow({'ID': 'ID_000039fa0_subdural', 'Label': 0.1})
            writer.writerow({'ID': 'ID_000039fa0_any', 'Label': 0.9})

        self.dataset_train = RSNAIntracranialHemorrhageDetection(
            self.base_path,
            training=True
        )

        self.dataset_test = RSNAIntracranialHemorrhageDetection(
            self.base_path,
            training=False
        )

    def test_getitem(self):
        for item in self.dataset_train:
            assert item['image'] == os.path.join('stage_2_train', 'ID_000012eaf.dcm') or \
                   item['image'] == os.path.join('stage_2_train', 'ID_000039fa0.dcm')

            if item['image'] == os.path.join('stage_2_train', 'ID_000012eaf.dcm'):
                assert np.all(item['label'] == np.asarray([0.1, 0.2, 0.1, 0, 0, 0.6]))

            if item['image'] == os.path.join('stage_2_train', 'ID_000039fa0.dcm'):
                assert np.all(item['label'] == np.asarray([0, 0, 0, 0, 0.1, 0.9]))

        for item in self.dataset_test:
            assert item['image'] == os.path.join('stage_2_test', 'ID_000000e27.dcm') or \
                   item['image'] == os.path.join('stage_2_test', 'ID_000009146.dcm')

    def test_len(self):
        assert len(self.dataset_train) == 2
        assert len(self.dataset_test) == 2


class TestJsonDataset:
    def setup_class(self):
        self.base_path = tempfile.mkdtemp()

        dataset = [
            {'image': '/path/to/image_1.png', 'label': '/path/to/label_1png'},
            {'image': '/path/to/image_2.png', 'label': '/path/to/label_2.png'},
        ]

        with open(os.path.join(self.base_path, 'json_file.json'), 'w') as outfile:
            json.dump(dataset, outfile)

        self.dataset = JsonDataset(self.base_path, 'json_file.json')

    def test_getitem(self):
        item = self.dataset[0]

        assert item['image'] == '/path/to/image_1.png'
        assert item['label'] == '/path/to/label_1png'

        item = self.dataset[1]

        assert item['image'] == '/path/to/image_2.png'
        assert item['label'] == '/path/to/label_2.png'

    def test_len(self):
        assert len(self.dataset) == 2
