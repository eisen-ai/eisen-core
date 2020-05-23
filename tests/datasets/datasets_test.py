import numpy as np
import tempfile
import h5py
import os
import csv
import json
import shutil

from eisen.datasets import PatchCamelyon
from eisen.datasets import JsonDataset
from eisen.datasets import MSDDataset
from eisen.datasets import CAMUS
from eisen.datasets import RSNABoneAgeChallenge
from eisen.datasets import RSNAIntracranialHemorrhageDetection
from eisen.datasets import PANDA
from eisen.datasets import ABCsDataset, ABCDataset
from eisen.datasets import EMIDEC


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

    def __del__(self):
        shutil.rmtree(self.base_path)

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

    def __del__(self):
        shutil.rmtree(self.base_path)

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

    def __del__(self):
        shutil.rmtree(self.base_path)

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

    def __del__(self):
        shutil.rmtree(self.base_path)

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

    def __del__(self):
        shutil.rmtree(self.base_path)

    def test_getitem(self):
        item = self.dataset[0]

        assert item['image'] == '/path/to/image_1.png'
        assert item['label'] == '/path/to/label_1png'

        item = self.dataset[1]

        assert item['image'] == '/path/to/image_2.png'
        assert item['label'] == '/path/to/label_2.png'

    def test_len(self):
        assert len(self.dataset) == 2


class TestMSDDataset:
    def setup_class(self):
        self.base_path = tempfile.mkdtemp()

        dataset = {
            "name": "Hippocampus",
            "description": "Left and right hippocampus segmentation",
            "reference": " Vanderbilt University Medical Center",
            "licence": "CC-BY-SA 4.0",
            "relase": "1.0 04/05/2018",
            "tensorImageSize": "3D",
            "modality": {
                "0": "MRI"
            },
            "labels": {
                "0": "background",
                "1": "Anterior",
                "2": "Posterior"
            },
            "numTraining": 260,
            "numTest": 130,
            "training": [
                {"image": "./imagesTr/hippocampus_367.nii.gz", "label": "./labelsTr/hippocampus_367.nii.gz"},
                {"image": "./imagesTr/hippocampus_304.nii.gz", "label": "./labelsTr/hippocampus_304.nii.gz"}
            ],
            "test": [
                "./imagesTs/hippocampus_267.nii.gz",
                "./imagesTs/hippocampus_379.nii.gz"
            ]
        }
        with open(os.path.join(self.base_path, 'json_file.json'), 'w') as outfile:
            json.dump(dataset, outfile)

        self.dataset_train = MSDDataset(self.base_path, 'json_file.json', phase='training')
        self.dataset_test = MSDDataset(self.base_path, 'json_file.json', phase='test')

    def __del__(self):
        shutil.rmtree(self.base_path)

    def test_getitem(self):
        item = self.dataset_train[0]

        assert item['image'] == './imagesTr/hippocampus_367.nii.gz'
        assert item['label'] == './labelsTr/hippocampus_367.nii.gz'

        item = self.dataset_train[1]

        assert item['image'] == './imagesTr/hippocampus_304.nii.gz'
        assert item['label'] == './labelsTr/hippocampus_304.nii.gz'

        item = self.dataset_test[0]

        assert item['image'] == './imagesTs/hippocampus_267.nii.gz'

        item = self.dataset_test[1]

        assert item['image'] == './imagesTs/hippocampus_379.nii.gz'

    def test_len(self):
        assert len(self.dataset_train) == 2
        assert len(self.dataset_test) == 2


class TestPANDA:
    def setup_class(self):
        self.base_path = tempfile.mkdtemp()

        self.csv_content = [
            'image_id,data_provider,isup_grade,gleason_score',
            '0005f7aaab2800f6170c399693a96917,karolinska,0,0+0',
            '000920ad0b612851f8e01bcc880d9b3d,karolinska,0,0+0',
            '0018ae58b01bdadc8e347995b69f99aa,radboud,4,4+4'
        ]

        with open(os.path.join(self.base_path, 'train.csv'), 'w') as f:

            f.write("%s\n%s\n%s\n%s\n" % (
                self.csv_content[0],
                self.csv_content[1],
                self.csv_content[2],
                self.csv_content[3]
            ))

        self.images_path = os.path.join(self.base_path, 'train_images')
        self.labels_path = os.path.join(self.base_path, 'train_label_masks')

        os.mkdir(self.images_path)
        os.mkdir(self.labels_path)

        filenames = [
            os.path.join(self.images_path, '0005f7aaab2800f6170c399693a96917.tiff'),
            os.path.join(self.images_path, '000920ad0b612851f8e01bcc880d9b3d.tiff'),
            os.path.join(self.images_path, '0018ae58b01bdadc8e347995b69f99aa.tiff'),
            os.path.join(self.labels_path, '0005f7aaab2800f6170c399693a96917_mask.tiff'),
            os.path.join(self.labels_path, '000920ad0b612851f8e01bcc880d9b3d_mask.tiff'),
            os.path.join(self.labels_path, '0018ae58b01bdadc8e347995b69f99aa_mask.tiff'),
        ]

        for filename in filenames:
            with open(filename, 'w') as f:
                pass

    def __del__(self):
        shutil.rmtree(self.base_path)

    def test_training_set(self):
        dataset = PANDA(self.base_path, 'train.csv', True)

        element = dataset[0]

        assert element['image'] == os.path.join('train_images', '0005f7aaab2800f6170c399693a96917.tiff')
        assert element['mask'] == os.path.join('train_label_masks', '0005f7aaab2800f6170c399693a96917_mask.tiff')

        assert element['provider'] == 'karolinska'
        assert element['isup'] == 0
        assert element['gleason'] == '0+0'

        assert os.path.exists(os.path.join(self.base_path, element['image']))
        assert os.path.exists(os.path.join(self.base_path, element['mask']))

        assert len(dataset) == 3


class TestABCs:
    def setup_class(self):
        self.flat_path = tempfile.mkdtemp()
        self.structured_path = tempfile.mkdtemp()

        self.part_1_dir = os.path.join(self.structured_path, 'ABCs_training_data_Part1')
        self.part_2_dir = os.path.join(self.structured_path, 'ABCs_training_data_Part2')

        os.mkdir(self.part_1_dir)
        os.mkdir(self.part_2_dir)

        flat_path_touch_file = [
            os.path.join(self.flat_path, '001_ct.mha'),
            os.path.join(self.flat_path, '001_t1.mha'),
            os.path.join(self.flat_path, '001_t2.mha'),
            os.path.join(self.flat_path, '001_labelmap_task2.mha'),
            os.path.join(self.flat_path, '001_labelmap_task1.mha'),
            os.path.join(self.flat_path, '021_ct.mha'),
            os.path.join(self.flat_path, '021_t1.mha'),
            os.path.join(self.flat_path, '021_t2.mha'),
            os.path.join(self.flat_path, '021_labelmap_task2.mha'),
            os.path.join(self.flat_path, '021_labelmap_task1.mha'),
            os.path.join(self.part_1_dir, '001_ct.mha'),
            os.path.join(self.part_1_dir, '001_t1.mha'),
            os.path.join(self.part_1_dir, '001_t2.mha'),
            os.path.join(self.part_1_dir, '001_labelmap_task2.mha'),
            os.path.join(self.part_1_dir, '001_labelmap_task1.mha'),
            os.path.join(self.part_2_dir, '021_ct.mha'),
            os.path.join(self.part_2_dir, '021_t1.mha'),
            os.path.join(self.part_2_dir, '021_t2.mha'),
            os.path.join(self.part_2_dir, '021_labelmap_task2.mha'),
            os.path.join(self.part_2_dir, '021_labelmap_task1.mha')
        ]

        for file in flat_path_touch_file:
            with open(file, 'w') as f:
                f.write('')

    def __del__(self):
        shutil.rmtree(self.structured_path)
        shutil.rmtree(self.flat_path)

    @staticmethod
    def check_training_content(basedir, element):
        assert 'ct' in element.keys()
        assert 't1' in element.keys()
        assert 't2' in element.keys()
        assert 'label_task1' in element.keys()
        assert 'label_task2' in element.keys()

        assert os.path.exists(os.path.join(basedir, element['ct']))
        assert os.path.exists(os.path.join(basedir, element['t1']))
        assert os.path.exists(os.path.join(basedir, element['t2']))
        assert os.path.exists(os.path.join(basedir, element['label_task1']))
        assert os.path.exists(os.path.join(basedir, element['label_task2']))

    @staticmethod
    def check_testing_content(basedir, element):
        assert 'ct' in element.keys()
        assert 't1' in element.keys()
        assert 't2' in element.keys()

        assert os.path.exists(os.path.join(basedir, element['ct']))
        assert os.path.exists(os.path.join(basedir, element['t1']))
        assert os.path.exists(os.path.join(basedir, element['t2']))

    def test_training_structured(self):
        dataset = ABCsDataset(
            data_dir=self.structured_path,
            training=True,
            flat_dir_structure=False,
            transform=None
        )

        assert len(dataset) == 2

        for i in range(len(dataset)):
            element = dataset[i]
            self.check_training_content(self.structured_path, element)

    def test_training_flat(self):
        dataset = ABCsDataset(
            data_dir=self.flat_path,
            training=True,
            flat_dir_structure=True,
            transform=None
        )

        assert len(dataset) == 2

        for i in range(len(dataset)):
            element = dataset[i]
            self.check_training_content(self.flat_path, element)

    def test_testing(self):
        dataset = ABCsDataset(
            data_dir=self.flat_path,
            training=True,
            flat_dir_structure=True,
            transform=None
        )

        assert len(dataset) == 2

        for i in range(len(dataset)):
            element = dataset[i]
            self.check_testing_content(self.flat_path, element)

class TestEMIDEC:
    def setup_class(self):
        self.path = tempfile.mkdtemp()

        os.makedirs(os.path.join(self.path, 'Case_P042', 'Images'))
        os.makedirs(os.path.join(self.path, 'Case_P042', 'Contours'))

        os.makedirs(os.path.join(self.path, 'Case_N069', 'Images'))
        os.makedirs(os.path.join(self.path, 'Case_N069', 'Contours'))

        self.paths = [
            os.path.join('Case_P042', 'Images', 'Case_P042.nii.gz'),
            os.path.join('Case_P042', 'Contours', 'Case_P042.nii.gz'),
            os.path.join('Case_N069', 'Images', 'Case_N069.nii.gz'),
            os.path.join('Case_N069', 'Contours', 'Case_N069.nii.gz'),
        ]

        touch_file = [
            os.path.join(self.path, self.paths[0]),
            os.path.join(self.path, self.paths[1]),
            os.path.join(self.path, self.paths[2]),
            os.path.join(self.path, self.paths[3]),
            os.path.join(self.path, 'Case N069.txt'),
            os.path.join(self.path, 'Case P042.txt'),
        ]

        for file in touch_file:
            with open(file, 'w') as f:
                f.write('dummy')

    def __del__(self):
        shutil.rmtree(self.path)

    def test_training_dataset(self):
        dataset = EMIDEC(
            data_dir=self.path,
            training=True,
            transform=None
        )

        assert len(dataset) == 2

        element = dataset[0]

        image_paths = [
            self.paths[0],
            self.paths[2]
        ]

        label_paths = [
            self.paths[1],
            self.paths[3]
        ]

        assert element['image'] in image_paths
        assert element['label'] in label_paths

        assert element['metadata'] == 'dummy'

        image_paths.remove(element['image'])
        label_paths.remove(element['label'])

        element = dataset[1]

        assert element['image'] in image_paths
        assert element['label'] in label_paths

        assert element['metadata'] == 'dummy'

    def test_test_dataset(self):
        dataset = EMIDEC(
            data_dir=self.path,
            training=False,
            transform=None
        )

        assert len(dataset) == 2

        image_paths = [
            self.paths[0],
            self.paths[2]
        ]

        element = dataset[0]

        assert element['image'] in image_paths

        assert 'label' not in element.keys()
        assert 'pathological' not in element.keys()

        assert element['metadata'] == 'dummy'

        image_paths.remove(element['image'])

        element = dataset[1]

        assert element['image'] in image_paths

        assert 'label' not in element.keys()
        assert 'pathological' not in element.keys()

        assert element['metadata'] == 'dummy'
