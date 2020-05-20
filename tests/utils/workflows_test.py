import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam

from eisen.utils import EisenModuleWrapper

from eisen.utils.workflows import GenericWorkflow
from eisen.utils.workflows import Training as WorkflowTraining
from eisen.utils.workflows import Validation as WorkflowValidation
from eisen.utils.workflows import Testing as WorkflowTesting


class Net(nn.Module):
    # dummy network for tests
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 3 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6 * 2 * 2, 16)  # 6*6 from image dimension
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1

        for s in size:
            num_features *= s

        return num_features


class TestGenericWorkflow:
    def setup_class(self):
        self.batch = {
            'x': torch.rand((2, 1, 8, 8)),  # batch size 2, 1 input channel, 8x8 pixels
            'y': torch.LongTensor([0, 1])  # class for those two images (0 and 1 respectively)
        }

        self.module = EisenModuleWrapper(Net(), input_names=['x'], output_names=['pred'])

        self.generic_module = GenericWorkflow(self.module, gpu=False)

        assert isinstance(self.generic_module, GenericWorkflow)

    def test_call(self):
        output, losses, metrics = self.generic_module(self.batch)

        assert isinstance(losses, list)
        assert len(losses) == 0

        assert isinstance(metrics, list)
        assert len(metrics) == 0

        assert isinstance(output, dict)

        pred = output['pred']

        assert isinstance(pred, torch.Tensor)

        assert pred.size()[0] == 2
        assert pred.size()[1] == 2


class DummyDataset(Dataset):
    def __init__(self):
        self.dataset = [
            {
                'x': torch.rand((1, 8, 8)),  # batch size 2, 1 input channel, 8x8 pixels
                'y': 0  # class for those two images (0 and 1 respectively)
            },
            {
                'x': torch.rand((1, 8, 8)),
                'y': 1
            },
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


class TestWorkflowTraining:
    def setup_class(self):
        self.batch = {
            'x': torch.rand((2, 1, 8, 8)),  # batch size 2, 1 input channel, 8x8 pixels
            'y': torch.LongTensor([0, 1])  # class for those two images (0 and 1 respectively)
        }

        self.data_loader = DataLoader(
            DummyDataset(),
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        self.module = EisenModuleWrapper(Net(), input_names=['x'], output_names=['pred'])

        self.optimizer = Adam(self.module.parameters(), 0.001)

        self.loss = EisenModuleWrapper(module=CrossEntropyLoss(), input_names=['pred', 'y'], output_names=['loss'])

        self.metric = EisenModuleWrapper(module=CrossEntropyLoss(), input_names=['pred', 'y'], output_names=['metric'])

        self.training_workflow = WorkflowTraining(
            self.module,
            self.data_loader,
            [self.loss],
            self.optimizer,
            [self.metric],
            gpu=False
        )

        assert isinstance(self.training_workflow, WorkflowTraining)

    def test_call(self):
        output, losses, metrics = self.training_workflow(self.batch)

        assert isinstance(losses, list)
        assert len(losses) == 1
        assert isinstance(losses[0], dict)
        assert isinstance(losses[0]['loss'], torch.Tensor)

        assert isinstance(metrics, list)
        assert len(metrics) == 1
        assert isinstance(metrics[0], dict)
        assert isinstance(metrics[0]['metric'], torch.Tensor)

        assert isinstance(output, dict)

        pred = output['pred']

        assert isinstance(pred, torch.Tensor)

        assert pred.size()[0] == 2
        assert pred.size()[1] == 2

    def test_run(self):
        self.training_workflow.run()

        assert self.training_workflow.epoch == 1

        self.training_workflow.run()

        assert self.training_workflow.epoch == 2


class TestWorkflowValidation:
    def setup_class(self):
        self.batch = {
            'x': torch.rand((2, 1, 8, 8)),  # batch size 2, 1 input channel, 8x8 pixels
            'y': torch.LongTensor([0, 1])  # class for those two images (0 and 1 respectively)
        }

        self.data_loader = DataLoader(
            DummyDataset(),
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        self.module = EisenModuleWrapper(Net(), input_names=['x'], output_names=['pred'])

        self.optimizer = Adam(self.module.parameters(), 0.001)

        self.loss = EisenModuleWrapper(module=CrossEntropyLoss(), input_names=['pred', 'y'], output_names=['loss'])

        self.metric = EisenModuleWrapper(module=CrossEntropyLoss(), input_names=['pred', 'y'], output_names=['metric'])

        self.validation_workflow = WorkflowValidation(
            self.module,
            self.data_loader,
            [self.loss],
            [self.metric],
            gpu=False
        )

        assert isinstance(self.validation_workflow, WorkflowValidation)

    def test_call(self):
        output, losses, metrics = self.validation_workflow(self.batch)

        assert isinstance(losses, list)
        assert len(losses) == 1
        assert isinstance(losses[0], dict)
        assert isinstance(losses[0]['loss'], torch.Tensor)

        assert isinstance(metrics, list)
        assert len(metrics) == 1
        assert isinstance(metrics[0], dict)
        assert isinstance(metrics[0]['metric'], torch.Tensor)

        assert isinstance(output, dict)

        pred = output['pred']

        assert isinstance(pred, torch.Tensor)

        assert pred.size()[0] == 2
        assert pred.size()[1] == 2

    def test_run(self):
        self.validation_workflow.run()

        assert self.validation_workflow.epoch == 1

        self.validation_workflow.run()

        assert self.validation_workflow.epoch == 2


class TestWorkflowTesting:
    def setup_class(self):
        self.batch = {
            'x': torch.rand((2, 1, 8, 8)),  # batch size 2, 1 input channel, 8x8 pixels
            'y': torch.LongTensor([0, 1])  # class for those two images (0 and 1 respectively)
        }

        self.data_loader = DataLoader(
            DummyDataset(),
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        self.module = EisenModuleWrapper(Net(), input_names=['x'], output_names=['pred'])

        self.optimizer = Adam(self.module.parameters(), 0.001)

        self.loss = EisenModuleWrapper(module=CrossEntropyLoss(), input_names=['pred', 'y'], output_names=['loss'])

        self.metric = EisenModuleWrapper(module=CrossEntropyLoss(), input_names=['pred', 'y'], output_names=['metric'])

        self.testing_workflow = WorkflowTesting(
            self.module,
            self.data_loader,
            [self.metric],
            gpu=False
        )

        assert isinstance(self.testing_workflow, WorkflowTesting)

    def test_call(self):
        output, losses, metrics = self.testing_workflow(self.batch)

        assert isinstance(losses, list)
        assert len(losses) == 0

        assert isinstance(metrics, list)
        assert len(metrics) == 1
        assert isinstance(metrics[0], dict)
        assert isinstance(metrics[0]['metric'], torch.Tensor)

        assert isinstance(output, dict)

        pred = output['pred']

        assert isinstance(pred, torch.Tensor)

        assert pred.size()[0] == 2
        assert pred.size()[1] == 2

    def test_run(self):
        self.testing_workflow.run()


class TestDataParallelTraining(TestWorkflowTraining):
    def setup_class(self):
        self.batch = {
            'x': torch.rand((2, 1, 8, 8)),  # batch size 2, 1 input channel, 8x8 pixels
            'y': torch.LongTensor([0, 1])  # class for those two images (0 and 1 respectively)
        }

        self.data_loader = DataLoader(
            DummyDataset(),
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        data_parallel_net = torch.nn.DataParallel(Net())

        self.module = EisenModuleWrapper(data_parallel_net, input_names=['x'], output_names=['pred'])

        self.optimizer = Adam(self.module.parameters(), 0.001)

        self.loss = EisenModuleWrapper(module=CrossEntropyLoss(), input_names=['pred', 'y'], output_names=['loss'])

        self.metric = EisenModuleWrapper(module=CrossEntropyLoss(), input_names=['pred', 'y'], output_names=['metric'])

        self.training_workflow = WorkflowTraining(
            self.module,
            self.data_loader,
            [self.loss],
            self.optimizer,
            [self.metric],
            gpu=False
        )

        assert isinstance(self.training_workflow, WorkflowTraining)
