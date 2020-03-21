import torch
import torch.nn.functional as F

from torch import nn


# ATTRIBUTION: the code contained in this file was obtained from https://github.com/mattiaspaul/OBELISK
# the code is licensed under MIT license which is also included below

"""
MIT License

Copyright (c) 2018 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

class ObeliskMIDL(nn.Module):
    def __init__(self, num_labels, full_res, outputs_activation='sigmoid'):
        """
        :param num_labels: number of output channels
        :type num_labels: int
        :param full_res: final resolution (sizes)
        :type full_res: list
        :param outputs_activation: output activation type either sigmoid, softmax or none
        :type outputs_activation: str

        <json>
        [
            {"name": "input_names", "type": "list:string", "value": "['images']"},
            {"name": "output_names", "type": "list:string", "value": "['output']"},
            {"name": "num_labels", "type": "int", "value": ""},
            {"name": "full_res", "type": "list:int", "value": ""},
            {"name": "outputs_activation", "type": "string", "value": ["sigmoid", "softmax", "none"]}
        ]
        </json>
        """
        super(ObeliskMIDL, self).__init__()
        self.num_labels = num_labels
        self.full_res = full_res

        D_in1 = full_res[0]
        H_in1 = full_res[1]
        W_in1 = full_res[2]
        D_in2 = (D_in1 + 1) // 2
        H_in2 = (H_in1 + 1) // 2
        W_in2 = (W_in1 + 1) // 2  # half resolution

        self.half_res = torch.Tensor([D_in2, H_in2, W_in2]).long()

        half_res = self.half_res
        D_in4 = (D_in2 + 1) // 2
        H_in4 = (H_in2 + 1) // 2
        W_in4 = (W_in2 + 1) // 2  # quarter resolution

        self.quarter_res = torch.Tensor([D_in4, H_in4, W_in4]).long()
        quarter_res = self.quarter_res

        # Obelisk Layer
        # sample_grid: 1 x    1     x #samples x 1 x 3
        # offsets:     1 x #offsets x     1    x 1 x 3

        self.sample_grid1 = F.affine_grid(
            torch.eye(3, 4).unsqueeze(0),
            torch.Size((1, 1, full_res[0], full_res[1], full_res[2]))
        )

        self.sample_grid1.requires_grad = False

        # in this model (binary-variant) two spatial offsets are paired
        self.offset1 = nn.Parameter(torch.randn(1, 1024, 1, 2, 3) * 0.2)

        # Dense-Net with 1x1x1 kernels
        self.LIN1 = nn.Conv3d(1024, 256, 1, bias=False, groups=4)  # grouped convolutions
        self.BN1 = nn.BatchNorm3d(256)
        self.LIN2 = nn.Conv3d(256, 128, 1, bias=False)
        self.BN2 = nn.BatchNorm3d(128)

        self.LIN3a = nn.Conv3d(128, 32, 1, bias=False)
        self.BN3a = nn.BatchNorm3d(128 + 32)
        self.LIN3b = nn.Conv3d(128 + 32, 32, 1, bias=False)
        self.BN3b = nn.BatchNorm3d(128 + 64)
        self.LIN3c = nn.Conv3d(128 + 64, 32, 1, bias=False)
        self.BN3c = nn.BatchNorm3d(128 + 96)
        self.LIN3d = nn.Conv3d(128 + 96, 32, 1, bias=False)
        self.BN3d = nn.BatchNorm3d(256)

        self.LIN4 = nn.Conv3d(256, num_labels, 1)

        if outputs_activation == 'sigmoid':
            self.outputs_activation = nn.Sigmoid()
        elif outputs_activation == 'softmax':
            self.outputs_activation = nn.Softmax()
        elif outputs_activation == 'none':
            self.outputs_activation = nn.Identity()

    def forward(self, images, sample_grid=None):
        """
        Computes output of the Obelisk network.

        :param images: Input tensor containing images
        :type images: torch.Tensor
        :param sample_grid: Optional parameter, sampling grid. can be obtained via F.affine_grid(...)
        :type sample_grid: torch.Tensor
        :return: prediction
        """
        B, C, D, H, W = images.size()

        if (sample_grid is None):
            sample_grid = self.sample_grid1

        sample_grid = sample_grid.to(images.device)
        # pre-smooth image (has to be done in advance for original models )
        # x00 = F.avg_pool3d(inputImg,3,padding=1,stride=1)

        _, D_grid, H_grid, W_grid, _ = sample_grid.size()

        factor1 = F.grid_sample(
            images,
            (sample_grid.view(1, 1, -1, 1, 3).repeat(B, 1, 1, 1, 1) + self.offset1[:, :, :, :, :])
        ).view(B, -1, D_grid, H_grid, W_grid)

        factor2 = F.grid_sample(
            images,
            (sample_grid.view(1, 1, -1, 1, 3).repeat(B, 1, 1, 1, 1) + self.offset1[:, :, :, 1:2, :])
        ).view(B, -1, D_grid, H_grid, W_grid)

        input = factor1 - factor2

        x1 = F.relu(self.BN1(self.LIN1(input)))
        x2 = self.BN2(self.LIN2(x1))

        x3a = torch.cat((x2, F.relu(self.LIN3a(x2))), dim=1)
        x3b = torch.cat((x3a, F.relu(self.LIN3b(self.BN3a(x3a)))), dim=1)
        x3c = torch.cat((x3b, F.relu(self.LIN3c(self.BN3b(x3b)))), dim=1)
        x3d = torch.cat((x3c, F.relu(self.LIN3d(self.BN3c(x3c)))), dim=1)

        x4 = self.LIN4(self.BN3d(x3d))
        # return half-resolution segmentation/prediction

        out = self.outputs_activation(x4)

        pred = F.interpolate(
            out,
            size=[self.full_res[0], self.full_res[1], self.full_res[2]],
            mode='trilinear',
            align_corners=False
        )

        return pred