import torch
import torch.nn as nn
import torch.nn.functional as F


# ATTRIBUTION: this implementation has been obtained from https://github.com/UdonDa/3D-UNet-PyTorch

"""
MIT License

Copyright (c) 2018 HoritaDaichi

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


class GroupNorm3D(nn.Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5):
        super(GroupNorm3D, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W, D = x.size()

        G = self.num_groups

        if C % G != 0:
            while C % G != 0 and G > 0:
                G -= 1
            print('Warning: a GroupNorm3D operation was requested num_groups {} but had to use {} instead'.format(
                self.num_groups,
                G
            ))
            self.num_groups = G

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W, D)
        return x * self.weight + self.bias


def conv_block_3d(in_dim, out_dim, activation, normalization):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        normalization(out_dim),
        activation, )


def conv_trans_block_3d(in_dim, out_dim, activation, normalization):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        normalization(out_dim),
        activation, )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation, normalization):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation, normalization),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        normalization(out_dim), )


class UNet3D(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            n_filters=16,
            outputs_activation='sigmoid',
            normalization='groupnorm'
    ):
        """
        :param input_channels: number of input channels
        :type input_channels: int
        :param output_channels: number of output channels
        :type output_channels: int
        :param n_filters: number of filters
        :type n_filters: int
        :param outputs_activation: output activation type either sigmoid, softmax or none
        :type outputs_activation: str
        :param normalization: normalization either groupnorm, batchnorm or none
        :type normalization: str

        <json>
        [
            {"name": "input_names", "type": "list:string", "value": "['images']"},
            {"name": "output_names", "type": "list:string", "value": "['output']"},
            {"name": "input_channels", "type": "int", "value": ""},
            {"name": "output_channels", "type": "int", "value": ""},
            {"name": "n_filters", "type": "int", "value": "16"},
            {"name": "outputs_activation", "type": "string", "value": ["sigmoid", "softmax", "none"]},
            {"name": "normalization", "type": "string", "value": ["groupnorm", "batchnorm", "none"]}
        ]
        </json>
        """
        super(UNet3D, self).__init__()

        self.in_dim = input_channels
        self.out_dim = output_channels
        self.num_filters = n_filters

        if normalization == 'groupnorm':
            normalization = GroupNorm3D
        elif normalization == 'batchnorm':
            normalization = nn.BatchNorm3d
        else:
            normalization = nn.Identity

        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation, normalization)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation, normalization)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation, normalization)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation, normalization)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation, normalization)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation, normalization)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation, normalization)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation, normalization)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation, normalization)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation, normalization)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation, normalization)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation, normalization)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation, normalization)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation, normalization)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation, normalization)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation, normalization)

        # Output
        self.out = conv_block_3d(self.num_filters, self.out_dim, activation, nn.Identity)

        if outputs_activation == 'sigmoid':
            self.outputs_activation_fn = nn.Sigmoid()
        elif outputs_activation == 'softmax':
            self.outputs_activation_fn = nn.Softmax()
        else:
            self.outputs_activation_fn = nn.Identity()

    def forward(self, x):
        """
        Computes output of the network.

        :param x: Input tensor containing images
        :type x: torch.Tensor
        :return: prediction
        """
        # Down sampling
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        down_5 = self.down_5(pool_4)
        pool_5 = self.pool_5(down_5)

        # Bridge
        bridge = self.bridge(pool_5)

        # Up sampling
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_5], dim=1)
        up_1 = self.up_1(concat_1)

        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_4], dim=1)
        up_2 = self.up_2(concat_2)

        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_3], dim=1)
        up_3 = self.up_3(concat_3)

        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_2], dim=1)
        up_4 = self.up_4(concat_4)

        trans_5 = self.trans_5(up_4)
        concat_5 = torch.cat([trans_5, down_1], dim=1)
        up_5 = self.up_5(concat_5)

        # Output
        out = self.out(up_5)

        return self.outputs_activation_fn(out)
