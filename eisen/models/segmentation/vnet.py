import torch
from torch import nn


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
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W, D)
        return x * self.weight + self.bias


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none', expand_chan=False):
        super(ResidualConvBlock, self).__init__()

        self.expand_chan = expand_chan
        if self.expand_chan:
            ops = []

            ops.append(nn.Conv3d(n_filters_in, n_filters_out, 1))

            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm3D(n_filters_out))

            ops.append(nn.ReLU(inplace=True))

            self.conv_expan = nn.Sequential(*ops)

        ops = []
        for i in range(n_stages):
            if normalization != 'none':
                ops.append(nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1))
                if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm3d(n_filters_out))
                if normalization == 'groupnorm':
                    ops.append(GroupNorm3D(n_filters_out))
            else:
                ops.append(nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1))

            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        if self.expand_chan:
            x = self.conv(x) + self.conv_expan(x)
        else:
            x = (self.conv(x) + x)

        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm3D(n_filters_out))
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm3D(n_filters_out))
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNet(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            n_filters=16,
            filter_size=3,
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
        :param filter_size: size of filter in voxels
        :type filter_size: int
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
            {"name": "filter_size", "type": "int", "value": "3"},
            {"name": "outputs_activation", "type": "string", "value": ["sigmoid", "softmax", "none"]},
            {"name": "normalization", "type": "string", "value": ["groupnorm", "batchnorm", "none"]}
        ]
        </json>
        """
        # Define operations
        super(VNet, self).__init__()

        if input_channels > 1:
            self.block_one = ResidualConvBlock(1, input_channels, n_filters, normalization=normalization, expand_chan=True)
        else:
            self.block_one = ResidualConvBlock(1, input_channels, n_filters, normalization=normalization)

        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ResidualConvBlock(filter_size, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ResidualConvBlock(filter_size, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ResidualConvBlock(filter_size, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ResidualConvBlock(filter_size, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ResidualConvBlock(filter_size, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ResidualConvBlock(filter_size, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ResidualConvBlock(filter_size, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ResidualConvBlock(1, n_filters, n_filters, normalization=normalization)

        self.out_conv = nn.Conv3d(n_filters, output_channels, 1, padding=0)

        if outputs_activation == 'sigmoid':
            self.outputs_activation_fn = nn.Sigmoid()
        elif outputs_activation == 'softmax':
            self.outputs_activation_fn = nn.Softmax()
        elif outputs_activation == 'none':
            self.outputs_activation_fn = nn.Identity()


    def forward(self, images):
        """
        Computes output of the network.

        :param images: Input tensor containing images
        :type images: torch.Tensor
        :return: prediction
        """
        x1 = self.block_one(images)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)

        out = self.out_conv(x9)

        output = self.outputs_activation_fn(out)

        return output