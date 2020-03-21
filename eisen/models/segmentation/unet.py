import torch
import torch.nn as nn
import torch.nn.functional as F


# This implementation of UNet has been obtained from https://github.com/milesial/Pytorch-UNet
# The implementation has been adapted to better fit Eisen framework.

# WARNING: The code contained in this file is licensed under the GNU General Public License v3.0
# which you can find here https://github.com/milesial/Pytorch-UNet/blob/master/LICENSE

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, normalization_fn):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            normalization_fn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            normalization_fn(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, normalization_fn):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, normalization_fn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, normalization_fn, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, normalization_fn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=16, **kwargs):
        super(GroupNorm, self).__init__(num_groups, num_channels, **kwargs)


class UNet(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            n_filters=64,
            bilinear=False,
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
            {"name": "bilinear", "type": "bool", "value": "false"},
            {"name": "outputs_activation", "type": "string", "value": ["sigmoid", "softmax", "none"]},
            {"name": "normalization", "type": "string", "value": ["groupnorm", "batchnorm", "none"]}
        ]
        </json>
        """
        super(UNet, self).__init__()
        self.n_channels = input_channels
        self.n_classes = output_channels
        self.bilinear = bilinear
        self.n_filters = n_filters

        if normalization == 'groupnorm':
            normalization_fn = GroupNorm
        elif normalization == 'batchnorm':
            normalization_fn = nn.BatchNorm2d
        else:
            normalization_fn = nn.Identity

        self.inc = DoubleConv(self.n_channels, self.n_filters, normalization_fn)
        self.down1 = Down(self.n_filters, self.n_filters * 2, normalization_fn)
        self.down2 = Down(self.n_filters * 2, self.n_filters * 4, normalization_fn)
        self.down3 = Down(self.n_filters * 4, self.n_filters * 8, normalization_fn)
        self.down4 = Down(self.n_filters * 8, self.n_filters * 8, normalization_fn)
        self.up1 = Up(self.n_filters * 16, self.n_filters * 4, normalization_fn, bilinear)
        self.up2 = Up(self.n_filters * 8, self.n_filters * 2, normalization_fn, bilinear)
        self.up3 = Up(self.n_filters * 4, self.n_filters, normalization_fn, bilinear)
        self.up4 = Up(self.n_filters * 2, self.n_filters, normalization_fn, bilinear)
        self.outc = OutConv(self.n_filters, self.n_classes)

        self.outputs_activation = outputs_activation
        self.normalization = normalization

        if self.outputs_activation == 'sigmoid':
            self.outputs_activation_fn = nn.Sigmoid()
        elif outputs_activation == 'softmax':
            self.outputs_activation_fn = nn.Softmax()
        elif outputs_activation == 'none':
            self.outputs_activation_fn = nn.Identity()

    def forward(self, x):
        """
        Computes output of the network.

        :param x: Input tensor containing images
        :type x: torch.Tensor
        :return: prediction
        """

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return self.outputs_activation_fn(logits)
