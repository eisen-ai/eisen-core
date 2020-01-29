import torch
import torch.nn as nn
import torch.nn.functional as F


# ATTRIBUTION: this implementation has been obtained from https://github.com/JielongZ/3D-UNet-PyTorch-Implementation
# The code contained in this file is licensed according to the original license

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

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W, D)
        return x * self.weight + self.bias


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = normalization(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        x = F.elu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth, n_filters, normalization, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = n_filters
        self.num_conv_blocks = 2
        # self.module_list = nn.ModuleList()
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):

                if depth == 0:

                    self.conv_block = ConvBlock(
                        in_channels=in_channels,
                        out_channels=feat_map_channels,
                        normalization=normalization
                    )
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:

                    self.conv_block = ConvBlock(
                        in_channels=in_channels,
                        out_channels=feat_map_channels,
                        normalization=normalization
                    )
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)

                if k.endswith("1"):
                    down_sampling_features.append(x)
            elif k.startswith("max_pooling"):
                x = op(x)

        return x, down_sampling_features


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, n_filters, model_depth, normalization):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = n_filters
        # user nn.ModuleDict() to store ops
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps

            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(
                        in_channels=feat_map_channels * 6,
                        out_channels=feat_map_channels * 2,
                        normalization=normalization
                    )
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(
                        in_channels=feat_map_channels * 2,
                        out_channels=feat_map_channels * 2,
                        normalization=normalization
                    )
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(
                    in_channels=feat_map_channels * 2,
                    out_channels=out_channels,
                    normalization=normalization
                )
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, down_sampling_features):
        """
        :param x: inputs
        :param down_sampling_features: feature maps from encoder path
        :return: output
        """
        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                x = op(x)
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
            elif k.startswith("conv"):
                x = op(x)
            else:
                x = op(x)
        return x


class UNet3D(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            n_filters=16,
            model_depth=4,
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

        if normalization == 'groupnorm':
            normalization = GroupNorm3D
        elif normalization == 'batchnorm':
            normalization = nn.BatchNorm2d
        else:
            normalization = nn.Identity

        self.encoder = EncoderBlock(
            in_channels=input_channels,
            n_filters=n_filters,
            model_depth=model_depth,
            normalization=normalization
        )
        self.decoder = DecoderBlock(
            out_channels=output_channels,
            n_filters=n_filters,
            model_depth=model_depth,
            normalization=normalization
        )

        if outputs_activation == 'sigmoid':
            self.outputs_activation_fn = nn.Sigmoid()
        elif outputs_activation == 'softmax':
            self.outputs_activation_fn = nn.Softmax()
        elif outputs_activation == 'none':
            self.outputs_activation_fn = nn.Identity()

    def forward(self, x):
        x, downsampling_features = self.encoder(x)

        x = self.decoder(x, downsampling_features)

        x = self.outputs_activation_fn(x)

        return x