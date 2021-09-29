import torch
import torch.nn as nn

# CONSTANTS
CONV_KERNEL_SIZE = 3
CONV_STRIDE = 1
CONV_PADDING = 1

MAX_POOL_KERNEL_SIZE = 2
MAX_POOL_STRIDE = 2

CONV_TRANS_KERNEL_SIZE = 2
CONV_TRANS_STRIDE = 2

OUTPUT_CONV_KERNEL_SIZE = 1

DEFAULT_CHANNELS = 1
FEATURES_PER_LEVEL = [64, 128, 256, 512]


class ConvolutionBlock(nn.Module):
    """
    This class implements the doubled convolutional layer,
    used in the U-net architecture, with batch normalization
    and ReLu activation function at the end.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__()
        self._block = nn.Sequential(
            # FIRST PART
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=CONV_KERNEL_SIZE,
                stride=CONV_STRIDE,
                padding=CONV_PADDING,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # SECOND PART
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=CONV_KERNEL_SIZE,
                stride=CONV_STRIDE,
                padding=CONV_PADDING,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, block_input):
        return self._block(block_input)


class Unet(nn.Module):
    """Implementation of the U-net architecture"""

    def __init__(
        self,
        in_channels=DEFAULT_CHANNELS,
        out_channels=DEFAULT_CHANNELS,
        features=FEATURES_PER_LEVEL,
    ):
        super(Unet, self).__init__()
        self._up_sampling = nn.ModuleList()
        self._down_sampling = nn.ModuleList()
        self._max_pool = nn.MaxPool2d(
            kernel_size=MAX_POOL_KERNEL_SIZE, stride=MAX_POOL_STRIDE
        )

        for feature in features:
            self._down_sampling.append(ConvolutionBlock(in_channels, feature))
            in_channels = feature

        joint_in_channels = features[-1]
        joint_out_channels = features[-1] * 2
        self._joint = ConvolutionBlock(
            in_channels=joint_in_channels, out_channels=joint_out_channels
        )

        for feature in reversed(features):
            up_steps = feature * 2
            self._up_sampling.append(
                nn.ConvTranspose2d(
                    in_channels=up_steps,
                    out_channels=feature,
                    kernel_size=CONV_TRANS_KERNEL_SIZE,
                    stride=CONV_TRANS_STRIDE,
                )
            )
            self._up_sampling.append(ConvolutionBlock(up_steps, feature))

        self._output_conv_layer = nn.Conv2d(
            features[0], out_channels, kernel_size=OUTPUT_CONV_KERNEL_SIZE
        )

    def forward(self, batch):

        skip_connections = []

        for down_sampling_block in self._down_sampling:
            batch = down_sampling_block(batch)
            skip_connections.append(batch)
            batch = self._max_pool(batch)

        batch = self._joint(batch)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self._up_sampling), 2):
            batch = self._up_sampling[idx](batch)
            skip_connection = skip_connections[idx // 2]

            batch = torch.cat((skip_connection, batch), dim=1)
            batch = self._up_sampling[idx + 1](batch)

        output = self._output_conv_layer(batch)

        return output
