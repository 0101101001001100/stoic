# CNNs for 3D data

import torch
import torch.nn as nn
from torchvision.models.video.resnet import Conv3DSimple, BasicBlock, Bottleneck, VideoResNet
from pytorchvideo.models import resnet, x3d
from pytorchvideo.layers.swish import Swish


def r50():
    '''
    Baseline model from pytorch video.
    '''
    return resnet.create_resnet(
        # model configs
        input_channel=1,
        model_depth=50,
        model_num_class=2,
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
        # stem configs
        stem_dim_out=64,
        stem_conv_kernel_size=(7, 7, 7),
        stem_conv_stride=(2, 2, 2),
        stem_pool_kernel_size=(3, 3, 3),
        stem_pool_stride = (2, 2, 2),
        # stage configs
        stage1_pool_kernel_size = (2, 2, 2),
        stage_conv_a_kernel_size=(
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
        ),
        stage_conv_b_kernel_size=(
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        stage_spatial_h_stride=(1, 2, 2, 2),
        stage_spatial_w_stride=(1, 2, 2, 2),
        stage_temporal_stride=(1, 2, 2, 2),
        # head configs
        head_pool_kernel_size=(7, 7, 7),
        head_output_size=(2)#,
        # head_activation=nn.Linear
    )


class ResNetStem(nn.Sequential):
    def __init__(self) -> None:
        """
        Conv-BatchNorm-ReLu-MaxPool stem modified from `torchvision.models.video.resnet.BasicStem`.
        """
        super(ResNetStem, self).__init__(
            nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            )


class R3D(VideoResNet):
    def __init__(self, 
        block, 
        conv_makers, 
        layers, 
        stem, 
        num_classes: int = 3, 
        dropout_rate: float = 0.5,
        zero_init_residual: bool = False
    ) -> None:
        super().__init__(block, conv_makers, layers, stem, num_classes, zero_init_residual)
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.drop_out(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def r18(pretrained=False, state_dict=None):
    '''
    3D ResNet with 18 layers for image classification.

    Args:
        'pretrained' (bool): if `True`, load weights adapted from TorchVision model pretrained on Kinetics-400.
    '''
    model = R3D(
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=ResNetStem,
        num_classes=3,
        dropout_rate=0.5
    )
    assert not (pretrained and state_dict)
    if pretrained:
        model.load_state_dict(torch.load('models/r18_k400_reshaped.pt', encoding='ascii'))
    if state_dict:
        model.load_state_dict(torch.load(state_dict, encoding='ascii'))
    return model


def rx3d():
    return x3d.create_x3d(
        # Input clip configs.
        input_channel = 1,
        input_clip_length = 256,
        input_crop_size = 256,
        # Model configs.
        model_num_class = 3,
        dropout_rate = 0.5,
        width_factor = 2.0,
        depth_factor = 2.0,
        activation = nn.ReLU,
        # Stem configs.
        stem_dim_in = 16,
        stem_conv_kernel_size = (5, 5, 5),
        stem_conv_stride = (2, 2, 2),
        # Stage configs.
        stage_conv_kernel_size = (
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        stage_spatial_stride = (2, 2, 2, 2),
        stage_temporal_stride = (2, 2, 2, 2),
        bottleneck = x3d.create_x3d_bottleneck_block,
        bottleneck_factor = 2.25,
        se_ratio = 0.0625,
        inner_act = Swish,
        # Head configs.
        head_dim_out = 2048,
        head_pool_act = nn.ReLU,
        head_bn_lin5_on = False,
        head_activation = nn.Softmax,
        head_output_with_global_average = True
    )