import torch
import torch.nn as nn
from collections.abc import Sequence
from monai.networks.blocks import Convolution, Upsample
from typing import Union
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep

__all__ = ["ModUnet", "Modunet", "modunet", "ModUNet"]

class ConvBlock(nn.Sequential):
    '''
    The Convolution Block maybe consist of two or three or more convolutions.
    If you want, you can freely modify the number of convolution by yourself.
    '''
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            kernel_size: Union[Sequence[int], int] = 3,
            dilation: Union[Sequence[int], int] = 1,
            dropout: Union[float, tuple] = 0.0,
    ):
        '''
        Arguments:
        :param spatial_dims: number of spatial dimensions.
        :param in_chns: number of input channels.
        :param out_chns: number of output channels.
        :param act: activation type and arguments. e.g. ('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}).
        :param norm: feature normalization type and arguments. e.g. ('instance', {'affine': True}).
        :param bias: whether to have a bias term in convolution blocks.
        :param kernel_size: size of the convolution kernel. Defaults to 3.
               Two Convolution for each layer. e.g. kernel_size=(3, 5) 1. used for 3*3, 2. used for 5*5
        :param dilation: dilation rate for convolutions. Defaults to 1.
                         It can be a sequence of int for different conv layers.
                         Two Convolution for each layer. e.g. dilation=(1, 2) 1. used for 1 and 2. used for 2
        :param dropout: dropout ratio. default to no dropout.
        by the way, act, norm, bias and dropout have default value in Convolution.
        '''
        super().__init__()

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        print(f"get_up_layer initialized with dilation={dilation} and kernel_size={kernel_size}")
        conv0 = Convolution(
            spatial_dims,
            in_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            dilation=dilation[0],
            kernel_size=kernel_size[0],
        )
        conv1 = Convolution(
            spatial_dims,
            out_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            dilation=dilation[1],
            kernel_size=kernel_size[1],
        )
        self.add_module('conv0', conv0)
        self.add_module('conV1', conv1)



class get_down_layer(nn.Sequential):
    '''performing the downsampling first, and then perform the convolution twice.'''

    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            kernel_size: Union[Sequence[int], int] = 3,
            dilation: Union[Sequence[int], int] = 1,
            dropout: Union[float, tuple] = 0.0,
    ):
        """
        :param spatial_dims: number of spatial dimensions.
        :param in_chns: number of input channels.
        :param out_chns: number of output channels.
        :param act: activation type and arguments. e.g. ('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}).
        :param norm: feature normalization type and arguments. e.g. ('instance', {'affine': True}).
        :param bias: whether to have a bias term in convolution blocks.
        :param kernel_size: size of the convolution kernel. Defaults to 3.
        :param dilation: dilation rate for convolutions. Defaults to 1.
                         It can be a sequence of int for different conv layers.
        :param dropout: dropout ratio. default to no dropout.
        """

        super().__init__()
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        print(f"get_up_layer initialized with dilation={dilation} and kernel_size={kernel_size}")
        max_pooling = Pool['MAX', spatial_dims](kernel_size=2)
        convs = ConvBlock(spatial_dims, in_chns, out_chns, act, norm, bias, kernel_size, dilation, dropout)
        self.add_module('max_pooling', max_pooling)
        self.add_module('convs', convs)


class get_up_layer(nn.Module):
    '''
    performing upsampling first, and then concatenation with the encoder feature map, lastly perform the conv twice.
    '''

    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            kernel_size: Union[Sequence[int], int] = 3,
            dilation: Union[Sequence[int], int] = 1,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = 'deconv',
            is_pad: bool = True,
    ):
        """
        :param spatial_dims: number of spatial dimensions.
        :param in_chns: number of input channels to be upsampled.
        :param cat_chns: number of channels from the encoder.
        :param out_chns: number of output channels.
        :param act: activation type and arguments.
        :param norm: feature normalization type and arguments.
        :param bias: whether to have a bias term in convolution blocks.
        :param kernel_size: size of convolution kernel in ConvBlocks.
        :param dilation: dilation rate for convolution in ConvBlocks
        :param dropout: dropout ratio. Defaults to no dropout.
        :param upsample: upsampling methods. Defaults to deconv.
        :param is_pad: whether to pad upsampling features to fit features from encoder. Defaults to True.
        """
        super().__init__()
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        print(f"get_up_layer initialized with dilation={dilation} and kernel_size={kernel_size}")
        up_chns = in_chns // 2
        self.upsample = Upsample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
        )
        self.convs = ConvBlock(
            spatial_dims,
            cat_chns + up_chns,
            out_chns,
            act,
            norm,
            bias,
            kernel_size,
            dilation,
            dropout,
        )
        self.is_pad = is_pad

    def forward(self, x: torch.Tensor, x_encoder: torch.Tensor):
        '''
        :param x: features will be upsampled.
        :param x_encoder: features from the encoder,
        :return: x_connect
        '''
        x_up = self.upsample(x)
        '''
        e.g. if image is 3D, dims is [H, W, D], sp=[0, 0, 0, 0, 0, 0].
             if depth of x_encoder != depth of x_up, but their height and width are in same, sp=[0, 1, 0, 0, 0, 0].
             means that we need to pad in the back of the depth direction.
             sp=[left_D, right_D, left_W, right_W, left_H, right_H]
        '''
        if self.is_pad:
            # if image size is odd, after upsampling we need to pad x_up.
            dims = len(x.shape) - 2  # 2 means the dims of batch_size and channel.
            sp = [0] * (dims * 2)
            for i in range(dims):
                if x_encoder.shape[-i - 1] != x_up.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_up = torch.nn.functional.pad(x_up, sp, 'replicate')
        x_connect = self.convs(torch.cat([x_up, x_encoder], dim=1))

        return x_connect


class ModUNet(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            features: Sequence[int],
            act: Union[str, tuple] = ('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}),
            norm: Union[str, tuple] = ('instance', {'affine': True}),
            bias: bool = True,
            kernel_size: Union[Sequence[int], int] = 3,
            dilation: Union[Sequence[int], int] = 1,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = 'deconv',
    ):
        """
        A modified UNet implementation with 2D/3D images.
        :param spatial_dims: number of spatial dimensions.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param features: number of features at each layer.
        :param act: activation type and arguments. Defaults to LeakyReLU.
        :param norm: feature normalization type and arguments. Defaults to instance norm.
        :param bias: whether to hace a bias term in convolution blocks. Defaults to True.
        :param kernel_size: size of convolution kernel in ConvBlocks.
        :param dilation: dilation rate for convolution in ConvBlocks.
        :param dropout: dropout ratio. Defaluts to no dropout.
        :param upsample: upsampling methods. Defaults to deconv
        """
        super().__init__()

        self.features = ensure_tuple_rep(features, len(features))

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        print(f"get_up_layer initialized with dilation={dilation} and kernel_size={kernel_size}")

        self.num_layers = len(features) - 1
        self.conv_0 = ConvBlock(
            spatial_dims,
            in_channels,
            self.features[0],
            act,
            norm,
            bias,
            kernel_size,
            dilation,
            dropout,
        )
        self.down_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            down_layer = get_down_layer(
                spatial_dims,
                self.features[i],
                self.features[i + 1],
                act,
                norm,
                bias,
                kernel_size,
                dilation,
                dropout,
            )
            self.down_layers.append(down_layer)
        self.up_layers = nn.ModuleList()
        for i in range(self.num_layers - 2, -1, -1):
            up_layer = get_up_layer(
                spatial_dims,
                self.features[i + 1],
                self.features[i],
                self.features[i],
                act,
                norm,
                bias,
                kernel_size,
                dilation,
                dropout,
                upsample,
            )
            self.up_layers.append(up_layer)
        self.final_conv = Conv['Conv', spatial_dims](self.features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        encoder_features = []
        x = self.conv_0(x)
        encoder_features.append(x)

        for down_layer in self.down_layers:
            x = down_layer(x)
            encoder_features.append(x)

        for i, up_layer in enumerate(self.up_layers):
            x = up_layer(x, encoder_features[-(i + 2)])

        x = self.final_conv(x)

        return x

ModUnet = Modunet = modunet = ModUNet




