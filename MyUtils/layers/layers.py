import torch 
import torch.nn as nn
import torch.nn.functional as F
from compressai.compressai.layers import*
from torch import Tensor
from typing import Tuple, Any

up_kwargs = {'mode': 'nearest'}

__all__=[
    "OctaveConv",
    "FirstOctaveConv",
    "LastOctaveConv",
    "octave3x3",
    "octave1x1",
    "octave5x5",
    "octave5x5s2",
    "OctaveAttn",
    "ResidualBottleneckBlock",
    "OctaveRBB",
    "conv",
    "deconv",
    "AttentionBlock",
    "CheckboardMaskedConv2d",
    "conv3x3",
    "conv1x1",
    "subpel_conv3x3"
]


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)

class CheckboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out



class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs=up_kwargs):
        super().__init__()
        self.weights = nn.parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels).cuda()
        
        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.in_channels = in_channels
        self.out_channels =out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):
        X_h,X_l = x
        
        X_h2l = self.h2g_pool(X_h)

        end_h_x = int(self.in_channels*(1-self.alpha_in))
        end_h_y = int(self.out_channels*(1-self.alpha_out))
        
        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], self.stride,
                        self.padding, self.dilation, self.groups)

        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], self.stride,
                        self.padding, self.dilation, self.groups)

        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], self.stride,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :,:], self.bias[0:end_h_y], self.stride,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.upsample(X_l2h, scale_factor=2, **self.up_kwargs)

        X_h = X_h2h + X_l2h
        X_l = X_l2l + X_h2l

        return X_h, X_l      
    

class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha_in=0.0, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = up_kwargs):
        super(FirstOctaveConv, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels).cuda()
        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):

        X_h2l = self.h2g_pool(x)
        X_h = x

        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0: end_h_x, :,:], self.bias[0:end_h_y], self.stride,
                        self.padding, self.dilation, self.groups)

        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], self.stride,
                        self.padding, self.dilation, self.groups)

        X_h = X_h2h
        X_l = X_h2l

        return X_h, X_l


class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.0, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = up_kwargs):
        super(LastOctaveConv, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels).cuda()
        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):
        X_h, X_l = x

        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[:end_h_y], self.stride,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :,:], self.bias[:end_h_y], self.stride,
                        self.padding, self.dilation, self.groups)
        X_l2h = F.upsample(X_l2h, scale_factor=2, **self.up_kwargs)

        X_h = X_h2h + X_l2h
        return X_h


def octave3x3(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return OctaveConv(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=1, padding=1, bias=True)

def octave1x1(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return OctaveConv(in_channels=in_ch, out_channels=out_ch, kernel_size=(1,1), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=1, padding=0, bias=True)

def octave5x5(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return OctaveConv(in_channels=in_ch, out_channels=out_ch, kernel_size=(5,5), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=1, padding=2, bias=True)

def octave5x5s2(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return OctaveConv(in_channels=in_ch, out_channels=out_ch, kernel_size=(5,5), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=2, padding=2, bias=True)


class OctaveAttn(nn.Module):
    """
    attention module from cheng2020, substitute all conv with octave conv
    input x:always tuple(x_h and x_l)
    """
    def __init__(self, N:int, alpha_in, alpha_out):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit"""
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    octave1x1(N,N//2, alpha_in=alpha_in, alpha_out=alpha_out),
                    nn.ReLU(inplace=True),
                    octave3x3(N//2, N//2, alpha_in=alpha_in, alpha_out=alpha_out),
                    nn.ReLU(inplace=True),
                    octave1x1(N//2, N,alpha_in=alpha_in, alpha_out=alpha_out)
                )
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x:Tuple):
                h_identity,l_identity = x
                h_out, l_out = self.conv(x)
                h_out += h_identity
                l_out += l_identity
                h_out = self.relu(h_out)
                l_out = self.relu(l_out)
                return h_out, l_out
        
        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            octave1x1(N,N,alpha_in=alpha_in, alpha_out=alpha_out)
        )

    def forward(self, x:Tuple) -> Tuple:
        h_identity, l_identity = x
        h_a, l_a = self.conv_a(x)
        h_b, l_b = self.conv_b(x)
        h_out, l_out = h_a*torch.sigmoid(h_b), l_a*torch.sigmoid(l_b)
        h_out, l_out = h_out + h_identity, l_out + l_identity
        return h_out, l_out


class ResidualBottleneckBlock(nn.Module):
    """
    original residualbottleneckblock in ELIC
        Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch//2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out   


class OctaveRBB(nn.Module):
    """
    Octave version for original ELIC rbb
    args:
    in_ch(int):number of input channels
    out_ch(int):number of output channels
    alpha_in(float)
    alpha_out(float)
    """
    def __init__(self, in_ch:int, alpha_in, alpha_out):
        super().__init__()
        self.conv1 = octave1x1(in_ch, in_ch//2, alpha_in=alpha_in, alpha_out=alpha_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = octave3x3(in_ch//2, in_ch//2, alpha_in=alpha_in, alpha_out=alpha_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = octave1x1(in_ch//2, in_ch, alpha_in=alpha_in, alpha_out=alpha_out)
    
    def forward(self, x:Tuple) -> Tuple:
        h_identity, l_identity = x
        h_out,l_out = self.conv1(x)
        h_out,l_out = self.relu(h_out), self.relu(l_out)
        h_out,l_out = self.conv2((h_out, l_out))
        h_out,l_out = self.relu2(h_out), self.relu2(l_out)
        h_out, l_out = self.conv3((h_out,l_out))

        h_out, l_out = h_out + h_identity, l_out + l_identity
        return h_out, l_out
    

#TODO:for Goctave activation, we use relu
class Goctave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super().__init__()
        self.weights = nn.parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels).cuda()

        self.in_channels = in_channels
        self.out_channels =out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):
        X_h,X_l = x

        end_h_x = int(self.in_channels*(1-self.alpha_in))
        end_h_y = int(self.out_channels*(1-self.alpha_out))
        
        X_h2h = F.relu(F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], self.stride,
                        self.padding, self.dilation, self.groups))

        X_l2l = F.relu(F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], self.stride,
                        self.padding, self.dilation, self.groups))

        X_h2l = F.relu(F.conv2d(X_h2h, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], self.stride*2,
                        self.padding, self.dilation, self.groups))
        #X_l2h = F.relu(F.conv_transpose2d(X_l2l, self.weights[]))
        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :,:], self.bias[0:end_h_y], self.stride,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.upsample(X_l2h, scale_factor=2, **self.up_kwargs)

        X_h = X_h2h + X_l2h
        X_l = X_l2l + X_h2l

        return X_h, X_l    