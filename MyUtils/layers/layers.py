import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Any
import time 
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms 
import random

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
    "subpel_conv3x3",
    "goctave3x3",
    "goctave1x1",
    "goctave5x5",
    "goctave5x5s2",
    "Goctave",
    "FirstGoctaveConv",
    "LastGoctaveConv",
    "GoctaveAttn",
    "GoctaveRBB",
    "toctave5x5s2",
    "TLastOctaveConv",
    "tgoctave3x3",
    "tgoctave1x1",
    "tgoctave5x5",
    "tgoctave5x5s2",
    "TLastGoctaveConv",
    "WTAtten",
    "ECA"
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


def conv1x1(in_ch: int, out_ch: int, stride: int = 1, group_num=1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, groups=group_num)


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


class ECA(nn.Module):
    def __init__(self, in_ch, k):
        super().__init__()
        self.k = k 
        self.C = in_ch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=k, padding=k//2, bias=False)
    
    def forward(self, x:Tensor):
        y = self.avg_pool(x)   #[N,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(-1,-2))  #[N,1,C]
        y = y.transpose(-1,-2).unsqueeze(-1)  #[N,C,1,1]
        return x*y.expand_as(x)


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs=up_kwargs):
        super().__init__()
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


class TOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs=up_kwargs):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(in_channels,out_channels,kernel_size[0], kernel_size[1]))
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
        self.out_channels =out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):
        X_h,X_l = x
        
        X_h2l = self.h2g_pool(X_h)

        end_h_x = int(self.in_channels*(1-self.alpha_in))
        end_h_y = int(self.out_channels*(1-self.alpha_out))
        
        X_h2h = F.conv_transpose2d(X_h, self.weights[0:end_h_x,0:end_h_y,:,:], self.bias[0:end_h_y], self.stride,
                        self.padding, self.dilation, self.groups)

        X_l2l = F.conv_transpose2d(X_l, self.weights[end_h_x:,end_h_y:,:,:], self.bias[end_h_y:], self.stride,
                        self.padding, self.dilation, self.groups)

        X_h2l = F.conv_transpose2d(X_h2l, self.weights[0:end_h_x,end_h_y:,:,:], self.bias[end_h_y:], self.stride,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.conv_transpose2d(X_l, self.weights[end_h_x:,0:end_h_y,:,:], self.bias[0:end_h_y], self.stride,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.upsample(X_l2h, scale_factor=2, **self.up_kwargs)

        X_h = X_h2h + X_l2h
        X_l = X_l2l + X_h2l

        return X_h, X_l 


class TLastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.0, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = up_kwargs):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size[0], kernel_size[1]))
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

        X_h2h = F.conv_transpose2d(X_h, self.weights[0:end_h_x, 0:end_h_y, :,:], self.bias[:end_h_y], self.stride,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.conv_transpose2d(X_l, self.weights[end_h_x:, 0:end_h_y, :,:], self.bias[:end_h_y], self.stride,
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

def toctave5x5s2(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return TOctaveConv(in_channels=in_ch, out_channels=out_ch, kernel_size=(5,5), alpha_in=alpha_in, alpha_out=alpha_out,
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
                self.octave1x1_1 = octave1x1(N,N//2, alpha_in=alpha_in, alpha_out=alpha_out)
                self.relu1 = nn.ReLU(inplace=True)
                self.octave3x3 = octave3x3(N//2, N//2, alpha_in=alpha_in, alpha_out=alpha_out)
                self.relu2 = nn.ReLU(inplace=True)
                self.octave1x1_2 = octave1x1(N//2, N,alpha_in=alpha_in, alpha_out=alpha_out)
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x:Tuple):
                h_identity,l_identity = x
                h_out, l_out = self.octave1x1_1(x)
                h_out, l_out = self.relu1(h_out), self.relu1(l_out)
                h_out, l_out = self.octave3x3((h_out, l_out))
                h_out, l_out = self.relu2(h_out), self.relu2(l_out)
                h_out, l_out = self.octave1x1_2((h_out, l_out))
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
#makesure alpha_in equals alpha_out
class Goctave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=True):
        super().__init__()
        self.weights_hh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out)),int(in_channels*(1-alpha_in)), kernel_size[0], kernel_size[1]))
        self.weights_hl = nn.Parameter(torch.FloatTensor(int(out_channels*alpha_out), int(out_channels*(1-alpha_out)), kernel_size[0], kernel_size[1]))
        self.weights_ll = nn.Parameter(torch.FloatTensor(int(out_channels*alpha_out), int(in_channels*alpha_in), kernel_size[0], kernel_size[1]))
        self.weights_lh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out)), int(out_channels*alpha_out),  kernel_size[0], kernel_size[1]))
        """
        self.gdn1 = GDN(in_channels=int(out_channels*(1-alpha_out)))
        self.gdn2 = GDN(in_channels=int(out_channels*alpha_out))
        self.gdn3 = GDN(in_channels=int(out_channels*alpha_out))
        self.gdn4 = GDN(in_channels=int(out_channels*(1-alpha_out)))
        """
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias_hh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out))))
            self.bias_hl = nn.Parameter(torch.FloatTensor(int(out_channels*alpha_out)))
            self.bias_ll = nn.Parameter(torch.FloatTensor(int(out_channels*alpha_out)))
            self.bias_lh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out))))
        else:
            self.bias_hh = torch.zeros(int(out_channels*(1-alpha_out))).cuda()
            self.bias_hl = torch.zeros(int(out_channels*alpha_out)).cuda()
            self.bias_ll = torch.zeros(int(out_channels*alpha_out)).cuda()
            self.bias_lh = torch.zeros(int(out_channels*(1-alpha_out))).cuda()

        self.in_channels = in_channels
        self.out_channels =out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out


    def forward(self, x):
        X_h,X_l = x
        X_h2h = F.relu((F.conv2d(X_h, self.weights_hh, self.bias_hh, self.stride, self.padding, self.dilation, self.groups)))
        X_h2l = F.relu((F.conv2d(X_h2h, self.weights_hl, self.bias_hl, 2, self.padding, self.dilation, self.groups)))
        X_l2l = F.relu((F.conv2d(X_l, self.weights_ll, self.bias_ll, self.stride, self.padding, self.dilation, self.groups)))
        X_l2h = F.relu((F.conv_transpose2d(X_l2l, self.weights_lh, self.bias_lh, 2, self.padding, self.dilation, self.groups)))
        X_h = X_h2h + X_l2h
        X_l = X_l2l + X_h2l
        return X_h, X_l      


class TGoctave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=True):
        super().__init__()
        self.weights_hh = nn.Parameter(torch.FloatTensor(int(in_channels*(1-alpha_in)),int(out_channels*(1-alpha_out)), kernel_size[0], kernel_size[1]))
        self.weights_hl = nn.Parameter(torch.FloatTensor(int(out_channels*alpha_out), int(out_channels*(1-alpha_out)),  kernel_size[0], kernel_size[1]))
        self.weights_ll = nn.Parameter(torch.FloatTensor(int(in_channels*alpha_in), int(out_channels*alpha_out), kernel_size[0], kernel_size[1]))
        self.weights_lh = nn.Parameter(torch.FloatTensor(int(out_channels*alpha_out),int(out_channels*(1-alpha_out)),  kernel_size[0], kernel_size[1]))
        """
        self.igdn1 = GDN(in_channels=int(in_channels*(1-alpha_in)), inverse=True)
        self.igdn2 = GDN(in_channels=int(out_channels*(1-alpha_out)), inverse=True)
        self.igdn3 = GDN(in_channels=int(in_channels*alpha_in), inverse=True)
        self.igdn4 = GDN(in_channels=int(out_channels*alpha_out), inverse=True)
        """
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias_hh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out))))
            self.bias_hl = nn.Parameter(torch.FloatTensor(int(out_channels*alpha_out)))
            self.bias_ll = nn.Parameter(torch.FloatTensor(int(out_channels*alpha_out)))
            self.bias_lh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out))))
        else:
            self.bias_hh = torch.zeros(int(out_channels*(1-alpha_out))).cuda()
            self.bias_hl = torch.zeros(int(out_channels*alpha_out)).cuda()
            self.bias_ll = torch.zeros(int(out_channels*alpha_out)).cuda()
            self.bias_lh = torch.zeros(int(out_channels*(1-alpha_out))).cuda()

        self.in_channels = in_channels
        self.out_channels =out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out


    def forward(self, x):
        X_h,X_l = x

        X_h2h = F.conv_transpose2d(F.relu(X_h), self.weights_hh, self.bias_hh, self.stride, self.padding, self.dilation, self.groups)
        X_h2l = F.conv2d(F.relu(X_h2h), self.weights_hl, self.bias_hl, 2, self.padding, self.dilation, self.groups)
        X_l2l = F.conv_transpose2d(F.relu(X_l), self.weights_ll, self.bias_ll, self.stride, self.padding, self.dilation, self.groups)
        X_l2h = F.conv_transpose2d(F.relu(X_l2l), self.weights_lh, self.bias_lh, 2, self.padding, self.dilation, self.groups)
        
        X_h = X_h2h + X_l2h
        X_l = X_l2l + X_h2l

        return X_h, X_l   


class FirstGoctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=True):
        super().__init__()
        self.weights_hh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out)),int(in_channels*(1-alpha_in)), kernel_size[0], kernel_size[1]))
        self.weights_hl = nn.Parameter(torch.FloatTensor(int(out_channels*alpha_out), int(out_channels*(1-alpha_out)), kernel_size[0], kernel_size[1]))
        """
        self.gdn1 = GDN(in_channels=int(out_channels*(1-alpha_out)))
        self.gdn2 = GDN(in_channels=int(out_channels*alpha_out))
        """
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias_hh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out))))
            self.bias_hl = nn.Parameter(torch.FloatTensor(int(out_channels*alpha_out)))
        else:
            self.bias_hh = torch.zeros(int(out_channels*(1-alpha_out))).cuda()
            self.bias_hl = torch.zeros(int(out_channels*alpha_out)).cuda()
        
        self.in_channels = in_channels
        self.out_channels =out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):
        X_h2h = F.relu(F.conv2d(x, self.weights_hh, self.bias_hh, self.stride, self.padding, self.dilation, self.groups))
        X_h2l = F.relu(F.conv2d(X_h2h, self.weights_hl, self.bias_hl, 2, self.padding, self.dilation, self.groups)) 
        X_h = X_h2h 
        X_l = X_h2l
        return X_h, X_l  


class LastGoctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super().__init__()
        self.weights_hh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out)),int(in_channels*(1-alpha_in)), kernel_size[0], kernel_size[1]))
        self.weights_ll = nn.Parameter(torch.FloatTensor(int(out_channels), int(in_channels*alpha_in), kernel_size[0], kernel_size[1]))
        self.weights_lh = nn.Parameter(torch.FloatTensor(int(out_channels), int(out_channels*(1-alpha_out)),kernel_size[0], kernel_size[1]))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias_hh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out))))
            self.bias_ll = nn.Parameter(torch.FloatTensor(int(out_channels)))
            self.bias_lh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out))))
        else:
            self.bias_hh = torch.zeros(int(out_channels*(1-alpha_out))).cuda()
            self.bias_ll = torch.zeros(int(out_channels)).cuda()
            self.bias_lh = torch.zeros(int(out_channels*(1-alpha_out))).cuda()

        self.in_channels = in_channels
        self.out_channels =out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out


    def forward(self, x):
        X_h,X_l = x
        
        X_h2h = F.relu(F.conv2d(X_h, self.weights_hh, self.bias_hh, self.stride, self.padding, self.dilation, self.groups))
        X_l2l = F.relu(F.conv2d(X_l, self.weights_ll, self.bias_ll, self.stride, self.padding, self.dilation, self.groups))
        X_l2h = F.relu(F.conv_transpose2d(X_l2l, self.weights_lh, self.bias_lh, 2, self.padding, self.dilation, self.groups)) 
        X_h = X_h2h + X_l2h
        return X_h


class TLastGoctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super().__init__()
        self.weights_hh = nn.Parameter(torch.FloatTensor(int(in_channels*(1-alpha_in)),int(out_channels*(1-alpha_out)), kernel_size[0], kernel_size[1]))
        self.weights_ll = nn.Parameter(torch.FloatTensor(int(in_channels*alpha_in),int(out_channels), kernel_size[0], kernel_size[1]))
        self.weights_lh = nn.Parameter(torch.FloatTensor(int(out_channels), int(out_channels*(1-alpha_out)), kernel_size[0], kernel_size[1]))
        """
        self.igdn1 = GDN(in_channels=int(in_channels*(1-alpha_in)), inverse=True)
        self.igdn2 = GDN(in_channels=int(in_channels*alpha_in), inverse=True)
        self.igdn3 = GDN(in_channels=int(out_channels), inverse=True)
        """
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias_hh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out))))
            self.bias_ll = nn.Parameter(torch.FloatTensor(int(out_channels)))
            self.bias_lh = nn.Parameter(torch.FloatTensor(int(out_channels*(1-alpha_out))))
        else:
            self.bias_hh = torch.zeros(int(out_channels*(1-alpha_out))).cuda()
            self.bias_ll = torch.zeros(int(out_channels)).cuda()
            self.bias_lh = torch.zeros(int(out_channels*(1-alpha_out))).cuda()

        self.in_channels = in_channels
        self.out_channels =out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out


    def forward(self, x):
        X_h,X_l = x
        X_h2h = F.conv_transpose2d(F.relu(X_h), self.weights_hh, self.bias_hh, self.stride, self.padding, self.dilation, self.groups)
        X_l2l = F.conv_transpose2d(F.relu(X_l), self.weights_ll, self.bias_ll, self.stride, self.padding, self.dilation, self.groups)
        X_l2h = F.conv_transpose2d(F.relu(X_l2l), self.weights_lh, self.bias_lh, 2, self.padding, self.dilation, self.groups) 
        X_h = X_h2h + X_l2h
        return X_h


def goctave3x3(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return Goctave(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=1, padding=1, bias=True)

def goctave1x1(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return Goctave(in_channels=in_ch, out_channels=out_ch, kernel_size=(1,1), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=1, padding=0, bias=True)

def goctave5x5(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return Goctave(in_channels=in_ch, out_channels=out_ch, kernel_size=(5,5), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=1, padding=2, bias=True)

def goctave5x5s2(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return Goctave(in_channels=in_ch, out_channels=out_ch, kernel_size=(5,5), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=2, padding=2, bias=True)

def tgoctave3x3(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return TGoctave(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=1, padding=1, bias=True)

def tgoctave1x1(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return TGoctave(in_channels=in_ch, out_channels=out_ch, kernel_size=(1,1), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=1, padding=0, bias=True)

def tgoctave5x5(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return TGoctave(in_channels=in_ch, out_channels=out_ch, kernel_size=(5,5), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=1, padding=2, bias=True)

def tgoctave5x5s2(in_ch, out_ch, alpha_in=0.5, alpha_out=0.5):
    return TGoctave(in_channels=in_ch, out_channels=out_ch, kernel_size=(5,5), alpha_in=alpha_in, alpha_out=alpha_out,
                      stride=2, padding=2, bias=True)

class GoctaveAttn(nn.Module):
    """
    attention module from cheng2020, substitute all conv with octave conv
    input x:always tuple(x_h and x_l)
    """
    def __init__(self, N:int, alpha_in=0.5, alpha_out=0.5):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit"""
            def __init__(self):
                super().__init__()
                self.goctave1x1_1 = goctave1x1(N,N//2, alpha_in=alpha_in, alpha_out=alpha_out)
                #self.relu1 = nn.ReLU(inplace=True)
                self.goctave3x3 = goctave3x3(N//2, N//2, alpha_in=alpha_in, alpha_out=alpha_out)
                #self.relu2 = nn.ReLU(inplace=True)
                self.goctave1x1_2 = goctave1x1(N//2, N,alpha_in=alpha_in, alpha_out=alpha_out)
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x:Tuple):
                h_identity,l_identity = x
                h_out, l_out = self.goctave1x1_1(x)
                #h_out, l_out  = self.relu1(h_out), self.relu1(l_out)
                h_out, l_out = self.goctave3x3((h_out, l_out))
                #h_out, l_out = self.relu2(h_out), self.relu2(l_out)
                h_out, l_out = self.goctave1x1_2((h_out, l_out))

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
            goctave1x1(N,N,alpha_in=alpha_in, alpha_out=alpha_out)
        )

    def forward(self, x:Tuple) -> Tuple:
        h_identity, l_identity = x
        h_a, l_a = self.conv_a(x)
        h_b, l_b = self.conv_b(x)
        h_out, l_out = h_a*torch.sigmoid(h_b), l_a*torch.sigmoid(l_b)
        h_out, l_out = (h_out + h_identity), (l_out + l_identity)
        return h_out, l_out


class GoctaveRBB(nn.Module):
    """
    Octave version for original ELIC rbb
    args:
    in_ch(int):number of input channels
    out_ch(int):number of output channels
    alpha_in(float)
    alpha_out(float)
    """
    def __init__(self, in_ch:int, alpha_in=0.5, alpha_out=0.5):
        super().__init__()
        self.conv1 = goctave1x1(in_ch, in_ch//2, alpha_in=alpha_in, alpha_out=alpha_out)
        #self.relu = nn.ReLU(inplace=True)
        self.conv2 = goctave3x3(in_ch//2, in_ch//2, alpha_in=alpha_in, alpha_out=alpha_out)
        #self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = goctave1x1(in_ch//2, in_ch, alpha_in=alpha_in, alpha_out=alpha_out)
    
    def forward(self, x:Tuple) -> Tuple:
        h_identity, l_identity = x
        h_out,l_out = self.conv1(x)
        #h_out,l_out = self.relu(h_out), self.relu(l_out)
        h_out,l_out = self.conv2((h_out, l_out))
        #h_out,l_out = self.relu2(h_out), self.relu2(l_out)
        h_out, l_out = self.conv3((h_out,l_out))

        h_out, l_out = (h_out + h_identity), (l_out + l_identity)
        return h_out, l_out


class TGoctaveRBB(nn.Module):
    """
    Octave version for original ELIC rbb
    args:
    in_ch(int):number of input channels
    out_ch(int):number of output channels
    alpha_in(float)
    alpha_out(float)
    """
    def __init__(self, in_ch:int, alpha_in=0.5, alpha_out=0.5):
        super().__init__()
        self.conv1 = goctave1x1(in_ch, in_ch//2, alpha_in=alpha_in, alpha_out=alpha_out)
        #self.relu = nn.ReLU(inplace=True)
        self.conv2 = tgoctave3x3(in_ch//2, in_ch//2, alpha_in=alpha_in, alpha_out=alpha_out)
        #self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = goctave1x1(in_ch//2, in_ch, alpha_in=alpha_in, alpha_out=alpha_out)
    
    def forward(self, x:Tuple) -> Tuple:
        h_identity, l_identity = x
        h_out,l_out = self.conv1(x)
        #h_out,l_out = self.relu(h_out), self.relu(l_out)
        h_out,l_out = self.conv2((h_out, l_out))
        #h_out,l_out = self.relu2(h_out), self.relu2(l_out)
        h_out, l_out = self.conv3((h_out,l_out))

        h_out, l_out = h_out + h_identity, l_out + l_identity
        return h_out, l_out

"""
class Ghost(nn.Module):
    def __init__(self, in_channels:int, kernel_size:int, s:int=2):
        
        in_channels: int, in_channel number
        s:int, expand rate, default 2
        
        super().__init__()
        self.linear = nn.Conv2d(in_channels, in_channels*(s-1), stride=1, padding=kernel_size//2,groups=)
"""
class DWTConv(nn.Module):
    def __init__(self):
        super().__init__()
        hh_weights = torch.tensor([[0.5,-0.5],
                      [-0.5,0.5]])
        hl_weights = torch.tensor([[0.5,0.5],
                      [-0.5,-0.5]])
        lh_weights = torch.tensor([[0.5,-0.5],
                      [0.5,-0.5]])
        ll_weights = torch.tensor([[0.5,0.5],
                      [0.5,0.5]])
        self.register_buffer("hh_weights", hh_weights)
        self.register_buffer("hl_weights", hl_weights)
        self.register_buffer("lh_weights", lh_weights)
        self.register_buffer("ll_weights", ll_weights)
        self.hh_weights = self.hh_weights.unsqueeze(0).unsqueeze(0)
        self.hl_weights = self.hl_weights.unsqueeze(0).unsqueeze(0)
        self.lh_weights = self.lh_weights.unsqueeze(0).unsqueeze(0)
        self.ll_weights = self.ll_weights.unsqueeze(0).unsqueeze(0)

    def forward(self, x:Tensor, inverse=False):
        B,C,H,W = x.shape
        x_hh = F.conv2d(x, self.hh_weights.expand([C,1,2,2]), None, stride=2, groups=C)
        x_hl = F.conv2d(x, self.hl_weights.expand([C,1,2,2]), None, stride=2, groups=C)
        x_lh = F.conv2d(x, self.lh_weights.expand([C,1,2,2]), None, stride=2, groups=C)
        x_ll = F.conv2d(x, self.ll_weights.expand([C,1,2,2]), None, stride=2, groups=C)
        return x_hh, x_hl, x_lh, x_ll 


class IDWTConv(nn.Module):
    def __init__(self):
        super().__init__()
        hh_weights = torch.tensor([[0.5,-0.5],
                      [-0.5,0.5]])
        hl_weights = torch.tensor([[0.5,0.5],
                      [-0.5,-0.5]])
        lh_weights = torch.tensor([[0.5,-0.5],
                      [0.5,-0.5]])
        ll_weights = torch.tensor([[0.5,0.5],
                      [0.5,0.5]])
        self.register_buffer("hh_weights", hh_weights)
        self.register_buffer("hl_weights", hl_weights)
        self.register_buffer("lh_weights", lh_weights)
        self.register_buffer("ll_weights", ll_weights)
        self.hh_weights = self.hh_weights.unsqueeze(0).unsqueeze(0)
        self.hl_weights = self.hl_weights.unsqueeze(0).unsqueeze(0)
        self.lh_weights = self.lh_weights.unsqueeze(0).unsqueeze(0)
        self.ll_weights = self.ll_weights.unsqueeze(0).unsqueeze(0)
    
    def forward(self, hh,hl,lh,ll):
        _,C,_,_ = hh.shape
        x_hh = F.conv_transpose2d(hh, self.hh_weights.expand([C,1,2,2]), bias=None, stride=2, groups=C)
        x_hl = F.conv_transpose2d(hl, self.hl_weights.expand([C,1,2,2]), bias=None, stride=2, groups=C)
        x_lh = F.conv_transpose2d(lh, self.lh_weights.expand([C,1,2,2]), bias=None, stride=2, groups=C)
        x_ll = F.conv_transpose2d(ll, self.ll_weights.expand([C,1,2,2]), bias=None, stride=2, groups=C)
        return x_hh + x_hl + x_lh + x_ll


class WTAtten(nn.Module):
    def __init__(self, N:int=192 ,depth:int=2):
        super().__init__()
        self.N = N
        self.depth = depth
        class ResidualUnit(nn.Module):
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
        
        self.conv_a = nn.Sequential(conv(in_channels=N, out_channels=N),ResidualUnit(), ResidualUnit(), ResidualUnit(), conv1x1(in_ch=N, out_ch=4*N))
            
        self.identity_conv = nn.Sequential(
                conv3x3(in_ch=N, out_ch=N, stride=1),
                nn.ReLU(inplace=True),
                conv3x3(in_ch=N, out_ch=N, stride=1),
            )
        
        self.DWT = DWTConv()
        self.iDWT = IDWTConv()

        self.hh_conv = nn.Sequential(
                conv3x3(in_ch=N, out_ch=N, stride=1),
                nn.ReLU(inplace=True),
                conv3x3(in_ch=N, out_ch=N)
            )        

        self.hl_conv = nn.Sequential(
                conv3x3(in_ch=N, out_ch=N, stride=1),
                nn.ReLU(inplace=True),
                conv3x3(in_ch=N, out_ch=N)
            )

        self.lh_conv =  nn.Sequential(
                conv3x3(in_ch=N, out_ch=N, stride=1),
                nn.ReLU(inplace=True),
                conv3x3(in_ch=N, out_ch=N)
            )

        self.ll_conv1 = nn.Sequential(
                conv(in_channels=N, out_channels=N, stride=1),
                nn.ReLU(inplace=True),
                conv(in_channels=N, out_channels=N, stride=1)
            )
        self.ll_conv2 = nn.Sequential(
                conv3x3(in_ch=N, out_ch=N, stride=1),
                nn.ReLU(inplace=True),
                conv3x3(in_ch=N, out_ch=N)
            )
        self.ll_fusion = conv1x1(in_ch=2*N, out_ch=N)


    def forward(self, x:Tensor):
        idetity_conv = self.identity_conv(x)
        attn_map = self.conv_a(x)
        hh_map, hl_map, lh_map, ll_map = torch.chunk(attn_map, 4, 1)
        hh, hl, lh, ll = self.DWT(x)
        #with skip connection
        hh_conv = self.hh_conv(hh) + hh
        hl_conv = self.hl_conv(hl) + hl
        lh_conv = self.lh_conv(lh) + lh
        ll_conv1 = self.ll_conv1(ll)
        ll_conv2 = self.ll_conv2(ll)
        ll_conv = self.ll_fusion(torch.concatenate([ll_conv1, ll_conv2], dim=1)) + ll
        #multiply with attention_map
        hh_attn = hh_map * hh_conv
        hl_attn = hl_map * hl_conv
        lh_attn = lh_map * lh_conv
        ll_attn = ll_map * ll_conv
        conv_result = self.iDWT(hh_attn, hl_attn, lh_attn, ll_attn)
        return conv_result + idetity_conv


def test():
    x = torch.rand((1,3,128,128))
    x = x.to("cuda")

    octaveconv1 = octave1x1(in_ch=12, out_ch=12).cuda()
    octaveconv2 = octave3x3(in_ch=12, out_ch=12).cuda()
    octaveconv3 = octave5x5(in_ch=12, out_ch=12).cuda()
    octavefirst1 = FirstOctaveConv(in_channels=3, out_channels=12, kernel_size=(3,3), padding=1).cuda()
    octavefirst2 = FirstOctaveConv(in_channels=3, out_channels=12, kernel_size=(5,5), stride=2, padding=2).cuda()
    octavelast1 = LastOctaveConv(in_channels=12, out_channels=6, kernel_size=(3,3), padding=1).cuda()
    octavelast2 = LastOctaveConv(in_channels=12, out_channels=6, kernel_size=(5,5), stride=2, padding=2).cuda()
    goctave1 = goctave1x1(in_ch=12, out_ch=12).cuda()
    goctave2 = goctave5x5s2(in_ch=12, out_ch=12).cuda()
    goctavefirst1 = FirstGoctaveConv(in_channels=3, out_channels=12, kernel_size=(3,3)).cuda()
    goctavefirst2 = FirstGoctaveConv(in_channels=3, out_channels=12, kernel_size=(5,5), stride=2, padding=2).cuda()
    goctavelast1 = LastGoctaveConv(in_channels=12, out_channels=6, kernel_size=(3,3)).cuda()
    goctavelast2 = LastGoctaveConv(in_channels=12, out_channels=6, kernel_size=(5,5), stride=2, padding=2).cuda()

    x_h,x_l = octavefirst1(x)
    x_h2, x_l2 = octavefirst2(x)

    out_octave1 = octaveconv1((x_h, x_l))
    out_octave2 = octaveconv3((x_h,x_l))
    out_octave_last1 = octavelast1(out_octave1)
    out_octave_last2 = octavelast2(out_octave2)
    
    x_gh, x_gl = goctavefirst1(x)
    x_hh2, x_gl2 = goctavefirst2(x)

    out_goctave1 = goctave1((x_gh, x_gl))
    out_goctave2 = goctave2((x_gh, x_gl))

    out_goctave_last1 = goctavelast1(out_goctave1)
    out_goctave_last2 = goctavelast2(out_goctave2)
    
    first_conv = conv3x3(in_ch=3, out_ch=12).cuda()
    residual_conv = ResidualBottleneckBlock(in_ch=12).cuda()
    residual_goctave = GoctaveRBB(in_ch=12).cuda()

    first_gconv = FirstGoctaveConv(in_channels=3, out_channels=12, kernel_size=(3,3), padding=1, bias=True).cuda()
    atten_conv = AttentionBlock(N=12).cuda()
    atten_goctave = GoctaveAttn(N=12).cuda()
    
    for i in range(10):
        conv_start = time.time()
        out_conv = first_conv(x)
        out_conv = residual_conv(out_conv)
        out_conv = residual_conv(out_conv)
        out_conv = atten_conv(out_conv)
        conv_time = time.time() - conv_start
        print("convtime:{:.4f}".format(conv_time))

    for i in range(10):
        gconv_start = time.time()
        gconv_out = first_gconv(x)
        gconv_out = residual_goctave(gconv_out)
        gconv_out = residual_goctave(gconv_out)
        gconv_out = atten_goctave(gconv_out)
        gconv_time = time.time() - gconv_start
        print("gconv_time:{:.4f}".format(gconv_time))

def test_wt():
    torch.manual_seed(8)
    random.seed(8)
    torch.cuda.manual_seed_all(8)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    img_path = "/mnt/data1/jingwengu/kodak1/kodim21.png"
    img = Image.open(img_path).convert("RGB")
    trans = transforms.ToTensor()
    img_tensor = trans(img).unsqueeze(0).to("cuda")
    dwt_conv = DWTConv().to("cuda")
    idwt_conv = IDWTConv().to("cuda")
    hh,hl,lh,ll = dwt_conv(img_tensor)
    hh_plt = hh[0,0,:,:].detach().to("cpu").numpy()
    hl_plt = hl[0,0,:,:].detach().to("cpu").numpy()
    lh_plt = lh[0,0,:,:].detach().to("cpu").numpy()
    ll_plt = ll[0,0,:,:].detach().to("cpu").numpy()
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(hh_plt)
    plt.title("hh")
    plt.subplot(2,2,2)
    plt.imshow(hl_plt)
    plt.title("hl")
    plt.subplot(2,2,3)
    plt.imshow(lh_plt)
    plt.title("lh")
    plt.subplot(2,2,4)
    plt.imshow(ll_plt)
    plt.title("ll")
    plt.savefig("test.png")
    plt.close()
    recon = idwt_conv(hh,hl,lh,ll)
    assert torch.isclose(recon, img_tensor,atol=1e-5).all()

    #test dwt-attn model
    features = torch.rand([1,192,128,128]).to("cuda")
    dwtattn = WTAtten().to("cuda")
    out = dwtattn(features)

    

if __name__=="__main__":
    test_wt()