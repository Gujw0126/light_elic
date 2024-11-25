import torch
import torch.nn as nn
from compressai.compressai.models import EntropyBottleneck, CompressionModel
from compressai.compressai.entropy_models import GaussianConditional
from MyUtils.layers import *
from torch import Tensor
from compressai.compressai.models.utils import update_registered_buffers
import math
import time
from typing import Optional
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
TINY = 1e-8

def standardized_cumulative(self, inputs: Tensor) -> Tensor:
    half = float(0.5)
    const = float(-(2**-0.5))
    # Using the complementary error function maximizes numerical precision.
    return half * torch.erfc(const * inputs)

def likelihood(
    inputs: Tensor, scales: Tensor) -> Tensor:
    half = float(0.5)
    values = inputs
    scales = torch.clamp(min=0.1)
    values = torch.abs(values)
    upper = standardized_cumulative((half - values) / scales)
    lower = standardized_cumulative((-half - values) / scales)
    likelihood = upper - lower
    return likelihood


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)
        

class SuperHyper(CompressionModel):
    """
    ELIC Model with super resulotion
    """
    def __init__(self, N=192, M=320):
        super().__init__(entropy_bottleneck_channels=192)
        self.N = int(N)
        self.M = int(M)
        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        self.g_a = nn.Sequential(
            conv(3, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, M),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N*3//2),
            nn.ReLU(inplace=True),
            conv3x3(N*3//2, 2*M),
        )

        self.quantizer = Quantizer()
        self.gaussian_conditional = GaussianConditional(None)
        self.tensor_scales = nn.Parameter(data=torch.zeros(3,dtype=torch.float32), requires_grad=True)

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod

    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    

    def get_gaussian_kernel(in_ch,out_ch,k,scale,dev):
        """
        get gaussian blurring kernel 
        in_ch: in_channels
        out_ch: out_channels,
        k:kernel size
        scale: scale  ,expand to[in_ch, out_ch, k, k]
        """
        scale = torch.clamp(min=0.05)
        gaussian_kernel = torch.zeros([in_ch, out_ch, k, k], device=dev)
        for row in range(k):
            gaussian_kernel[:,:,row,:] += torch.abs(row-k//2)
        
        for col in range(k):
            gaussian_kernel[:,:,:,col] += torch.abs(row-k//2)
        
        gaussian_kernel = likelihood(inputs=gaussian_kernel, scales=scale)
        gaussian_kernel_norm = gaussian_kernel/torch.sum(gaussian_kernel, dim=1, keepdim=True)
        return gaussian_kernel_norm



    def forward(self, x:Tensor, noisequant=False):  
        kernel_scales = self.tensor_scales.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        kernel_scales = kernel_scales.expand([1,3,5,5])
        gaussian_kernel = self.get_gaussian_kernel(in_ch=1, out_ch=3, k=5, scale=kernel_scales, dev=x.device)
        x_blurred = torch.nn.functional.conv2d(input=x, weight=gaussian_kernel, bias=None, stride=1, padding=2, groups=3)

        y = self.g_a(x_blurred)
        B, C, H, W = y.size() ## The shape of y to generate the mask

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = self.quantizer.quantize(z_tmp,"ste") + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, latent_scales, latent_means)
        if noisequant:
            y_hat = self.quantizer.quantize(y, quantize_type="ste")
        else:
            y_hat = self.quantizer.quantize(y-latent_means, quantize_type="ste")+latent_means
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "latent_norm":(y-latent_means)/(latent_scales+TINY),
        }
    

    def compress(self,x:Tensor):
        whole_time = time.time()
        y_enc_start = time.time()
        kernel_scales = self.tensor_scales.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        kernel_scales = kernel_scales.expand([1,3,5,5])
        gaussian_kernel = self.get_gaussian_kernel(in_ch=1, out_ch=3, k=5, scale=kernel_scales, dev=x.device)
        x_blurred = torch.nn.functional.conv2d(input=x, weight=gaussian_kernel, bias=None, stride=1, padding=2, groups=3)

        y = self.g_a(x_blurred)
        y_enc = time.time() - y_enc_start
        B, C, H, W = y.size()  ## The shape of y to generate the mask

        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start
        z_compress_time = time.time()
        z_strings = self.entropy_bottleneck.compress(z)
        z_compress_time = time.time() - z_compress_time
        
        z_offset = self.entropy_bottleneck._get_medians().detach()
        z_tmp = z - z_offset
        z_hat = torch.round(z_tmp) + z_offset

        #z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        y_compress_time = time.time()
        indexes = self.gaussian_conditional.build_indexes(latent_scales)
        y_strings = self.gaussian_conditional.compress(y, indexes=indexes, means=latent_means)
        y_compress_time = time.time()
        whole_time = time.time() - whole_time
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec,"whole":whole_time,
                "z_compress_time":z_compress_time, "y_compress_time":y_compress_time},
                "latent": y
                }


    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        whole_time = time.time()
        z_decompress_time = time.time()
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_decompress_time = time.time() -  z_decompress_time
        B, _, _, _ = z_hat.size()
        z_dec = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec

        y_strings = strings[0]
        indexes = self.gaussian_conditional.build_indexes(latent_scales)
        y_decompress_time = time.time()
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=latent_means)
        y_decompress_time = time.time() - y_decompress_time

        y_dec = time.time()
        x_hat = self.g_s(y_hat)
        y_dec = time.time() - y_dec
        whole_time = time.time() - whole_time
        return {"x_hat": x_hat, "time":{"z_dec":z_dec, "y_dec": y_dec, "whole":whole_time,
                                        "z_decompress_time":z_decompress_time,"y_decompress_time":y_decompress_time}}
    