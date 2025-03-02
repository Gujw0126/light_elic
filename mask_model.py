import torch
import torch.nn as nn
from compressai.compressai.models import CompressionModel
from compressai.compressai.entropy_models import GaussianConditional, GaussianMixtureConditional, EntropyBottleneck
from MyUtils.layers import*
from compressai.compressai.ops import quantize_ste
from compressai.compressai.models.utils import update_registered_buffers,remap_old_keys
import math
import compressai.compressai
from torch import Tensor
import time
from ptflops import get_model_complexity_info
from torchsummary import summary
from thop import profile, clever_format
from typing import cast
import pdb
from compressai.compressai.zoo import load_state_dict
from PIL import Image
from torchvision import transforms


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
TINY = 1e-8
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

def ste_sign(inputs, limit, mode="soft"):
    """
    mode: hard or soft
    """
    if mode=="soft":
        return torch.where(inputs>limit, 0.9, 0.1) - inputs.detach() + inputs
    else:
        return torch.where(inputs>limit, 1.0, 0.0).detach()
    

class MaskSkip(CompressionModel):
    """
    for a gain pretrained model, add a simple transform coding
    transform (y_hat-latent_means) for last important 160 channels,
    compress them with an entropy_bottelneck, then get the mask with its reconstruction and all latent information
    when inference, skip unimportant elements in y according to mask  
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

        #进入mask的是个二值化的残差，表示y中需要编码的位置
        self.mask_a = nn.Sequential(
            conv3x3(M//2, N//2),
            nn.ReLU(inplace=True),
            conv(N//2, N//2),
            nn.ReLU(inplace=True),
            conv(N//2, N//2),
        )

        self.mask_g =  nn.Sequential(
            deconv(N//2, N//2),
            nn.ReLU(inplace=True),
            deconv(N//2, M//2),
            nn.ReLU(inplace=True),
            conv3x3(M//2, M//2),
        )


        self.mask_bottleneck = EntropyBottleneck(channels=N//2)        
        self.quantizer = Quantizer()
        self.gaussian_conditional = GaussianConditional(None) 
        gain = torch.ones([1,M,1,1],dtype=torch.float32)
        self.gain = nn.Parameter(gain, requires_grad=True)
        inverse_gain = torch.ones([1,M,1,1], dtype=torch.float32)
        self.inverse_gain = nn.Parameter(inverse_gain, requires_grad=True)
        self.register_buffer(name="gain_indices", tensor=torch.zeros(M, dtype=torch.long), persistent=True) 
        """
        h_gain = torch.ones([1,N,1,1],dtype=torch.float32)
        h_inverse_gain = torch.ones([1,N,1,1],dtype=torch.float32)
        self.h_gain = nn.Parameter(h_gain, requires_grad=True)
        self.h_inverse_gain = nn.Parameter(h_inverse_gain, requires_grad=True)
        """

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

    def update_indices(self):
        _, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        self.gain_indices = sort_indices.squeeze(-1).squeeze(-1)
        return True
    

    def forward(self, x:Tensor, noisequant=False):
        y = self.g_a(x)
        B,C,H,W = y.shape
        y = self.gain.expand_as(y)*y

        z = self.h_a(y)
        #z = self.h_gain.expand_as(z)*z
        _, z_likelihoods = self.entropy_bottleneck(z)
        if noisequant:
            z_hat = self.quantizer.quantize(z, "noise")
        else:
            offset = self.entropy_bottleneck._get_medians()
            z_hat = self.quantizer.quantize(z - offset, "ste") + offset
        
        #z_hat = self.h_inverse_gain.expand_as(z_hat)*z_hat
        latent_means, latent_scales = self.h_s(z_hat).chunk(2,1)
        y_hat = self.quantizer.quantize(y - latent_means,"ste") + latent_means
        
        means_sort = torch.index_select(input=latent_means, dim=1, index=self.gain_indices)
        scales_sort = torch.index_select(input=latent_scales, dim=1, index=self.gain_indices)
        y_hat_sort = torch.index_select(input=y_hat, dim=1, index=self.gain_indices)
        y_sort = torch.index_select(input=y, dim=1, index=self.gain_indices)
        _,y_likelihoods = self.gaussian_conditional(y_sort[:,:self.M//2,:,:], scales_sort[:,:self.M//2,:,:], means_sort[:,:self.M//2, :,:])

        y_encode = y_hat_sort - means_sort
        y_encode_h = self.mask_a(y_encode[:,self.M//2:,:,:])
        _, mask_likelihoods = self.mask_bottleneck(y_encode_h)

        mask_offset = self.mask_bottleneck._get_medians()
        mask_hat = self.quantizer.quantize(y_encode_h - mask_offset,"ste") + mask_offset
        residual = self.mask_g(mask_hat)

        y_hat_sort_hat = torch.zeros_like(y)
        y_hat_sort_hat[:,:self.M//2,:,:] = y_hat_sort[:,:self.M//2,:,:]
        y_hat_sort_hat[:,self.M//2:,:,:] = means_sort[:,self.M//2:,:,:] + residual

        y_hat_hat = torch.zeros_like(y)
        y_hat_hat = y_hat_hat.index_copy(dim=1, index=self.gain_indices, source=y_hat_sort_hat)
        y_hat_hat = self.inverse_gain.expand_as(y)*y_hat_hat
        x_hat = self.g_s(y_hat_hat)  


        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods, "mask":mask_likelihoods},
            #"latent_norm":(y-latent_means)/(latent_scales+TINY),
        }


    def inference(self, x:Tensor, use_num):
        y = self.g_a(x)
        B, C, H, W = y.size() ## The shape of y to generate the mask
        y = self.gain.expand([B,C,H,W])*y

        z = self.h_a(y)
        z_offset = self.entropy_bottleneck._get_medians().detach()
        z_tmp = z - z_offset
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = torch.round(z_tmp) + z_offset
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        #calculate importance map and skip mask
        importance_map = self.importance_pred(z_hat)
        means_sort = torch.index_select(latent_means, dim=1, index=self.gain_indices)
        scales_sort = torch.index_select(latent_scales, dim=1, index=self.gain_indices)

        d_skip_means = means_sort[:,self.n_skip:self.n_skip+self.d_skip,:,:]
        d_skip_scales = scales_sort[:,self.n_skip:self.n_skip+self.d_skip,:,:]

        y_sort = torch.index_select(y, dim=1, index=self.gain_indices)
        skip_mask = self.skip_prediction(torch.concatenate([d_skip_means, d_skip_scales, importance_map], dim=1))
        skip_mask_tmp = ste_sign(torch.sigmoid(skip_mask),"soft")
        all_mask = torch.zeros_like(y)
        all_mask[:,self.n_skip:self.n_skip+self.d_skip,:,:] = skip_mask_tmp
        all_mask[:,:use_num,:,:] = 1.
        encode_index = torch.nonzero(all_mask.reshape(B*C*H*W)).squeeze(-1)
        skip_index = torch.nonzero((1.-all_mask).reshape(B*C*H*W)).squeeze(-1)

        #select encode elements
        encode_elements = torch.index_select(y_sort.reshape(B*C*H*W), dim=0, index=encode_index)
        encode_means = torch.index_select(means_sort.reshape(B*C*H*W), dim=0, index=encode_index)
        encode_scales = torch.index_select(scales_sort.reshape(B*C*H*W), dim=0, index=encode_index)
        _, encode_likelihoods = self.gaussian_conditional(encode_elements, encode_scales, encode_means)
        encode_hat = torch.round(encode_elements - encode_means) + encode_means
        #select skip elements
        skip_elements = torch.index_select(means_sort.reshape(B*C*H*W), dim=0, index=skip_index)
        y_hat_sort = torch.zeros(B*C*H*W,device=y.device)
        y_hat_sort = y_hat_sort.index_copy(dim=0, index=encode_index, source=encode_hat)
        y_hat_sort = y_hat_sort.index_copy(dim=0, index=skip_index, source=skip_elements)
        y_hat_sort = y_hat_sort.reshape([B,C,H,W])
        y_hat = torch.zeros_like(y)
        y_hat.index_copy_(dim=1, index=self.gain_indices, source=y_hat_sort)
        y_hat = self.inverse_gain.expand([B,C,H,W])*y_hat
        x_hat = self.g_s(y_hat.reshape([B,C,H,W]))
        return{
            "likelihoods":{"z":z_likelihoods, "y":encode_likelihoods},
            "x_hat":x_hat,
            "norm":skip_mask_tmp
        }



    def compress(self,x:Tensor,use_num):
        whole_time = time.time()
        y_enc = time.time()
        y = self.g_a(x)
        B, C, H, W = y.size() ## The shape of y to generate the mask
        y = self.gain.expand([B,C,H,W])*y
        y_enc = time.time() - y_enc

        z_enc = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc
        z_offset = self.entropy_bottleneck._get_medians().detach()
        z_tmp = z - z_offset
        z_compress_time = time.time()
        z_strings = self.entropy_bottleneck.compress(z)
        z_compress_time = time.time() - z_compress_time
        z_hat = torch.round(z_tmp) + z_offset
        z_dec = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec

        #calculate importance map and skip mask
        importance_map = self.importance_pred(z_hat)
        means_sort = torch.index_select(latent_means, dim=1, index=self.gain_indices)
        scales_sort = torch.index_select(latent_scales, dim=1, index=self.gain_indices)

        d_skip_means = means_sort[:,self.n_skip:self.n_skip+self.d_skip,:,:]
        d_skip_scales = scales_sort[:,self.n_skip:self.n_skip+self.d_skip,:,:]

        y_sort = torch.index_select(y, dim=1, index=self.gain_indices)
        skip_mask = self.skip_prediction(torch.concatenate([d_skip_means, d_skip_scales, importance_map], dim=1))
        skip_mask_tmp = ste_sign(torch.sigmoid(skip_mask),"hard")
        all_mask = torch.zeros_like(y)
        all_mask[:,self.n_skip:self.n_skip+self.d_skip,:,:] = skip_mask_tmp
        all_mask[:,:use_num,:,:] = 1.
        encode_index = torch.nonzero(all_mask.reshape(B*C*H*W)).squeeze(-1)

        #select encode elements
        encode_elements = torch.index_select(y_sort.reshape(B*C*H*W), dim=0, index=encode_index)
        encode_means = torch.index_select(means_sort.reshape(B*C*H*W), dim=0, index=encode_index)
        encode_scales = torch.index_select(scales_sort.reshape(B*C*H*W), dim=0, index=encode_index)
        y_index = self.gaussian_conditional.build_indexes(encode_scales.unsqueeze(0))
        y_compress_time = time.time()
        y_strings = self.gaussian_conditional.compress(encode_elements.unsqueeze(0), y_index, encode_means.unsqueeze(0))
        y_compress_time = time.time() - y_compress_time
        whole_time = time.time() - whole_time
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec,"whole":whole_time,
                "z_compress_time":z_compress_time, "y_compress_time":y_compress_time},
                "skip_encode":torch.round(y_sort[:,self.n_skip:self.n_skip+self.d_skip,:,:] - d_skip_means),
                "mask":skip_mask_tmp
                }


    def decompress(self, strings, shape, use_num):
        assert isinstance(strings, list) and len(strings) == 2
        whole_time = time.time()
        z_decompress_time = time.time()
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_decompress_time = time.time() -  z_decompress_time
        B, _, _, _ = z_hat.size()
        z_dec = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec
        
        get_mask_start = time.time()
        importance_map = self.importance_pred(z_hat)
        means_sort = torch.index_select(latent_means, dim=1, index=self.gain_indices)
        scales_sort = torch.index_select(latent_scales, dim=1, index=self.gain_indices)

        d_skip_means = means_sort[:,self.n_skip:self.n_skip+self.d_skip,:,:]
        d_skip_scales = scales_sort[:,self.n_skip:self.n_skip+self.d_skip,:,:]

        B,C,H,W = latent_means.shape
        skip_mask = self.skip_prediction(torch.concatenate([d_skip_means, d_skip_scales, importance_map], dim=1))
        skip_mask_tmp = ste_sign(torch.sigmoid(skip_mask),"hard")
        all_mask = torch.zeros_like(latent_means)
        all_mask[:,self.n_skip:self.n_skip+self.d_skip,:,:] = skip_mask_tmp
        all_mask[:,:use_num,:,:] = 1.
        non_zero_time = time.time()
        encode_index = torch.nonzero(all_mask.reshape(B*C*H*W)).squeeze(-1)
        skip_index = torch.nonzero((1.-all_mask).reshape(B*C*H*W)).squeeze(-1)
        print("non_zero_time:{:.5f}".format(time.time()-non_zero_time))
        #select encode elements
        encode_means = torch.index_select(means_sort.reshape(B*C*H*W), dim=0, index=encode_index)
        encode_scales = torch.index_select(scales_sort.reshape(B*C*H*W), dim=0, index=encode_index)
        print("get_mask_time:{:.5f}".format(time.time() - get_mask_start))
        y_strings = strings[0]
        y_index = self.gaussian_conditional.build_indexes(encode_scales.unsqueeze(0))
        y_decompress_time = time.time()
        encode_hat = self.gaussian_conditional.decompress(y_strings, y_index, means=encode_means.unsqueeze(0)).squeeze(0)
        y_decompress_time = time.time() - y_decompress_time

        #select skip elements
        skip_elements = torch.index_select(means_sort.reshape(B*C*H*W), dim=0, index=skip_index)
        y_hat_sort = torch.zeros(B*C*H*W,device=latent_means.device)
        y_hat_sort = y_hat_sort.index_copy(dim=0, index=encode_index, source=encode_hat)
        y_hat_sort = y_hat_sort.index_copy(dim=0, index=skip_index, source=skip_elements)
        y_hat_sort = y_hat_sort.reshape([B,C,H,W])
        y_hat = torch.zeros_like(latent_means)
        y_hat.index_copy_(dim=1, index=self.gain_indices, source=y_hat_sort)
        y_hat = self.inverse_gain.expand([B,C,H,W])*y_hat
        y_dec = time.time()
        x_hat = self.g_s(y_hat.reshape([B,C,H,W]))
        y_dec = time.time() - y_dec
        whole_time = time.time() - whole_time
        return {"x_hat": x_hat, "time":{"z_dec":z_dec, "y_dec": y_dec, "whole":whole_time,
                                        "z_decompress_time":z_decompress_time,"y_decompress_time":y_decompress_time}}
