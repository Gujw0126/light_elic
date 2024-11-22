import torch
import torch.nn as nn
from compressai.compressai.models import CompressionModel,Cheng2020Attention, ScaleHyperprior
from compressai.compressai.entropy_models import GaussianConditional, GaussianMixtureConditional, EntropyBottleneck
from MyUtils.layers import*
from compressai.compressai.ops import quantize_ste
from compressai.compressai.models.utils import update_registered_buffers
import math
import compressai.compressai
from torch import Tensor
import time
from ptflops import get_model_complexity_info
from torchsummary import summary
from thop import profile, clever_format
from typing import cast

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
        

class ELIC_original(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5, **kwargs):
        super().__init__(entropy_bottleneck_channels=192)
        self.N = int(N)
        self.M = int(M)
        self.num_slices = num_slices

        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        self.groups = [0, 16, 16, 32, 64, 192] #support depth
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

        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.groups[min(1, i) if i > 0 else 0] + self.groups[i if i > 1 else 0], 224, stride=1,
                     kernel_size=5),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.groups[i + 1]*2, stride=1, kernel_size=5),
            ) for i in range(1,  num_slices)
        ) ## from https://github.com/tensorflow/compression/blob/master/models/ms2020.py

        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
            self.groups[i+1], 2*self.groups[i+1], kernel_size=5, padding=2, stride=1
            ) for i in range(num_slices)
        )## from https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression/blob/main/version2/layers/CheckerboardContext.py

        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(640 + self.groups[i+1 if i > 0 else 0] * 2 + self.groups[
                        i + 1] * 2, 640),
                nn.ReLU(inplace=True),
                conv1x1(640, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[i + 1]*2),
            ) for i in range(num_slices)
        ) ##from checkboard "Checkerboard Context Model for Efficient Learned Image Compression"" gep网络参数

        self.quantizer = Quantizer()

        self.gaussian_conditional = GaussianConditional(None)  
    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, noisequant=False):
        y = self.g_a(x)
        B, C, H, W = y.size() ## The shape of y to generate the mask

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)

        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * i for i in self.groups[1:]], 1)
        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []
        y_means = []
        y_scales = []
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index-1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(x.device)
            means_hat_split = torch.zeros_like(y_anchor).to(x.device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]
            if noisequant:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor, "noise")
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor, "ste")
            else:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor


            y_anchor_quantilized[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized[:, :, 1::2, 0::2] = 0
            y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            # entropy estimation
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)
            y_means.append(means_hat_split)
            y_scales.append(scales_hat_split)
            y_non_anchor = non_anchor_split[slice_index]
            if noisequant:
                y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor, "noise")
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor, "ste")
            else:
                y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                          "ste") + means_non_anchor
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                          "ste") + means_non_anchor

            y_non_anchor_quantilized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized[:, :, 1::2, 1::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized + y_non_anchor_quantilized
            y_hat_slice_for_gs = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            y_hat_slices.append(y_hat_slice)
            ### ste for synthesis model
            y_hat_slices_for_gs.append(y_hat_slice_for_gs)
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_means_all = torch.cat(y_means, dim=1)
        y_scales_all =  torch.cat
        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "latent_norm":(y-latent_means)/(latent_scales+TINY)
        }

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

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        whole_time = time.time()
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start
        B, C, H, W = y.size()  ## The shape of y to generate the mask

        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start
        z_compress_time = time.time()
        z_strings = self.entropy_bottleneck.compress(z)
        z_compress_time = time.time() - z_compress_time

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        y_slices = torch.split(y, self.groups[1:], 1)

        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device), [2 * i for i in self.groups[1:]], 1)

        y_strings = []
        y_hat_slices = []
        y_compress_time = 0
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = y_slices[slice_index].clone()
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()

            y_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)

            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]
            compress_anchor = time.time()
            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = self.gaussian_conditional.compress(y_anchor_encode, indexes_anchor, means=means_anchor_encode)
            compress_anchor = time.time() - compress_anchor
            y_compress_time += compress_anchor

            anchor_quantized = self.quantizer.quantize(y_anchor_encode-means_anchor_encode, "ste") + means_anchor_encode
            #anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor_encode)
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]


            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            y_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)

            non_anchor = y_slices[slice_index].clone()
            y_non_anchor_encode[:, :, 0::2, :] = non_anchor[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = non_anchor[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]
            compress_non_anchor = time.time()
            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor_encode, indexes_non_anchor,
                                                                    means=means_non_anchor_encode)
            compress_non_anchor = time.time() - compress_non_anchor
            y_compress_time += compress_non_anchor
            non_anchor_quantized = self.quantizer.quantize(y_non_anchor_encode-means_non_anchor_encode, "ste") + means_non_anchor_encode
            #non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
            #                                                            means=means_non_anchor_encode)

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)

            y_strings.append([anchor_strings, non_anchor_strings])
        whole_time = time.time()-whole_time
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec, "y_compress_time":y_compress_time,
                         "z_compress_time":z_compress_time,"whole":whole_time}}


    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        # FIXME: we don't respect the default entropy coder and directly call thse
        # range ANS decoder
        whole_time = time.time()
        z_decompress_time = time.time()
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_decompress_time = time.time() - z_decompress_time

        B, _, _, _ = z_hat.size()
        z_dec = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_strings = strings[0]

        ctx_params_anchor = torch.zeros((B, self.M*2, z_hat.shape[2] * 4, z_hat.shape[3] * 4)).to(z_hat.device)
        ctx_params_anchor_split = torch.split(ctx_params_anchor, [2 * i for i in self.groups[1:]], 1)

        y_hat_slices = []
        y_decompress_time = 0
        for slice_index in range(len(self.groups) - 1):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()

            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(z_hat.device)

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            anchor_decompress = time.time()
            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = y_strings[slice_index][0]
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor,
                                                                    means=means_anchor_encode)
            anchor_decompress = time.time() - anchor_decompress
            y_decompress_time += anchor_decompress

            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)

            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]
            
            non_anchor_decompress = time.time()
            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = y_strings[slice_index][1]
            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)
            non_anchor_decompress = time.time() - non_anchor_decompress
            y_decompress_time += non_anchor_decompress

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)
        y_hat = torch.cat(y_hat_slices, dim=1)

        y_dec_start = time.time()
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        y_dec = time.time() - y_dec_start
        whole_time = time.time() - whole_time
        return {"x_hat": x_hat, "time":{"z_dec":z_dec,"y_dec": y_dec, "z_decompress_time":z_decompress_time,"y_decompress_time":y_decompress_time,"whole":whole_time}}


class ELICHyper(CompressionModel):
    """
    ELIC Model with only a hyperprior entropy model
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

    def forward(self, x:Tensor, noisequant=False):
        y = self.g_a(x)
        B, C, H, W = y.size() ## The shape of y to generate the mask

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset

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
        y = self.g_a(x)
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


class ELICExtract(CompressionModel):
    """
    ELIC Model with only a hyperprior entropy model
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
            conv1x1(M,M)         #extract layer
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

    def forward(self, x:Tensor, noisequant=False):
        y = self.g_a(x)
        B, C, H, W = y.size() ## The shape of y to generate the mask

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset

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
        y = self.g_a(x)
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
                "latent":y
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
    

#ELICHyper model with Octave convolution
class ELICOctave(CompressionModel):
    def __init__(self, N=192, M=320, alpha_in=0.5, alpha_out=0.5):
        super().__init__()
        self.entropy_bottleneck_h = EntropyBottleneck(channels=int(N*(1-alpha_out)))
        self.entropy_bottleneck_l = EntropyBottleneck(channels=int(N*alpha_out))
        self.N = int(N)
        self.M = int(M)
        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        class H_A(nn.Module):
            def __init__(self):
                super().__init__()
                self.octave1 = octave3x3(M,N, alpha_in=alpha_in, alpha_out=alpha_out)
                self.relu1 = nn.ReLU(inplace=True)
                self.octave2 = octave5x5s2(N,N, alpha_in=alpha_in, alpha_out=alpha_out)
                self.relu2 = nn.ReLU(inplace=True)
                self.octave3 = octave5x5s2(N,N, alpha_in=alpha_in, alpha_out=alpha_out)

            def forward(self, x):
                h_out, l_out = self.octave1(x)
                h_out, l_out = self.relu1(h_out), self.relu1(l_out)
                h_out, l_out = self.octave2((h_out, l_out))
                h_out, l_out = self.relu2(h_out), self.relu2(l_out)
                h_out, l_out = self.octave3((h_out, l_out))
                return h_out, l_out

        class H_S(nn.Module):
            def __init__(self):
                super().__init__()
                self.octave1 = toctave5x5s2(N,N, alpha_in=alpha_in, alpha_out=alpha_out)
                self.relu1 = nn.ReLU(inplace=True)
                self.octave2 = toctave5x5s2(N, N*3//2, alpha_in=alpha_in, alpha_out=alpha_out)
                self.relu2 = nn.ReLU(inplace=True)
                self.octave3 = octave3x3(N*3//2, 2*M, alpha_in=alpha_in, alpha_out=alpha_out)
            
            def forward(self, x):
                h_out, l_out = self.octave1(x)
                h_out, l_out = self.relu1(h_out), self.relu2(l_out)
                h_out, l_out = self.octave2((h_out, l_out))
                h_out, l_out = self.relu2(h_out), self.relu2(l_out)
                h_out, l_out = self.octave3((h_out, l_out))
                return h_out, l_out

        self.g_a = nn.Sequential(
            FirstOctaveConv(in_channels=3, out_channels=N, kernel_size=(5,5), stride=2,
                             padding=2,bias=True,alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            octave5x5s2(N,N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveAttn(N, alpha_in=alpha_in, alpha_out=alpha_out),
            octave5x5s2(N,N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            octave5x5s2(N,M, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveAttn(M, alpha_in=alpha_in, alpha_out=alpha_out)
        )

        self.g_s = nn.Sequential(
            OctaveAttn(M, alpha_in=alpha_in, alpha_out=alpha_out),
            toctave5x5s2(in_ch=M, out_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),    
            toctave5x5s2(in_ch=N, out_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveAttn(N,alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            toctave5x5s2(in_ch=N, out_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            OctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            TLastOctaveConv(in_channels=N,out_channels=3,kernel_size=(5,5), padding=2, stride=2, bias=True, alpha_in=alpha_in),
        )

        self.h_a = H_A()
         
        self.h_s = H_S()

        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.quantizer = Quantizer()
        #self.entropy_bottleneck = EntropyBottleneck()
        self.gaussian_conditional = GaussianConditional(None)
        self.shuffle = nn.PixelShuffle(2)
        self.unshuffle = nn.PixelUnshuffle(2)
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim()>1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                nn.init._calculate_fan_in_and_fan_out
            else:
                nn.init.uniform_(p, a=-0.1, b=0.1)  #1/sqrt(192*3*3)


    def forward(self, x, noisequant=False):
        y_h, y_l = self.g_a(x)
        z_h, z_l = self.h_a((y_h, y_l))
        """
        z_h_unshuffle = self.unshuffle(z_h)
        z_encode = torch.concatenate([z_h_unshuffle, z_l], dim=1)
        z_hat, z_likelihoods = self.entropy_bottleneck(z_encode)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z_encode - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset

        z_hat_h = self.shuffle(z_hat[:,0:int(4*self.N*(1-self.alpha_out)),:,:])
        z_hat_l = z_hat[:,int(4*self.N*(1-self.alpha_out)):,:,:]
        """
        z_h_hat, z_h_likelihoods = self.entropy_bottleneck_h(z_h)
        z_l_hat, z_l_likelihoods = self.entropy_bottleneck_l(z_l)
        if not noisequant:
            z_h_offset = self.entropy_bottleneck_h._get_medians()
            z_l_offset = self.entropy_bottleneck_l._get_medians()
            z_h_tmp = z_h - z_h_offset
            z_l_tmp = z_l - z_l_offset
            z_h_hat = quantize_ste(z_h_tmp) + z_h_offset
            z_l_hat = quantize_ste(z_l_tmp) + z_l_offset

        params_h, params_l = self.h_s((z_h_hat, z_l_hat))
        h_means, h_scales = torch.chunk(params_h,2,dim=1)
        l_means, l_scales = torch.chunk(params_l,2,dim=1)

        _, h_likelihoods = self.gaussian_conditional(y_h, h_scales, h_means)
        _, l_likelihoods = self.gaussian_conditional(y_l, l_scales, l_means)

        if noisequant:
            h_hat = self.quantizer.quantize(y_h, quantize_type="ste")
            l_hat = self.quantizer.quantize(y_l, quantize_type="ste")
        else:
            h_hat = self.quantizer.quantize(y_h - h_means, quantize_type="ste") + h_means
            l_hat =  self.quantizer.quantize(y_l - l_means, quantize_type="ste") + l_means

        x_hat = self.g_s((h_hat, l_hat))
        #y_likelihoods = torch.concatenate([self.unshuffle(h_likelihoods), l_likelihoods], dim=1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"h":h_likelihoods, "l":l_likelihoods, "z_h": z_h_likelihoods, "z_l":z_l_likelihoods},
            #"latent_norm":{"h_norm":(y_h-h_means)/(h_scales+TINY),"l_norm":(y_l-l_means)/(l_scales+TINY)}
        }
    

    def compress(self, x):
        """
        y_enc:g_a
        z_enc:h_a
        z_compress_time:encode z with rANS
        z_dec:h_s
        y_compress_time:encode y with rANS
        whole_time:compress function time
        """
        whole_time = time.time()
        y_enc_start = time.time()
        y_h, y_l = self.g_a(x)
        y_enc = time.time() - y_enc_start
        
        z_enc = time.time()
        z_h, z_l = self.h_a((y_h, y_l))
        #z_h_unshuffle = self.unshuffle(z_h)
        #z_encode = torch.concatenate([z_h_unshuffle, z_l], dim=1)
        z_enc = time.time() - z_enc

        z_compress_time = time.time()
        #z_strings = self.entropy_bottleneck.compress(z_encode)
        z_h_strings = self.entropy_bottleneck_h.compress(z_h)
        z_l_strings = self.entropy_bottleneck_l.compress(z_l)
        z_compress_time = time.time() - z_compress_time

        z_h_offset = self.entropy_bottleneck_h._get_medians().detach()
        z_tmp_h = z_h - z_h_offset
        z_h_hat = torch.round(z_tmp_h) + z_h_offset
        
        z_l_offset = self.entropy_bottleneck_l._get_medians().detach()
        z_tmp_l = z_l - z_l_offset
        z_l_hat = torch.round(z_tmp_l) + z_l_offset
        #z_hat_h = self.shuffle(z_hat[:,0:int(4*self.N*(1-self.alpha_out)),:,:])
        #z_hat_l = z_hat[:,int(4*self.N*(1-self.alpha_out)):,:,:]
        
        z_dec = time.time()
        params_h, params_l = self.h_s((z_h_hat, z_l_hat))
        z_dec = time.time() - z_dec

        h_means, h_scales = torch.chunk(params_h, 2, dim=1)
        l_means, l_scales = torch.chunk(params_l, 2, dim=1)
        
        y_compress_time = time.time()
        h_index = self.gaussian_conditional.build_indexes(h_scales)
        l_index = self.gaussian_conditional.build_indexes(l_scales)
        h_strings = self.gaussian_conditional.compress(inputs=y_h, indexes=h_index, means=h_means)
        l_strings = self.gaussian_conditional.compress(inputs=y_l, indexes=l_index, means=l_means)
        y_compress_time = time.time() - y_compress_time
        whole_time = time.time() - whole_time

        return {"strings": [h_strings, l_strings, z_h_strings,z_l_strings], "shape": [z_h.size()[-2:], z_l.size()[-2:]],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec, "params": 0,"whole":whole_time,"z_compress_time":z_compress_time,"y_compress_time":y_compress_time}}


    def decompress(self, strings, shape):
        #assert isinstance(strings, list) and len(strings) == 2
        whole_time = time.time()
        z_decompress_time = time.time()
        z_h_hat = self.entropy_bottleneck_h.decompress(strings[2], shape[0])
        z_l_hat = self.entropy_bottleneck_l.decompress(strings[3], shape[1])
        z_decompress_time = time.time() - z_decompress_time

        #z_hat_h = self.shuffle(z_hat[:,0:int(4*self.N*(1-self.alpha_out)),:,:])
        #z_hat_l = z_hat[:,int(4*self.N*(1-self.alpha_out)):,:,:]

        z_dec = time.time()
        params_h, params_l = self.h_s((z_h_hat, z_l_hat))
        z_dec = time.time() - z_dec
        
        h_means, h_scales = torch.chunk(params_h, 2, dim=1)
        l_means, l_scales = torch.chunk(params_l, 2, dim=1)

        y_decompress_time = time.time()
        h_index = self.gaussian_conditional.build_indexes(h_scales)
        l_index = self.gaussian_conditional.build_indexes(l_scales)
        h_hat = self.gaussian_conditional.decompress(strings[0], h_index, means=h_means)
        l_hat = self.gaussian_conditional.decompress(strings[1], l_index, means=l_means)
        y_decompress_time = time.time() - y_decompress_time

        y_dec = time.time()
        x_hat = self.g_s((h_hat, l_hat))
        y_dec = time.time() - y_dec
        params_time = 0
        whole_time = time.time() - whole_time
        return {"x_hat": x_hat, "time":{"z_dec":z_dec, "y_dec": y_dec, "z_decompress_time":z_decompress_time,
                                        "y_decompress_time":y_decompress_time,"params":params_time, "whole":whole_time}}
    
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
        for module in self.named_parameters():
            if isinstance(module, EntropyBottleneck):
                updated |= module.update()
        return updated
    
    def aux_loss(self) -> Tensor:
        loss = self.entropy_bottleneck_h.loss() + self.entropy_bottleneck_l.loss()
        return cast(Tensor, loss)

    

#ELICHyper model with GOctave convolution alpha_in必须等于alpha_out
class ELICGOctave(ELICOctave):
    def __init__(self, N=192, M=320, alpha_in=0.5, alpha_out=0.5):
        super().__init__(alpha_in=alpha_in, alpha_out=alpha_out)
        self.N = int(N)
        self.M = int(M)
        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        class H_A(nn.Module):
            def __init__(self):
                super().__init__()
                self.goctave1 = goctave3x3(M,N, alpha_in=alpha_in, alpha_out=alpha_out)
                #self.relu1 = nn.ReLU(inplace=True)
                self.goctave2 = goctave5x5s2(N,N, alpha_in=alpha_in, alpha_out=alpha_out)
                #self.relu2 = nn.ReLU(inplace=True)
                self.goctave3 = goctave5x5s2(N,N, alpha_in=alpha_in, alpha_out=alpha_out)

            def forward(self, x):
                h_out, l_out = self.goctave1(x)
                #h_out, l_out = self.relu1(h_out), self.relu1(l_out)
                h_out, l_out = self.goctave2((h_out, l_out))
                #h_out, l_out = self.relu2(h_out), self.relu2(l_out)
                h_out, l_out = self.goctave3((h_out, l_out))
                return h_out, l_out

        class H_S(nn.Module):
            def __init__(self):
                super().__init__()
                self.goctave1 = tgoctave5x5s2(N,N, alpha_in=alpha_in, alpha_out=alpha_out)
                #self.relu1 = nn.ReLU(inplace=True)
                self.goctave2 = tgoctave5x5s2(N, N*3//2, alpha_in=alpha_in, alpha_out=alpha_out)
                #self.relu2 = nn.ReLU(inplace=True)
                self.goctave3 = goctave3x3(N*3//2, 2*M, alpha_in=alpha_in, alpha_out=alpha_out)
            
            def forward(self, x):
                h_out, l_out = self.goctave1(x)
                #h_out, l_out = self.relu1(h_out), self.relu2(l_out)
                h_out, l_out = self.goctave2((h_out, l_out))
                #h_out, l_out = self.relu2(h_out), self.relu2(l_out)
                h_out, l_out = self.goctave3((h_out, l_out))
                return h_out, l_out


        self.g_a = nn.Sequential(
            FirstGoctaveConv(in_channels=3, out_channels=N, kernel_size=(5,5), stride=2,
                             padding=2,bias=True,alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            goctave5x5s2(N,N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveAttn(N, alpha_in=alpha_in, alpha_out=alpha_out),
            goctave5x5s2(N,N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            goctave5x5s2(N,M, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveAttn(M, alpha_in=alpha_in, alpha_out=alpha_out)
        )

        self.g_s = nn.Sequential(
            GoctaveAttn(M, alpha_in=alpha_in, alpha_out=alpha_out),
            tgoctave5x5s2(in_ch=M, out_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),    
            tgoctave5x5s2(in_ch=N, out_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveAttn(N,alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            tgoctave5x5s2(in_ch=N, out_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            GoctaveRBB(in_ch=N, alpha_in=alpha_in, alpha_out=alpha_out),
            TLastGoctaveConv(N,3,kernel_size=(5,5), padding=2, stride=2, bias=True, alpha_in=alpha_in),
        )
        
        self.h_a = H_A()
        self.h_s = H_S()

        self.quantizer = Quantizer()
        self.gaussian_conditional = GaussianConditional(None)
        self.shuffle = nn.PixelShuffle(2)
        self.unshuffle = nn.PixelUnshuffle(2)
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
    

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim()>1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                nn.init._calculate_fan_in_and_fan_out
            else:
                nn.init.uniform_(p, a=-0.1, b=0.1)  #1/sqrt(192*3*3)

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
        updated |= self.entropy_bottleneck_h.update()
        updated |= self.entropy_bottleneck_l.update()
        return updated


class ELICSkip(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5, quality=1,**kwargs):
        super().__init__(entropy_bottleneck_channels=192)
        self.N = int(N)
        self.M = int(M)
        self.num_slices = num_slices
        self.quality = quality
        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        self.groups = [0, 16, 16, 32, 64, 192] #support depth
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

        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.groups[min(1, i) if i > 0 else 0] + self.groups[i if i > 1 else 0], 224, stride=1,
                     kernel_size=5),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.groups[i + 1]*2, stride=1, kernel_size=5),
            ) for i in range(1,  num_slices)
        ) ## from https://github.com/tensorflow/compression/blob/master/models/ms2020.py

        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
            self.groups[i+1], 2*self.groups[i+1], kernel_size=5, padding=2, stride=1
            ) for i in range(num_slices)
        )## from https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression/blob/main/version2/layers/CheckerboardContext.py

        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(640 + self.groups[i+1 if i > 0 else 0] * 2 + self.groups[
                        i + 1] * 2, 640),
                nn.ReLU(inplace=True),
                conv1x1(640, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[i + 1]*2),
            ) for i in range(num_slices)
        ) ##from checkboard "Checkerboard Context Model for Efficient Learned Image Compression"" gep网络参数

        self.quantizer = Quantizer()

        self.gaussian_conditional = GaussianConditional(None)  
    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, noisequant=False):
        y = self.g_a(x)
        B, C, H, W = y.size() ## The shape of y to generate the mask

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)

        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * i for i in self.groups[1:]], 1)
        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index-1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            
            ### checkboard process 1
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(x.device)
            means_hat_split = torch.zeros_like(y_anchor).to(x.device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]
            if noisequant:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor, "noise")
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor, "ste")
            else:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor


            y_anchor_quantilized[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized[:, :, 1::2, 0::2] = 0
            y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)
            
            #noise finetune according to model quality
            if self.quality <4: 
                scales_non_anchor = torch.sigmoid(scales_non_anchor)*0.5
            elif self.quality==4:
                if slice_index>1:#skip slice index 2,3,4
                    scales_non_anchor = torch.sigmoid(scales_non_anchor)*0.5
            else:
                if slice_index>2:#skip slice index 3,4
                    scales_non_anchor = torch.sigmoid(scales_non_anchor)*0.5

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]

            # entropy estimation
            if noisequant:
                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)
            else:
                #anchor only
                _,y_slice_likelihood_all = self.gaussian_conditional(y_slice, scales_hat_split, means_hat_split)
                B,C,H,W = y_slice.shape
                y_slice_likelihood = torch.ones((B,C,H,W)).to(x.device)
                if self.quality < 4: 
                    y_slice_likelihood[:,:,0::2,0::2] = y_slice_likelihood_all[:,:,0::2,0::2]
                    y_slice_likelihood[:,:,1::2,1::2] = y_slice_likelihood_all[:,:,1::2,1::2]
                else:
                    if (self.quality == 4 and slice_index > 1) or (self.quality>4 and slice_index>2):
                        y_slice_likelihood[:,:,0::2,0::2] = y_slice_likelihood_all[:,:,0::2,0::2]
                        y_slice_likelihood[:,:,1::2,1::2] = y_slice_likelihood_all[:,:,1::2,1::2]
                    else:
                        y_slice_likelihood = y_slice_likelihood_all
                """
                B,C,H,W = y_slice.shape
                y_anchor_estimation = torch.zeros((B, C, H, W//2)).to(x.device)
                scales_anchor_estimation = torch.zeros_like(y_anchor_estimation)
                means_anchor_estimation = torch.zeros_like(y_anchor_estimation)
                y_anchor_estimation[:,:,0::2,:] = y_slice[:,:,0::2,0::2]
                y_anchor_estimation[:,:,1::2,:] = y_slice[:,:,1::2,1::2]
                scales_anchor_estimation[:,:,0::2,:] = scales_anchor[:,:,0::2,0::2]
                scales_anchor_estimation[:,:,1::2,:] = scales_anchor[:,:,1::2,1::2]
                means_anchor_estimation[:,:,0::2,:] = means_anchor[:,:,0::2,0::2]
                means_anchor_estimation[:,:,1::2,:] = means_anchor[:,:,1::2,1::2]
                _,y_slice_likelihood = self.gaussian_conditional(y_anchor_estimation, scales_anchor_estimation, means_anchor_estimation)
                """
            y_non_anchor = non_anchor_split[slice_index]

            #quantize
            if noisequant:
                y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor, "noise")
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor, "ste")
            else:

                """
                y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                          "ste") + means_non_anchor
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                          "ste") + means_non_anchor
                """
                if self.quality < 4:
                    y_non_anchor_quantilized = torch.zeros_like(means_anchor).to(x.device)
                    y_non_anchor_quantilized_for_gs = torch.zeros_like(means_anchor).to(x.device)
                    y_non_anchor_quantilized[:,:,0::2,1::2] = means_non_anchor[:,:,0::2,1::2]
                    y_non_anchor_quantilized[:,:,1::2,0::2] = means_non_anchor[:,:,1::2,0::2]
                    y_non_anchor_quantilized_for_gs[:,:,0::2,1::2] = means_non_anchor[:,:,0::2,1::2]
                    y_non_anchor_quantilized_for_gs[:,:,1::2,0::2] = means_non_anchor[:,:,1::2,0::2]
                else: 
                    if (self.quality == 4 and slice_index>1) or (self.quality>4 and slice_index>2):
                        y_non_anchor_quantilized = torch.zeros_like(means_anchor).to(x.device)
                        y_non_anchor_quantilized_for_gs = torch.zeros_like(means_anchor).to(x.device)
                        y_non_anchor_quantilized[:,:,0::2,1::2] = means_non_anchor[:,:,0::2,1::2]
                        y_non_anchor_quantilized[:,:,1::2,0::2] = means_non_anchor[:,:,1::2,0::2]
                        y_non_anchor_quantilized_for_gs[:,:,0::2,1::2] = means_non_anchor[:,:,0::2,1::2]
                        y_non_anchor_quantilized_for_gs[:,:,1::2,0::2] = means_non_anchor[:,:,1::2,0::2]     
                    else:
                        y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor - means_non_anchor, "ste") + means_non_anchor
                        y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,"ste") + means_non_anchor                              
            
            y_non_anchor_quantilized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized[:, :, 1::2, 1::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0
            
            y_hat_slice = y_anchor_quantilized + y_non_anchor_quantilized
            y_hat_slice_for_gs = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            y_hat_slices.append(y_hat_slice)
            ### ste for synthesis model
            y_hat_slices_for_gs.append(y_hat_slice_for_gs)
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

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

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        whole_time = time.time()
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start
        B, C, H, W = y.size()  ## The shape of y to generate the mask

        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start
        z_compress_time = time.time()
        z_strings = self.entropy_bottleneck.compress(z)
        z_compress_time = time.time() - z_compress_time

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        y_slices = torch.split(y, self.groups[1:], 1)

        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device), [2 * i for i in self.groups[1:]], 1)

        y_strings = []
        y_hat_slices = []
        y_compress_time = 0
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = y_slices[slice_index].clone()
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()

            y_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)

            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]
            compress_anchor = time.time()
            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = self.gaussian_conditional.compress(y_anchor_encode, indexes_anchor, means=means_anchor_encode)
            compress_anchor = time.time() - compress_anchor
            y_compress_time += compress_anchor

            anchor_quantized = self.quantizer.quantize(y_anchor_encode-means_anchor_encode, "ste") + means_anchor_encode
            #anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor_encode)
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)
            
            y_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)

            non_anchor = y_slices[slice_index].clone()
            y_non_anchor_encode[:, :, 0::2, :] = non_anchor[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = non_anchor[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            if self.quality < 4:
                y_non_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)
                y_non_anchor_decode[:,:,1::2,0::2] = means_non_anchor[:,:,1::2, 0::2]
                y_non_anchor_decode[:,:,0::2,1::2] = means_non_anchor[:,:,0::2, 1::2]
                non_anchor_strings = ""
            else: 
                if (self.quality == 4 and slice_index>1) or (self.quality > 4 and slice_index > 2):
                    y_non_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)
                    y_non_anchor_decode[:,:,1::2,0::2] = means_non_anchor[:,:,1::2, 0::2]
                    y_non_anchor_decode[:,:,0::2,1::2] = means_non_anchor[:,:,0::2, 1::2]
                    non_anchor_strings = ""
                else:
                    indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
                    non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor_encode, indexes_non_anchor,
                                                                    means=means_non_anchor_encode)
                    y_non_anchor_decode = torch.round(non_anchor-means_non_anchor) + means_non_anchor
                    y_non_anchor_decode[:,:,0::2,0::2] = 0
                    y_non_anchor_decode[:,:,1::2,1::2] = 0

            y_slice_hat = y_anchor_decode + y_non_anchor_decode
            y_hat_slices.append(y_slice_hat)
            y_strings.append([anchor_strings,non_anchor_strings])
            
        whole_time = time.time()-whole_time
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec, "y_compress_time":y_compress_time,
                         "z_compress_time":z_compress_time,"whole":whole_time}}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        # FIXME: we don't respect the default entropy coder and directly call thse
        # range ANS decoder
        whole_time = time.time()
        z_decompress_time = time.time()
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_decompress_time = time.time() - z_decompress_time

        B, _, _, _ = z_hat.size()
        z_dec = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_strings = strings[0]

        ctx_params_anchor = torch.zeros((B, self.M*2, z_hat.shape[2] * 4, z_hat.shape[3] * 4)).to(z_hat.device)
        ctx_params_anchor_split = torch.split(ctx_params_anchor, [2 * i for i in self.groups[1:]], 1)

        y_hat_slices = []
        y_decompress_time = 0
        for slice_index in range(len(self.groups) - 1):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()

            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(z_hat.device)

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            anchor_decompress = time.time()
            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = y_strings[slice_index][0]
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor,
                                                                    means=means_anchor_encode)
            anchor_decompress = time.time() - anchor_decompress
            y_decompress_time += anchor_decompress

            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            if self.quality < 4:
                y_non_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(z_hat.device)
                y_non_anchor_decode[:,:,0::2,1::2] = means_non_anchor[:,:,0::2,1::2]
                y_non_anchor_decode[:,:,1::2,0::2] = means_non_anchor[:,:,1::2,0::2]
            else:
                if (self.quality==4 and slice_index > 1) or (self.quality>4 and slice_index > 2):
                    y_non_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(z_hat.device)
                    y_non_anchor_decode[:,:,0::2,1::2] = means_non_anchor[:,:,0::2,1::2]
                    y_non_anchor_decode[:,:,1::2,0::2] = means_non_anchor[:,:,1::2,0::2]
                else:
                    means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
                    scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)

                    means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
                    means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
                    scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
                    scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]
                    
                    indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
                    non_anchor_strings = y_strings[slice_index][1]
                    non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,means=means_non_anchor_encode)
                    y_non_anchor_decode = torch.zeros_like(means_anchor)
                    y_non_anchor_decode[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
                    y_non_anchor_decode[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_decode
            y_hat_slices.append(y_slice_hat)
        y_hat = torch.cat(y_hat_slices, dim=1)

        y_dec_start = time.time()
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        y_dec = time.time() - y_dec_start
        whole_time = time.time() - whole_time
        return {"x_hat": x_hat, "time":{"z_dec":z_dec,"y_dec": y_dec, "z_decompress_time":z_decompress_time,"y_decompress_time":y_decompress_time,"whole":whole_time}}


class NARModel(CompressionModel):
    def __init__(self,N:int=192, M:int=256, b_num:int=3, num_slices=5):
        super().__init__(entropy_bottleneck_channels=N)
        self.groups = [0, 16, 16, 32, 64, 192] #support depth
        self.num_slices=5
        self.bottleneck_ch = N
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
            conv3x3(N*3//2, 2*N+32),
        )

        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv((2*N+self.groups[1] if i==1 else 2*N+self.groups[1]+self.groups[i]), 224, stride=1,
                     kernel_size=5),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.groups[i + 1], stride=1, kernel_size=5),
            ) for i in range(1,  num_slices)
        ) ## from https://github.com/tensorflow/compression/blob/master/models/ms2020.py

        self.gaussian_conditional = GaussianConditional(None)
        self.quantizer = Quantizer()
    

    def forward(self, x:Tensor, noisequant=False):
        y = self.g_a(x)
        y_slices = torch.split(y, self.groups[1:], dim=1)
        #first slice
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset =  self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = torch.round(z_tmp) + z_offset
        
        latent_params = self.h_s(z_hat)
        latent_means, latent_scales = torch.chunk(latent_params[:,0:32,:,:],chunks=2, dim=1)
        _, y_likelihoods = self.gaussian_conditional(y_slices[0], latent_means, latent_scales)
        y_slices_hat = []
        if noisequant:
            y_0_hat = self.quantizer.quantize(y_slices[0], "noise")
        else:
            y_0_hat = self.quantizer.quantize(y_slices[0] - latent_means, "ste") + latent_means
        
        y_slices_hat.append(y_0_hat)
        for slice_idx in range(1, self.num_slices):
            if slice_idx==1:
                ctx = torch.concatenate([latent_params[:,32:,:,:], y_slices_hat[0]], dim=1)
                y_slice_hat = self.cc_transforms[0](ctx)
            else:
                ctx = torch.concatenate([latent_params[:,32:,:,:], y_slices_hat[0], y_slices_hat[slice_idx-1]], dim=1)
                y_slice_hat = self.cc_transforms[slice_idx-1](ctx)
            y_slices_hat.append(y_slice_hat)
        
        y_hat = torch.concat(y_slices_hat,dim=1)
        x_hat = self.g_s(y_hat)
        return{
            "likelihoods":{"y":y_likelihoods, "z":z_likelihoods},
            "x_hat":x_hat,
            "y":y,
            "y_hat":y_hat
        }
    

    def compress(self, x:Tensor):
        y = self.g_a(x)
        y_slices = torch.split(y, self.groups[1:], dim=1)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = torch.round(z_tmp) + z_offset

        latent_params = self.h_s(z_hat)
        latent_means, latent_scales = torch.chunk(latent_params[:,0:32,:,:], chunks=2, dim=1)
        y_index = self.gaussian_conditional.build_indexes(latent_scales)
        y_strings = self.gaussian_conditional.compress(y_slices[0], y_index, means=latent_means)

        return{
            "strings":{"y":y_strings, "z":z_strings},
            "shape":z.size()[-2:]
        }


    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings=strings["z"], size=shape)
        latent_params = self.h_s(z_hat)
        latent_means, latent_scales = torch.chunk(latent_params[:,0:32,:,:], chunks=2, dim=1)
        y_index = self.gaussian_conditional.build_indexes(scales=latent_scales)
        y_0_hat = self.gaussian_conditional.decompress(strings=strings["y"], indexes=y_index, means=latent_means)
        y_slices_hat = []
        y_slices_hat.append(y_0_hat)
        for slice_idx in range(1, self.num_slices):
            if slice_idx==1:
                ctx = torch.concatenate([latent_params[:,32:,:,:], y_slices_hat[0]], dim=1)
                y_slice_hat = self.cc_transforms[0](ctx)
            else:
                ctx = torch.concatenate([latent_params[:,32:,:,:], y_slices_hat[0], y_slices_hat[slice_idx-1]], dim=1)
                y_slice_hat = self.cc_transforms[slice_idx-1](ctx)
            y_slices_hat.append(y_slice_hat)
        
        y_hat = torch.concat(y_slices_hat,dim=1)
        x_hat = self.g_s(y_hat)
        return{"x_hat":x_hat}

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

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net


class ELICDWT(ELICHyper):
    def __init__(self, N=192, M=320):
        super().__init__(192, 320)
        self.g_a = nn.Sequential(
            conv(3, N),
            WTAtten(N),
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
            AttentionBlock(M)
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
            WTAtten(N),
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


if __name__=="__main__":
    """
    net = ELIC_original().cuda()
    net.update()
    entropy_coder = compressai.compressai.available_entropy_coders()[0]
    compressai.compressai.set_entropy_coder(entropy_coder)
    out_compress = net.compress(x)
    out_decompress = net.decompress(out_compress["strings"], out_compress["shape"])
    flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    
    net_h = ELICHyper().cuda()
    net_h.update()
    out_compress = net_h.compress(x)
    out_decompress = net_h.decompress(out_compress["strings"], out_compress["shape"])
    flops, params = get_model_complexity_info(net_h, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    flops, params = profile(net, inputs=(x,))
    flops, params = clever_format([flops, params],"%.3f")
    print("FLOPs: %s" %(flops))
    print("params:%s" %(params))
    """
    """
    net = ELICSkip(quality=5).to("cuda")
    updated = net.update()
    print(updated)
    entropy_coder = compressai.compressai.available_entropy_coders()[0]
    x = torch.rand((1,3,256,256)).to("cuda")
    compress_time = time.time()
    forward_out = net(x)
    out_compress = net.compress(x)
    compress_time = time.time() - compress_time
    decompress_time = time.time()
    out_decompress = net.decompress(out_compress["strings"], out_compress["shape"])
    decompress_time = time.time() - decompress_time
    print("compress_time:{:.4f}".format(compress_time))
    print("decopmress_time:{:.4f}".format(decompress_time))
    #flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    #print('flops: ', flops, 'params: ', params)
    """
    net = Cheng2020Attention(N=192).to("cuda")
    net.update()
    x = torch.rand((1,3,256,256)).to("cuda")
    flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    """
    entropy_coder = compressai.compressai.available_entropy_coders()[0]
    compressai.compressai.set_entropy_coder(entropy_coder)
    with torch.no_grad():
        compress_time = time.time()
        forward_out = net(x)
        out_compress = net.compress(x)
        compress_time = time.time() - compress_time
        decompress_time = time.time()
        out_decompress = net.decompress(out_compress["strings"], out_compress["shape"])
        decompress_time = time.time() - decompress_time
        print("compress_time:{:.4f}".format(compress_time))
        print("decopmress_time:{:.4f}".format(decompress_time))
    """



    