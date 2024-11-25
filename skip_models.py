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


class SKModel(CompressionModel):
    def __init__(self,M=320, N=192, groups=[32,32,64,192]):
        super().__init__()
        self.pred_num = len(groups)
        self.groups = groups
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
            conv3x3(in_ch=self.groups[0], out_ch=self.groups[0]),
            nn.ReLU(inplace=True),
            conv(in_channels=self.groups[0], out_channels=self.groups[0]),
            nn.ReLU(inplace=True),
            conv(in_channels=self.groups[0], out_channels=self.groups[0])
        )
        
        self.h_s = nn.Sequential(
            deconv(in_channels=self.groups[0], out_channels=self.groups[0]),
            nn.ReLU(inplace=True),
            deconv(in_channels=self.groups[0], out_channels=self.groups[0]*3//2),
            nn.ReLU(inplace=True),
            conv3x3(in_ch=self.groups[0]*3//2, out_ch=self.groups[0]*2)
        )

        self.h_a_nets = nn.ModuleList(
            nn.Sequential(
                conv3x3(in_ch=self.groups[i], out_ch=self.groups[i]),
                nn.ReLU(inplace=True),
                conv(in_channels=self.groups[i], out_channels=self.groups[i]),
                nn.ReLU(inplace=True),
                conv(in_channels=self.groups[i], out_channels=self.groups[i])
            )
            for i in range(1,self.pred_num)
        )

        self.h_s_nets = nn.ModuleList(
            nn.Sequential(
                deconv(in_channels=self.groups[i], out_channels=self.groups[i]),
                nn.ReLU(inplace=True),
                deconv(in_channels=self.groups[i], out_channels=self.groups[i]),
                nn.ReLU(inplace=True),
                deconv(self.groups[i], self.groups[i],kernel_size=3,stride=1)   #TODO:改成反卷积
            )
            for i in range(1, self.pred_num)
        )

        self.pred_nets = nn.ModuleList(
            nn.Sequential(
                conv(in_channels=self.groups[0]+ self.groups[i]+(0 if i==1 else self.groups[i-1]), out_channels=self.groups[i]*2,stride=1),
                nn.ReLU(inplace=True),
                conv(in_channels=self.groups[i]*2, out_channels=self.groups[i]*3//2, stride=1),
                nn.ReLU(inplace=True),
                conv(in_channels=self.groups[i]*3//2, out_channels=self.groups[i], stride=1)
            )
            for i in range(1, self.pred_num)
        )
        

        self.entropy_bottlenecks = nn.ModuleList(
            EntropyBottleneck(channels=groups[i])
            for i in range(self.pred_num)
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.quantizer = Quantizer()

    def warm_up(self, x:Tensor):
        y = self.g_a(x)
        y_slices = torch.split(y,self.groups, dim=1)
        y_encode = y_slices[0]
        z_encode = self.h_a(y_encode)
        z_hat, z_likelihood = self.entropy_bottlenecks[0](z_encode)
        latent_means, latent_scales = torch.chunk(self.h_s(z_hat), 2, dim=1)
        _, y_likelihood = self.gaussian_conditional(y_encode, scales=latent_scales, means=latent_means)
        y_decode = self.quantizer.quantize(y_encode, "noise")
        y_slices_hat = []
        y_slices_hat.append(y_decode) 
        #extract information from other slices
        for i in range(1, self.pred_num):
            y_slice_aux = y_slices[i]
            z_aux = self.h_a_nets[i-1](y_slice_aux)
            y_slice_hat = self.h_s_nets[i-1](z_aux)
            y_slices_hat.append(y_slice_hat)
        
        #prediction
        y_preds = []
        y_preds.append(y_decode)
        for i in range(1, self.pred_num):
            if i==1:
                ctx = torch.concatenate([y_preds[0], y_slices_hat[i]],dim=1)
            else:
                ctx = torch.concatenate([y_preds[0], y_preds[i-1], y_slices_hat[i]],dim=1)
            y_pred = self.pred_nets[i-1](ctx)
            y_preds.append(y_pred)
        
        #get x_hat
        y_hat = torch.concat(y_preds,dim=1)
        x_hat = self.g_s(y_hat)
        return{
            "x_hat":x_hat,
            "likelihoods":{"y":y_likelihood,"z":z_likelihood},
            "preds":y_preds[1:],
            "y":y_slices[1:]
        }


    def forward(self, x:Tensor, noisequant=False):
        y = self.g_a(x)
        y_slices = torch.split(y, self.groups, dim=1)
        y_encode = y_slices[0]
        z_encode = self.h_a(y_encode)
        z_hat, z_likelihood = self.entropy_bottlenecks[0](z_encode)
        if not noisequant:
            z_offset = self.entropy_bottlenecks[0]._get_medians()
            z_tmp = z_encode - z_offset
            z_hat = torch.round(z_tmp) + z_offset
        
        latent_means, latent_scales = torch.chunk(self.h_s(z_hat), 2, dim=1)
        _, y_likelihood = self.gaussian_conditional(y_encode, scales=latent_scales, means=latent_means)
        if noisequant:
            y_decode = self.quantizer.quantize(y_encode, "noise")
        else:
            y_decode = self.quantizer.quantize(y_encode - latent_means,"ste") + latent_means

        aux_likelihoods = []
        y_slices_hat = []
        y_slices_hat.append(y_decode) 
        for i in range(1, self.pred_num):
            y_slice_aux = y_slices[i]
            z_aux = self.h_a_nets[i-1](y_slice_aux)
            z_aux_hat, aux_likelihood = self.entropy_bottlenecks[i](z_aux)
            aux_likelihoods.append(aux_likelihood)
            if not noisequant:
                offset = self.entropy_bottlenecks[i]._get_medians()
                tmp = z_aux - offset
                z_aux_hat = torch.round(tmp) + offset
            y_slice_hat = self.h_s_nets[i-1](z_aux_hat)
            y_slices_hat.append(y_slice_hat)
        
        #prediction
        y_preds = []
        y_preds.append(y_decode)
        for i in range(1, self.pred_num):
            if i==1:
                ctx = torch.concatenate([y_preds[0], y_slices_hat[i]],dim=1)
            else:
                ctx = torch.concatenate([y_preds[0], y_preds[i-1], y_slices_hat[i]],dim=1)
            y_pred = self.pred_nets[i-1](ctx)
            y_preds.append(y_pred)
        
        #get x_hat
        y_hat = torch.concat(y_preds,dim=1)
        x_hat = self.g_s(y_hat)
        slice_likelihood = torch.concat(aux_likelihoods,dim=1)

        return{
            "x_hat":x_hat,
            "likelihoods":{"z":z_likelihood, "y":y_likelihood, "slice":slice_likelihood},
            "preds":y_preds[1:],
            "y":y_slices[1:]
        }
    

    def compress(self, x:Tensor):
        y = self.g_a(x)
        y_slices = torch.split(y, self.groups, dim=1)
        y_encode = y_slices[0]
        z = self.h_a(y_encode)
        z_strings = self.entropy_bottlenecks[0].compress(z)
        
        z_offset = self.entropy_bottlenecks[0]._get_medians()
        z_tmp = z-z_offset
        z_hat = torch.round(z_tmp) + z_offset
        
        #z_hat = self.entropy_bottlenecks[0].decompress(z_strings,z.shape[-2:])
        latent_means, latent_scales = torch.chunk(self.h_s(z_hat), chunks=2, dim=1)
        y_index = self.gaussian_conditional.build_indexes(latent_scales)
        y_strings = self.gaussian_conditional.compress(y_encode, indexes=y_index, means=latent_means)
        #encode other slices
        aux_strings = []
        z_auxs = []
        for i in range(1, self.pred_num):
            y_slice_aux = y_slices[i]
            z_aux = self.h_a_nets[i-1](y_slice_aux)
            z_auxs.append(z_aux)
            aux_string = self.entropy_bottlenecks[i].compress(z_aux)
            aux_strings.append(aux_string)
        z_aux = torch.concat(z_auxs,dim=1)
        return{
            "strings":{"z":z_strings, "y":y_strings,"slice":aux_strings},
            "shape":z.size()[-2:],
            "latent":{"y":y,"z":z, "z_aux":z_aux}
        }
    

    def decompress(self, strings, shape):
        z_strings = strings["z"]
        z_hat = self.entropy_bottlenecks[0].decompress(z_strings,shape)
        latent_means, latent_scales, = torch.chunk(self.h_s(z_hat), 2, dim=1)
        y_index = self.gaussian_conditional.build_indexes(latent_scales)
        y_decode = self.gaussian_conditional.decompress(strings["y"], indexes=y_index, means=latent_means)
        
        #decode aux
        y_slices_hat = []
        y_slices_hat.append(y_decode)
        for i in range(1,self.pred_num):
            slice_string = strings["slice"][i-1]
            aux_decode = self.entropy_bottlenecks[i].decompress(slice_string, shape)
            y_slice_hat = self.h_s_nets[i-1](aux_decode)
            y_slices_hat.append(y_slice_hat)
        
        #pred
        y_preds = []
        y_preds.append(y_decode)
        for i in range(1, self.pred_num):
            if i==1:
                ctx = torch.concatenate([y_preds[0], y_slices_hat[i]],dim=1)
            else:
                ctx = torch.concatenate([y_preds[0], y_preds[i-1], y_slices_hat[i]],dim=1)
            y_pred = self.pred_nets[i-1](ctx)
            y_preds.append(y_pred)
        
        #get x_hat
        y_hat = torch.concat(y_preds,dim=1)
        x_hat = self.g_s(y_hat)
        return{
            "x_hat":x_hat
        }
    

    def load_state_dict(self, state_dict):
        for i in range(self.pred_num):
            update_registered_buffers(
                self.entropy_bottlenecks[i],
                "entropy_bottlenecks.{}".format(i),
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )
            state_dict = remap_old_keys("entropy_bottlenecks.{}".format(i), state_dict)

        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        nn.Module.load_state_dict(self,state_dict)

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
        for i in range(self.pred_num):
            updated |= self.entropy_bottlenecks[i].update()
        return updated
    

    def aux_loss(self) -> Tensor:
        loss = sum(self.entropy_bottlenecks[i].loss() for i in range(self.pred_num))
        return cast(Tensor, loss)


#先尝试以下两个分段(均分)的channel-ar的模型
#隔一个取一个 2024、11、22
class AnnelModel(CompressionModel):
    def __init__(self, M=320, N=192):
        super().__init__()
        self.M = M
        self.N = N
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
            conv3x3(in_ch=M, out_ch=N),
            nn.ReLU(inplace=True),
            conv(in_channels=N, out_channels=N),
            nn.ReLU(inplace=True),
            conv(in_channels=N, out_channels=N)
        )

        self.h_s = nn.Sequential(
            deconv(in_channels=N, out_channels=N),
            nn.ReLU(inplace=True),
            deconv(in_channels=N, out_channels=N*3//2),
            nn.ReLU(inplace=True),
            conv3x3(in_ch=N*3//2, out_ch=2*M)
        )

        self.cc_transform = nn.Sequential(
            conv(in_channels=M//2,out_channels=M//2*3, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            conv(in_channels=M//2*3, out_channels=M//2*3, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            conv(in_channels=M//2*3, out_channels=M, kernel_size=5, stride=1)
        )

        self.ParamAggregation = nn.Sequential(
            conv1x1(in_ch=3*M, out_ch=M*2),
            nn.ReLU(inplace=True),
            conv1x1(in_ch=M*2, out_ch=M*2),
            nn.ReLU(inplace=True),
            conv1x1(in_ch=M*2, out_ch=M)
        )

        self.prediction = nn.Sequential(  #use y2_means and y2_scales and y1_hat to predict latent residual
            conv3x3(in_ch=M*3//2, out_ch=M),
            nn.ReLU(inplace=True),
            conv3x3(in_ch=M, out_ch=M),
            nn.ReLU(inplace=True),
            conv3x3(in_ch=M, out_ch=M//2)
        )
    
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)
        self.quantizer = Quantizer()
    
    
    def soft(self, x, tmp=1.0):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihood = self.entropy_bottleneck(z)
        latent = self.h_s(z_hat)
        y1, y2 = torch.chunk(y, 2, 1)
        #y1_likelihood
        cc_ctx = torch.zeros_like(y)
        ctx = torch.concatenate([latent, cc_ctx],dim=1)
        y1_means, y1_scales = torch.chunk(self.ParamAggregation(ctx), chunks=2, dim=1)
        _, y1_likelihood = self.gaussian_conditional(y1, y1_scales, y1_means)
        y1_hat = self.quantizer.quantize(y1,"noise")
        #y2_likelihood
        cc_ctx = self.cc_transform(y1_hat)
        ctx = torch.concatenate([latent, cc_ctx],dim=1)
        y2_means, y2_scales = torch.chunk(self.ParamAggregation(ctx), chunks=2, dim=1)
        y2_scales = torch.sigmoid(y2_scales)*tmp
        _, y2_likelihood = self.gaussian_conditional(y2, y2_scales, y2_means)
        y2_hat = self.quantizer.quantize(y2, "noise")
        #latent residual prediction
        res = self.prediction(torch.concatenate([y1_hat, y2_means, y2_scales], dim=1))
        y2_hat = y2_hat + res
        y_hat = torch.concatenate([y1_hat, y2_hat],dim=1)
        y_likelihood = torch.concatenate([y1_likelihood, y2_likelihood],dim=1)
        y_means = torch.concatenate([y1_means, y2_means], dim=1)
        y_scales = torch.concatenate([y1_scales, y2_scales], dim=1)
        x_hat = self.g_s(y_hat)
        return{
            "x_hat":x_hat,
            "likelihoods":{"y":y_likelihood,"z":z_likelihood},
            "y_norm":(y - y_means)/(y_scales+TINY),
            "latent":y
        }
    
    def forward(self, x, tmp=1.0):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihood = self.entropy_bottleneck(z)
        latent = self.h_s(z_hat)
        y1, y2 = torch.chunk(y, 2, 1)
        #y1_likelihood
        cc_ctx = torch.zeros_like(y)
        ctx = torch.concatenate([latent, cc_ctx],dim=1)
        y1_means, y1_scales = torch.chunk(self.ParamAggregation(ctx), chunks=2, dim=1)
        _, y1_likelihood = self.gaussian_conditional(y1, y1_scales, y1_means)
        y1_hat = self.quantizer.quantize(y1,"noise")
        #y2_likelihood
        cc_ctx = self.cc_transform(y1_hat)
        ctx = torch.concatenate([latent, cc_ctx],dim=1)
        y2_means, y2_scales = torch.chunk(self.ParamAggregation(ctx), chunks=2, dim=1)
        y2_scales = torch.sigmoid(y2_scales)*tmp
        _, y2_likelihood = self.gaussian_conditional(y2, y2_scales, y2_means)
        y2_hat = self.quantizer.quantize(y2, "noise")
        #latent residual prediction
        res = self.prediction(torch.concatenate([y1_hat, y2_means, y2_scales], dim=1))
        y2_hat = y2_hat + res
        y_hat = torch.concatenate([y1_hat, y2_hat],dim=1)
        y_likelihood = torch.concatenate([y1_likelihood, y2_likelihood],dim=1)
        x_hat = self.g_s(y_hat)
        return{
            "x_hat":x_hat,
            "likelihoods":{"y":y_likelihood,"z":z_likelihood}
        }
    
    def hard(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihood = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = self.quantizer.quantize(z_tmp,"ste") + z_offset
        latent = self.h_s(z_hat)
        y1,y2 = torch.chunk(y, 2, 1)
        #y1_likelihood
        cc_ctx = torch.zeros_like(y)
        ctx = torch.concatenate([latent, cc_ctx], dim=1)
        y1_means, y1_scales = torch.chunk(self.ParamAggregation(ctx), chunks=2, dim=1)
        _, y1_likelihood =self.gaussian_conditional(y1, y1_scales, y1_means)
        y1_hat = self.quantizer.quantize(y1-y1_means,"ste") + y1_means
        #y2_likelihood
        cc_ctx = self.cc_transform(y1_hat)
        ctx = torch.concatenate([latent, cc_ctx],dim=1)
        y2_means, y2_scales = torch.chunk(self.ParamAggregation(ctx),2,1)
        #latent residual prediction
        res = self.prediction(torch.concatenate([y1_hat, y2_means, y2_scales],dim=1))
        y2_hat = y2_means + res
        y_likelihood = y1_likelihood
        y_hat = torch.concatenate([y1_hat, y2_hat],dim=1)
        x_hat = self.g_s(y_hat)
        y_means = torch.concatenate([y1_means, y2_means], dim=1)
        y_scales = torch.concatenate([y1_scales, y2_scales], dim=1)
        return{
            "x_hat":x_hat, 
            "likelihoods":{"y":y_likelihood, "z":z_likelihood},
            "y_norm":(y  - y_means)/(y_scales+TINY)
        }
    
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_string = self.entropy_bottleneck.compress(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = torch.round(z_tmp) + z_offset
        latent = self.h_s(z_hat)
        y1,y2 = torch.chunk(y, 2, 1)
        #y1_likelihood
        cc_ctx = torch.zeros_like(y)
        ctx = torch.concatenate([latent, cc_ctx], dim=1)
        y1_means, y1_scales = torch.chunk(self.ParamAggregation(ctx), chunks=2, dim=1)
        y1_index = self.gaussian_conditional.build_indexes(y1_scales)
        y_string = self.gaussian_conditional.compress(y1, indexes=y1_index, means=y1_means)
        return{
            "shape":z.size()[-2:],
            "strings":[y_string,z_string]
        }
    

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent = self.h_s(z_hat)
        #y1_likelihood
        B,C,H,W = latent.shape
        cc_ctx = torch.zeros((B,C//2,H,W)).to(z_hat.device)
        ctx = torch.concatenate([latent, cc_ctx], dim=1)
        y1_means, y1_scales = torch.chunk(self.ParamAggregation(ctx), chunks=2, dim=1)
        y1_index = self.gaussian_conditional.build_indexes(y1_scales)
        y1_hat = self.gaussian_conditional.decompress(strings=strings[0], indexes=y1_index, means=y1_means)
        #y2
        cc_ctx = self.cc_transform(y1_hat)
        ctx = torch.concatenate([latent, cc_ctx],dim=1)
        y2_means, y2_scales = torch.chunk(self.ParamAggregation(ctx),2,1)
        #res
        res = self.prediction(torch.concatenate([y1_hat, y2_means, y2_scales],dim=1))
        y2_hat = y2_means + res
        y_hat = torch.concatenate([y1_hat, y2_hat],dim=1)
        x_hat = self.g_s(y_hat)
        return{
            "x_hat":x_hat
        }
        
    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        state_dict = remap_old_keys("entropy_bottleneck", state_dict)

        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        nn.Module.load_state_dict(self,state_dict)

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
        updated |= self.entropy_bottleneck.update()
        return updated
    

    def aux_loss(self) -> Tensor:
        loss = self.entropy_bottleneck.loss() 
        return cast(Tensor, loss)



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
"""
    def soft(self, x, noisequant=False,tmp=1.0):
        y = self.g_a(x)
        y1,y2 = torch.chunk(y, 2, 1)
        z1 = self.h_a1(y1)
        z2 = self.h_a2(y2)
        z1_hat,z1_likelihood = self.entropy_bottleneck1(z1)
        z2_hat,z2_likelihood = self.entropy_bottleneck2(z2)
        if not noisequant:
            z1_offset = self.entropy_bottleneck1._get_medians()
            z1_tmp = z1 - z1_offset
            z1_hat = self.quantizer.quantize(z1_tmp,"ste") + z1_offset
            z2_offset = self.entropy_bottleneck2._get_medians()
            z2_tmp = z2 - z2_offset
            z2_hat = self.quantizer.quantize(z2_tmp,"ste") + z2_offset
        z_likelihood = torch.cat([z1_likelihood, z2_likelihood],dim=1)

        latent_means1, latent_scales1 = self.h_s1(z1_hat)
        latent_means2, latent_scales2 = self.h_s2(z2_hat)
        latent_scales2 = torch.sigmoid(latent_scales2)*tmp
        _, y1_likelihood = self.gaussian_conditional(y1,latent_scales1, latent_means1)
        _, y2_likelihood = self.gaussian_conditional(y2,latent_scales2, latent_means2)
        if noisequant:
            y1_hat = self.quantizer.quantize(y1,"noise")
            y2_hat = self.quantizer.quantize(y2,"noise")
        else:
            y1_hat = self.quantizer.quantize(y1-latent_means1,"ste") + latent_means1
            y2_hat = self.quantizer.quantize(y2-latent_means2,"ste") + latent_means2

        y_hat = torch.concatenate([y1_hat,y2_hat],dim=1)
        y_likelihood = torch.concatenate([y1_likelihood, y2_likelihood],dim=1)
        z_likelihood = torch.concatenate([z1_likelihood, z2_likelihood],dim=1)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat":x_hat,
            "likelihoods":{"y":y_likelihood, "z":z_likelihood},            
        }


    def hard(self,x):
        y = self.g_a(x)
        y1,y2 = torch.chunk(y,2,1)
        z1 = self.h_a1(y1)
        z2 = self.h_a2(y2)
        _, z1_likeilhood = self.entropy_bottleneck1(z1)
        _, z2_likelihood = self.entropy_bottleneck2(z2)
        z1_offset = self.entropy_bottleneck1._get_medians()
        z1_tmp = z1 - z1_offset
        z1_hat = self.quantizer.quantize(z1_tmp,"ste") + z1_offset

        z2_offset = self.entropy_bottleneck2._get_medians()
        z2_tmp = z2 - z2_offset
        z2_hat = self.quantizer.quantize(z2_tmp,"ste") + z2_offset

        latent_means1, latent_scales1 = self.h_s1(z1_hat)
        latent_means2, latent_scales2 = self.h_s2(z2_hat)
        _,y1_likelihood = self.gaussian_conditional(y1, latent_scales1, latent_means1)
        y1_hat = self.quantizer.quantize(y1-latent_means1,"ste")+latent_means1
        ctx = torch.concatenate([latent_means2, latent_scales2, y1_hat],dim=1)
        y2_hat = self.prediction(ctx)
        y_hat = torch.concatenate([y1_hat, y2_hat],dim=1)
        x_hat = self.g_s(y_hat)
        z_likelihood = torch.concatenate([z1_likeilhood, z2_likelihood],dim=1)
        return{
            "x_hat":x_hat, 
            "likelihoods":{"y":y1_likelihood,"z":z_likelihood}
        }
"""
 

class HyperGain(CompressionModel):
    """
    ELIC Model with only a hyperprior entropy model
    """
    def __init__(self, N=192, M=320):
        super().__init__(entropy_bottleneck_channels=N)
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

        gain = torch.ones([1,M,1,1],dtype=torch.float32)
        self.gain = nn.Parameter(gain, requires_grad=True)
        inverse_gain = torch.ones([1,M,1,1], dtype=torch.float32)
        self.inverse_gain = nn.Parameter(inverse_gain, requires_grad=True)
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
        y = self.gain.expand([B,C,H,W])*y

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
        
        y_hat = self.inverse_gain.expand([B,C,H,W])*y_hat
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
        y = self.gain.expand([B,C,H,W])*y

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
                "latent": y,
                "latent_enc":torch.round(y-latent_means)
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
        B,C,H,W = y_hat.shape
        y_hat = self.inverse_gain.expand([B,C,H,W])*y_hat
        x_hat = self.g_s(y_hat)
        y_dec = time.time() - y_dec
        whole_time = time.time() - whole_time
        return {"x_hat": x_hat, "time":{"z_dec":z_dec, "y_dec": y_dec, "whole":whole_time,
                                        "z_decompress_time":z_decompress_time,"y_decompress_time":y_decompress_time}}
    
    #get bpp and psnr
    def mask_inference(self, x, ch_idx):
        y = self.g_a(x)
        B,C,H,W = y.shape
        y = self.gain.expand([B,C,H,W])*y
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = torch.round(z - z_offset) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2,1)
        #mask
        y[:,ch_idx,:,:] = latent_means[:,ch_idx,:,:]
        y_hat = torch.round(y - latent_means) + latent_means
        y_hat = self.inverse_gain.expand([B,C,H,W])*y_hat
        x_hat = self.g_s(y_hat)
        return x_hat


if __name__=="__main__":
    x = torch.rand((1,3,256,256)).to("cuda")
    net = HyperGain().to("cuda")
    checkpoint_path = "/mnt/data3/jingwengu/ELIC_light/hyper_gain/lambda4/checkpoint_last_99.pth.tar"
    state_dict = torch.load(checkpoint_path,map_location="cuda")
    state_dict = state_dict["state_dict"]
    net.load_state_dict(state_dict)
    net.update()
    net.cuda()
    out_forward = net.soft(x)
    out_forward = net.hard(x)
    compress_time = time.time()
    out_compress = net.compress(x)
    compress_time = time.time() - compress_time
    decompress_time = time.time()
    out_decompress = net.decompress(out_compress["strings"], out_compress["shape"])
    decompress_time = time.time() - decompress_time
    print("compress_time:{:.4f}".format(compress_time))
    print("decompress_time:{:.4f}".format(decompress_time))
    x = torch.rand((1,3,256,256)).to("cuda")
    flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)