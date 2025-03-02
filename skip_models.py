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

def ste_sign(inputs, mode="soft"):
    """
    mode: hard or soft
    """
    if mode=="soft":
        return torch.where(inputs>0.5,0.999,0.001) - inputs.detach() + inputs
    else:
        return torch.where(inputs>0.5, 1.0, 0.0).detach()
    

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
            "latent":torch.round(y-y_means)
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
        y_means = torch.cat([y1_means, y2_means],dim=1)
        x_hat = self.g_s(y_hat)
        return{
            "x_hat":x_hat,
            "likelihoods":{"y":y_likelihood,"z":z_likelihood},
            "latent":torch.round(y-y_means)
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
            "strings":[y_string,z_string],
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


    def mask_inference(self, x, ch_idx):
        y = self.g_a(x)
        z = self.h_a(y)
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
        y_hat = torch.concatenate([y1_hat, y2_hat],dim=1)
        means = torch.concatenate([y1_means, y2_means],dim=1)
        if ch_idx<160:
            y_hat[:,ch_idx,:,:] = means[:,ch_idx,:,:]
        x_hat = self.g_s(y_hat)
        return x_hat

    

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
    def __init__(self, N=192, M=320, r_num=6):
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
        """
        gain = torch.ones([r_num,M,1,1],dtype=torch.float32)
        inverse_gain = torch.ones([r_num,M,1,1], dtype=torch.float32)

        h_gain = torch.ones([r_num,N,1,1],dtype=torch.float32)
        h_inverse_gain = torch.ones([r_num,N,1,1], dtype=torch.float32)

        self.gains = nn.Parameter(gain, requires_grad=True)
        self.inverse_gains = nn.Parameter(inverse_gain, requires_grad=True)
        
        self.r_num = r_num
        """
        gain = torch.ones([1,M,1,1],dtype=torch.float32)
        inverse_gain = torch.ones([1,M,1,1], dtype=torch.float32)
        self.gain = nn.Parameter(gain, requires_grad=True)
        self.inverse_gain = nn.Parameter(inverse_gain, requires_grad=True)
        self.quantizer = Quantizer()
        self.gaussian_conditional = GaussianConditional(None)
        self.register_buffer(name="gain_indices", tensor=torch.zeros(M, dtype=torch.long), persistent=True) 


    def update_indices(self):
        _, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        self.gain_indices = sort_indices.squeeze(-1).squeeze(-1)
        return True
    

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
            y_hat = self.quantizer.quantize(y, quantize_type="noise")
        else:
            y_hat = self.quantizer.quantize(y-latent_means, quantize_type="ste") + latent_means
        
        y_hat = self.inverse_gain.expand([B,C,H,W])*y_hat
        x_hat = self.g_s(y_hat) 
        return{
            "x_hat":x_hat,
            "likelihoods":{"y":y_likelihoods, "z":z_likelihoods}
        }
        
    

    def compress(self,x:Tensor, use_num):
        whole_time = time.time()
        y_enc_start = time.time()
        y = self.g_a(x)

        y_enc = time.time() - y_enc_start
        B, C, H, W = y.size()  ## The shape of y to generate the mask
        y = self.gain.expand([B,C,H,W])*y

        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start
        
        z_e_enc = time.time()
        z_strings = self.entropy_bottleneck.compress(z)
        z_e_enc = time.time() - z_e_enc
        
        z_offset = self.entropy_bottleneck._get_medians().detach()
        z_tmp = z - z_offset
        z_hat = torch.round(z_tmp) + z_offset

        #z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start
        
        #sort important channels
        sort_start = time.time()
        sort_results, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        
        sort_indices = sort_indices.squeeze(-1).squeeze(-1)
        y_first = torch.index_select(y, dim=1, index=sort_indices[:use_num])
        means_first = torch.index_select(latent_means, dim=1, index=sort_indices[:use_num])
        scales_first = torch.index_select(latent_scales, dim=1, index=sort_indices[:use_num])

        sort_time = time.time() - sort_start
        print("sort_time:{:.4f}\n".format(sort_time))

        y_e_enc = time.time()
        indexes = self.gaussian_conditional.build_indexes(scales_first)
        y_strings = self.gaussian_conditional.compress(y_first, indexes=indexes, means=means_first)
        y_e_enc = time.time() - y_e_enc
        
        whole_time = time.time() - whole_time
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec,"whole":whole_time,
                "z_e_enc":z_e_enc, "y_e_enc":y_e_enc},
                }


    def decompress(self, strings, shape, use_num):
        assert isinstance(strings, list) and len(strings) == 2
        whole_time = time.time()
        z_e_dec = time.time()
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_e_dec = time.time() -  z_e_dec
        
        B, _, _, _ = z_hat.size()
        z_dec = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec

        sort_results, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        sort_indices = sort_indices.squeeze(-1).squeeze(-1)
        means_first = torch.index_select(latent_means, dim=1, index=sort_indices[:use_num])
        means_last = torch.index_select(latent_means, dim=1, index=sort_indices[use_num:])
        scales_first = torch.index_select(latent_scales, dim=1, index=sort_indices[:use_num])

        y_strings = strings[0]
        indexes = self.gaussian_conditional.build_indexes(scales_first)
        y_e_dec = time.time()
        y_first_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_first)
        y_e_dec = time.time() - y_e_dec
        
        y_last_hat = means_last
        y_hat_sort = torch.concatenate([y_first_hat, y_last_hat],dim=1)
        y_hat_resort = torch.zeros_like(y_hat_sort)
        y_hat_resort.index_copy_(dim=1, index=sort_indices, source=y_hat_sort)
        y_dec = time.time()
        B,C,H,W = y_hat_resort.shape
        y_hat = self.inverse_gain.expand([B,C,H,W])*y_hat_resort
        x_hat = self.g_s(y_hat)
        y_dec = time.time() - y_dec
        whole_time = time.time() - whole_time
        return {"x_hat": x_hat, "time":{"z_dec":z_dec, "y_dec": y_dec, "whole":whole_time,
                                        "z_e_dec":z_e_dec,"y_e_dec":y_e_dec}}
    
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


class GainSkip(CompressionModel):
    """
    ELIC Model with only a hyperprior entropy model
    """
    def __init__(self, N=192, M=320, encode_ch=160):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = int(N)
        self.M = int(M)
        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        self.encode_ch = encode_ch
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
        
        #use important channels to predict other channels' residual
        self.cc_transform = nn.Sequential(
            conv(in_channels=encode_ch,out_channels=encode_ch*3//2, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            conv(in_channels=encode_ch*3//2, out_channels=encode_ch*3//2, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            conv(in_channels=encode_ch*3//2, out_channels=M-encode_ch, kernel_size=5, stride=1)
        )

        self.ParamAggregation = nn.Sequential(
            conv1x1(in_ch=2*(M-encode_ch), out_ch=2*(M-encode_ch)),
            nn.ReLU(inplace=True),
            conv1x1(in_ch=2*(M-encode_ch), out_ch=M-encode_ch),
            nn.ReLU(inplace=True),
            conv1x1(in_ch=M-encode_ch, out_ch=M-encode_ch)
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

        #sort important channels
        sort_results, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        sort_indices = sort_indices.squeeze(-1).squeeze(-1)
        y_important = torch.zeros_like(y)
        means_important = torch.zeros_like(latent_means)
        scales_important = torch.zeros_like(latent_scales)
        count = 0
        for index in sort_indices:
            y_important[:,count,:,:] = y[:,index,:,:]
            means_important[:,count,:,:] = latent_means[:,index, :,:]
            scales_important[:,count,:,:] = latent_scales[:,index,:,:]
            count +=1

        #get important latent part and their parameters
        
        y_first = y_important[:,:self.encode_ch,:,:]
        y_last = y_important[:,self.encode_ch:,:,:]
        means_first = means_important[:,:self.encode_ch,:,:]
        means_last = means_important[:,self.encode_ch:,:,:]
        scales_first = scales_important[:,:self.encode_ch,:,:]
        scales_last = scales_important[:,self.encode_ch:,:,:]
        
        _, y_first_likelihoods = self.gaussian_conditional(y_first, scales_first, means_first)
        if noisequant:
            y_first_hat = self.quantizer.quantize(y_first, quantize_type="ste")
        else:
            y_first_hat = self.quantizer.quantize(y_first - means_first, quantize_type="ste") + means_first
        
        cc_results = self.cc_transform(y_first_hat)
        y_last_hat = self.ParamAggregation(torch.concatenate([cc_results, scales_last], dim=1)) + means_last
        #y_last_hat = last_means
        y_hat_sort = torch.concatenate([y_first_hat, y_last_hat],dim=1)
        #resort y_hat
        y_hat_resort = torch.zeros_like(y)
        count = 0
        for idx in sort_indices:
            y_hat_resort[:,idx,:,:] = y_hat_sort[:,count,:,:]
            count +=1 

        y_hat = self.inverse_gain.expand([B,C,H,W])*y_hat_resort
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_first_likelihoods, "z": z_likelihoods},
            "latent_norm":(y-latent_means)/(latent_scales+TINY),
        }


    def compress(self,x:Tensor, use_num):
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
        
        #sort important channels
        sort_results, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        sort_indices = sort_indices.squeeze(-1).squeeze(-1)
        y_important = torch.zeros_like(y)
        means_important = torch.zeros_like(latent_means)
        scales_important = torch.zeros_like(latent_scales)
        count = 0
        for index in sort_indices:
            y_important[:,count,:,:] = y[:,index,:,:]
            means_important[:,count,:,:] = latent_means[:,index, :,:]
            scales_important[:,count,:,:] = latent_scales[:,index,:,:]
            count +=1

        #get important latent part and their parameters
        y_first = y_important[:,:self.encode_ch,:,:]
        y_last = y_important[:,self.encode_ch:,:,:]
        means_first = means_important[:,:self.encode_ch,:,:]
        means_last = means_important[:,self.encode_ch:,:,:]
        scales_first = scales_important[:,:self.encode_ch,:,:]
        scales_last = scales_important[:,self.encode_ch:,:,:]

        y_compress_time = time.time()
        indexes = self.gaussian_conditional.build_indexes(scales_first)
        y_strings = self.gaussian_conditional.compress(y_first, indexes=indexes, means=means_first)
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

        #sort important channels
        sort_results, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        sort_indices = sort_indices.squeeze(-1).squeeze(-1)
        means_important = torch.zeros_like(latent_means)
        scales_important = torch.zeros_like(latent_scales)
        count = 0
        for index in sort_indices:
            means_important[:,count,:,:] = latent_means[:,index, :,:]
            scales_important[:,count,:,:] = latent_scales[:,index,:,:]
            count +=1

        means_first = means_important[:,:self.encode_ch,:,:]
        means_last = means_important[:,self.encode_ch:,:,:]
        scales_first = scales_important[:,:self.encode_ch,:,:]
        scales_last = scales_important[:,self.encode_ch:,:,:]

        y_strings = strings[0]
        indexes = self.gaussian_conditional.build_indexes(scales_first)
        y_decompress_time = time.time()
        y_first_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_first)
        y_decompress_time = time.time() - y_decompress_time
        #predict
        cc_results = self.cc_transform(y_first_hat)
        y_last_hat = self.ParamAggregation(torch.concatenate([cc_results, scales_last], dim=1)) + means_last

        y_hat_sort = torch.concatenate([y_first_hat, y_last_hat],dim=1)
        #resort y_hat
        y_hat_resort = torch.zeros_like(y_hat_sort)
        count = 0
        for idx in sort_indices:
            y_hat_resort[:,idx,:,:] = y_hat_sort[:,count,:,:]
 
        y_dec = time.time()
        B,C,H,W = y_hat_resort.shape
        y_hat = self.inverse_gain.expand([B,C,H,W])*y_hat_resort
        x_hat = self.g_s(y_hat)
        y_dec = time.time() - y_dec
        whole_time = time.time() - whole_time
        return {"x_hat": x_hat, "time":{"z_dec":z_dec, "y_dec": y_dec, "whole":whole_time,
                                        "z_decompress_time":z_decompress_time,"y_decompress_time":y_decompress_time}}


class GainCAR(CompressionModel):
    """
    Gain model with channel autoregressive
    """
    def __init__(self, N=192, M=320, num_slices=5):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = int(N)
        self.M = int(M)
        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        self.num_slices = num_slices

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

        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(2*M + self.groups[i+1 if i > 0 else 0] * 2 , 512),
                nn.ReLU(inplace=True),
                conv1x1(512, 480),
                nn.ReLU(inplace=True),
                conv1x1(480, self.groups[i + 1]*2),
            ) for i in range(num_slices)
        ) ##from checkboard "Checkerboard Context Model for Efficient Learned Image Compression"" gep网络参数

        gain = torch.ones([1,M,1,1],dtype=torch.float32)
        inverse_gain = torch.ones([1,M,1,1], dtype=torch.float32)
        self.gain = nn.Parameter(gain, requires_grad=True)
        self.inverse_gain = nn.Parameter(inverse_gain, requires_grad=True)
        self.quantizer = Quantizer()
        self.gaussian_conditional = GaussianConditional(None)
        self.register_buffer(name="gain_indices", tensor=torch.zeros(M, dtype=torch.long), persistent=True)


    def update_indices(self):
        _, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        self.gain_indices = sort_indices.squeeze(-1).squeeze(-1)
        return True
    

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
        y_sort = torch.index_select(input=y, dim=1, index=self.gain_indices)

        z = self.h_a(y_sort)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = self.quantizer.quantize(z_tmp,"ste") + z_offset
        
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        #channel_autoregressive based on gain 
        y_slices = torch.split(y_sort, self.groups[1:],dim=1)
        y_sort_hats = []
        y_sort_hats_gs = []
        y_sort_likelihoods = []
        for slice_idx, y_slice in enumerate(y_slices):
            if slice_idx==0:
                all_ctx = torch.concatenate([latent_means, latent_scales], dim=1)
            elif slice_idx==1:
                cc_result = self.cc_transforms[slice_idx-1](y_sort_hats[0])
                all_ctx = torch.concatenate([latent_means, latent_scales, cc_result], dim=1)
            else:
                cc_result = self.cc_transforms[slice_idx-1](torch.concatenate([y_sort_hats[0], y_sort_hats[slice_idx-1]],dim=1))
                all_ctx = torch.concatenate([latent_means, latent_scales, cc_result], dim=1)

            slice_means, slice_scales = torch.chunk(self.ParamAggregation[slice_idx](all_ctx), 2, 1)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, slice_scales, slice_means)
            #quantization
            if noisequant:
                y_slice_hat = self.quantizer.quantize(y_slice, quantize_type="noise")
                y_slice_gs = self.quantizer.quantize(y_slice, quantize_type="ste")
            else:
                y_slice_hat = self.quantizer.quantize(y_slice - slice_means, quantize_type="ste") + slice_means
                y_slice_gs = self.quantizer.quantize(y_slice - slice_means, quantize_type="ste") + slice_means

            y_sort_hats.append(y_slice_hat)
            y_sort_hats_gs.append(y_slice_gs)
            y_sort_likelihoods.append(y_slice_likelihood)

        y_likelihoods = torch.concat(y_sort_likelihoods,dim=1)
        y_hat_gs = torch.concat(y_sort_hats_gs, dim=1)
        
        y_likelihoods_resort = torch.zeros_like(y_likelihoods)
        y_hat_resort = torch.zeros_like(y_hat_gs)
        y_hat_resort.index_copy_(dim=1, index=self.gain_indices, source=y_hat_gs)
        y_likelihoods_resort.index_copy_(dim=1, index=self.gain_indices, source=y_likelihoods)
        
        y_hat = self.inverse_gain.expand([B,C,H,W])*y_hat_resort
        x_hat = self.g_s(y_hat) 
        return{
            "x_hat":x_hat,
            "likelihoods":{"y":y_likelihoods_resort, "z":z_likelihoods}
        }
        
    
    def compress(self,x:Tensor, use_num=160):
        #use_num:[0,192]
        assert use_num <= 192, print("use_num must >= 192")

        whole_time = time.time()
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start
        B, C, H, W = y.size() ## The shape of y to generate the mask
        y = self.gain.expand([B,C,H,W])*y
        y_sort = torch.index_select(input=y, dim=1, index=self.gain_indices)
        z = self.h_a(y_sort)  
        #channel_autoregressive based on gain 
        z_enc_start = time.time()
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
        
        y_compress_start = time.time()
        #sort important channels
       
        #channel_autoregressive based on gain 
        y_slices = torch.split(y_sort, self.groups[1:],dim=1)
        y_sort_hats = []
        y_strings = []
        for slice_idx, y_slice in enumerate(y_slices):
            if slice_idx==0:
                all_ctx = torch.concatenate([latent_means, latent_scales], dim=1)
            elif slice_idx==1:
                cc_result = self.cc_transforms[slice_idx-1](y_sort_hats[0])
                all_ctx = torch.concatenate([latent_means, latent_scales, cc_result], dim=1)
            else:
                cc_result = self.cc_transforms[slice_idx-1](torch.concatenate([y_sort_hats[0], y_sort_hats[slice_idx-1]],dim=1))
                all_ctx = torch.concatenate([latent_means, latent_scales, cc_result], dim=1)
            
            #compress y_slice
            slice_means, slice_scales = torch.chunk(self.ParamAggregation[slice_idx](all_ctx), 2, 1)
            if slice_idx == self.num_slices-1:
                if use_num==0:
                    continue
                else:
                    slice_means_first = slice_means[:,:use_num,:,:]
                    slice_scales_first = slice_scales[:,:use_num,:,:]
                    slice_index_first = self.gaussian_conditional.build_indexes(slice_scales_first)
                    y_string = self.gaussian_conditional.compress(y_slice[:,:use_num,:,:], slice_index_first, slice_means_first)
                    y_strings.append(y_string)
                    continue

            slice_index = self.gaussian_conditional.build_indexes(slice_scales)
            y_slice_string = self.gaussian_conditional.compress(y_slice, slice_index, slice_means)
            y_strings.append(y_slice_string)
            #quantization
            y_slice_hat = self.quantizer.quantize(y_slice - slice_means, quantize_type="ste") + slice_means
            y_sort_hats.append(y_slice_hat)
        
        y_compress_time = time.time() - y_compress_start
        whole_time = time.time() - whole_time
        
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec,"whole":whole_time,
                "z_compress_time":z_compress_time, "y_compress_time":y_compress_time},
                #"latent": y,
                #"latent_enc":torch.round(y-latent_means)
                }


    def decompress(self, strings, shape, use_num=160):
        assert isinstance(strings, list) and len(strings) == 2
        whole_time = time.time()
        z_decompress_time = time.time()
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_decompress_time = time.time() -  z_decompress_time
        z_dec = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec

        y_decompress_start = time.time()
        #sort important channels
        y_sort_hats = []
        for slice_idx in range(self.num_slices):
            if slice_idx==0:
                all_ctx = torch.concatenate([latent_means, latent_scales], dim=1)
            elif slice_idx==1:
                cc_result = self.cc_transforms[slice_idx-1](y_sort_hats[0])
                all_ctx = torch.concatenate([latent_means, latent_scales, cc_result], dim=1)
            else:
                cc_result = self.cc_transforms[slice_idx-1](torch.concatenate([y_sort_hats[0], y_sort_hats[slice_idx-1]],dim=1))
                all_ctx = torch.concatenate([latent_means, latent_scales, cc_result], dim=1)
            
            #compress y_slice
            slice_means, slice_scales = torch.chunk(self.ParamAggregation[slice_idx](all_ctx), 2, 1)
            if slice_idx == self.num_slices-1:
                if use_num==0:
                    y_slice_hat = slice_means
                    y_sort_hats.append(y_slice_hat)
                    continue
                else:
                    slice_means_first = slice_means[:,:use_num,:,:]
                    slice_scales_first = slice_scales[:,:use_num,:,:]
                    slice_index_first = self.gaussian_conditional.build_indexes(slice_scales_first)
                    y_slice_first = self.gaussian_conditional.decompress(strings[0][slice_idx], slice_index_first, means=slice_means_first)
                    y_slice_hat = torch.concatenate([y_slice_first, slice_means[:,use_num:,:,:]],dim=1)
                    y_sort_hats.append(y_slice_hat)
                    continue

            slice_index = self.gaussian_conditional.build_indexes(slice_scales)
            #quantization
            y_slice_hat = self.gaussian_conditional.decompress(strings[0][slice_idx], slice_index, means=slice_means)
            y_sort_hats.append(y_slice_hat) 
        y_decompress_time = time.time() - y_decompress_start

        y_sort_hat = torch.concatenate(y_sort_hats, dim=1)
        y_hat_resort = torch.zeros_like(y_sort_hat)
        y_hat_resort.index_copy_(dim=1, index=self.gain_indices, source=y_sort_hat)
        B,C,H,W = y_hat_resort.shape
        y_hat = self.inverse_gain.expand([B,C,H,W])*y_hat_resort

        y_dec_start = time.time()
        x_hat = self.g_s(y_hat) 
        y_dec = time.time() - y_dec_start
        whole_time = time.time() - whole_time
        return {"x_hat": x_hat, "time":{"z_dec":z_dec, "y_dec": y_dec, "whole":whole_time,
                                        "z_decompress_time":z_decompress_time,"y_decompress_time":y_decompress_time}}


class HyperSkip(CompressionModel):
    """
    ELIC Model with only a hyperprior entropy model
    """
    def __init__(self, N=192, M=320, n_skip=200, d_skip=60, s_skip=40):
        super().__init__(entropy_bottleneck_channels=192)
        self.N = int(N)
        self.M = int(M)
        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        self.d_skip = d_skip
        self.s_skip = s_skip
        self.n_skip = n_skip
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
        gain = torch.ones([1,M,1,1],dtype=torch.float32)
        self.gain = nn.Parameter(gain, requires_grad=True)
        inverse_gain = torch.ones([1,M,1,1], dtype=torch.float32)
        self.inverse_gain = nn.Parameter(inverse_gain, requires_grad=True)
        self.register_buffer(name="gain_indices", tensor=torch.zeros(M, dtype=torch.long), persistent=True) 

        self.importance_pred = nn.Sequential(
            deconv(N, N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            deconv(N, N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            conv3x3(N, N),
        )

        self.skip_prediction = nn.Sequential(
            conv3x3(2*self.d_skip+N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            conv3x3(N, N//2),
            ResidualBottleneckBlock(N//2),
            ResidualBottleneckBlock(N//2),
            AttentionBlock(N//2),
            conv3x3(N//2, self.d_skip),
        )


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
        B, C, H, W = y.size() ## The shape of y to generate the mask
        y = self.gain.expand([B,C,H,W])*y
        
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        importance_map = self.importance_pred(z_hat)
        means_sort = torch.index_select(latent_means, dim=1, index=self.gain_indices)
        scales_sort = torch.index_select(latent_scales, dim=1, index=self.gain_indices)

        d_skip_means = means_sort[:,self.n_skip:self.n_skip+self.d_skip,:,:]
        d_skip_scales = scales_sort[:,self.n_skip:self.n_skip+self.d_skip,:,:]
        skip_mask = self.skip_prediction(torch.concatenate([d_skip_means, d_skip_scales, importance_map], dim=1))
        skip_mask_tmp = ste_sign(torch.sigmoid(skip_mask),"soft")
        all_mask = torch.ones_like(y)
        all_mask[:,self.n_skip:self.n_skip+self.d_skip,:,:] = skip_mask_tmp

        _, y_likelihoods = self.gaussian_conditional(y, latent_scales, latent_means)
        y_likelihoods_sort = torch.index_select(y_likelihoods, dim=1, index=self.gain_indices)
        y_likelihoods_sort = all_mask *y_likelihoods_sort+ (1.0 - all_mask)

        if noisequant:
            y_hat = self.quantizer.quantize(y, quantize_type="noise")
        else:
            y_hat = self.quantizer.quantize(y-latent_means, quantize_type="ste") + latent_means
        
        y_hat_sort = torch.index_select(y_hat, dim=1, index=self.gain_indices)
        y_hat_sort = y_hat_sort * all_mask + (1 - all_mask) * means_sort
        y_hat_resort = torch.zeros_like(y_hat)
        y_hat_resort = y_hat_resort.index_copy(dim=1, index=self.gain_indices, source=y_hat_sort)
        y_hat_resort = self.inverse_gain.expand([B,C,H,W])*y_hat_resort
        x_hat = self.g_s(y_hat_resort)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods_sort, "z": z_likelihoods},
            "norm": skip_mask_tmp
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

        #want to know compress time and decompress time for y and z
        z_compress_start = time.time()
        z_string = self.entropy_bottleneck.compress(z)
        print("z_compress_time:{:.4f}".format(time.time()-z_compress_start))

        z_decompress_start = time.time()
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])
        print("z_decompress_time:{:.4f}".format(time.time() - z_decompress_start))


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


class GainCheckerboard(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = int(N)
        self.M = int(M)
        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        self.num_slices = num_slices

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

        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
            self.groups[i+1], 2*self.groups[i+1], kernel_size=5, padding=2, stride=1
            ) for i in range(num_slices)
        )## from https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression/blob/main/version2/layers/CheckerboardContext.py

        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(640  + self.groups[i + 1] * 2, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, 480),
                nn.ReLU(inplace=True),
                conv1x1(480, self.groups[i + 1]*2),
            ) for i in range(num_slices)
        ) ##from checkboard "Checkerboard Context Model for Efficient Learned Image Compression"" gep网络参数

        gain = torch.ones([1,M,1,1],dtype=torch.float32)
        inverse_gain = torch.ones([1,M,1,1], dtype=torch.float32)
        self.gain = nn.Parameter(gain, requires_grad=True)
        self.inverse_gain = nn.Parameter(inverse_gain, requires_grad=True)
        self.quantizer = Quantizer()
        self.gaussian_conditional = GaussianConditional(None)
        self.register_buffer(name="gain_indices", tensor=torch.zeros(M, dtype=torch.long), persistent=True)


    def update_indices(self):
        _, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        self.gain_indices = sort_indices.squeeze(-1).squeeze(-1)
        return True
    

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
    
    #改变y的顺序后要重新训练熵模型
    def forward(self, x, noisequant, ctx_idx):
        y = self.g_a(x)
        y = self.gain.expand_as(y) * y
        y_sort = torch.index_select(y, 1, self.gain_indices)

        z = self.h_a(y_sort)
        _, z_likelihoods = self.entropy_bottleneck(z)
        if noisequant:
            z_hat = self.quantizer.quantize(z,"noise")
        else:
            z_offset = self.entropy_bottleneck._get_medians()
            z_hat = self.quantizer.quantize(z - z_offset, "ste") + z_offset
        
        latent_means_sort, latent_scales_sort = self.h_s(z_hat).chunk(2,1)
        y_sort_slices = torch.split(y_sort, self.groups[1:], dim=1)
        y_slice_likelihoods = []
        y_slice_hats = []
        y_slice_hats_gs = []

        for slice_idx, sort_slice in enumerate(y_sort_slices):
            latent_ctx = torch.concatenate([latent_means_sort, latent_scales_sort], dim=1)
            B,C,H,W = sort_slice.shape
            sort_slice_anchor = torch.zeros_like(sort_slice)
            sort_slice_na = torch.zeros_like(sort_slice)
            sort_slice_anchor[:,:,0::2,0::2] = sort_slice[:,:,0::2,0::2]
            sort_slice_anchor[:,:,1::2,1::2] = sort_slice[:,:,1::2,1::2]
            sort_slice_na[:,:,0::2,1::2] = sort_slice[:,:,0::2,1::2]
            sort_slice_na[:,:,1::2,0::2] = sort_slice[:,:,1::2,0::2]

            #checkerboard process1
            anchor_spatial_ctx = torch.zeros([B, 2*C, H, W], device=sort_slice.device)
            ctx = torch.concatenate([latent_ctx, anchor_spatial_ctx], dim=1)
            slice_a_means, slice_a_scales = self.ParamAggregation[slice_idx](ctx).chunk(2,1)
            slice_anchor_means = torch.zeros_like(slice_a_means)
            slice_anchor_scales = torch.zeros_like(slice_a_scales)
            slice_anchor_means[:,:,0::2,0::2] = slice_a_means[:,:,0::2,0::2]
            slice_anchor_means[:,:,1::2,1::2] = slice_a_means[:,:,1::2,1::2]
            slice_anchor_scales[:,:,0::2,0::2] = slice_a_scales[:,:,0::2,0::2]
            slice_anchor_scales[:,:,1::2,1::2] = slice_a_scales[:,:,1::2,1::2]
            if noisequant:
                anchor_hat = self.quantizer.quantize(sort_slice_anchor, "noise")
                anchor_hat_gs = self.quantizer.quantize(sort_slice_anchor, "ste")
            else:
                anchor_hat = self.quantizer.quantize(sort_slice_anchor - slice_anchor_means, "ste") + slice_anchor_means 
                anchor_hat_gs = self.quantizer.quantize(sort_slice_anchor - slice_anchor_means, "ste") + slice_anchor_means
                
            #checkerboard process2
            na_spatial_ctx = torch.zeros([B, 2*C, H, W], device=sort_slice.device)
            if slice_idx == ctx_idx:
                na_spatial_ctx = self.context_prediction[slice_idx](anchor_hat)
            ctx = torch.concatenate([latent_ctx, na_spatial_ctx], dim=1)
            slice_n_means, slice_n_scales = self.ParamAggregation[slice_idx](ctx).chunk(2,1)
            slice_na_means = torch.zeros_like(slice_n_means)
            slice_na_scales = torch.zeros_like(slice_n_scales)
            slice_na_means[:,:,0::2,1::2] = slice_n_means[:,:,0::2,1::2]
            slice_na_means[:,:,1::2,0::2] = slice_n_means[:,:,1::2,0::2]
            slice_na_scales[:,:,0::2,1::2] = slice_n_scales[:,:,0::2,1::2]
            slice_na_scales[:,:,1::2,0::2] = slice_n_scales[:,:,1::2,0::2]
            if noisequant:
                na_hat = self.quantizer.quantize(sort_slice_na, "noise")
                na_hat_gs = self.quantizer.quantize(sort_slice_na, "ste")
            else:
                na_hat = self.quantizer.quantize(sort_slice_na - slice_na_means, "ste") + slice_na_means 
                na_hat_gs = self.quantizer.quantize(sort_slice_na -  slice_na_means,"ste") + slice_na_means

            slice_means = slice_anchor_means + slice_na_means
            slice_scales = slice_anchor_scales + slice_na_scales
            y_sort_slice_hat = anchor_hat + na_hat
            y_sort_slice_hat_gs = anchor_hat_gs + na_hat_gs
            _, y_slice_likelihood = self.gaussian_conditional(sort_slice, slice_scales, slice_means)
            y_slice_likelihoods.append(y_slice_likelihood)
            y_slice_hats.append(y_sort_slice_hat)
            y_slice_hats_gs.append(y_sort_slice_hat_gs)
        
        y_sort_hat = torch.concat(y_slice_hats,dim=1)
        y_sort_hat_gs = torch.concat(y_slice_hats_gs, dim=1)
        y_sort_likelihoods = torch.concat(y_slice_likelihoods, dim=1)
        y_hat = torch.zeros_like(y)
        y_hat = y_hat.index_copy(dim=1, index=self.gain_indices, source=y_sort_hat_gs)
        y_hat = self.inverse_gain.expand_as(y)*y_hat
        x_hat = self.g_s(y_hat)
        return{
            "likelihoods":{"y":y_sort_likelihoods, "z":z_likelihoods},
            "x_hat":x_hat
        }





class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, alpha=0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.alpha = alpha



#gain + ELIC
class ELIC_sort(CompressionModel):
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
        gain = torch.ones([1,M,1,1],dtype=torch.float32)
        inverse_gain = torch.ones([1,M,1,1], dtype=torch.float32)
        self.gain = nn.Parameter(gain, requires_grad=True)
        self.inverse_gain = nn.Parameter(inverse_gain, requires_grad=True)
        self.register_buffer(name="gain_indices", tensor=torch.zeros(M, dtype=torch.long), persistent=True)
        

        self.quantizer = Quantizer()
        self.gaussian_conditional = GaussianConditional(None)

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
            ) for i in range(num_slices-1)
        )

        #last context prediction
        self.context_prediction.append(
            nn.Sequential(
                CheckboardMaskedConv2d(self.groups[-1], 2*self.groups[-1], kernel_size=5, padding=2, stride=1, groups=self.groups[-1]//4),
                conv1x1(in_ch=2*self.groups[-1],out_ch=624,group_num=self.groups[-1]//4),
                nn.ReLU(inplace=True), 
                conv1x1(in_ch=624, out_ch=480, group_num=self.groups[-1]//4), 
                nn.ReLU(inplace=True), 
                conv1x1(480, self.groups[-1]*2, group_num=self.groups[-1]//4)
                )
        )
        ## from https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression/blob/main/version2/layers/CheckerboardContext.py

        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(640 + self.groups[i+1 if i > 0 else 0] * 2 + self.groups[
                        i + 1] * 2, 640),
                nn.ReLU(inplace=True),
                conv1x1(640, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[i + 1]*2),
            ) for i in range(num_slices-1)
        ) ##from checkboard "Checkerboard Context Model for Efficient Learned Image Compression"" gep网络参数

        ### last self aggregation, only for latent and channel-context
        self.ParamAggregation.append(
            nn.Sequential(
                conv1x1(640 + self.groups[-1]*2, 640),
                nn.ReLU(inplace=True), 
                conv1x1(640, 512), 
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[-1]*2)
            )
        ) 
 
    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)


    def update_indices(self):
        _, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        self.gain_indices = sort_indices.squeeze(-1).squeeze(-1)
        return True


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
    

    def forward(self, x, noisequant=False):
        y = self.g_a(x)
        B, C, H, W = y.size() ## The shape of y to generate the mask
        y = y*self.gain.expand_as(y)
        y = torch.index_select(index=self.gain_indices, dim=1, input=y)

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
            if slice_index==self.num_slices-1:
                means_anchor, scales_anchor = self.ParamAggregation[slice_index](support).chunk(2,1)
            else:
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
            if slice_index==self.num_slices-1:
                means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](support).chunk(2,1)
                means_checkboard = masked_context[:,0::2, :,:]
                scales_checkboard = masked_context[:,1::2,:,:]
                means_non_anchor = means_non_anchor + means_checkboard
                scales_non_anchor = scales_non_anchor + scales_checkboard
            else:
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
        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        y_hat = y_hat.index_copy(dim=1, index=self.gain_indices, source=y_hat)*self.inverse_gain.expand_as(y)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    

    def compress(self, x, use_num=320):
        assert use_num>128 and use_num<=320


        whole_time = time.time()
        y_enc_start = time.time()
        y = self.g_a(x)
        y = self.gain.expand_as(y)*y

        y_sort = torch.index_select(input=y, dim=1, index=self.gain_indices)
        B, C, H, W = y_sort.size() ## The shape of y to generate the mask

        y_enc = time.time() - y_enc_start
        z_enc_start = time.time()
        z = self.h_a(y_sort)
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

        y_slices = torch.split(y_sort, self.groups[1:], 1)

        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device), [2 * i for i in self.groups[1:]], 1)
        y_need_encode = []
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
            

            ##support mean and scale, including channel autoregression and latent scales
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            
            #split
            anchor_part = torch.zeros_like(y_slice)
            non_anchor_part  = torch.zeros_like(y_slice)
            anchor_part[:,:,0::2,0::2] = y_slice[:,:,0::2,0::2]
            anchor_part[:,:,1::2,1::2] = y_slice[:,:,1::2,1::2]
            non_anchor_part[:,:,0::2,1::2] = y_slice[:,:,0::2,1::2]
            non_anchor_part[:,:,1::2,0::2] = y_slice[:,:,1::2, 0::2]
            #need pass
            if slice_index==self.num_slices-1 and use_num < 320:
                #checkerboard process 1
                y_anchor = y_slices[slice_index].clone()
                means_anchor, scales_anchor, = self.ParamAggregation[slice_index](support).chunk(2, 1)
                means_anchor_zero = torch.zeros_like(means_anchor)
                means_anchor_zero[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
                means_anchor_zero[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

                B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()
                use_ch_num = use_num - 128
                
                y_anchor_encode = torch.zeros(B_anchor, use_ch_num, H_anchor, W_anchor//2).to(x.device)
                means_anchor_encode = torch.zeros(B_anchor, use_ch_num, H_anchor, W_anchor//2).to(x.device)
                scales_anchor_encode = torch.zeros(B_anchor, use_ch_num, H_anchor, W_anchor // 2).to(x.device)
                y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)

                y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :use_ch_num, 0::2, 0::2]
                y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :use_ch_num, 1::2, 1::2]
                means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :use_ch_num, 0::2, 0::2]
                means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :use_ch_num, 1::2, 1::2]
                scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :use_ch_num, 0::2, 0::2]
                scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :use_ch_num, 1::2, 1::2]
                compress_anchor = time.time()
                indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
                anchor_strings = self.gaussian_conditional.compress(y_anchor_encode, indexes_anchor, means=means_anchor_encode)
                compress_anchor = time.time() - compress_anchor
                y_compress_time += compress_anchor

                anchor_quantized = self.quantizer.quantize(y_anchor_encode - means_anchor_encode, "ste") + means_anchor_encode
                #anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor_encode)
                y_anchor_decode[:, :use_ch_num, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
                y_anchor_decode[:, :use_ch_num, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]
                y_anchor_decode[:, use_ch_num:, 0::2, 0::2] = means_anchor[:, use_ch_num:, 0::2, 0::2]
                y_anchor_decode[:, use_ch_num:, 1::2, 1::2] = means_anchor[:, use_ch_num, 1::2, 1::2]
                
                #checkerboard process 2
                masked_context = self.context_prediction[slice_index](y_anchor_decode)
                means_checkboard = masked_context[:,0::2,:,:]
                scales_checkboard = masked_context[:,1::2,:,:]
                means_non_anchor = means_anchor + means_checkboard
                scales_non_anchor = scales_anchor + scales_checkboard

                y_non_anchor_encode = torch.zeros(B_anchor, use_ch_num, H_anchor, W_anchor // 2).to(x.device)
                means_non_anchor_encode = torch.zeros(B_anchor, use_ch_num, H_anchor, W_anchor // 2).to(x.device)
                scales_non_anchor_encode = torch.zeros(B_anchor, use_ch_num, H_anchor, W_anchor // 2).to(x.device)

                means_na_zero = torch.zeros_like(means_non_anchor)
                means_na_zero[:,:,0::2,0::2] = means_non_anchor[:,:,0::2,0::2]
                means_na_zero[:,:,1::2,1::2] = means_non_anchor[:,:,1::2,1::2]


                non_anchor = y_slices[slice_index].clone()
                y_non_anchor_encode[:, :, 0::2, :] = non_anchor[:, :use_ch_num, 0::2, 1::2]
                y_non_anchor_encode[:, :, 1::2, :] = non_anchor[:, :use_ch_num, 1::2, 0::2]
                means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :use_ch_num, 0::2, 1::2]
                means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :use_ch_num, 1::2, 0::2]
                scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :use_ch_num, 0::2, 1::2]
                scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :use_ch_num, 1::2, 0::2]
                compress_non_anchor = time.time()
                indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
                non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor_encode, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)
                compress_non_anchor = time.time() - compress_non_anchor
                y_compress_time += compress_non_anchor
                non_anchor_quantized = self.quantizer.quantize(y_non_anchor_encode - means_non_anchor_encode, "ste") + means_non_anchor_encode
                #non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                #                                                            means=means_non_anchor_encode)

                y_non_anchor_quantized = torch.zeros_like(means_anchor)
                y_non_anchor_quantized[:, :use_ch_num, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
                y_non_anchor_quantized[:, :use_ch_num, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]
                y_non_anchor_quantized[:, use_ch_num:, 0::2, 1::2] = means_non_anchor[:, use_ch_num:, 0::2, 1::2]
                y_non_anchor_quantized[:, use_ch_num:, 1::2, 0::2] = means_non_anchor[:, use_ch_num:, 1::2, 0::2]
        
                y_slice_encode = torch.round(anchor_part-means_anchor_zero) + torch.round(non_anchor_part - means_na_zero) 
                y_need_encode.append(y_slice_encode)
                y_slice_hat = y_anchor_decode + y_non_anchor_quantized
                y_hat_slices.append(y_slice_hat)

                y_strings.append([anchor_strings, non_anchor_strings])
                y_encode = torch.cat(y_need_encode, dim=1)

            else:
                ### checkboard process 1
                y_anchor = y_slices[slice_index].clone()
                if slice_index==self.num_slices-1:
                    means_anchor, scales_anchor = self.ParamAggregation[slice_index](support).chunk(2,1)
                else:
                    means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                        torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)
                    
                B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()
                y_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
                means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
                scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
                y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)

                means_anchor_zero = torch.zeros_like(means_anchor)
                means_anchor_zero[:,:,0::2,0::2] = means_anchor[:,:,0::2,0::2]
                means_anchor_zero[:,:,1::2,1::2] = means_anchor[:,:,1::2,1::2]

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
                if slice_index==self.num_slices-1:
                    means_checkboard = masked_context[:,0::2,:,:]
                    scales_checkboard = masked_context[:,1::2,:,:]
                    means_non_anchor = means_anchor + means_checkboard
                    scales_non_anchor = scales_anchor + scales_checkboard
                else:
                    means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                        torch.concat([masked_context, support], dim=1)).chunk(2, 1)
                y_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
                means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
                scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)

                means_na_zero = torch.zeros_like(means_non_anchor)
                means_na_zero[:,:,0::2,1::2] = means_non_anchor[:,:,0::2,1::2]
                means_na_zero[:,:,1::2,0::2] = means_non_anchor[:,:,1::2,0::2]

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
        
                y_slice_encode = torch.round(anchor_part - means_anchor_zero) + torch.round(non_anchor_part - means_na_zero) 
                y_need_encode.append(y_slice_encode)
                y_slice_hat = y_anchor_decode + y_non_anchor_quantized
                y_hat_slices.append(y_slice_hat)

                y_strings.append([anchor_strings, non_anchor_strings])
                y_encode = torch.cat(y_need_encode, dim=1)

        whole_time = time.time() - whole_time
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec, "y_compress_time":y_compress_time,
                         "z_compress_time":z_compress_time,"whole":whole_time},
                         "latent":y_encode}



    def decompress(self, strings, shape, use_num=320):
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
            
            if slice_index==self.num_slices-1 and use_num < 320:
                ### checkboard process 1
                means_anchor, scales_anchor, = self.ParamAggregation[slice_index](support).chunk(2,1)

                B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()
                use_ch_num = use_num - 128
                means_anchor_encode = torch.zeros(B_anchor, use_ch_num, H_anchor, W_anchor // 2).to(z_hat.device)
                scales_anchor_encode = torch.zeros(B_anchor, use_ch_num, H_anchor, W_anchor // 2).to(z_hat.device)
                y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(z_hat.device)

                means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :use_ch_num, 0::2, 0::2]
                means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :use_ch_num, 1::2, 1::2]
                scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :use_ch_num, 0::2, 0::2]
                scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :use_ch_num, 1::2, 1::2]

                anchor_decompress = time.time()
                indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
                anchor_strings = y_strings[slice_index][0]
                anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor,
                                                                        means=means_anchor_encode)
                anchor_decompress = time.time() - anchor_decompress
                y_decompress_time += anchor_decompress

                y_anchor_decode[:, :use_ch_num, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
                y_anchor_decode[:, :use_ch_num, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]
                y_anchor_decode[:, use_ch_num:, 0::2, 0::2] = means_anchor[:, use_ch_num:, 0::2, 0::2]
                y_anchor_decode[:, use_ch_num:, 1::2, 1::2] = means_anchor[:, use_ch_num:, 1::2, 1::2]

                ### checkboard process 2
                masked_context = self.context_prediction[slice_index](y_anchor_decode)
                means_checkboard = masked_context[:,0::2,:,:]
                scales_checkboard = masked_context[:,1::2,:,:]
                means_non_anchor = means_anchor + means_checkboard
                scales_non_anchor = scales_anchor + scales_checkboard

                means_non_anchor_encode = torch.zeros(B_anchor, use_ch_num, H_anchor, W_anchor // 2).to(z_hat.device)
                scales_non_anchor_encode = torch.zeros(B_anchor, use_ch_num, H_anchor, W_anchor // 2).to(z_hat.device)

                means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :use_ch_num, 0::2, 1::2]
                means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :use_ch_num, 1::2, 0::2]
                scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :use_ch_num, 0::2, 1::2]
                scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :use_ch_num, 1::2, 0::2]
                
                non_anchor_decompress = time.time()
                indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
                non_anchor_strings = y_strings[slice_index][1]
                non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                            means=means_non_anchor_encode)
                non_anchor_decompress = time.time() - non_anchor_decompress
                y_decompress_time += non_anchor_decompress

                y_non_anchor_quantized = torch.zeros_like(means_anchor)
                y_non_anchor_quantized[:, :use_ch_num, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
                y_non_anchor_quantized[:, :use_ch_num, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]
                y_non_anchor_quantized[:, use_ch_num:, 0::2, 1::2] = means_non_anchor[:, use_ch_num:, 0::2, 1::2]
                y_non_anchor_quantized[:, use_ch_num:, 1::2, 0::2] = means_non_anchor[:, use_ch_num:, 1::2, 0::2]

                y_slice_hat = y_anchor_decode + y_non_anchor_quantized
                y_hat_slices.append(y_slice_hat)
            else:
                ### checkboard process 1
                if slice_index==self.num_slices-1:
                    means_anchor, scales_anchor = self.ParamAggregation[slice_index](support).chunk(2,1)
                else:
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
                if slice_index==self.num_slices-1:
                    means_checkboard = masked_context[:,0::2,:,:]
                    scales_checkboard = masked_context[:,1::2,:,:]
                    means_non_anchor = means_anchor + means_checkboard
                    scales_non_anchor = scales_anchor + scales_checkboard
                else: 
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
        y_hat_resort = torch.zeros_like(y_hat)
        y_hat_resort = y_hat_resort.index_copy(dim=1, source=y_hat, index=self.gain_indices)*self.inverse_gain.expand_as(y_hat)
        y_dec_start = time.time()
        x_hat = self.g_s(y_hat_resort).clamp_(0, 1)
        y_dec = time.time() - y_dec_start
        whole_time = time.time() - whole_time
        return {"x_hat": x_hat, "time":{"z_dec":z_dec,"y_dec": y_dec, "z_decompress_time":z_decompress_time,"y_decompress_time":y_decompress_time,"whole":whole_time}}


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, alpha=0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.alpha = alpha

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["y_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["y"]
        )
        out["z_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["z"]
        )
        out["mse_loss"] = self.mse(output["x_hat"], target) * 255 ** 2
        #out["norm"] = torch.sum(output["norm"]) 
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]

        return out


class ELIC_gain(CompressionModel):
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
        gain = torch.ones([1,M,1,1],dtype=torch.float32)
        inverse_gain = torch.ones([1,M,1,1], dtype=torch.float32)
        self.gain = nn.Parameter(gain, requires_grad=True)
        self.inverse_gain = nn.Parameter(inverse_gain, requires_grad=True)


    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)


    def forward(self, x, noisequant=False):
        y = self.g_a(x)
        y = self.gain.expand_as(y)*y
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
        
        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        y_hat = self.inverse_gain.expand_as(y_hat)*y_hat
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


    def compress(self, x, use_num=320):
        assert use_num>128 and use_num<=320
        whole_time = time.time()
        y_enc_start = time.time()
        y = self.g_a(x)
        y = self.gain.expand_as(y)*y

        _, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        gain_indices = sort_indices.squeeze(-1).squeeze(-1)

        y_skip_ch = gain_indices[use_num:]

        B, C, H, W = y.size() ## The shape of y to generate the mask

        y_enc = time.time() - y_enc_start
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
        y_need_encode = []
        y_strings = []
        y_hat_slices = []
        y_compress_time = 0

        group_cumulative = torch.cumsum(torch.tensor(self.groups), dim=0)
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
            

            ##support mean and scale, including channel autoregression and latent scales
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            
            #split
            anchor_part = torch.zeros_like(y_slice)
            non_anchor_part  = torch.zeros_like(y_slice)
            anchor_part[:,:,0::2,0::2] = y_slice[:,:,0::2,0::2]
            anchor_part[:,:,1::2,1::2] = y_slice[:,:,1::2,1::2]
            non_anchor_part[:,:,0::2,1::2] = y_slice[:,:,0::2,1::2]
            non_anchor_part[:,:,1::2,0::2] = y_slice[:,:,1::2, 0::2]

            ### checkboard process 1
            y_anchor = y_slices[slice_index].clone()
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)
                
            B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()
            #find channels who need skip
            slice_skip_ch = [ch_idx - group_cumulative[slice_index] 
                             for ch_idx in y_skip_ch
                             if ch_idx >= group_cumulative[slice_index] and ch_idx < group_cumulative[slice_index+1]]
            need_encode_ch = [ch_idx for ch_idx in range(self.groups[slice_index+1]) if ch_idx not in slice_skip_ch]
            need_encode_ch = torch.tensor(need_encode_ch, dtype=torch.long, device=y.device)

            y_anchor_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(x.device)
            y_anchor_encode = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor//2).to(x.device)
            means_anchor_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(x.device)
            means_anchor_encode = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor//2).to(x.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor // 2).to(x.device)
            scales_anchor_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(x.device)
            y_anchor_decode_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(x.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)

            means_anchor_zero = torch.zeros_like(means_anchor)
            means_anchor_zero[:,:,0::2,0::2] = means_anchor[:,:,0::2,0::2]
            means_anchor_zero[:,:,1::2,1::2] = means_anchor[:,:,1::2,1::2]

            y_anchor_tmp = torch.index_select(input=y_anchor, dim=1, index=need_encode_ch)
            means_anchor_tmp = torch.index_select(input=means_anchor, dim=1, index=need_encode_ch)
            scales_anchor_tmp = torch.index_select(input=scales_anchor, dim=1, index=need_encode_ch)

            y_anchor_encode[:, :, 0::2, :] = y_anchor_tmp[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor_tmp[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor_tmp[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor_tmp[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor_tmp[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor_tmp[:, :, 1::2, 1::2]

            compress_anchor = time.time()
            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = self.gaussian_conditional.compress(y_anchor_encode, indexes_anchor, means=means_anchor_encode)
            compress_anchor = time.time() - compress_anchor
            y_compress_time += compress_anchor

            anchor_quantized = self.quantizer.quantize(y_anchor_encode-means_anchor_encode, "ste") + means_anchor_encode
            #anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor_encode)
            y_anchor_decode_tmp[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode_tmp[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]
            y_anchor_decode = y_anchor_decode.index_copy(dim=1, index=need_encode_ch, source=y_anchor_decode_tmp)
            check = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor
            check[:,:,0::2,1::2] = 0
            check[:,:,1::2,0::2] = 0
            if len(slice_skip_ch)>0:
                for skip_ch in slice_skip_ch:
                    y_anchor_decode[:, skip_ch, 0::2, 0::2] = means_anchor[:, skip_ch, 0::2, 0::2]
                    y_anchor_decode[:, skip_ch, 1::2, 1::2] = means_anchor[:, skip_ch, 1::2, 1::2]


            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)
                
            y_non_anchor_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(x.device)
            y_non_anchor_encode = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor // 2).to(x.device)
            means_na_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(x.device)
            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor // 2).to(x.device)
            scales_na_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(x.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor // 2).to(x.device)
            y_na_decode_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(x.device)

            means_na_zero = torch.zeros_like(means_non_anchor)
            means_na_zero[:,:,0::2,1::2] = means_non_anchor[:,:,0::2,1::2]
            means_na_zero[:,:,1::2,0::2] = means_non_anchor[:,:,1::2,0::2]

            non_anchor = y_slices[slice_index].clone()
            y_non_anchor_tmp = torch.index_select(input=non_anchor, dim=1, index=need_encode_ch)
            means_na_tmp = torch.index_select(input=means_non_anchor, dim=1, index=need_encode_ch)
            scales_na_tmp = torch.index_select(input=scales_non_anchor, dim=1, index=need_encode_ch)

            y_non_anchor_encode[:, :, 0::2, :] = y_non_anchor_tmp[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = y_non_anchor_tmp[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_na_tmp[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_na_tmp[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_na_tmp[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_na_tmp[:, :, 1::2, 0::2]

            compress_non_anchor = time.time()
            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor_encode, indexes_non_anchor,
                                                                    means=means_non_anchor_encode)
            compress_non_anchor = time.time() - compress_non_anchor
            y_compress_time += compress_non_anchor
            non_anchor_quantized = self.quantizer.quantize(y_non_anchor_encode-means_non_anchor_encode, "ste") + means_non_anchor_encode
            #non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
            #                                                            means=means_non_anchor_encode)

            y_na_decode_tmp[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_na_decode_tmp[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]
            y_na_decode = y_anchor_decode.index_copy(dim=1, index=need_encode_ch, source=y_na_decode_tmp)
            if len(slice_skip_ch)>0:
                for skip_ch in slice_skip_ch:
                    y_na_decode[:, skip_ch, 0::2, 1::2] = means_non_anchor[:, skip_ch, 0::2, 1::2]
                    y_na_decode[:, skip_ch, 1::2, 0::2] = means_non_anchor[:, skip_ch, 1::2, 0::2]
    
            y_slice_encode = torch.round(anchor_part - means_anchor_zero) + torch.round(non_anchor_part - means_na_zero) 
            y_need_encode.append(y_slice_encode)
            y_slice_hat = y_anchor_decode + y_na_decode
            y_hat_slices.append(y_slice_hat)

            y_strings.append([anchor_strings, non_anchor_strings])
            y_encode = torch.cat(y_need_encode, dim=1)

        whole_time = time.time() - whole_time
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec, "y_compress_time":y_compress_time,
                         "z_compress_time":z_compress_time,"whole":whole_time},
                         "latent":y_encode}



    def decompress(self, strings, shape, use_num=320):
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
        group_cumulative = torch.cumsum(torch.tensor(self.groups), dim=0)
        _, sort_indices = torch.sort(self.gain[0],dim=0, descending=True)
        gain_indices = sort_indices.squeeze(-1).squeeze(-1)
        y_skip_ch = gain_indices[use_num:]

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
            slice_skip_ch = [ch_idx - group_cumulative[slice_index] 
                             for ch_idx in y_skip_ch
                             if ch_idx >= group_cumulative[slice_index] and ch_idx < group_cumulative[slice_index+1]]
            need_encode_ch = [ch_idx for ch_idx in range(self.groups[slice_index+1]) if ch_idx not in slice_skip_ch]
            need_encode_ch = torch.tensor(need_encode_ch, dtype=torch.long, device=support.device)

            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()

            means_anchor_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(means_anchor.device)
            means_anchor_encode = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor//2).to(means_anchor.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor // 2).to(means_anchor.device)
            scales_anchor_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(means_anchor.device)
            y_anchor_decode_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(means_anchor.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(means_anchor.device)

            means_anchor_zero = torch.zeros_like(means_anchor)
            means_anchor_zero[:,:,0::2,0::2] = means_anchor[:,:,0::2,0::2]
            means_anchor_zero[:,:,1::2,1::2] = means_anchor[:,:,1::2,1::2]

            means_anchor_tmp = torch.index_select(input=means_anchor, dim=1, index=need_encode_ch)
            scales_anchor_tmp = torch.index_select(input=scales_anchor, dim=1, index=need_encode_ch)
            means_anchor_encode[:, :, 0::2, :] = means_anchor_tmp[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor_tmp[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor_tmp[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor_tmp[:, :, 1::2, 1::2]

            anchor_decompress = time.time()
            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = y_strings[slice_index][0]
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor,
                                                                    means=means_anchor_encode)
            y_anchor_decode_tmp[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode_tmp[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]
            y_anchor_decode = y_anchor_decode.index_copy(dim=1, index=need_encode_ch, source=y_anchor_decode_tmp)
            if len(slice_skip_ch)>0:
                for skip_ch in slice_skip_ch:
                    y_anchor_decode[:, skip_ch, 0::2, 0::2] = means_anchor[:, skip_ch, 0::2, 0::2]
                    y_anchor_decode[:, skip_ch, 1::2, 1::2] = means_anchor[:, skip_ch, 1::2, 1::2]            

            anchor_decompress = time.time() - anchor_decompress
            y_decompress_time += anchor_decompress

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            means_na_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(means_non_anchor.device)
            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor // 2).to(means_non_anchor.device)
            scales_na_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(means_non_anchor.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor // 2).to(means_non_anchor.device)
            y_na_decode_tmp = torch.zeros(B_anchor, C_anchor-len(slice_skip_ch), H_anchor, W_anchor).to(means_non_anchor.device)
            y_na_decode = torch.zeros_like(means_non_anchor)

            means_na_zero = torch.zeros_like(means_non_anchor)
            means_na_zero[:,:,0::2,1::2] = means_non_anchor[:,:,0::2,1::2]
            means_na_zero[:,:,1::2,0::2] = means_non_anchor[:,:,1::2,0::2]

            means_na_tmp = torch.index_select(input=means_non_anchor, dim=1, index=need_encode_ch)
            scales_na_tmp = torch.index_select(input=scales_non_anchor, dim=1, index=need_encode_ch)

            means_non_anchor_encode[:, :, 0::2, :] = means_na_tmp[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_na_tmp[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_na_tmp[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_na_tmp[:, :, 1::2, 0::2]
            
            non_anchor_decompress = time.time()
            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = y_strings[slice_index][1]
            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)
            non_anchor_decompress = time.time() - non_anchor_decompress
            y_decompress_time += non_anchor_decompress

            y_na_decode_tmp[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_na_decode_tmp[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]
            y_na_decode = y_anchor_decode.index_copy(dim=1, index=need_encode_ch, source=y_na_decode_tmp)
            if len(slice_skip_ch)>0:
                for skip_ch in slice_skip_ch:
                    y_na_decode[:, skip_ch, 0::2, 1::2] = means_non_anchor[:, skip_ch, 0::2, 1::2]
                    y_na_decode[:, skip_ch, 1::2, 0::2] = means_non_anchor[:, skip_ch, 1::2, 0::2]

            y_slice_hat = y_anchor_decode + y_na_decode
            y_hat_slices.append(y_slice_hat)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = self.inverse_gain.expand_as(y_hat)*y_hat
        y_dec_start = time.time()
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        y_dec = time.time() - y_dec_start
        whole_time = time.time() - whole_time
        return {"x_hat": x_hat, "time":{"z_dec":z_dec,"y_dec": y_dec, "z_decompress_time":z_decompress_time,"y_decompress_time":y_decompress_time,"whole":whole_time}}
    

if __name__=="__main__":
    """
    checkpoint_path = "/mnt/data3/jingwengu/ELIC_light/hyper_gain/lambda4/checkpoint_last_99.pth.tar"
    state_dict = torch.load(checkpoint_path,map_location="cuda")
    state_dict = state_dict["state_dict"]
    net.load_state_dict(state_dict)
    """
   
    """
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
    """
    model_path = "/mnt/data3/jingwengu/ELIC_light/ELIC_sort/lambda1/noise_last_337.pth.tar"
    model_state = torch.load(model_path, map_location="cuda:1")
    model_cls = ELIC_sort()
    model_cls.load_state_dict(model_state["state_dict"])
    print(model_cls.named_parameters)
    #model_cls.update_indices()
    model_cls.to("cuda:1").eval()
    img_path = "/mnt/data1/jingwengu/kodak/kodim01.png"
    img = Image.open(img_path).convert("RGB")
    trans = transforms.ToTensor()
    img_tensor = trans(img).unsqueeze(0).to("cuda:1")
    criterion = RateDistortionLoss(lmbda=4e-3)
    forward_start = time.time()
    out_forward = model_cls(img_tensor)
    forward_time = time.time() - forward_start
    out_criterion = criterion(out_forward, img_tensor)
    print("forward results")
    print("bpp:{:.5f}".format(out_criterion["bpp_loss"].item()))
    print("mse:{:.5f}".format(out_criterion["mse_loss"].item()))
    print("z_bpp:{:.5f}".format(out_criterion["z_bpp_loss"].item()))
    print("y_bpp:{:.5f}".format(out_criterion["y_bpp_loss"].item()))
    print("total_loss:{:.5f}".format(out_criterion["loss"].item()))
    print("time:{:.5f}\n".format(forward_time))


    forward_start = time.time()
    out_forward = model_cls(img_tensor)
    forward_time = time.time() - forward_start
    out_criterion = criterion(out_forward, img_tensor)
    print("forward results")
    print("bpp:{:.5f}".format(out_criterion["bpp_loss"].item()))
    print("mse:{:.5f}".format(out_criterion["mse_loss"].item()))
    print("z_bpp:{:.5f}".format(out_criterion["z_bpp_loss"].item()))
    print("y_bpp:{:.5f}".format(out_criterion["y_bpp_loss"].item()))
    print("norm:{:.5f}".format(out_criterion["norm"].item()))
    print("total_loss:{:.5f}".format(out_criterion["loss"].item()))
    print("time:{:.5f}\n".format(forward_time))


    inference_start = time.time()
    out_inference = model_cls.inference(img_tensor,300)
    inference_time = time.time() - inference_start
    inference_criterion = criterion(out_inference,img_tensor)
    print("inference results")
    print("bpp:{:.5f}".format(inference_criterion["bpp_loss"].item()))
    print("mse:{:.5f}".format(inference_criterion["mse_loss"].item()))
    print("z_bpp:{:.5f}".format(inference_criterion["z_bpp_loss"].item()))
    print("y_bpp:{:.5f}".format(inference_criterion["y_bpp_loss"].item()))
    print("norm:{:.5f}".format(inference_criterion["norm"].item()))
    print("total_loss:{:.5f}".format(inference_criterion["loss"].item()))
    print("time:{:.5f}\n".format(inference_time))