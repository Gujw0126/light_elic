#show feature maps in a model
import torch
import torch.nn as nn
from teacher_models import*
import matplotlib.pyplot as plt
from skip_models import SKModel, AnnelModel, HyperGain, HyperSkip
from compressai.compressai.zoo import load_state_dict
from torch import Tensor
from PIL import Image
import numpy as np
import os
import argparse
import torchvision.transforms as transforms
import math
import sys

#对elic_sort, elic_hyper_gain提供某张图像encode内容， 通道熵， 每个通道的重要性， 前n个通道的熵和psnr
def plot_latent(latent:Tensor, savename, savepath):
    """
    画latent的绝对值的每个channel,分通道存在savepath里面
    """
    B,C,H,W = latent.shape
    all_max = torch.max(torch.abs(latent)).detach().cpu().numpy()
    latent = torch.abs(latent)
    for ch in range(C):
        latent_map = latent[0,ch,:,:].detach().cpu().numpy()
        #latent_map_norm = latent_map/np.max(latent_map)
        plt.figure(figsize=(8,4))
        plt.imshow(latent_map/all_max,vmin=0,vmax=1)
        plt.colorbar()
        savename = os.path.join(savepath, savename + "ch{}.png".format(ch))
        plt.savefig(savename)
        plt.close()


def plot_latent_sort(latent:Tensor, savepath, sort_index):
    """
    画latent的绝对值得每个channel, 按照gain排序， 存在savepath里面
    """
    B,C,H,W = latent.shape
    sort_indices = sort_index.squeeze(-1).squeeze(-1)
    all_max = torch.max(torch.abs(latent)).detach().cpu().numpy()
    latent = torch.abs(latent)
    count = 0
    for ch in sort_indices:
        latent_map = latent[0,ch,:,:].detach().cpu().numpy()
        #latent_map_norm = latent_map/np.max(latent_map)
        plt.figure(figsize=(8,4))
        plt.imshow(latent_map/all_max, vmin=0, vmax=1)
        plt.colorbar()
        savename = os.path.join(savepath,"sort{}_index{}.png".format(count, ch))
        count+=1
        plt.savefig(savename)
        plt.close()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def mask_reference(model, img_tensor, start_num, savepath):
    rate_list = []
    psnr_list = []
    compress_time_list = []
    decompress_time_list = []
    for use_num in range(start_num, 320):
        out_inference = model.inference(img_tensor, use_num)
        B,C,H,W = img_tensor
        pixel_num = B*H*W
        rate=sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixel_num))
            for likelihoods in out_inference["likelihoods"].values()
        )
        psnr_list.append(psnr(out_inference["x_hat"], img_tensor))
        rate_list.append(rate)
        compress_time_list.append(out_inference["compress_time"])
        decompress_time_list.append(out_inference["decompress_time"])
    
    #画5个图
    #use_num和rate之间的关系
    #use_num和psnr之间的关系
    #use_num和rd之间的关系
    #use_num和两个time之间的关系
    rate_list = np.array(rate_list)
    psnr_list = np.array(psnr_list)
    compress_time_list = np.array(compress_time_list)
    decompress_time_list = np.array(decompress_time_list)

    use_nums = np.arange(start=start_num, stop=320)
    #use_num和rate之间的关系
    plt.figure(figsize=(8,4))
    plt.title("rate_use_num")
    plt.bar(x=use_nums, height=rate_list)
    plt.savefig(os.path.join(savepath, "rate.png"))
    plt.close()

    #use_num和psnr之间的关系
    plt.figure(figsize=(8,4))
    plt.title("psnr_use_num")
    plt.bar(x=use_nums, height=psnr_list)
    plt.savefig(os.path.join(savepath, "psnr.png"))
    plt.close()

    #use_num和compress_time之间的关系
    plt.figure(figsize=(8,4))
    plt.title("compress_time_use_num")
    plt.bar(x=use_nums, height=compress_time_list)
    plt.savefig(os.path.join(savepath, "compress_time.png"))
    plt.close()

    #use_num和decompress_time之间的关系
    plt.figure(figsize=(8,4))
    plt.title("decompress_time_use_num")
    plt.bar(x=use_nums, height=decompress_time_list)
    plt.savefig(os.path.join(savepath, "decompress.png"))
    plt.close()