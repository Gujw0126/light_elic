#show feature maps in a model
import torch
import torch.nn as nn
from teacher_models import*
import matplotlib.pyplot as plt
from skip_models import SKModel, AnnelModel, HyperGain
from compressai.compressai.zoo import load_state_dict
from torch import Tensor
from PIL import Image
import numpy as np
import os
import argparse
import torchvision.transforms as transforms
import math
import sys

model_generate = {"ELICHyper":ELICHyper, "SKModel":SKModel,"ELIC":ELIC_original, "Annel":AnnelModel}
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return -10 * math.log10(mse)

def plot_latent(latent:Tensor, savepath):
    B,C,H,W = latent.shape
    all_max = torch.max(torch.abs(latent)).detach().cpu().numpy()
    latent = torch.abs(latent)
    for ch in range(C):
        latent_map = latent[0,ch,:,:].detach().cpu().numpy()
        #latent_map_norm = latent_map/np.max(latent_map)
        plt.figure(figsize=(8,4))
        plt.imshow(latent_map/all_max,vmin=0,vmax=1)
        plt.colorbar()
        savename = os.path.join(savepath,"ch{}.png".format(ch))
        plt.savefig(savename)
        plt.close()

def plot_latent_sort(latent:Tensor, savepath, sort_index):
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


def set_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dir",type=str, required=True, help="path to test image(one image)")
    parser.add_argument("-m","--model",type=str,choices=["SKModel","ELICHyper","Annel"])
    parser.add_argument("-p","--path",type=str, help="model path")
    parser.add_argument("-s","--savepath",type=str, required=True)
    return parser.parse_args(argv)


def PCA_analysis(data:Tensor, save, savepath=None):
    """
    data: tensor[1,C,H,W]
    save: save or not
    savepath: path to save results
    """
    

def main(argv):
    args = set_args(argv)
    net = HyperGain()
    state_dict = load_state_dict(torch.load(args.path,map_location="cuda")) 
    net = net.from_state_dict(state_dict).eval().to("cuda")
    img = Image.open(args.dir).convert("RGB")
    trans = transforms.ToTensor()
    img = trans(img).unsqueeze(0)
    B,C,H,W = img.shape
    new_h = math.ceil(H/64)*64
    new_w = math.ceil(W/64)*64
    padding_left = (new_w - W)//2
    padding_right = new_w - W - padding_left
    padding_top = (new_h - H)//2
    padding_down = new_h - H - padding_top
    img_pad = nn.functional.pad(
        img,
        [padding_left, padding_right, padding_top, padding_down],
        mode="constant",
        value=0
    ).to("cuda")

    gain, sort_indexs = torch.sort(net.gain[0],dim=0, descending=True)
    out_compress = net.compress(img_pad)
    plot_latent_sort(out_compress["latent_enc"],savepath=args.savepath,sort_index=sort_indexs)
    """
    psnr_list = np.zeros(320)
    count = 0
    with torch.no_grad():
        for ch in sort_indexs:
            psnr_list[count]=psnr(net.mask_inference(img_pad, ch), img_pad)
            count+=1
        
    x_axis = np.arange(320)
    psnr_min = np.min(psnr_list)
    plt.figure(figsize=(160,8))
    plt.bar(x=x_axis, height=psnr_list-psnr_min, width=0.8)
    plt.xticks(range(0,320,1))
    savename = os.path.join(args.savepath,"psnr_mask.png")
    plt.savefig(savename)
    """

    print("gain shape:{}".format(gain.shape))
    for i in range(320):
        print("ch: {}, gain:{}".format(sort_indexs[i], gain[i].data))
    """
    #calculate channel entropy
    entropy = -torch.log(out_forward["likelihoods"]["y"])/math.log(2)
    total_entropy = torch.sum(-torch.log(out_forward["likelihoods"]["y"])/math.log(2))/H/W
    print("total_entropy: {:.4f}".format(total_entropy))
    channel_entropy = torch.sum(torch.sum(entropy,dim=-1),dim=-1)
    for ch_idx in range(320):
        print("channel{}: entropy {:.4f}".format(ch_idx, channel_entropy[0,ch_idx]))
    print("test: {:.4f}".format(torch.sum(channel_entropy)/H/W))
    assert torch.sum(channel_entropy)/H/W == total_entropy
    """

if __name__=="__main__":
    main(sys.argv[1:])


        