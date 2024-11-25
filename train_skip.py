# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from MyUtils.datasets import ImageFolder

from tensorboardX import SummaryWriter
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
from MyUtils.utils.utils import DelfileList, load_checkpoint
from teacher_models import*
import logging
from skip_models import*
my_logger = logging.getLogger(__name__)



#prediction loss and todo:distangle loss
class RateDistortionLoss2(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2, alpha=1):
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
        out["slice_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["slice"]
        )

        #out["slice_bpp_loss"] = torch.clamp(out["slice_bpp_loss"],0.001)
        out["pred_loss"] = 0
        for i in range(len(output["preds"])):
            out["pred_loss"] += torch.abs(output["y"][i] - output["preds"][i]).sum()/num_pixels

        out["mse_loss"] = self.mse(output["x_hat"], target) * 255 ** 2
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"] + self.alpha*out["pred_loss"]
        return out


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

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
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"] 
        return out


class WarmUpLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2, alpha=1):
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
        #out["slice_bpp_loss"] = torch.clamp(out["slice_bpp_loss"],0.001)
        out["pred_loss"] = 0
        for i in range(len(output["preds"])):
            out["pred_loss"] += torch.abs(output["y"][i] - output["preds"][i]).sum()/num_pixels

        out["mse_loss"] = self.mse(output["x_hat"], target) * 255 ** 2
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"] + self.alpha*out["pred_loss"]
        return out

#TODO:distangle loss
class SpatialDistangleLoss(nn.Module):
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size=window_size
        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size//2, stride=1)
    
    def forward(self,y_norm:Tensor):
        """
        y_norm:latent y normalized by miu and sigma
        """
        B,C,H,W = y_norm.shape
        y_unfold = self.unfold(y_norm).reshape(B,C,self.window_size,self.window_size,H,W).permute(0,1,4,5,2,3)#y_unfold[B,C,H,W,self.window, self.window]
        result = torch.zeros_like(y_unfold)
        for i in range(self.window_size):
            for j in range(self.window_size):
                result[:,:,:,:,i,j] = torch.abs(y_norm*y_unfold[:,:,:,:,i,j])
        relation_map = torch.sum(result,dim=1)  #[B,H,W,self.window, self.window]
        relation_map[:,:,:,self.window_size//2, self.window_size//2] = 0  #except myself
        relation_map = torch.sum(torch.sum(relation_map,dim=-1),dim=-1)/self.window_size**2
        loss = torch.sum(relation_map)/(H*W*B)
        return loss
                
class RateDistortionLoss_S(nn.Module):
    """
    rate-distortion loss with spatial distangle loss 
    """
    def __init__(self, lmbda, beta, window):
        """
        lmbda:rate-distortion tradeoff
        beta:rate-distortion and spatial-distortion tradeoff
        """
        super().__init__()
        self.lmbda = lmbda
        self.beta = beta
        self.mse = nn.MSELoss()
        self.spatial = SpatialDistangleLoss(window_size=window)

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
        out["spatial_loss"] = self.spatial(output["y_norm"])
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"] + self.beta*out["spatial_loss"]
        return out


class ChannelDistangleLoss(nn.Module):
    pass


class BiDistangleLoss(nn.Module):
    pass

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate, betas=(0.9, 0.999),
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate, betas=(0.9, 0.999),
    )
    return optimizer, aux_optimizer


def configure_finetune_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if (not n.endswith(".quantiles") and p.requires_grad)
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate, betas=(0.9, 0.999),
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate, betas=(0.9, 0.999),
    )
    return optimizer, aux_optimizer


def warm_up_one_epoch(
        model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger
):
    model.train()
    device = next(model.parameters()).device
    train_loss = AverageMeter()
    train_bpp_loss = AverageMeter()
    train_y_bpp_loss = AverageMeter()
    train_z_bpp_loss = AverageMeter()
    train_mse_loss = AverageMeter()
    train_pred_loss = AverageMeter()

    start = time.time()
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model.warm_up(d)
        out_criterion = criterion(out_net, d)
        train_bpp_loss.update(out_criterion["bpp_loss"].item())
        train_y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
        train_z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
        train_loss.update(out_criterion["loss"].item())
        train_mse_loss.update(out_criterion["mse_loss"].item())
        train_pred_loss.update(out_criterion["pred_loss"].item())

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10000 == 0:
            info = f"Train epoch {epoch}: ["\
                f"{i*len(d)}/{len(train_dataloader.dataset)}"\
                f" ({100. * i / len(train_dataloader):.0f}%)]"\
                f'\tLoss: {out_criterion["loss"].item():.3f}|'\
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f}|'\
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f}|'\
                f'\ty_Bpp loss: {out_criterion["y_bpp_loss"].item():.4f}|'\
                f'\tz_Bpp loss: {out_criterion["z_bpp_loss"].item():.4f}|'\
                f'\tpred loss:{out_criterion["pred_loss"].item():.4f}|'
            print(info)
            logger.info(info)

    info = f"Train epoch {epoch}: Average losses:"\
        f"\tLoss: {train_loss.avg:.3f}|"\
        f"\tMSE loss: {train_mse_loss.avg:.3f}|"\
        f"\tBpp loss: {train_bpp_loss.avg:.4f}|"\
        f"\ty_Bpp loss: {train_y_bpp_loss.avg:.5f}|"\
        f"\tz_Bpp loss: {train_z_bpp_loss.avg:.5f}|"\
        f"\tpred loss:{train_pred_loss.avg:.5f}|"\
        f"\tTime (s) : {time.time()-start:.4f}" 
    print(info)
    logger.info(info)
    return train_loss.avg, train_bpp_loss.avg, train_mse_loss.avg, train_pred_loss.avg
    
def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger, noisequant, temp
):
    model.train()
    device = next(model.parameters()).device
    train_loss = AverageMeter()
    train_bpp_loss = AverageMeter()
    train_y_bpp_loss = AverageMeter()
    train_z_bpp_loss = AverageMeter()
    train_mse_loss = AverageMeter()
    train_spatial_loss = AverageMeter()

    start = time.time()
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        if noisequant:
            out_net = model.soft(d, temp)
        else:
            out_net = model.hard(d)

        out_criterion = criterion(out_net, d)
        train_bpp_loss.update(out_criterion["bpp_loss"].item())
        train_y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
        train_z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
        train_loss.update(out_criterion["loss"].item())
        train_mse_loss.update(out_criterion["mse_loss"].item())
        if "spatial_loss" in out_criterion.keys():
            train_spatial_loss.update(out_criterion["spatial_loss"].item())
        

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10000 == 0:
            info = f"Train epoch {epoch}: ["\
                f"{i*len(d)}/{len(train_dataloader.dataset)}"\
                f" ({100. * i / len(train_dataloader):.0f}%)]"\
                f"\ttemp: {temp:.4f}|"\
                f'\tLoss: {out_criterion["loss"].item():.3f}|'\
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f}|'\
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f}|'\
                f'\ty_Bpp loss: {out_criterion["y_bpp_loss"].item():.4f}|'\
                f'\tz_Bpp loss: {out_criterion["z_bpp_loss"].item():.4f}|'
            if "spatial_loss" in out_criterion.keys():
                info += f'\tspatial loss:{out_criterion["spatial_loss"].item():.4f}'            
            print(info)
            logger.info(info)

    info = f"Train epoch {epoch}: Average losses:"\
        f"\tLoss: {train_loss.avg:.3f}|"\
        f"\tMSE loss: {train_mse_loss.avg:.3f}|"\
        f"\tBpp loss: {train_bpp_loss.avg:.4f}|"\
        f"\ty_Bpp loss: {train_y_bpp_loss.avg:.5f}|"\
        f"\tz_Bpp loss: {train_z_bpp_loss.avg:.5f}|"\
        f'\tspatial loss:{train_spatial_loss.avg:.4f}'\
        f"\tTime (s) : {time.time()-start:.4f}" 

    print(info)
    logger.info(info)
    return train_loss.avg, train_bpp_loss.avg, train_mse_loss.avg, train_spatial_loss.avg


def test_epoch(epoch, test_dataloader, model, criterion, logger,noisequant,temp):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    y_bpp_loss = AverageMeter()
    z_bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    spatial_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            if noisequant:
                out_net = model.soft(d,temp)
            else:
                out_net = model.hard(d)
            out_criterion = criterion(out_net, d)
            aux_loss.update(model.aux_loss().item())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
            z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
            loss.update(out_criterion["loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())
            if "spatial_loss" in out_criterion.keys():
                spatial_loss.update(out_criterion["spatial_loss"].item())
    
        info =  f"Test epoch {epoch}: Average losses:"\
            f"\tLoss: {loss.avg:.3f}|"\
            f"\tMSE loss: {mse_loss.avg:.3f}|"\
            f"\tBpp loss: {bpp_loss.avg:.4f}|"\
            f"\ty_Bpp loss: {y_bpp_loss.avg:.4f}|"\
            f"\tz_Bpp loss: {z_bpp_loss.avg:.4f}|"\
            f"\tspatial loss: {spatial_loss.avg:.4f}"\
            f"\tAux loss: {aux_loss.avg:.4f}\n" 
        
    print(info)
    logger.info(info)
    return loss.avg, bpp_loss.avg, mse_loss.avg, spatial_loss.avg


def warm_up_epoch(epoch, test_dataloader, model, criterion, logger):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    y_bpp_loss = AverageMeter()
    z_bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    pred_loss = AverageMeter()
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            aux_loss.update(model.aux_loss().item())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
            z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
            loss.update(out_criterion["loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())
            pred_loss.update(out_criterion["pred_loss"].item())
    
        info =  f"Test epoch {epoch}: Average losses:"\
            f"\tLoss: {loss.avg:.3f}|"\
            f"\tMSE loss: {mse_loss.avg:.3f}|"\
            f"\tBpp loss: {bpp_loss.avg:.4f}|"\
            f"\ty_Bpp loss: {y_bpp_loss.avg:.4f}|"\
            f"\tz_Bpp loss: {z_bpp_loss.avg:.4f}|"\
            f"\tpred loss: {pred_loss.avg:.4f}"\
            f"\tAux loss: {aux_loss.avg:.4f}\n" 
    print(info)
    logger.info(info)
    return loss.avg, bpp_loss.avg, mse_loss.avg, pred_loss.avg

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "--N",
        default=192,
        type=int,
        help="Number of channels of main codec",
    )
    parser.add_argument(
        "--M",
        default=320,
        type=int,
        help="Number of channels of latent",
    )
    parser.add_argument(
        "--beta",
        default=1e-13,
        type=float
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=4000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=15e-3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", default=1926, type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="use the pretrain model to refine the models",
    )
    parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--savepath', default='./checkpoint', type=str, help='Path to save the checkpoint')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--spatial", action="store_true")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    log_file = os.path.join(args.savepath, "train.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    
    net = AnnelModel()
    if args.spatial:
        criterion = RateDistortionLoss_S(lmbda=args.lmbda, beta=args.beta,window=5)
    else:
        criterion = RateDistortionLoss(lmbda=args.lmbda)

    #criterion = RateDistortionLoss(lmbda=args.lmbda)

    #TODO:other octave models
    net = net.to(device)
    if not os.path.exists(args.savepath):
        try:
            os.mkdir(args.savepath)
        except:
            os.makedirs(args.savepath)
    writer = SummaryWriter(args.savepath)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    
    if args.finetune:
        premodel = torch.load(args.premodel, map_location=device)
        net_state_dict = net.state_dict()   
        for key in premodel:
            if key.startswith("g_a") or key.startswith("g_s"):
                net_state_dict[key] = premodel[key]
                net_state_dict[key].requires_grad = False
        net.load_state_dict(net_state_dict)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=8)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3800], gamma=0.1)
    temp = 2.0
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        temp = checkpoint["temp"]

    stemode = False ##set the pretrained flag
    if args.pretrained:
        stemode = True
    if args.checkpoint and args.pretrained:
        optimizer.param_groups[0]['lr'] = args.learning_rate
        aux_optimizer.param_groups[0]['lr'] = args.aux_learning_rate
        del lr_scheduler
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=10)
        last_epoch = 0
        stemode = True

    noisequant = True
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        if temp>0.1:
            if epoch % 50==0 and epoch>0:
                temp *= 0.9
        if epoch > 3800 or stemode:
            noisequant = False
        print("noisequant: {}, stemode:{}".format(noisequant, stemode))
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"temp: {temp:.5f}")
        #assert isinstance(criterion, RateDistortionLoss2)
        """
        if args.warmup:
            train_loss, train_bpp, train_mse, train_pred = warm_up_one_epoch(
                net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, my_logger
            )
            writer.add_scalar('Train/loss', train_loss, epoch)
            writer.add_scalar('Train/mse', train_mse, epoch)
            writer.add_scalar('Train/bpp', train_bpp, epoch)
            writer.add_scalar('Train/pred', train_pred, epoch)
            loss, bpp, mse, pred = warm_up_epoch(epoch, test_dataloader, net, criterion, my_logger)
            writer.add_scalar('Test/loss', loss, epoch)
            writer.add_scalar('Test/mse', mse, epoch)
            writer.add_scalar('Test/bpp', bpp, epoch)
            writer.add_scalar('Test/pred',pred, epoch)
        """
        train_loss, train_bpp, train_mse, train_spatial = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            my_logger,
            noisequant,
            temp
        )
        loss, bpp, mse, spatial = test_epoch(epoch, test_dataloader, net, criterion, my_logger,noisequant,temp)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            DelfileList(args.savepath, "checkpoint_last")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "temp":temp
                },
                filename=os.path.join(args.savepath, "checkpoint_last_{}.pth.tar".format(epoch))
            )
            if is_best:
                DelfileList(args.savepath, "checkpoint_best")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "temp":temp
                    },
                    filename=os.path.join(args.savepath, "checkpoint_best_loss_{}.pth.tar".format(epoch))
                )

if __name__ == "__main__":
    main(sys.argv[1:])
