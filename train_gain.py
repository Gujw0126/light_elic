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
import pdb

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from MyUtils.datasets import ImageFolder
from typing import List, Dict

from tensorboardX import SummaryWriter
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
from MyUtils.utils.utils import DelfileList, load_checkpoint
from teacher_models import*
import logging
from skip_models import HyperGain, GainSkip
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

my_logger = logging.getLogger(__name__)



class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["y_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["y"]
        )
        out["z_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["z"]
        )
        out["mse"] = self.mse(output["x_hat"], target) * 255 ** 2
        out["loss"] = lmbda * out["mse"] + out["bpp"]

        return out


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


def train_one_epoch(
        model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger, lambdas, noisequant=True
):
    model.train()
    device = next(model.parameters()).device
    lambda_len = len(lambdas)
    train_loss = [AverageMeter() for _ in range(lambda_len)]
    train_mse = [AverageMeter() for _ in range(lambda_len)]
    train_bpp = [AverageMeter() for _ in range(lambda_len)]
    train_y_bpp = [AverageMeter() for _ in range(lambda_len)]
    train_z_bpp = [AverageMeter() for _ in range(lambda_len)]

    start = time.time()
    for i, d in enumerate(train_dataloader):
        s = random.randint(0,lambda_len-1)
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d, noisequant)
        out_criterion = criterion(out_net, d, lambdas[s])

        train_loss[s].update(out_criterion["loss"].item())
        train_mse[s].update(out_criterion["mse"].item())
        train_bpp[s].update(out_criterion["bpp"].item())
        train_y_bpp[s].update(out_criterion["y_bpp"].item())
        train_z_bpp[s].update(out_criterion["z_bpp"].item())

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        #TODO: auxloss
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        if i % 8000 == 0:
            info = f"Train epoch {epoch}: ["\
                f"{i*len(d)}/{len(train_dataloader.dataset)}"\
                f" ({100. * i / len(train_dataloader):.0f}%)]"\
                f'\tlambda={lambdas[s]:.4f}|'\
                f'\tLoss: {out_criterion["loss"].item():.4f}|'\
                f'\tMSE loss: {out_criterion["mse"].item():.4f}|'\
                f'\tBpp loss: {out_criterion["bpp"].item():.4f}|'\
                f'\ty_bpp loss: {out_criterion["y_bpp"].item():.4f}|'\
                f'\tz_bpp loss: {out_criterion["z_bpp"].item():.4f}|'\
                f'\tAux loss: {aux_loss.item():.4f}'
            print(info)
            logger.info(info)
    
    train_loss_avg = [meter.avg for meter in train_loss]
    train_mse_avg = [meter.avg for meter in train_mse]
    train_bpp_avg = [meter.avg for meter in train_bpp]
    train_y_avg = [meter.avg for meter in train_y_bpp]
    train_z_avg = [meter.avg for meter in train_z_bpp]

    info = f"Train epoch {epoch}:"\
        f"\tLoss: {sum(train_loss_avg):.4f}"\
        f"\tMSE loss: {sum(train_mse_avg):.4f}"\
        f'\tBpp loss: {sum(train_bpp_avg):.4f}'\
        f'\ty_bpp loss: {sum(train_y_avg):.4f}'\
        f'\tz_bpp loss: {sum(train_z_avg):.4f}'\
        f"\tTime(s) :{time.time() - start:.4f}"
    
    print(info)
    logger.info(info)
    return train_loss_avg, train_mse_avg, train_bpp_avg, train_y_avg, train_z_avg


def test_epoch(
        model, criterion, test_dataloader, epoch, logger, lambdas ,noisequant=True
):
    model.eval()
    device = next(model.parameters()).device
    lambda_len = len(lambdas)
    test_loss = [AverageMeter() for _ in range(lambda_len)]
    test_mse = [AverageMeter() for _ in range(lambda_len)]
    test_bpp = [AverageMeter() for _ in range(lambda_len)]
    test_y_bpp = [AverageMeter() for _ in range(lambda_len)]
    test_z_bpp = [AverageMeter() for _ in range(lambda_len)]

    with torch.no_grad():
        start = time.time()
        for i, d in enumerate(test_dataloader):
            s = random.randint(0,lambda_len-1)
            d = d.to(device)
            out_net = model(d, noisequant)
            out_criterion = criterion(out_net, d, lambdas[s])           
            test_loss[s].update(out_criterion["loss"].item())
            test_bpp[s].update(out_criterion["bpp"].item())
            test_mse[s].update(out_criterion["mse"].item())
            test_y_bpp[s].update(out_criterion["y_bpp"])
            test_z_bpp[s].update(out_criterion["z_bpp"])
            aux_loss = model.aux_loss()

    test_loss_avg = [meter.avg for meter in test_loss]
    test_mse_avg = [meter.avg for meter in test_mse]
    test_bpp_avg = [meter.avg for meter in test_bpp]
    test_y_avg = [meter.avg for meter in test_y_bpp]
    test_z_avg = [meter.avg for meter in test_z_bpp]

    info = f"Test epoch {epoch}:"\
        f"\tLoss:{sum(test_loss_avg):.4f}"\
        f"\tMSE loss:{sum(test_mse_avg):.4f}"\
        f'\tBpp loss:{sum(test_bpp_avg):.4f}'\
        f'\ty_bpp loss:{sum(test_y_avg):.4f}'\
        f'\tz_bpp loss:{sum(test_z_avg):.4f}'\
        f"\tTime(s) :{time.time() - start:.4f}"
    print(info)
    logger.info(info)
    return test_loss_avg, test_mse_avg, test_bpp_avg, test_y_avg, test_z_avg


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
        "-e",
        "--epochs",
        default=1000,
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
        "--lambdas",
        dest="lmbdas",
        type=float,
        nargs=8,
        default=(0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.18),
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
        default=(256, 256),
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
    discrete_num = len(args.lmbdas)
    net = HyperGain(N=192,M=320, r_num=discrete_num)
    print(net.M)
    criterion = RateDistortionLoss()
    #TODO:other octave models
    net = net.to(device)
    if not os.path.exists(args.savepath):
        try:
            os.mkdir(args.savepath)
        except:
            os.makedirs(args.savepath)
    writer = SummaryWriter(args.savepath)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3800], gamma=0.1)
    
    last_epoch = 0

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    stemode = False ##set the pretrained flag
    if args.checkpoint and args.pretrained:
        optimizer.param_groups[0]['lr'] = args.learning_rate
        aux_optimizer.param_groups[0]['lr'] = args.aux_learning_rate
        del lr_scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=10)
        last_epoch = 0
        stemode = True

    noisequant = True
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        if epoch > 1000 or stemode:
            noisequant = False
        print("noisequant: {}, stemode:{}".format(noisequant, stemode))
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss, train_mse, train_bpp, train_y_bpp, train_z_bpp = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            my_logger,
            args.lmbdas,
            noisequant
        )

        for i, lmbda in enumerate(args.lmbdas):
            writer.add_scalar('Train/loss_lambda={}'.format(lmbda), train_loss[i], epoch)
            writer.add_scalar('Train/mse_lambda={}'.format(lmbda), train_mse[i], epoch)
            writer.add_scalar('Train/bpp_lambda={}'.format(lmbda), train_bpp[i], epoch)
            writer.add_scalar('Train/y_bpp_lambda={}'.format(lmbda), train_y_bpp[i], epoch)
            writer.add_scalar('Train/z_bpp_lambda={}'.format(lmbda), train_z_bpp[i], epoch)

        test_loss, test_mse, test_bpp, test_y_bpp, test_z_bpp = test_epoch(
            net, 
            criterion,
            test_dataloader,
            epoch, 
            my_logger,
            args.lmbdas, 
            noisequant
            )

        for i, lmbda in enumerate(args.lmbdas):
            writer.add_scalar('Test/loss_lambda={}'.format(lmbda), test_loss[i], epoch)
            writer.add_scalar('Test/mse_lambda={}'.format(lmbda), test_mse[i], epoch)
            writer.add_scalar('Test/bpp_lambda={}'.format(lmbda), test_bpp[i], epoch)
            writer.add_scalar('Test/y_bpp_lambda={}'.format(lmbda), test_y_bpp[i], epoch)
            writer.add_scalar('Test/z_bpp_lambda={}'.format(lmbda), test_z_bpp[i], epoch)
        
        total_test_loss = sum(test_loss)
        lr_scheduler.step(total_test_loss)

        is_best = total_test_loss < best_loss
        best_loss = min(total_test_loss, best_loss)

        if args.save:
            DelfileList(args.savepath, "checkpoint_last")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": total_test_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                filename=os.path.join(args.savepath, "checkpoint_last_{}.pth.tar".format(epoch))
            )
            if is_best:
                DelfileList(args.savepath, "checkpoint_best")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": total_test_loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    filename=os.path.join(args.savepath, "checkpoint_best_loss_{}.pth.tar".format(epoch))
                )

if __name__ == "__main__":
    main(sys.argv[1:])
