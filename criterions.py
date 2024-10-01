import torch
import torch.nn as nn
import math

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


#TODO:distangle loss
class SpatialDistangleLoss(nn.Module):
    def __init__(self, lmbda, alpha, window_size):
        super().__init__()
        self.window_size = window_size
        self.rd_loss = RateDistortionLoss(lmbda)
        self.alpha = alpha

    def forward(self, output, target):
        """
        output:{"likelihoods":{"y":y_likelihoods,"z":z_likelihoods}
                "x_hat":x_hat,
                "latent":}
        
        """
        rd_loss = self.rd_loss(output, target)
        latent_norm = output["latent_norm"]
        



        return out

class ChannelDistangleLoss(nn.Module):
    pass


class BiDistangleLoss(nn.Module):
    pass
