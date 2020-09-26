import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

def compute_KL_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    loss = 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
    return loss 

def compute_reconstruct_loss(x_real, x_recon):
    loss = F.mse_loss(x_real, x_recon)
    return loss