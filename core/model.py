import argparse
import os
import numpy as np
import math
import itertools
from munch import Munch

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

# ten = tensor
# encoder block (used in encoder and discriminator)
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, tensor, out=False, t = False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            tensor = self.conv(tensor)
            tensor_out = tensor
            tensor = self.bn(tensor)
            tensor = F.relu(tensor, False)
            return tensor, tensor_out
        else:
            tensor = self.conv(tensor)
            tensor = self.bn(tensor)
            tensor = F.relu(tensor, True)
            return tensor


# decoder block (used in the decoder)
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        # transpose convolution to double the dimensions
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, output_padding=1,
                                       bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, tensor):
        tensor = self.conv(tensor)
        tensor = self.bn(tensor)
        tensor = F.relu(tensor, True)
        return tensor


class Encoder(nn.Module):
    def __init__(self, img_size:int=256, 
                       channel_in:int=32, 
                       target_size:int=8,
                       max_channel:int=256,
                       latent_dim:int=64):
        super().__init__()
        repeat_num = int(np.log2(img_size) - np.log2(target_size)) - 1
        blocks = []
        # the first time 3->64, for every other double the channel size
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        blocks += [nn.Conv2d(3, channel_in, 5, 1, 2)]
        blocks += [nn.BatchNorm2d(channel_in)]
        blocks += [nn.LeakyReLU(0.2)]
        for _ in range(repeat_num):
            channel_out = min(2*channel_in, max_channel)
            blocks += [EncoderBlock(channel_in=channel_in, channel_out=channel_out)]
            channel_in = channel_out
        # final shape Bx256x8x8
        blocks += [nn.Conv2d(channel_out, channel_out, 3, 2, 1)]
        blocks += [nn.BatchNorm2d(channel_out)]
        blocks += [nn.LeakyReLU(0.2)]

        
        self.main = nn.Sequential(*blocks)
        self.fc = nn.Sequential(nn.Linear(in_features=target_size * target_size * channel_out, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.LeakyReLU(0.2))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=1024, out_features=latent_dim)
        self.l_var = nn.Linear(in_features=1024, out_features=latent_dim)

    def forward(self, ten):
        ten = self.main(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, img_size:int=256, 
                       channel_in:int=256, 
                       start_size:int=8,
                       min_channel:int=16,
                       latent_dim:int=64):
        super(Decoder, self).__init__()
        self.start_size = start_size
        # start from B*latent_dim
        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=start_size * start_size * channel_in, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * channel_in, momentum=0.9),
                                nn.ReLU(True))
        repeat_num = int(np.log2(img_size) - np.log2(start_size)) - 1
        blocks = []
        blocks += [DecoderBlock(channel_in=channel_in, channel_out=channel_in)]
        for _ in range(repeat_num):
            channel_out = max(channel_in//2, min_channel)
            blocks += [DecoderBlock(channel_in=channel_in, channel_out=channel_out)]
            channel_in = channel_out

        # final conv to get 3 channels and tanh layer
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*blocks)

    def forward(self, ten):

        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, self.start_size, self.start_size)
        ten = self.conv(ten)
        return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


def build_nets(args):
    encoder = Encoder(img_size=args.img_size,
                      channel_in=args.encoder_channel_in,
                      target_size=args.target_size,
                      max_channel=args.max_channel,
                      latent_dim=args.latent_dim)
    decoder = Decoder(img_size=args.img_size,
                      channel_in=args.decoder_channel_in,
                      start_size=args.start_size,
                      min_channel=args.min_channel,
                      latent_dim=args.latent_dim)
    nets = Munch(encoder=encoder, decoder=decoder)
    return nets

# if __name__ == "__main__":
#     args = Munch(img_size=256, channel_in=32, target_size=8, max_channel=256, latent_dim=64, start_size=8, min_channel=16)
#     nets = build_nets(args)
#     for name, module in nets.items():
#         print(name, module)
