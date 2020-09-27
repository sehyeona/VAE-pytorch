import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join as ospj
from pathlib import Path
import numpy as np
from munch import Munch
from PIL import Image
import time
from copy import deepcopy

from torchvision import transforms
from torchvision.utils import save_image
from functools import wraps
from torchsummary import summary
import matplotlib.pyplot as plt

from core.model import build_nets
from core.checkpoint import CheckpointIO
from core.utils import tensor2ndarray255


## decorator for time checker
def checkLoggingTime(func):
    @wraps(func)
    def wrapper(*arg, **kwargs):
        start = time.time()
        result = func(*arg, **kwargs)
        end = time.time()
        print("{:.2f}".format(end-start))
        return result
    return wrapper

class Vectorization(object):
    @torch.no_grad()
    def __init__(self, img_size, target_size, start_size, latent_dim, resume_iter, checkpoint_dir):
        args = Munch()
        args.img_size = img_size
        args.encoder_channel_in = 32
        args.decoder_channel_in = 256
        args.max_channel = 256
        args.min_channel = 16
        args.target_size = 8
        args.start_size = 8
        args.latent_dim = 64
        args.resume_iter = resume_iter
        args.checkpoint_dir = checkpoint_dir
        self.args = args
        self.nets = build_nets(args)
        

        # below setattrs are to make networks be children of Vetorization, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            # utils.print_network(module, name)
            setattr(self, name, module)
        self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets)]
        self._load_checkpoint(args.resume_iter)
        self.nets.encoder.eval()
        self.nets.decoder.eval()

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def resize_img(self, imgPath):
        img = Image.open(imgPath).convert('RGB')
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([
                    transforms.Resize([self.args.img_size, self.args.img_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
        img = transform(img)
        return img
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps

    @torch.no_grad()
    def vectorization_one_img(self, imgPath):
        # summary(self.nets.encoder, input_size=(3, 256, 256), device='cpu')
        img = self.resize_img(imgPath)
        plt.imshow(img.numpy()[0])
        img = img.unsqueeze(0)
        mu, var = self.nets.encoder(img)
        # vector = self.reparameterization(mu, var)
        # recon_img = self.nets.decoder(vector)
        # recon_img = tensor2ndarray255(recon_img)
        # save_image(recon_img[1:], 'recon_img.png')
        # vector = np.random.normal(0, 1, (mu.size(0), self.args.latent_dim)) * std + mu.detach().numpy()
        return mu.detach().numpy()

## example
if __name__ == '__main__': 
    vectorMachine = Vectorization(256, 8, 8, 64, 200, './')
    # v1 = vectorMachine.vectorization_one_img("/home/ubuntu/VAE-pytorch/상의_긴팔 티셔츠_1145.png")
    # v2 = vectorMachine.vectorization_one_img("/home/ubuntu/VAE-pytorch/상의_긴팔 티셔츠_1145.png")
    # print(v1)
    # print(v2)