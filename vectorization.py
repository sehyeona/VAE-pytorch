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

from numpy import dot
from numpy.linalg import norm
from torchvision import transforms
from torchvision.utils import save_image
from functools import wraps
from torchsummary import summary
import matplotlib.pyplot as plt

from core.model import build_nets
from core.checkpoint import CheckpointIO
from core.utils import tensor2ndarray255
from core.solver import VAE
from core.loss import compute_reconstruct_loss


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
    def __init__(self, args):
        self.args = args

        self.vae = VAE(args)
        # self.vae.device = 'cpu'
        self.load_checkpoint()
        self.vae.encoder.eval()
        self.vae.decoder.eval()

    def load_checkpoint(self):
        fname = self.args.checkpoint_fname
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))
        for name in module_dict.keys():
            self.vae.__getattr__(name).load_state_dict(module_dict[name])

    def resize_img(self, imgPath):
        img = Image.open(imgPath).convert('RGB')
        transform = transforms.Compose([
                    transforms.Resize([self.args.img_size, self.args.img_size]),
                    transforms.ToTensor(),])
        img = transform(img)
        return img

    def reconstruct_img(self, imgPath):
        real_img = self.resize_img(imgPath)
        real_img = real_img.unsqueeze(0).to(device='cuda')
        recon_img = self.vae(real_img)
        recon_img = recon_img[0]
        save_image(recon_img, '/'.join(imgPath.split('/')[:-1])+'/recon_img.png')
        print(1000 * compute_reconstruct_loss(real_img,recon_img))
        return 'recon_' + imgPath


    @torch.no_grad()
    def vectorization_one_img(self, imgPath):
        real_img = self.resize_img(imgPath)
        real_img = real_img.unsqueeze(0)
        if torch.cuda.is_available:
            real_img = real_img.to(device='cuda')
        vector = self.vae.reparameterize(*self.vae.encoder(real_img)).to(device='cpu')
        return vector.detach().numpy()

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

## example
if __name__ == '__main__': 
    args = Munch()
    args.img_size = 256
    args.encoder_channel_in = 32
    args.decoder_channel_in = 128
    args.max_channel = 256
    args.min_channel = 16
    args.target_size = 8
    args.start_size = 8
    args.latent_dim = 64
    args.checkpoint_fname = '/home/ubuntu/VAE-pytorch/expr/recon_1000_kl_10_channel_256/000300_nets.ckpt'
    args.mode = 'vector'
    vectorMachine = Vectorization(args)
    vectorMachine.reconstruct_img("/home/ubuntu/VAE-pytorch/heeloo.png")
    vectorMachine.reconstruct_img("/home/ubuntu/VAE-pytorch/heeloo.png")
    v1 = vectorMachine.vectorization_one_img("/home/ubuntu/VAE-pytorch/heeloo.png")
    v2 = vectorMachine.vectorization_one_img("/home/ubuntu/VAE-pytorch/heeloo.png")
    print(v1)
    print(v2)
    print("Cosine similarity of two same image: ", cosine_similarity(v1[0], v2[0]))
    # v2 = vectorMachine.vectorization_one_img("/home/ubuntu/VAE-pytorch/상의_긴팔 티셔츠_1145.png")
    # print(v1)
    # print(v2)
