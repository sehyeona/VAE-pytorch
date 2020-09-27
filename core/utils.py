import numpy as np

from torch.autograd import Variable
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print(cuda)

def reparameterization(latent_dim, mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = 0
    try:
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    except: 
        sampled_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z

def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255