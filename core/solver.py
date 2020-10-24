import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

# import utils
from .model import build_nets
from .checkpoint import CheckpointIO
from .loss import compute_KL_loss, compute_reconstruct_loss
from .utils import reparameterization


class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        print(self.args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets = build_nets(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            print(name, module)
            # set attribute to solver obejct name , module
            setattr(self, name, module)

            
            # train mode
        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                # encoder 와 decoder에 optimizer 부여
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets)]

        self.to(self.device)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
   
    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def compute_VAE_loss(self, x):
        mu, logvar = self.nets.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.nets.decoder(z)
        KL_loss = self.args.kl_weight * compute_KL_loss(mu, logvar)
        r_loss = self.args.recon_weight * compute_reconstruct_loss(x, recon_x)
        loss = KL_loss + r_loss
        return loss, Munch(KL_loss = KL_loss, r_loss=r_loss, reg=loss)

    def train(self, loaders):
        args = self.args
        nets = self.nets
        optims = self.optims
        print('1) nets: ', nets,'\n', '2) optims :',optims)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        print('Start training...')
        start_time = time.time()
        train_loader = loaders.src
        for i in range(args.resume_iter, args.total_iters):
            for x in train_loader:
                x = x.to(self.device)
                # train the VAE
                vae_loss, vae_loss_ref = self.compute_VAE_loss(x)
                self._reset_grad()
                vae_loss.backward()
                optims.encoder.step()
                optims.decoder.step()
            
            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)
            
            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([vae_loss_ref],
                                        ['VAE_loss',]):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

    def forward(self, x):
        mu, logvar = self.nets.encoder(x)
        vector = self.reparameterize(mu, logvar)
        recon_img = self.nets.decoder(vector)
        return recon_img    
    

