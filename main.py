import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from core.dataloader import get_train_loader
from core.solver import VAE


def str2bool(v):
    return v.lower() in ('true')

def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]

def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    vae = VAE(args)

    if args.mode == 'train':
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                            which='train',
                                            img_size=args.img_size,
                                            batch_size=args.batch_size,
                                            prob=args.randcrop_prob,
                                            num_workers=args.num_workers,))
        vae.train(loaders)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    # image path for vectorization
    parser.add_argument('--img_path', type=str, default=None,
                        help='Image path for vectorization')

    # model arguments
    # img_size and latent_dim
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent vector dimension')

    # encoder
    parser.add_argument('--target_size', type=int, default=8,
                        help='Image resolution')
    parser.add_argument('--max_channel', type=int, default=256,
                        help='Max channel for Encoder')
    parser.add_argument('--encoder_channel_in', type=int, default=32,
                        help='# of channels on first layer')
    

    # decoder
    parser.add_argument('--start_size', type=int, default=8,
                    help='Image resolution')
    parser.add_argument('--min_channel', type=int, default=16,
                        help='min channel for Encoder')
    parser.add_argument('--decoder_channel_in', type=int, default=256,
                        help='# of channels on first layer')


    # weight for objective functions
    parser.add_argument('--recon_weight', type=float, default=100,
                        help='Weight for R1 regularization')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=200,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval','use'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/',
                        help='Directory containing training images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=20)

    args = parser.parse_args()
    main(args)