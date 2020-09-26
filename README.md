# VAE pytorch
## computer vision project with pytorch (VAE)
```
python main.py --mode train \
               --lambda_reg 1 \
               --train_img_dir data/상의 \
               --img_size 256 --latent_dim 64\
               --max_channel 128 --decoder_channel_in 128\
               --print_every 1 --save_every 10 --total_iters 200\
               --batch_size 128 --lr 1e-4
````
 
