# DATA
batch_size = 64

# MODEL
nz = 100
ngf = 64
ndf = 64

# TRAIN
#   Generator
g_lr = 5e-4
g_beta1, g_beta2 = 0.5, 0.999
#   Fashion Discreminator
d_lr = 4e-6
d_beta1, d_beta2 = 0.5, 0.999
#   Sketch Disciminator
sg_lr = 1e-4
sg_beta1, sg_beta2 = 0.5, 0.999
sd_lr = 5e-6
sd_beta1, sd_beta2 = 0.5, 0.999
import torch

pretrained_opt = {
    'use_gpu': torch.cuda.is_available(),

    'gan_mode': None,

    'size': 256,
    'z_dim': 512,
    'n_mlp': 8,
    'lr_mlp': 0.01,
    'channel_multiplier': 2,
    'batch_size': 32,
    'mixing': 0,

    'optim_param_g': 'w_shift',
    'lr': 1e-3,
    'beta1': 0.5,
    'beta2': 0.99,
    'no_d_regularize': True,
    'd_reg_every': 1.0,

    'g_pretrained': '',
}