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
    'data_dir': 'E:/HKUST/2021F/COMP4471/project/data/img/G2.h5',
    'data_len': 10,

    'print_freq': 10,

    'resume_iter': None,

    'use_gpu': torch.cuda.is_available(),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'gan_mode': 'softplus',
    'max_epoch': 1,

    'image_size': 256,                                          #image size
    'z_dim': 512,
    'n_mlp': 8,
    'lr_mlp': 0.01,
    'channel_multiplier': 2,
    'batch_size': 2,
    'mixing': 0,

    'optim_param_g': 'w_shift',
    'lr': 1e-3,
    'beta1': 0.5,
    'beta2': 0.99,
    'no_d_regularize': True,
    'd_reg_every': 1.0,

    'g_pretrained': '',
    'd_pretrained': '',
}