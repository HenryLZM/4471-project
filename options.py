import torch

pretrained_opt = {
    'name': 'pretrain_wgp',
    'checkpoints_dir': './weights/',
    'purpose': 'pretrain/wgp',

    'data_dir': 'E:/HKUST/2021F/COMP4471/project/data/img/G2.h5',
    'visual_dir': './visual',
    'log_dir': './tensorboard',
    'data_len': 4000,
    'num_workers': 0,

    'print_freq': 10,
    'save_freq': 250,
    'vis_freq': 100,

    'resume_iter': 8750,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'loss': 'wgp',
    'max_epoch': 50,
                                         
    'nz': 128,
    'ngf': 256,
    'nc': 3,
    'ndf': 128,
    'batch_size': 80,

    'lr': 1e-4,
    'beta1': 0.0,
    'beta2': 0.9,
    'ncritic': 2,

    'lambda_gp': 10,
    'g_pretrained': None,
    'd_pretrained': None,
}