import torch

pretrained_opt = {
    'name': 'pretrain_wgp',
    'checkpoints_dir': './weights/',
    'pretrained': None,

    'data_dir': 'E:/HKUST/2021F/COMP4471/project/data/img/G2.h5',
    'visual_dir': './visual',
    'log_dir': './tensorboard',
    'data_len': 75000,
    'num_workers': 0,

    'print_freq': 10,
    'save_freq': 250,
    'vis_freq': 100,

    'resume_iter': 71250,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'loss': 'wgp',
    'max_epoch': 50,
                                         
    'nz': 128,
    'ngf': 256,
    'nc': 3,
    'ndf': 128,
    'batch_size': 80,

    'lr': 5e-5,
    'beta1': 0.0,
    'beta2': 0.9,
    'ncritic': 2,
    'lambda_gp': 10,
}

sketch_opt = {
    'name': 'sketch',
    'checkpoints_dir': './weights/',
    'pretrained': '76250_net_G.pth',

    'hed_weight': './weights/hed-bsds500.pth',
    'data_dir': 'E:/HKUST/2021F/COMP4471/project/data/img/G2.h5',
    'sketch_dir': 'exp\input_sketch',
    'visual_dir': './visual',
    'log_dir': './tensorboard/sketch',
    'data_len': 75000,
    'num_workers': 0,


    'print_freq': 1,
    'save_freq': 1000000,
    'vis_freq': 5,

    'resume_iter': 0,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'loss': 'wgp',
    'max_epoch': 50,
                                         
    'nz': 128,
    'ngf': 256,
    'nc_sketch': 1,
    'ndf_sketch': 128,
    'nc_image': 3,
    'ndf_image': 128,
    'image_batch': 50,
    'sketch_batch': 1,

    'lr': 5e-5,
    'beta1': 0.0,
    'beta2': 0.9,
    'ncritic': 2,

    'lambda_gp': 10,
    'l_image': 0,
    'l_weight': 0,
    'r1': 0.1,
    'd_reg_every': 0,
}