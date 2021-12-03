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
    'save_freq': 200,
    'vis_freq': 100,

    'resume_iter': 87000,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'loss': 'wgp',
    'max_epoch': 50,
                                         
    'nz': 128,
    'ngf': 256,
    'nc': 3,
    'ndf': 128,
    'batch_size': 80,

    'lr': 1e-5,
    'beta1': 0.0,
    'beta2': 0.9,
    'ncritic': 2,
    'lambda_gp': 10,
}

sketch_opt = {
    'name': 'wgp',
    'checkpoints_dir': './weights/sketch',
    'pretrained': '84600_net_',

    'hed_weight': './weights/hed-bsds500.pth',
    'data_dir': 'E:/HKUST/2021F/COMP4471/project/data/img/G2.h5',
    'sketch_dir': './input/sketch',
    'visual_dir': './visual/sketch',
    'log_dir': './tensorboard/sketch',
    'data_len': 75000,
    'num_workers': 0,


    'print_freq': 10,
    'save_freq': 500,
    'vis_freq': 100,

    'resume_iter': 0,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'loss': 'wgp',
    'max_epoch': 50000,
                                         
    'nz': 128,
    'ngf': 256,
    'trainable_G_layers': 22, # 2 for first trans_conv, 4 for first trans_conv + first bn, # 
    'nc_sketch': 1,
    'ndf_sketch': 32,
    'nc_image': 3,
    'ndf_image': 128,
    'image_batch': 10,
    'sketch_batch': 5,

    'g_lr': 5e-5,
    'g_beta1': 0.0,
    'g_beta2': 0.9,
    'd_s_lr' : 2e-4,
    'd_s_beta1': 0.0,
    'd_s_beta2': 0.9,
    'd_i_lr': 2e-4,
    'd_i_beta1': 0.0,
    'd_i_beta2': 0.9,
    'n_critic': 2,

    'lambda_gp_sketch': 10,
    'lambda_gp_image': 10,
    'l_image': 0,
    # 'l_weight': 0,
    # 'r1': 0.0,
    # 'd_reg_every': 0,
}

