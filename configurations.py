import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

path_g = './weights/g.pth'
path_d = './weights/d.pth'
data_dir = 'E:/HKUST/2021F/COMP4471/project/data/img/G2.h5'

generated_dir = './test/'

hed_weight_dir = './weights/hed-bsds500'

