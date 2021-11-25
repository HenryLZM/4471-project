from cleanfid import fid
from training.networks.dcgan import Generator
from training.networks.hed import HedNet
import torch
from training.dataset import create_dataloader, create_dataset

from torchvision.transforms import ToPILImage
import os
import shutil

real_dir = './fid_cal/real_img/'
fake_dir = './fid_cal/fake_img/'
real_sketch_dir = './fid_cal/real_sketch/'
fake_sketch_dir = './fid_cal/fake_sketch/'


def fid_metric(num, keep):
    for dir_ in [real_dir, fake_dir, fake_sketch_dir]:
        shutil.rmtree(dir_)
        os.mkdir(dir_)
        
    hed = HedNet('./weights/network-bsds500.pytorch')
    g = Generator(128,256)
    g.load_state_dict(torch.load('./weights/sketch/70250_net_G.pth'))
    noise = torch.randn(num, 128, 1, 1)
    o = g(noise)
    fake_sketch = hed(o)
    train_set = create_dataset('E:/HKUST/2021F/COMP4471/project/data/img/G2.h5',0,75000,'cuda')
    for i in range(num):
        idx = torch.randint(low=0, high=len(train_set),size=(1,)).item()
        ToPILImage()((train_set[idx]+1)/2).save(f'{real_dir}real{i}.png')
        ToPILImage()((o[i]+1)/2).save(f'{fake_dir}fake{i}.png')
        
    # num = len([l for l in os.listdir(real_sketch_dir)])
    # for i in range(num):
    #     ToPILImage()(fake_sketch[i]).save(f'{fake_sketch_dir}fake_sketch{i}.png')

    score_img = fid.compute_fid(fdir1=fake_dir, fdir2=real_dir, num_workers=0)
    # score_sketch = fid.compute_fid(fdir1=fake_sketch_dir, fdir2=real_sketch_dir, num_workers=0)

    if keep:
        for dir_ in [real_dir, fake_dir, fake_sketch_dir]:
            shutil.rmtree(dir_)
            os.mkdir(dir_)

    return score_img