from cleanfid import fid
from gan_model import Generator
import torch
from hyperparameters import nz,ngf
from hed import HedNet
from data import train_set
from torchvision.transforms import ToPILImage
import os
import shutil

real_dir = './fid_cal/real_img/'
fake_dir = './fid_cal/fake_img/'
real_sketch_dir = './fid_cal/real_sketch/'
fake_sketch_dir = './fid_cal/fake_sketch/'

def fid_metric(num):
    for dir_ in [real_dir, fake_dir, fake_sketch_dir]:
        shutil.rmtree(dir_)
        os.mkdir(dir_)
        
    hed = HedNet()
    g = Generator(nz,ngf)
    g.load_state_dict(torch.load('./weights/g_sketch.pth'))
    noise = torch.randn(num, nz, 1, 1)
    o = g(noise)
    fake_sketch = hed(o)
    for i in range(num):
        idx = torch.randint(low=0, high=len(train_set),size=(1,)).item()
        ToPILImage()((train_set[idx]+1)/2).save(f'{real_dir}real{i}.png')
        ToPILImage()((o[i]+1)/2).save(f'{fake_dir}fake{i}.png')
        
    num = len([l for l in os.listdir(real_sketch_dir)])
    for i in range(num):
        ToPILImage()(fake_sketch[i]).save(f'{fake_sketch_dir}fake_sketch{i}.png')

    score_img = fid.compute_fid(fdir1=fake_dir, fdir2=real_dir, num_workers=0)
    score_sketch = fid.compute_fid(fdir1=fake_sketch_dir, fdir2=real_sketch_dir, num_workers=0)

    for dir_ in [real_dir, fake_dir, fake_sketch_dir]:
        shutil.rmtree(dir_)
        os.mkdir(dir_)

    return score_img, score_sketch

# x = fid_metric(250)
# print(x)