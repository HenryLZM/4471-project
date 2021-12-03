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
    with torch.no_grad():
        for dir_ in [real_dir, fake_dir, fake_sketch_dir]:
            shutil.rmtree(dir_)
            os.mkdir(dir_)
            
        #hed = HedNet('./weights/hed-bsds500.pth').to('cuda')
        g = Generator(128,256).to('cuda')
        g.load_state_dict(torch.load('./weights/pretrain_wgp/85000_net_G.pth'))

        for i in range(num//256):
            print(i)
            noise = torch.randn(256, 128, 1, 1, device='cuda')
            o = g(noise)
            #fake_sketch = hed(o)
            for j in range(256):
                ToPILImage()((o[j]+1)/2).save(f'{fake_dir}fake{i*num + j}.png')

        train_set = create_dataset('E:/HKUST/2021F/COMP4471/project/data/img/G2.h5',0,75000,'cuda')
        for i in range(num):
            if i % 256 == 0: print(i//256)
            idx = torch.randint(low=0, high=len(train_set),size=(1,)).item()
            ToPILImage()((train_set[idx]+1)/2).save(f'{real_dir}real{i}.png')
            
        # num = len([l for l in os.listdir(real_sketch_dir)])
        # for i in range(num):
        #     ToPILImage()(fake_sketch[i]).save(f'{fake_sketch_dir}fake_sketch{i}.png')

        score_img = fid.compute_fid(fdir1=fake_dir, fdir2=real_dir, num_workers=0)
        # score_sketch = fid.compute_fid(fdir1=fake_sketch_dir, fdir2=real_sketch_dir, num_workers=0)

        if not keep:
            for dir_ in [real_dir, fake_dir, fake_sketch_dir]:
                shutil.rmtree(dir_)
                os.mkdir(dir_)

    return score_img

score = fid_metric(12800, False)
print(score)

"""
Iter: 78500, 4096 pics FID: 36.249426410132486
Iter: 78500, 8192 pics FID: 34.41659904826156
Iter: 82600, 12800 pics FID: 31.929555223681632
Iter: 83400, 12800 pics FID: 26.978774738040073
Iter: 84600, 12800 pics FID: 25.84643173267301 25.805210110838885
Iter: 85000, 12800 pics FID: 
Iter: 86600, 12800 pics FID: 28.165358907215733
"""