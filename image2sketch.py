import os
from configurations import photo_sketch_cmd, generated_dir
from gan_model import Generator
from hyperparameters import nz, ngf
import torch
from torchvision.transforms import ToPILImage

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_sketch(num):
    g = Generator(nz, ngf).to(device)
    g.load_state_dict(torch.load('./checkpoints/iter28800.tar')['g_state_dict'])
    noise = torch.randn((num,nz,1,1), device=device)
    out = g(noise)
    for i in range(num):
        ToPILImage()((out[i]+1)/2).save(f'{generated_dir}out{i}.png')
    os.system(photo_sketch_cmd)