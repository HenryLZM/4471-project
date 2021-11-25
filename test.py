from training.networks.dcgan import Generator
from options import pretrained_opt
import torch
import torchvision.utils as vutil
from torchvision.transforms import ToPILImage
from training.networks.hed import HedNet

def show(num):
    with torch.no_grad():
        g = Generator(pretrained_opt['nz'], pretrained_opt['ngf']).to('cuda')
        g.load_state_dict(torch.load('./weights/sketch/76250_net_G.pth'))
        noise = torch.randn((num, pretrained_opt['nz'], 1, 1),  device='cuda')
        fake = g(noise)
        hed = HedNet('./weights/hed-bsds500.pth').to('cuda')
        fake_sketch = hed((fake+1)/2*255)
        ToPILImage()(vutil.make_grid((fake+1)/2, nrow=int(num**0.5))).show()
        ToPILImage()(vutil.make_grid(fake_sketch, nrow=int(num**0.5))).show()

show(256)