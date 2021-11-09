from data import train_loader, val_loader, test_loader
from gan_model import Generator,Discriminator, weights_init
from hyperparameters import nz, ngf, ndf
from hyperparameters import g_lr, d_lr
import torch
from tqdm import tqdm
from configurations import path_g, path_d, device
from torchvision.transforms import ToPILImage
from torch.utils.tensorboard import SummaryWriter
from hed import HedNet
#tensorboard --logdir=./tensorboard --port 8080


real_label = 1.0
fake_label = 0.0

g = Generator(nz, ngf)
d = Discriminator(3, ndf)

cp = torch.load('./checkpoints/iter20800.tar')

g.load_state_dict(cp['g_state_dict'])
d.load_state_dict(cp['d_state_dict'])

g_opt = torch.optim.Adam(g.parameters(), g_lr)
d_opt = torch.optim.Adam(d.parameters(), d_lr)

loss_fn = torch.nn.BCELoss()

def train(g, d, g_optimizer, d_optimizer, criterion, train_loader, val_loader, epoch, printevery, device='cuda'):
    g.to(device)
    d.to(device)
    writer = SummaryWriter('./tensorboard')

    iteration = cp['iteration']
    for e in range(epoch):
        print(f'Epoch {e+1}')
        # all real
        for i,real in enumerate(tqdm(train_loader)):
            iteration += 1

            real = real.to(device)
            N = real.shape[0]
            label = torch.full(size=(N,), fill_value=real_label, device=device)
            output = d(real)
            loss_d_real = criterion(output, label)
            D_x = output.mean().item() # real item classified as real
            d_optimizer.zero_grad()
            loss_d_real.backward(retain_graph=True)
            
            # all fake
            noise = torch.randn(N, nz, 1,1, device=device)
            fake = g(noise)
            label.fill_(fake_label)
            output = d(fake)
            D_G_z1 = output.mean().item() # fake item classified as real
            loss_d_fake = criterion(output, label)
            loss_d_fake.backward(retain_graph=True)
            loss_d = (loss_d_real + loss_d_fake).item()
            d_optimizer.step()

            # Generator        
            label.fill_(real_label)
            output = d(fake)
            D_G_z2 = output.mean().item()
            loss_g = criterion(output, label)
            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()
            loss_g = loss_g.item()
            if loss_g > 30.0:
                break
            writer.add_scalar('generator loss', loss_g, iteration)
            writer.add_scalar('discriminator loss', loss_d, iteration)
            if iteration % 100 == 0:
                ToPILImage()((fake[0]+1)/2).save(f'./output/iteration{iteration}.png')
            if iteration % 1600 == 0:
                torch.save({
                    'epoch': e,
                    'iteration': iteration,
                    'g_state_dict': g.state_dict(),
                    'd_state_dict': d.state_dict(),
                    'info': f'Iteration {iteration}: LOSS_G={loss_g:.4f}, LOSS_D={loss_d:.4f}, D(x)={D_x:.4f}, D(G(z))={D_G_z1:.4f} / {D_G_z2:.4f}'
                }, f'./checkpoints/iter{iteration}.tar')
            if i % printevery == 0:
                tqdm.write(f'Iteration {iteration}: LOSS_G={loss_g:.4f}, LOSS_D={loss_d:.4f}, D(x)={D_x:.4f}, D(G(z))={D_G_z1:.4f} / {D_G_z2:.4f}')

        # g_schedular.step(loss_g)
        # d_schedular.step(loss_d)
    torch.save(g.state_dict(), './weights/g.pth')
    torch.save(d.state_dict(), './weights/d.pth')

def train_with_sketch(g, d_sketch, sketches, g_opt, d_opt, num, iteration, silence, device='cuda', printevery=10):
    hed = HedNet().to(device)
    g.to(device)
    train_next = True
    for i in range(iteration):
        noise = torch.randn(num, nz, 1,1, device=device)
        fake = g(noise)
        with torch.no_grad(): fake_sketches = hed((fake+1)/2)

        # all real
        label = torch.full(size=(sketches.shape[0],), fill_value=real_label, device=device)
        D_x = d_sketch(sketches)
        loss_d_real = loss_fn(D_x, label)

        # all fake
        label = torch.full(size=(num,), fill_value=fake_label, device=device)
        D_G_z1 = d_sketch(fake_sketches)
        loss_d_fake = loss_fn(D_G_z1, label)

        loss_d = loss_d_fake + loss_d_real
        if train_next:
            d_opt.zero_grad()
            loss_d.backward(retain_graph=True)
            d_opt.step()
        loss_d = loss_d.item()
        D_x = D_x.mean().item()
        D_G_z1 = D_G_z1.mean().item()

        # generator
        label = torch.full(size=(num,), fill_value=real_label, device=device)
        D_G_z2 = d_sketch(fake_sketches)
        loss_g = loss_fn(D_G_z2, label)
       # print(D_G_z2, label, loss_g)
        g_opt.zero_grad()
        loss_g.backward(retain_graph=True)
        g_opt.step()

        loss_g = loss_g.item()

        D_G_z2 = D_G_z2.mean().item()
        if i % 500 == 0:
            ToPILImage()((fake[0]+1)/2).save(f'./exp/sketch_guide/iter{i}.png')
            ToPILImage()(fake_sketches[0]).save(f'./exp/sketch_guide/iter{i}_fakesketch.png')

        if i % printevery == 0:
            # if train_next:
            print(f'Iter {i}: LOSS_G={loss_g:.4f}, LOSS_D={loss_d:.4f}, D(x)={D_x:.4f}, D(G(z))={D_G_z1:.4f} / {D_G_z2:.4f}')
            # else:
            #     print(f'Iter {i}: LOSS_G={loss_g:.4f}, D(G(z))={D_G_z2:.4f}')
        if D_G_z2<silence: train_next=False
    torch.save(g.state_dict(), './weights/g_sketch.pth')
    torch.save(d.state_dict(), './weights/d_sketch.pth')


import hyperparameters
from PIL import Image
from torchvision.transforms import ToTensor
d_sketch = Discriminator(1, ndf).to(device)
g_opt = torch.optim.Adam(g.parameters(), hyperparameters.sg_lr, (hyperparameters.sg_beta1, hyperparameters.sg_beta2))
d_opt = torch.optim.Adam(d_sketch.parameters(), hyperparameters.sd_lr, (hyperparameters.sd_beta1, hyperparameters.sd_beta2))
sketch = ToTensor()(Image.open('./exp/sketch_cropped.png')).unsqueeze(0).to(device)
train_with_sketch(g, d_sketch, sketch, g_opt, d_opt, 5, 5000, 0.1)