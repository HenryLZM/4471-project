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
d = Discriminator(ndf)

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


hed = HedNet().to(device)
d_sketch = Discriminator(ndf).to(device)
def train_with_sketch(g, d_sketch, sketches, g_opt, d_opt, num, epoch, silence, device='cuda', printevery=10):
    g.to(device)
    for e in range(epoch):
        noise = torch.randn(num, nz, ngf, device=device)
        fake = g(noise)
        fake_sketches = hed(fake)

        # all real
        label = torch.full(size=(N,), fill_value=real_label, device=device)
        if e % silence != 0:
            D_x = d_sketch(sketches)
            loss_d_real = loss_fn(D_x, label)

            # all fake
            label.fill_(fake_label)
            D_G_z1 = d_sketch(fake_sketches)
            loss_d_fake = loss_fn(D_G_z1, label)

            loss_d = loss_d_fake + loss_d_real
            d_opt.zero_grad()
            loss_d.backward()
            d_opt.step()

        # generator
        label.fill_(real_label)
        D_G_z2 = d_sketch(fake_sketches)
        loss_g = loss_fn(D_G_z2, label)
        g_opt.zero_grad()
        loss_g.backward()
        g_opt.step()

        loss_g = loss_g.item()
        loss_d = loss_d.item()
        D_x = D_x.item()
        D_G_z1 = D_G_z1.item()
        D_G_z2 = D_G_z2.item()

        if i % printevery == 0:
            print(f'Epoch {e+1}: LOSS_G={loss_g:.4f}, LOSS_D={loss_d:.4f}, D(x)={D_x:.4f}, D(G(z))={D_G_z1:.4f} / {D_G_z2:.4f}')

    torch.save(g.state_dict(), './weights/g_modified.pth')