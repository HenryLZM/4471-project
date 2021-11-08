from data import train_loader, val_loader, test_loader
from gan_model import Generator,Discriminator
from hyperparameters import nz, ngf, ndf
from hyperparameters import g_lr, d_lr
import torch
from tqdm import tqdm

real_label = 1.0
fake_label = 0.0

g = Generator(nz, ngf)
d = Discriminator(ndf)

g_opt = torch.optim.Adam(g.parameters(), g_lr)
d_opt = torch.optim.Adam(d.parameters(), d_lr)

g_sche = torch.optim.lr_scheduler.ReduceLROnPlateau(g_opt)
d_sche = torch.optim.lr_scheduler.ReduceLROnPlateau(d_opt)

loss_fn = torch.nn.BCELoss()

def train(g, d, g_optimizer, d_optimizer, g_schedular, d_schedular, criterion, train_loader, val_loader, epoch, printevery, device='cuda'):
    g.to(device)
    d.to(device)
    d_losses = []
    g_losses = []
    iteration = 0
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

            if i % printevery == 0:
                tqdm.write(f'Iteration {iteration}: LOSS_G= {loss_g}, LOSS_D= {loss_d}, D(x)= {D_x}, D(G(z))= {D_G_z1} / {D_G_z2}')
            g_losses.append(loss_g)
            d_losses.append(loss_d)
        #g_schedular.step()
        #d_schedular.step()
    return g_losses, d_losses

#train(g, d, g_opt, d_opt, g_sche, d_sche, loss_fn, train_loader, val_loader, 1, 10)
