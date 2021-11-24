import torch
from training import networks
import os

class GANModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = opt['device']
        self.BCELoss = torch.nn.BCELoss()
        self.netG, self.netD = self.initialize_networks(opt)
        # if self.purpose == 'pretrain':
        #     self.tf_fake = networks.transforms.Retrieve()
        #     self.tf_real = networks.transforms.Retrieve()
        # elif self.purpose == 'modify':
        #     pass
        # else:
        #     raise ValueError("purpose is invalid")

    def forward(self, data, mode):
        if mode == 'generator':
            g_loss = self.compute_generator_loss()
            return g_loss
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(data)
            return d_loss
        # elif mode == 'discriminator-regularize':
        #     assert not self.opt['no_d_regularize'], "Discriminator shouldn't be regularized with no_d_regularize applied"
        #     d_reg_loss = self.compute_discriminator_regularization(data)
        #     return d_reg_loss
        else:
            raise ValueError("|mode| is invalid")


    def initialize_networks(self, opt):
        netG = networks.Generator(
            opt['nz'],
            opt['ngf'],
        )
        netD = networks.Discriminator(
            opt['nc'],
            opt['ndf'],
        )
        
        return netG, netD

    def compute_generator_loss(self):
        fake_image = self.generate_fake()
        pred_fake = self.netD(fake_image)
        mode = self.opt['loss']
        if mode == 'bce':
            g_loss = self.BCELoss(pred_fake, torch.ones(pred_fake.shape, device=self.device))
        elif mode == 'wgp':
            g_loss = -1 * pred_fake.mean()
        return g_loss

    def compute_discriminator_loss(self, real):
        with torch.no_grad():
            fake = self.generate_fake(real.shape[0]).detach()

        pred_fake = self.netD(fake)
        pred_real = self.netD(real)
        
        self.dx = pred_real.mean().item()
        self.dgz = pred_fake.mean().item()
        mode = self.opt['loss']
        if  mode == 'bce':
            d_fake = self.BCELoss(pred_fake, torch.zeros(pred_fake.shape, device=self.device))
            d_real = self.BCELoss(pred_real, torch.ones(pred_fake.shape, device=self.device))
            return d_fake+d_real
        elif mode == 'wgp':
            d_fake = pred_fake.mean()
            d_real = -pred_real.mean()   
            gp = self.gradient_penalty(real, fake) * self.opt['lambda_gp']
            return d_fake + d_real + gp

        
        # 
        # print(f'd_fake: {d_fake.item():.4f}, d_real: {d_real.item():.4f}, gp: {gp.item():.4f}')
        

    def gradient_penalty(self, real, fake):
        alpha = torch.rand(real.shape[0], 1,1,1, device=self.device)
        interpolates = (alpha * real + (1-alpha) * fake).requires_grad_(True)
        # interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        d_interpolates = self.netD(interpolates)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(d_interpolates.size()).to(self.device),
                            create_graph=True, retain_graph=True)[0].reshape(real.shape[0], -1)

        gradient_penalty = ((gradients.norm(p=2, dim=1)-1)**2).mean() * self.opt['lambda_gp']

        return gradient_penalty

    # def compute_discriminator_regularization(self, data):
    #     data.requires_grad = True
    #     pred_real = self.netD(data)
    #     r1_loss = self.d_regularize(pred_real, data)
    #     d_reg_loss = self.opt['r1'] / 2 * r1_loss * self.opt['d_reg_every']
    #     return d_reg_loss
    
    def generate_fake(self, batch_size=None):
        if batch_size is None:
            batch_size = self.opt['batch_size']
        
        device = self.device
        noises = make_noise(batch_size, self.opt['nz'], self.device)
        fake = self.netG(noises)
        return fake

    # def create_loss_fns(self, opt):
    #     if self.device == 'cuda':
    #         tensor = torch.cuda.FloatTensor
    #     else:
    #         tensor = torch.FloatTensor
    #     self.criterionGAN = networks.loss.GANLoss(opt['gan_mode'], tensor=tensor, opt=opt)

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        self.G_params = G_params
        
        D_params = list(self.netD.parameters())
        self.D_params = D_params

        lr = opt['lr']
        beta1, beta2 = opt['beta1'], opt['beta2']
        G_lr, D_lr = lr, lr
        G_beta1, D_beta1 = beta1, beta1
        G_beta2, D_beta2 = beta2, beta2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(G_beta1, G_beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(D_beta1, D_beta2))
        return optimizer_G, optimizer_D

    def set_requires_grad(self, g_requires_grad=None, d_requires_grad=None):
        if g_requires_grad is not None:
            for p in self.G_params:
                p.requires_grad = g_requires_grad

        if d_requires_grad is not None:
            for p in self.D_params:
                p.requires_grad = d_requires_grad

    def save(self, iters):
        save_path = os.path.join(self.opt['checkpoints_dir'], self.opt['name'], f"{iters}_net_")
        torch.save(self.netG.state_dict(), save_path + 'G.pth')
        torch.save(self.netD.state_dict(), save_path + 'D.pth')

    def load(self, iters):
        load_path = os.path.join(self.opt['checkpoints_dir'], self.opt['name'], f"{iters}_net_")
        state_dict_g = torch.load(load_path + "G.pth", map_location=self.device)
        self.netG.load_state_dict(state_dict_g)
        state_dict_d = torch.load(load_path + "D.pth", map_location=self.device)
        self.netD.load_state_dict(state_dict_d)

def make_noise(batch_size, z_dim, device):
    return torch.randn(batch_size, z_dim, 1,1, device=device)
