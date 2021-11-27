from training.gan_model import GANModel
import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from training.networks.dcgan import Generator, Discriminator, DiscriminatorSig
from torch.optim import Adam
from training.networks.hed import HedNet
from training.networks.loss import RegularizeD, WeightLoss

class GANTrainer():
    def __init__(self, opt):
        self.opt = opt
        self.device = opt['device']
        self.gan_model = GANModel(opt).to(self.device)
        self.optimizer_G, self.optimizer_D = self.gan_model.create_optimizers(opt)
        self.gan_model.set_requires_grad(False, False)
        self.g_loss = 0
        self.d_loss = 0
        self.writer = SummaryWriter(opt['log_dir'])

    def run_generator_one_step(self, data):
        g_loss = self.gan_model(data, mode='generator')
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_loss = g_loss.item()

    def run_discriminator_one_step(self, data):
        d_loss = self.gan_model(data, mode='discriminator')
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_loss = d_loss.item()
        

    def train_one_step(self, data, iters):
        self.gan_model.set_requires_grad(False, True)
        self.run_discriminator_one_step(data)
        if self.opt['loss'] == 'wgp' and iters % self.opt['ncritic'] == 0:
            self.gan_model.set_requires_grad(True, False)
            self.run_generator_one_step(data)

        self.writer.add_scalar('generator', self.g_loss, global_step=iters)
        self.writer.add_scalar('discriminator', self.d_loss, global_step=iters)

    def get_latest_losses(self):
        return f"g_loss: {self.g_loss:.4f}, d_loss: {self.d_loss:.4f}, dx: {self.gan_model.dx:.4f}, dgz: {self.gan_model.dgz:.4f}"

    def get_latest_generated(self):
        return self.generated

    def save(self, iters):
        self.gan_model.save(iters)
        misc = {
            "g_optim": self.optimizer_G.state_dict(),
            "d_optim": self.optimizer_D.state_dict(),
            "opt": self.opt,
        }
        save_path = os.path.join(self.opt['checkpoints_dir'], self.opt['name'], f"{iters}_net_")
        torch.save(misc, save_path + "misc.pth")

    def load(self, iters):
        print(f"Resuming model at iteration {iters}")
        self.gan_model.load(iters)
        load_path = os.path.join(self.opt['checkpoints_dir'], self.opt['name'], f"{iters}_net_")
        state_dict = torch.load(load_path + "misc.pth", map_location=self.device)
        self.optimizer_G.load_state_dict(state_dict["g_optim"])
        self.optimizer_D.load_state_dict(state_dict["d_optim"])


class SketchTrainer():
    def __init__(self, opt):
        self.opt = opt
        self.device = opt['device']
        self.netG = Generator(opt['nz'], opt['ngf']).to(self.device)
        if opt['resume_iter'] is not None:
            self.netG.load_state_dict(torch.load(opt['checkpoints_dir'] + '/' + opt['name'] + '/' + opt['pretrained']))
        for p in self.netG.parameters():
            p.requires_grad = False # Freeze all layers

        self.hed = HedNet(opt['hed_weight']).to(self.device)
        self.optim_G = Adam(self.netG.parameters(), lr = opt['g_lr'], betas=(opt['g_beta1'], opt['g_beta2']))
        if opt['loss'] == 'bce':
            self.netD_sketch = DiscriminatorSig(opt['nc_sketch'], opt['ndf_sketch']).to(self.device)
        elif opt['loss'] == 'wgp':
            self.netD_sketch = Discriminator(opt['nc_sketch'], opt['ndf_sketch']).to(self.device)
        self.optim_D_sketch = Adam(self.netD_sketch.parameters(), lr=opt['d_s_lr'], betas=(opt['d_s_beta1'], opt['d_s_beta2']))
        self.bce_loss = torch.nn.BCELoss()

        if opt['l_weight'] > 0:
            self.weight_loss = WeightLoss(list(self.netG.parameters()))
            
        self.image_regularization = opt['l_image'] > 0
        
        if self.image_regularization:
            if opt['loss'] == 'bce':
                self.netD_image = DiscriminatorSig(opt['nc_image'], opt['ndf_image']).to(self.device)
            elif opt['loss'] == 'wgp':
                self.netD_image = Discriminator(opt['nc_image'], opt['ndf_image']).to(self.device)
            self.optim_D_image = Adam(self.netD_image.parameters(), lr=opt['d_i_lr'], betas=(opt['d_i_beta1'], opt['d_i_beta2']))

        self.g_loss = 0
        self.d_sketch_loss = 0
        self.d_image_loss = 0
        
        #self.writer = SummaryWriter(opt['log_dir'])
        
    def train_one_step(self, real_sketch, iters, real_image=None):
        noise = torch.randn(size=(real_sketch.shape[0], self.opt['nz'], 1,1), device=self.device)
        with torch.no_grad():
            fake_image = self.netG(noise)   
            fake_sketch = self.hed((fake_image+1)/2*255)

        # discriminators
        self.set_requires_grad(False, True, real_image is not None)

        # sketch discriminator loss
        pred_fake_sketch = self.netD_sketch(fake_sketch)
        pred_real_sketch = self.netD_sketch(real_sketch)
        if self.opt['loss'] == 'bce':
            fake_sketch_loss = self.bce_loss(pred_fake_sketch, torch.zeros(size=pred_fake_sketch.shape, device=self.device))
            real_sketch_loss = self.bce_loss(pred_real_sketch, torch.ones(size=pred_real_sketch.shape, device=self.device))
            d_sketch_loss = fake_sketch_loss + real_sketch_loss
        elif self.opt['loss'] == 'wgp':
            gp = self.gradient_penalty(self.netD_sketch, real_sketch, fake_sketch)
            d_sketch_loss = pred_fake_sketch.mean() - pred_real_sketch.mean() + gp
        
        self.optim_D_sketch.zero_grad()
        d_sketch_loss.backward()
        self.optim_D_sketch.step()
        self.d_sketch_loss = d_sketch_loss.item()

        # image discriminator loss
        d_image_loss = 0
        if real_image is not None:
            noise = torch.randn(size=(real_image.shape[0], self.opt['nz'], 1,1), device=self.device)
            fake_image = self.netG(noise)
            pred_fake_image = self.netD_image(fake_image)
            pred_real_image = self.netD_image(real_image)
            if self.opt['loss'] == 'bce':
                fake_image_loss = self.bce_loss(pred_fake_image, torch.zeros(size=pred_fake_image.shape, device=self.device))
                real_image_loss = self.bce_loss(pred_real_image, torch.ones(size=pred_real_image.shape, device=self.device))
                d_image_loss = fake_image_loss + real_image_loss
            elif self.opt['loss'] == 'wgp':
                gp = self.gradient_penalty(self.netD_image, real_image, fake_image)
                d_image_loss = pred_fake_image.mean() - pred_real_image.mean() + gp

            self.optim_D_image.zero_grad()
            d_image_loss.backward()
            self.optim_D_image.step()
            self.d_image_loss = d_image_loss.item()
        
        # image regularization
        if self.opt['d_reg_every'] != 0 and iters % self.opt['d_reg_every'] == 0:
            real_sketch.requires_grad = True
            pred_real = self.netD_sketch(self.tf_real(real_sketch))
            r1_loss = self.d_regularize(pred_real, real_sketch)
            d_reg_loss = self.opt['r1'] / 2 * r1_loss * self.opt['d_reg_every']

            # R1 regularization for D_image (if image regularization is applied)
            if self.opt.l_image > 0:
                real_image.requires_grad = True
                pred_real2 = self.netD_image(real_image)
                r1_loss2 = self.d_regularize(pred_real2, real_image)
                d_reg_loss += self.opt.l_image * \
                    self.opt['r1'] / 2 * r1_loss2 * self.opt['d_reg_every']
            
            self.optim_D_sketch.zero_grad()
            if self.image_regularization: self.optim_D_image.zero_grad()
            d_reg_loss.mean().backward()
            self.optim_D_sketch.step()
            if self.image_regularization: self.optim_D_image.step()

        # Generator
        self.set_requires_grad(True, False, False if self.image_regularization else None)
        noise = torch.randn(size=(self.opt['image_batch'], self.opt['nz'], 1,1), device=self.device)
        fake_image = self.netG(noise)

        # weight reg
        l_weight = torch.tensor(0, device=self.device)
        if self.opt['l_weight'] > 0:
            # weight loss
            l_weight = self.opt['l_weight'] * self.weight_loss(list(self.netG.parameters()))

        l_image = torch.tensor(0, device=self.device)
        if self.image_regularization:
            pred_image = self.netD_image(fake_image)
            if self.opt['loss'] == 'bce':
                l_image = self.bce_loss(pred_image, torch.ones(pred_image.shape, device=self.device))
            elif self.opt['loss'] == 'wgp':
                l_image = -pred_image.mean()  

        fake_sketch = self.hed(fake_image)
        pred_sketch = self.netD_sketch(fake_sketch)
        if self.opt['loss'] == 'bce':
            loss_g = self.bce_loss(pred_sketch, torch.ones(pred_sketch.shape, device=self.device))
        elif self.opt['loss'] == 'wgp':
            loss_g = -pred_sketch.mean()

        # if l_weight != 0: loss_g += l_weight
        #if l_image != 0: loss_g += l_image
        self.optim_G.zero_grad()
        loss_g.backward()
        self.optim_G.step()
        self.g_loss = loss_g.item()

            # color loss?
        # self.writer.add_scalar('generator', , global_step=iters)
        # self.writer.add_scalar('discriminator', self.d_loss, global_step=iters)

    def set_requires_grad(self, G=None, D1=None, D2=None):
        if G is not None:
            params = list(self.netG.parameters())
            params[0].requires_grad = G

        if D1 is not None:
            for p in self.netD_sketch.parameters():
                p.requires_grad = D1
        
        if self.image_regularization and D2 is not None:
            for p in self.netD_image.parameters():
                p.requires_grad = D2

    def get_brief(self, iters):
        if self.image_regularization:
            return f'g_loss {self.g_loss:.4f}, d_sketch_loss: {self.d_sketch_loss:.4f}, d_image_loss: {self.d_image_loss:.4f}'
        else:
            return f'g_loss {self.g_loss:.4f}, d_sketch_loss: {self.d_sketch_loss:.4f}'

    def save(self, iters):
        netG = self.netG.state_dict()
        savings = {
            "netD_sketch": self.netD_sketch.state_dict(),
            "netD_image": self.netD_image.state_dict() if self.image_regularization else None,
            "g_optim": self.optim_G.state_dict(),
            "d_sketch_optim": self.optim_D_sketch.state_dict(),
            "d_image_optim": self.optim_D_image.state_dict() if self.image_regularization else None,
        }
        save_path = os.path.join(self.opt['checkpoints_dir'], self.opt['name'], f"{iters}_")
        torch.save(netG, save_path + 'netG.pth')
        torch.save(savings, save_path + 'others.pth')

    def load_pretrain(self):
        self.gan_model.load_state_dict(self.opt['pretrained'])

    def load(self, iters):
        print(f"Resuming model at iteration {iters}")
        load_path = os.path.join(self.opt['checkpoints_dir'], self.opt['name'], f"{iters}_")
        device = self.opt['device']
        self.netG.load_state_dict(torch.load(load_path + 'netG.pth', map_location=device))

        savings = torch.load(load_path + 'others.pth', map_location=device)
        self.netD_sketch.load_state_dict(savings['netD_sketch'])
        self.optim_G.load_state_dict(savings['g_optim'])
        self.optim_D_sketch.load_state_dict(savings['d_sketch_optim'])

        if self.image_regularization:
            self.netD_image.load_state_dict(savings['netD_image'])
            self.optim_D_image.load_state_dict(savings['d_image_optim'])
    
    def d_regularize(self, real_pred, real_img):
        outputs = real_pred.reshape(real_pred.shape[0], -1).mean(1).sum()
        grad_real, = autograd.grad(
            outputs=outputs, inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty
    
    def generate_fake(self, batch_size = None):
        if batch_size is None:
            batch_size = self.opt['']
        noise = torch.randn(batch_size, self.opt['nz'],1,1, device=self.device)
        return self.netG(noise)

    def gradient_penalty(self, net, real, fake):
        alpha = torch.rand(real.shape[0], 1,1,1, device=self.device)
        interpolates = (alpha * real + (1-alpha) * fake).requires_grad_(True)
        # interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        d_interpolates = net(interpolates)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(d_interpolates.size()).to(self.device),
                            create_graph=True, retain_graph=True)[0].reshape(real.shape[0], -1)

        gradient_penalty = ((gradients.norm(p=2, dim=1)-1)**2).mean() * self.opt['lambda_gp']

        return gradient_penalty


        