import torch
from training.networks import GANLoss
from training.networks import stylegan2
from training.networks import RegularizeD
import os

class PretrainedGANModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.use_gpu = opt['use_gpu']
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        
        self.netG, self.netD = self.initialize_networks(opt)
    
        # self.tf_real = networks.OutputTransform(opt, process=opt.transform_real, diffaug_policy=opt.diffaug_policy)
        # self.tf_fake = networks.OutputTransform(opt, process=opt.transform_fake, diffaug_policy=opt.diffaug_policy)

    def forward(self, data, mode):
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss()
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(data)
            return d_loss
        elif mode == 'discriminator-regularize':
            assert not self.opt['no_d_regularize'], "Discriminator shouldn't be regularized with no_d_regularize applied"
            d_reg_loss = self.compute_discriminator_regularization(data)
            return d_reg_loss
        else:
            raise ValueError("|mode| is invalid")


    def initialize_networks(self, opt):
        w_shift = opt['optim_param_g'] == 'w_shift'
        netG = stylegan2.Generator(
            opt['image_size'], opt['z_dim'], opt['n_mlp'],lr_mlp=opt['lr_mlp'],channel_multiplier=opt['channel_multiplier'],w_shift=w_shift
        )
        netD = stylegan2.Discriminator(
            opt['image_size'], opt['channel_multiplier']
        )

        if opt['g_pretrained'] != '':
            netG.load_state_dict(torch.load(opt['g_pretrained']))

        if opt['d_pretrained'] != '':
            netD.load_state_dict(torch.load(opt['d_pretrained']))
        
        return netG, netD

    def compute_generator_loss(self):
        fake_image = self.generate_fake()
        
        pred_fake = self.netD(fake_image)
        g_loss = self.criterionGAN(pred_fake, True, for_discriminator=False)
        
        return g_loss, fake_image.detach()

    def compute_discriminator_loss(self, real):
        with torch.no_grad():
            fake = self.generate_fake().detach()

        # fake = self.tf_fake(fake)
        # real = self.tf_real(real)
        pred_fake = self.netD(fake)
        pred_real = self.netD(real)
        d_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        d_real = self.criterionGAN(pred_real, True, for_discriminator=True)

        return d_fake+d_real

    def compute_discriminator_regularization(self, data):
        data.requires_grad = True
        pred_real = self.netD(data)
        r1_loss = self.d_regularize(pred_real, data)
        d_reg_loss = self.opt['r1'] / 2 * r1_loss * self.opt['d_reg_every']
        return d_reg_loss
    
    def generate_fake(self, batch_size=None, style_mix=True, return_latents=False):
        if batch_size is None:
            batch_size = self.opt['batch_size']
        
        device = 'cuda' if self.use_gpu else 'cpu'
        style_mix_prob = self.opt['mixing'] if style_mix else 0
        noises = mixing_noise(batch_size, self.opt['z_dim'], style_mix_prob, device)
        
        fake_image, latents = self.netG(noises, return_latents=return_latents)
        if return_latents:
            return fake_image, latents
        return fake_image

    def create_loss_fns(self, opt):
        self.criterionGAN = GANLoss(opt['gan_mode'], tensor=self.FloatTensor, opt=self.opt)
        if not opt['no_d_regularize']:
            self.d_regularize = RegularizeD()

    def create_optimizers(self, opt):
        optim_param_g = opt['optim_param_g']
        if optim_param_g == 'style':
            G_param_names, G_params = get_param_by_name(self.netG, 'style')
        elif optim_param_g == 'w_shift':
            G_param_names, G_params = get_param_by_name(self.netG, 'w_shift')
        else:
            raise ValueError("optim_param_g should be 'style' or 'w_shift', but get ", optim_param_g)
        self.G_param_names, self.G_params = G_param_names, G_params
        
        D_params = list(self.netD.parameters())
        self.D_params = D_params


        d_reg_ratio = opt['d_reg_every'] / (opt['d_reg_every'] + 1) if not opt['no_d_regularize'] else 1.
        lr = opt['lr']
        beta1, beta2 = opt['beta1'], opt['beta2']
        G_lr, D_lr = lr, lr**d_reg_ratio
        G_beta1, D_beta1 = beta1, beta1 ** d_reg_ratio
        G_beta2, D_beta2 = beta2, beta2 ** d_reg_ratio

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
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt['name'], f"{iters}_net_")
        torch.save(self.netG.state_dict(), save_path + 'G.pth')
        torch.save(self.netD.state_dict(), save_path + 'D.pth')

    @torch.no_grad()
    def inference(self, noise, trunc_psi=1.0, mean_latent=None, with_tf=False):
        # If mean_latent is None, make one
        if trunc_psi < 1 and mean_latent is None:
            mean_latent = self.get_mean_latent(n_samples=self.opt.latent_avg_samples)

        img = self.netG([noise], truncation=trunc_psi, truncation_latent=mean_latent)[0]

        # if with_tf is True, output the transformed version (im2sketch) as well
        if with_tf:
            return img, self.tf_fake(img, apply_aug=False)
        return img
    
    def get_mean_latent(self, n_samples=8192):
        return self.netG.mean_latent(n_samples)


def mixing_noise(batch, latent_dim, prob, device):
    """Generate 1 or 2 set of noises for style mixing."""
    if prob > 0 and random.random() < prob:
        return torch.randn(2, batch, latent_dim, device=device).unbind(0)
    else:
        return [torch.randn(batch, latent_dim, device=device)]

def get_param_by_name(net, tgt_param):
    """Get parameters (and their names) that contain tgt_param in net."""
    name_list, param_list = [], []
    for name, param in net.named_parameters():
        if tgt_param in name:  # target layer
            name_list.append(name)
            param_list.append(param)
    return name_list, param_list
