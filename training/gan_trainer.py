from training.gan_model import GANModel
import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

class GANTrainer():
    def __init__(self, opt):
        self.opt = opt
        self.device = opt['device']
        self.gan_model = GANModel(opt).to(self.device)
        self.optimizer_G, self.optimizer_D = self.gan_model.create_optimizers(opt)
        # self.gan_model.create_loss_fns(opt)
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
        

    # def run_discriminator_regularization_one_step(self, data):
    #     d_reg_loss = self.gan_model(data, mode='discriminator-regularize')
    #     self.optimizer_D.zero_grad()
    #     d_reg_loss.backward()
    #     self.optimizer_D.step()
    #     self.d_loss = d_loss.item()

    def train_one_step(self, data, iters):
        self.gan_model.set_requires_grad(False, True)
        self.run_discriminator_one_step(data)
        # if not self.opt['no_d_regularize'] and iters % self.opt['d_reg_every'] == 0:
        #     self.run_discriminator_regularization_one_step(data)
        
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



