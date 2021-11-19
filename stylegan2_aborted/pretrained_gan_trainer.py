from training.pretrained_gan_model import PretrainedGANModel

class PretrainedGANTrainer():
    def __init__(self, opt):
        self.opt = opt
        self.device = 'cuda' if self.opt['use_gpu'] else 'cpu'
        self.gan_model = PretrainedGANModel(opt).to(self.device)
        self.optimizer_G, self.optimizer_D = \
            self.gan_model.create_optimizers(opt)
        self.gan_model.create_loss_fns(opt)
        self.gan_model.set_requires_grad(False, False)
        self.g_loss = None,
        self.d_loss = None,

    def run_generator_one_step(self, data):
        g_loss, generated = self.gan_model(data, mode='generator')
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()
        self.generated = generated
        self.g_loss = g_loss.item()

    def run_discriminator_one_step(self, data):
        d_loss = self.gan_model(data, mode='discriminator')
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_loss = d_loss.item()

    def run_discriminator_regularization_one_step(self, data):
        d_reg_loss = self.gan_model(data, mode='discriminator-regularize')
        self.optimizer_D.zero_grad()
        d_reg_loss.backward()
        self.optimizer_D.step()
        self.d_loss = d_loss.item()

    def train_one_step(self, data):
        self.gan_model.set_requires_grad(False, True)
        self.run_discriminator_one_step(data)
        if not self.opt['no_d_regularize'] and iters % self.opt['d_reg_every'] == 0:
            self.run_discriminator_regularization_one_step(data)
        self.gan_model.set_requires_grad(True, False)
        self.run_generator_one_step(data)

    def get_latest_losses(self):
        return f"g_loss: {self.g_loss:.4f}, d_loss: {self.d_loss:.4f}"

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