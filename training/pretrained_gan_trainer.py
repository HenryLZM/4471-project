from gan_model import GANModel

class PretrainedGANTrainer():
    def __init__(self, opt):
        self.opt = opt
        self.device = 'cuda' if self.opt['use_gpu'] else 'cpu'
        self.gan_model = GANModel(opt).to(self.device)
        self.optimizer_G, self.optimizer_D = \
            self.gan_model.create_optimizers(opt)
        self.gan_model.create_loss_fns(opt)
        self.gan_model.set_requires_grad(False, False)
        
    def run_generator_one_step(self, data):
        g_loss, generated = self.gan_model(data, mode='generator')
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()
        self.generated = generated
        self.g_loss = g_loss

    def run_discriminator_one_step(self, data):
        d_loss = self.gan_model(data, mode='discriminator')
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_loss = d_loss

    def train_one_step(self, data):
        self.gan_model.set_requires_grad(False, True)
        self.run_discriminator_one_step(data)
        self.gan_model.set_requires_grad(True, False)
        self.run_generator_one_step(data)