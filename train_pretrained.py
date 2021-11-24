from options import pretrained_opt
from training.dataset import create_dataloader
from training.gan_trainer import GANTrainer
from torchvision.transforms import ToPILImage
import torchvision.utils as vutils
import torch

def train_loop():
    opt = pretrained_opt

    dataloader = create_dataloader(
        data_dir = opt['data_dir'],
        base = 0,
        data_len = opt['data_len'],
        batch_size = opt['batch_size'],
        num_workers = opt['num_workers'],
        device =  opt['device']
    )
    real = next(iter(dataloader))[:16]
    grid = vutils.make_grid((real+1)/2, nrow=4, padding=2)
    ToPILImage()(grid).save(opt['visual_dir'] + '/' + opt['name'] + '/00_real.png')

    trainer = GANTrainer(opt)

    resume_iter = opt['resume_iter']
    if resume_iter is None or resume_iter == 0:
        total_iters = 0
    else:
        total_iters = resume_iter
        trainer.load(resume_iter)

    for epoch in range(opt['max_epoch']):
        print(f'epoch: {epoch}')
        for i, data in enumerate(dataloader):
            total_iters += 1
            trainer.train_one_step(data, total_iters)
            if total_iters % opt['print_freq'] == 0:
                print(f'iteration {total_iters}')
                print(trainer.get_latest_losses())

            if total_iters % opt['save_freq'] == 0:
                print("saving")
                trainer.save(total_iters)

            if total_iters % opt['vis_freq'] == 0:
                print("visual")
                with torch.no_grad():
                    fake = trainer.gan_model.generate_fake(16)
                grid = vutils.make_grid((fake+1)/2, nrow=4, padding=2)
                ToPILImage()(grid).save(opt['visual_dir'] + '/' + opt['name'] + f'/{total_iters}_fake.png')
    trainer.save(total_iters)

if __name__ == "__main__":
    print("Start Training\n========================")
    train_loop()
    print("==========================")
    print("Train Finished")
