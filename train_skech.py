from options import sketch_opt
from training.dataset import create_dataloader, sketch_dataloader
from training.gan_trainer import SketchTrainer
from torchvision.transforms import ToPILImage
import torchvision.utils as vutils
import torch

def train_loop():
    opt = sketch_opt

    image_loader = None
    if opt['l_image'] > 0:
        image_loader = create_dataloader(
            data_dir = opt['data_dir'],
            base = 0,
            data_len = opt['data_len'],
            batch_size = opt['batch_size'],
            num_workers = opt['num_workers'],
            device =  opt['device']
        )
    
    sketch_loader = sketch_dataloader(
        opt['sketch_dir'],
        opt['sketch_batch'],
        opt['device']
    )
    # real = next(iter(dataloader))[:16]
    # grid = vutils .make_grid((real+1)/2, nrow=4, padding=2)
    # ToPILImage()(grid).save(opt['visual_dir'] + '/' + opt['name'] + '/00_real.png')

    trainer = SketchTrainer(opt)

    resume_iter = opt['resume_iter']
    if resume_iter is None or resume_iter == 0:
        total_iters = 0
    else:
        total_iters = resume_iter
        trainer.load(resume_iter)

    for epoch in range(opt['max_epoch']):
        print(f'epoch: {epoch}')
        for i, data_sketch in enumerate(sketch_loader):
            total_iters += 1
            data_image = None
            if image_loader is not None:
                data_image = next(iter(image_loader))
            trainer.train_one_step(data_sketch, total_iters, data_image)

            if total_iters % opt['print_freq'] == 0:
                print(f'iteration {total_iters}')
                print(trainer.get_brief(total_iters))

            if total_iters % opt['save_freq'] == 0:
                print("saving")
                trainer.save(total_iters)

            if total_iters % opt['vis_freq'] == 0:
                print("visual")
                with torch.no_grad():
                    fake = trainer.generate_fake(16)
                    fake = (fake+1)/2
                    fake_sketch = trainer.hed(fake*255)

                grid_image = vutils.make_grid(fake, nrow=4, padding=2)
                ToPILImage()(grid_image).save(opt['visual_dir'] + '/' + opt['name'] + f'/{total_iters}_fake.png')
                grid_sketch = vutils.make_grid(fake_sketch, nrow=4, padding=2)
                ToPILImage()(grid_sketch).save(opt['visual_dir'] + '/' + opt['name'] + f'/{total_iters}_sketch.png')

    trainer.save(total_iters)

if __name__ == "__main__":
    print("Start Training\n========================")
    train_loop()
    print("==========================")
    print("Train Finished")
