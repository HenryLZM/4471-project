from hyperparameters import pretrained_opt
from training.pretrained_dataset import create_dataloader
from training.pretrained_gan_trainer import PretrainedGANTrainer

def train_loop():
    opt = pretrained_opt

    dataloader = create_dataloader(
        opt['data_dir'],
        opt['image_size'],
        opt['batch_size'],
        opt['data_len'],
        opt['device']
    )

    if opt['resume_iter'] is None:
        total_iters = 0
    else:
        total_iters = opt['resume_iter']

    trainer = PretrainedGANTrainer(opt)

    for epoch in range(opt['max_epoch']):
        print(f'epoch: {epoch}')
        for i, data in enumerate(dataloader):
            trainer.train_one_step(data)
            if total_iters % opt['print_freq'] == 0:
                print(trainer.get_latest_losses())

if __name__ == "__main__":
    print("Start Training\n========================")
    train_loop()
    print("==========================")
    print("Train Finished")
