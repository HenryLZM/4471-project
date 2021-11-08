from torch.utils.data import Dataset, DataLoader
from hyperparameters import batch_size
import h5py
import torch

class FashionDataset(Dataset):
    def __init__(self, dir, purpose, device):
        f = h5py.File(dir)
        self.images = f['ih']
        self.mean = f['ih_mean']
        self.purpose = purpose
        self.device = device

    def __len__(self):
        if self.purpose == 'train':
            return 60000
        elif self.purpose == 'val':
            return 10000
        elif self.purpose == 'test':
            return 8979

    def __getitem__(self, idx):
        if self.purpose == 'train':
            base = 0
        elif self.purpose == 'val':
            base = 60000
        elif self.purpose == 'test':
            base = 70000

        pic = self.images[base+idx] + self.mean
        pic = pic.clip(0,1)
        return torch.tensor(pic, device=self.device)

directory = 'E:/HKUST/2021F/COMP4471/project/data/img/G2.h5'
train_set = FashionDataset(directory, 'train', 'cuda')
val_set = FashionDataset(directory, 'val', 'cuda')
test_set = FashionDataset(directory, 'test', 'cuda')

train_loader = DataLoader(train_set, batch_size, shuffle=True)
val_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(train_set, batch_size, shuffle=True)