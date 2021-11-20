from torch.utils.data import Dataset, DataLoader
import h5py
import torch
from torchvision.transforms import Normalize, ToPILImage, functional
import numpy as np

class FashionDataset(Dataset):
    def __init__(self, dir, base, length, device):
        f = h5py.File(dir)
        self.masks = f['b_']
        self.images = f['ih']
        self.mean = np.array(f['ih_mean'])
        self.device = device
        self.length = length
        self.base = base

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        base = self.base
        pic = self.images[base+idx] + self.mean
        pic = pic + (self.masks[base+idx] == 0)
        pic = pic.clip(0,1)
        pic = torch.tensor(pic, device=self.device)
        pic = functional.rotate(pic, -90)
        pic = Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(pic)
        return pic

def create_dataloader(data_dir, base, data_len, batch_size, num_workers, device):
    dataset = FashionDataset(data_dir, base, data_len, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader