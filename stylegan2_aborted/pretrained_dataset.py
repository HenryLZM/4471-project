from torch.utils.data import Dataset, DataLoader
import h5py
import torch
from torchvision.transforms import Normalize, ToPILImage
import numpy as np

class FashionDataset(Dataset):
    def __init__(self, dir, length, device):
        f = h5py.File(dir)
        self.masks = f['b_']
        self.images = f['ih']
        self.mean = np.array(f['ih_mean'])
        self.device = device
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        base = 0

        pic = self.images[base+idx] + self.mean
        pic = pic.clip(0,1) * ((self.masks[base+idx] != 0))
        pic = torch.tensor(pic, device=self.device)
        pic = Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(pic)
        return pic

def create_dataloader(data_dir, img_size, batch_size, data_len, device):
    dataset = FashionDataset(data_dir, data_len, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader