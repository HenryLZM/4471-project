from torch.utils.data import Dataset, DataLoader
import h5py
import torch
from torchvision.transforms import Normalize, ToPILImage, functional, ToTensor
import numpy as np
import os
from PIL import Image


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

class SketchDataset(Dataset):
    def __init__(self, data_dir, device):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.data_dir + '/' + self.files[idx]
        pic = Image.open(path)
        pic = ToTensor()(pic).to(self.device)
        pic = pic*2 - 1 
        return pic

def create_dataloader(data_dir, base, data_len, batch_size, num_workers, device):
    dataset = FashionDataset(data_dir, base, data_len, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader

def create_dataset(data_dir, base, data_len, device):
    return FashionDataset(data_dir, base, data_len, device)

def sketch_dataset(data_dir, device):
    return SketchDataset(data_dir, device)

def sketch_dataloader(data_dir, batch_size, device):
    dataset = SketchDataset(data_dir, device)
    return DataLoader(dataset, batch_size=batch_size)