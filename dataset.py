import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=1, data_dir='../data'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True,
                                                   transform=self.train_transform)
            self.train_dataset, self.val_dataset = random_split(dataset, [45000, 5000])
        if stage == 'test' or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True,
                                                             transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          persistent_workers=True)


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=1, data_dir='../data'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dataset = torchvision.datasets.CIFAR100(root=self.data_dir, train=True, download=True,
                                                   transform=self.train_transform)
            self.train_dataset, self.val_dataset = random_split(dataset, [45000, 5000])
        if stage == 'test' or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR100(root=self.data_dir, train=False, download=True,
                                                             transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          persistent_workers=True)


class RxRx1Dataset(Dataset):
    def __init__(self, data_dir, transform=None, stage='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.stage = stage

        self.data = torch.load(os.path.join(data_dir, f'embeddings.pt'))
        self.metadata = pd.read_csv(os.path.join(data_dir, f'metadata.csv'))

        self.data_idx = self.metadata[self.metadata['dataset'] == stage].index
        self.data = self.data[self.data_idx]
        self.labels = self.metadata[self.metadata['dataset'] == stage]['sirna_id'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

class RxRx1DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=1, data_dir='../data'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          persistent_workers=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = RxRx1Dataset(self.data_dir, transform=self.transform, stage='train')
            n_train = int(len(self.train_dataset) * 0.8)
            n_val = len(self.train_dataset) - n_train
            self.train_dataset, self.val_dataset = random_split(self.train_dataset, [n_train, n_val])
        if stage == 'test' or stage is None:
            self.test_dataset = RxRx1Dataset(self.data_dir, transform=self.transform, stage='test')
