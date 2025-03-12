import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torchaudio
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.transforms import v2


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

        return sample.float(), label

class RxRx1DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=1, data_dir='../data'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

        self.transform = None

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



class LUMADataset(Dataset):
    def __init__(self, image_path, image_transform=None, target_transform=None, ood=False):
        self.image_path = image_path
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.ood = ood
        self._load_data()

        self.label_mapping = {'man': 0, 'boy': 1, 'house': 2, 'woman': 3, 'girl': 4, 'table': 5, 'road': 6, 'horse': 7,
                              'dog': 8, 'ship': 9, 'bird': 10, 'mountain': 11, 'bed': 12, 'train': 13, 'bridge': 14,
                              'fish': 15, 'cloud': 16, 'chair': 17, 'cat': 18, 'baby': 19, 'castle': 20, 'forest': 21,
                              'television': 22, 'bear': 23, 'camel': 24, 'sea': 25, 'fox': 26, 'plain': 27, 'bus': 28,
                              'snake': 29, 'lamp': 30, 'clock': 31, 'lion': 32, 'tank': 33, 'palm': 34, 'rabbit': 35,
                              'pine': 36, 'cattle': 37, 'oak': 38, 'mouse': 39, 'frog': 40, 'ray': 41, 'bicycle': 42,
                              'truck': 43, 'elephant': 44, 'roses': 45, 'wolf': 46, 'telephone': 47, 'bee': 48,
                              'whale': 49}

    def _load_data(self):
        # Load data from file
        self.image_data = pd.read_pickle(self.image_path)
        self.targets = self.image_data['label'].values

    def __getitem__(self, index):
        image = self.image_data.loc[:, 'image'].iloc[index]
        target = self.label_mapping[self.targets[index]] if not self.ood else 0

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.image_data)


class LUMADataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=1, data_dir='../LUMA/data'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.test_transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(*stats)
        ])

        self.train_transform = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(),
            v2.Normalize(*stats)
        ])
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dataset = LUMADataset(os.path.join(self.data_dir, 'image_data.pickle'), image_transform=self.train_transform)
            print(len(dataset))
            self.train_dataset, self.val_dataset = random_split(dataset, [91000, 10000])
        if stage == 'test' or stage is None:
            self.test_dataset = LUMADataset(os.path.join(self.data_dir, 'image_data_test.pickle'),
                                            image_transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          persistent_workers=True)