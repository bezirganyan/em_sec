import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split
from torchvision.transforms import v2
import scipy.io as sio



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


class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.num_views = data_X.shape[0]
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X[v])

        self.Y = data_Y
        self.Y = np.squeeze(self.Y)
        if np.min(self.Y) == 1:
            self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int64)
        self.num_classes = len(np.unique(self.Y))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = (self.X[0][index]).astype(np.float32)
        target = self.Y[index]
        return data, target

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
        if addConflict:
            self.addConflict(index, ratio_conflict)
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        pass

    def addConflict(self, index, ratio):
        records = dict()
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0]
            temp = dict()
            for v in range(self.num_views):
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.num_views)
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
        pass


def CalTech():
    # dims of views: 484 256 279
    data_path = "data/2view-caltech101-8677sample.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    # Take the first 10 categories
    data_Y = data_Y - 1
    data_X[0] = data_X[0][:, data_Y.reshape(-1) < 10]
    data_X[1] = data_X[1][:, data_Y.reshape(-1) < 10]
    data_Y = data_Y[data_Y < 10]
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("CalTech", data_X, data_Y)


class CalTechDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = CalTech()
        num_samples = len(dataset)
        self.num_classes = dataset.num_classes
        self.num_views = dataset.num_views
        self.dims = dataset.dims
        index = np.arange(num_samples)
        #set seed
        np.random.seed(42)
        np.random.shuffle(index)
        train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]
        train_index, val_index = train_index[:int(0.8 * len(train_index))], train_index[int(0.8 * len(train_index)):]
        dataset.addNoise(train_index, 1.0, 5)
        dataset.addNoise(val_index, 1.0, 5)
        dataset.addNoise(test_index, 1.0, 5)
        self.train_dataset = Subset(dataset, train_index)
        self.test_dataset = Subset(dataset, test_index)
        self.val_dataset = Subset(dataset, val_index)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
