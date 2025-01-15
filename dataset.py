import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=1, data_dir='../data'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True,
                                                   transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(dataset, [45000, 5000])
        if stage == 'test' or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True,
                                                             transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          persistent_workers=True)
