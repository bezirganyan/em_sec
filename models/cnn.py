import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import Adam
from torchmetrics import Accuracy


class CIFAR10Model(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super(CIFAR10Model, self).__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=False)
        self.num_classes = num_classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.set_metrics()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.train_acc(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.val_acc(y_hat, y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        self.test_acc(y_hat, y)

    def on_train_epoch_end(self) -> None:
        self.log('train_acc', self.train_acc.compute())

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)

    def on_test_epoch_end(self) -> None:
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def set_metrics(self):
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)

