import torch

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from models.conv_models import BasicBlock, FitNet4, ResNet
import torchvision.models as models
from torch.optim import Adam
from torchmetrics import Accuracy

from models.ds_utils import DM, DSNet


class CIFAR10DSModel(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=0.001, prototypes=200):
        super(CIFAR10DSModel, self).__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.num_classes = num_classes
        self.DS = DSNet(prototypes, num_classes, self.model.linear.in_features)
        self.model.linear = nn.Identity()
        self.set_metrics()

    def forward(self, x):
        features = self.model(x)
        outputs = self.DS(features)
        # evidences = F.elu(mass_Dempster_normalize)
        # beliefs = evidences / (evidences + 1).sum(dim=1, keepdim=True)
        # uncertainty = self.num_classes / (evidences + 1).sum(dim=1, keepdim=True)
        # mass_Dempster_normalize = torch.cat([beliefs, uncertainty], dim=1)
        # outputs = self.utility(mass_Dempster_normalize)
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        log_probs = torch.log(y_hat + 1e-8)
        loss = F.nll_loss(log_probs, y)
        self.log('train_loss', loss, prog_bar=True)
        self.train_acc(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
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
        return Adam(self.parameters(), lr=self.learning_rate)

    def set_metrics(self):
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)

