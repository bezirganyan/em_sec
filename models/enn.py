import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import wandb
from torch.optim import Adam
from torchmetrics import Accuracy, CalibrationError

from losses import get_evidential_loss
from metrics import CorrectIncorrectUncertaintyPlotter
from models.conv_models import BasicBlock, ResNet


class CIFAR10EnnModel(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, uncertainty_calibration=False):
        super(CIFAR10EnnModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.num_classes = num_classes
        self.set_metrics()
        self.uncertainty_calibration = uncertainty_calibration

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        evidence = torch.nn.functional.softplus(logits)
        loss = get_evidential_loss(evidence, y, self.current_epoch, self.num_classes, 10, self.device,
                                   uncertainty_calibration=self.uncertainty_calibration)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss)
        y_hat = torch.argmax(logits, dim=1)
        self.train_acc(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss)
        y_hat = torch.argmax(logits, dim=1)
        self.val_acc(y_hat, y)

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch, batch_idx)
        self.log('test_loss', loss)
        y_hat = torch.argmax(logits, dim=1)
        self.test_acc(y_hat, y)
        self.cor_unc_plot.update(torch.exp(logits), y)
        self.test_ece.update(logits, y)

    def on_train_epoch_end(self) -> None:
        self.log('train_acc', self.train_acc.compute())
        wandb.log({'train_acc': self.train_acc.compute()}, step=self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        wandb.log({'val_acc': self.val_acc.compute()}, step=self.current_epoch)

    def on_test_epoch_end(self) -> None:
        self.log('test_acc', self.test_acc.compute())
        self.log('test_ece', self.test_ece.compute())
        self.cor_unc_plot.plot()
        wandb.log({'test_acc': self.test_acc.compute()}, step=self.current_epoch)
        wandb.log({'test_ece': self.test_ece.compute()}, step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def set_metrics(self):
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_ece = CalibrationError(task='multiclass', num_classes=self.num_classes)

        self.cor_unc_plot = CorrectIncorrectUncertaintyPlotter()
