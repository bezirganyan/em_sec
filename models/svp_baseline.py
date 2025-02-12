import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import Adam
from torchmetrics import Accuracy
from svp.multiclass import SVPNet

from metrics import AverageUtility


class CIFAR10SVPModel(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super(CIFAR10SVPModel, self).__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=False)
        self.num_classes = num_classes
        hs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.set_metrics()
        self.set_params = {
            "c": 10,
            "svptype": "fb",
            "beta": 2
        }
        self.flat = SVPNet(phi=self.model, hidden_size=hs, classes=list(range(num_classes)), hierarchy="none")

    def forward(self, x, y=None):
        return self.flat(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x, y)
        self.log('train_loss', loss)
        y_hat = torch.tensor(self.flat.predict(x)).to(y.device)
        self.train_acc(y_hat, y)
        svp_preds_f = self.flat.predict_set(x, self.set_params)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x, y)
        self.log('val_loss', loss)
        y_hat = torch.tensor(self.flat.predict(x)).to(y.device)
        self.val_acc(y_hat, y)
        svp_preds_f = self.flat.predict_set(x, self.set_params)
        y_one_hot = F.one_hot(y, self.num_classes)
        self.val_utility.update(svp_preds_f, y_one_hot)


    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x, y)
        self.log('test_loss', loss)
        y_hat = torch.tensor(self.flat.predict(x)).to(y.device)
        self.test_acc(y_hat, y)
        svp_preds_f = self.flat.predict_set(x, self.set_params)
        y_one_hot = F.one_hot(y, self.num_classes)
        self.test_utility.update(svp_preds_f, y_one_hot)

    def on_train_epoch_end(self) -> None:
        self.log('train_acc', self.train_acc.compute())

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_utility', self.val_utility.compute(), prog_bar=True)

    def on_test_epoch_end(self) -> None:
        self.log('test_acc', self.test_acc.compute())
        self.log('test_utility', self.test_utility.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def set_metrics(self):
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)

        self.train_utility = AverageUtility(self.num_classes, tolerance=0.7)
        self.val_utility = AverageUtility(self.num_classes, tolerance=0.7)
        self.test_utility = AverageUtility(self.num_classes, tolerance=0.7)