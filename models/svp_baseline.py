import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import wandb
from torch.optim import Adam
from torchmetrics import Accuracy
from svp_py.multiclass import SVPNet

from metrics import AverageUtility, SetSize
from models.conv_models import BasicBlock, ResNet


class CIFAR10SVPModel(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, beta=1):
        super(CIFAR10SVPModel, self).__init__()
        self.save_hyperparameters()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.num_classes = num_classes
        self.beta_param = beta
        hs = self.model.linear.in_features
        self.model.linear = nn.Identity()
        self.set_metrics()
        self.set_params = {
            "c": self.num_classes,
            "svptype": "fb",
            "beta": self.beta_param
        }
        try:
            self.flat = SVPNet(phi=self.model, hidden_size=hs, classes=list(range(num_classes)), hierarchy="none")
        except TypeError:
            self.flat = SVPNet(phi=self.model, hidden_size=hs, classes=list(range(num_classes)), hierarchy="none", dropout=0.0)

    def forward(self, x, y=None):
        return self.flat(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x, y)
        self.log('train_loss', loss)
        y_hat = torch.tensor(self.flat.predict(x)).to(y.device)
        self.train_acc(y_hat, y)
        # svp_preds_f = self.flat.predict_set(x, self.set_params)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x, y)
        self.log('val_loss', loss)
        y_hat = torch.tensor(self.flat.predict(x)).to(y.device)
        self.val_acc(y_hat, y)
        svp_preds_f = self.flat.predict_set(x, self.set_params)
        svp_preds_f = [torch.tensor(p).to(y.device) for p in svp_preds_f]
        y_one_hot = F.one_hot(y, self.num_classes)
        self.val_set_size.update(svp_preds_f, y_one_hot)
        for k, v in self.val_utility_dict.items():
            if v.device != y_one_hot.device:
                v.to(y_one_hot.device)
            v.update(svp_preds_f, y_one_hot)


    def test_step(self, batch, batch_idx):
        x, y = batch
        # loss = self(x, y)
        # self.log('test_loss', loss)
        # y_hat = torch.tensor(self.flat.predict(x)).to(y.device)
        # self.test_acc(y_hat, y)
        svp_preds_f = self.flat.predict_set(x, self.set_params)
        svp_preds_f = [torch.tensor(p).to(y.device) for p in svp_preds_f]
        y_one_hot = F.one_hot(y, self.num_classes)
        self.test_set_size.update(svp_preds_f, y_one_hot)
        for k, v in self.test_utility_dict.items():
            if v.device != y_one_hot.device:
                v.to(y_one_hot.device)
            v.update(svp_preds_f, y_one_hot)

    def on_train_epoch_end(self) -> None:
        self.log('train_acc', self.train_acc.compute())

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_set_size', self.val_set_size.compute(), prog_bar=True)
        wandb.log({"val_set_size": self.val_set_size.compute()}, step=self.current_epoch)
        wandb.log({"val_acc": self.val_acc.compute()}, step=self.current_epoch)
        for k, v in self.val_utility_dict.items():
            self.log(f'val_{k}', v.compute())
            wandb.log({f'val_{k}': v.compute()}, step=self.current_epoch)

    def on_test_epoch_end(self) -> None:
        self.log('test_acc', self.test_acc.compute())
        self.log('test_set_size', self.test_set_size.compute())
        wandb.log({"test_set_size": self.test_set_size.compute()}, step=self.current_epoch)
        wandb.log({"test_acc": self.test_acc.compute()}, step=self.current_epoch)
        for k, v in self.test_utility_dict.items():
            self.log(f'test_{k}', v.compute())
            wandb.log({f'test_{k}': v.compute()}, step=self.current_epoch)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def set_metrics(self):
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)

        self.val_utility_dict = {
            'fb_1': AverageUtility(self.num_classes, utility='fb', beta=1),
            'fb_2': AverageUtility(self.num_classes, utility='fb', beta=2),
            'fb_3': AverageUtility(self.num_classes, utility='fb', beta=3),
            'fb_4': AverageUtility(self.num_classes, utility='fb', beta=4),
            'fb_5': AverageUtility(self.num_classes, utility='fb', beta=5),
            # 'owa_0.5': AverageUtility(self.num_classes, utility='owa', tolerance=0.5),
            # 'owa_0.6': AverageUtility(self.num_classes, utility='owa', tolerance=0.6),
            # 'owa_0.7': AverageUtility(self.num_classes, utility='owa', tolerance=0.7),
            # 'owa_0.8': AverageUtility(self.num_classes, utility='owa', tolerance=0.8),
            # 'owa_0.9': AverageUtility(self.num_classes, utility='owa', tolerance=0.9)
        }

        self.test_utility_dict = {
            'fb_1': AverageUtility(self.num_classes, utility='fb', beta=1),
            'fb_2': AverageUtility(self.num_classes, utility='fb', beta=2),
            'fb_3': AverageUtility(self.num_classes, utility='fb', beta=3),
            'fb_4': AverageUtility(self.num_classes, utility='fb', beta=4),
            'fb_5': AverageUtility(self.num_classes, utility='fb', beta=5),
            # 'owa_0.5': AverageUtility(self.num_classes, utility='owa', tolerance=0.5),
            # 'owa_0.6': AverageUtility(self.num_classes, utility='owa', tolerance=0.6),
            # 'owa_0.7': AverageUtility(self.num_classes, utility='owa', tolerance=0.7),
            # 'owa_0.8': AverageUtility(self.num_classes, utility='owa', tolerance=0.8),
            # 'owa_0.9': AverageUtility(self.num_classes, utility='owa', tolerance=0.9)
        }

        self.val_set_size = SetSize()
        self.test_set_size = SetSize()