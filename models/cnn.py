import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
from torch.optim import Adam
from torchmetrics import Accuracy


class StandardModel(pl.LightningModule):
    def __init__(self, model, num_classes=10, learning_rate=1e-3):
        super(StandardModel, self).__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.model = model
        self.num_classes = num_classes
        self.set_metrics()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        acc = self.train_acc(y_hat, y)
        self.log('train_acc', acc, prog_bar=True)
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
        wandb.log({'train_acc': self.train_acc.compute()}, step=self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        wandb.log({'val_acc': self.val_acc.compute()}, step=self.current_epoch)

    def on_test_epoch_end(self) -> None:
        self.log('test_acc', self.test_acc.compute())
        wandb.log({'test_acc': self.test_acc.compute()}, step=self.current_epoch)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        # optimizer = SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        # scheduler = CosineAnnealingLR(optimizer, T_max=200)
        # return [optimizer], [scheduler]

    def set_metrics(self):
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
