import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from torch.optim import Adam
from torchmetrics import Accuracy

from models.conv_models import BasicBlock, ResNet
from models.ds_utils import DM_test, DSNet  # DSNet: your DS-based network implementation
from metrics import AverageUtility, HyperAccuracy, SetSize, TimeLogger, compute_weights


# ---------------- Helper: Power Set ----------------
def power_set(items):
    N = len(items)
    set_all = []
    for i in range(1, 2 ** N):  # skip empty set (i == 0)
        combo = []
        for j in range(N):
            if (i >> j) & 1:
                combo.append(items[j])
        set_all.append(combo)
    return sorted(set_all, key=lambda s: (len(s), s))

# ---------------- CIFAR10DSModel Lightning Module ----------------
class CIFAR10DSModel(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, prototypes=200, nu=0.9, tol_i=2):
        super(CIFAR10DSModel, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Backbone: ResNet for feature extraction.
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        hs = self.model.linear.in_features
        self.model.linear = nn.Identity()

        # DS network: converts features into evidential outputs.
        self.DS = DSNet(prototypes, num_classes, hs)  # outputs shape: (batch, num_classes+1)

        # Compute the nonempty power set for set-valued predictions.
        self.act_set = power_set(list(range(num_classes)))
        cache_file = f'cache/weights_{num_classes}.pt'
        if os.path.exists(cache_file):
            weight_matrix = torch.load(cache_file, weights_only=False)
        else:
            print(f"Computing weights for {num_classes} classes for OWA utility metric.")
            weight_matrix = compute_weights(num_classes)
            torch.save(weight_matrix, cache_file)
        # DM_test: computes utility scores for each possible set.
        self.DM_test = DM_test(num_classes, self.act_set, weight_matrix, tol_i, nu)

        self.set_params = {"c": num_classes, "svptype": "ds", "nu": nu}
        self.set_metrics()

    def forward(self, x, y=None):
        features = self.model(x)
        outputs = self.DS(features)  # DS evidences (batch, num_classes+1)
        return outputs

    def predict(self, x):
        # Use the "precise" part of DS outputs (first num_classes entries) for class prediction.
        outputs = self(x)
        preds = outputs[:, :self.num_classes].argmax(dim=1)
        return preds.cpu().numpy()

    def predict_set(self, x, set_params):
        # Apply DM_test to DS outputs to get utility scores.
        outputs = self(x)  # (batch, num_classes+1)
        utility_scores = self.DM_test(outputs)  # (batch, num_set)
        set_indices = utility_scores.argmax(dim=1).cpu().numpy()
        set_preds = [self.act_set[idx] for idx in set_indices]
        return set_preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)  # DS evidences
        # Compute loss using negative log likelihood on log-probabilities (from the "precise" part)
        log_probs = torch.log(outputs[:, :self.num_classes] + 1e-8)
        loss = F.nll_loss(log_probs, y)
        self.log('train_loss', loss)
        preds = outputs[:, :self.num_classes].argmax(dim=1)
        self.train_multiclass_acc(preds, y)
        _ = self.predict_set(x, self.set_params)  # For consistency with baseline.
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs[:, :self.num_classes], y)
        self.log('val_loss', loss, prog_bar=True)
        preds = outputs[:, :self.num_classes].argmax(dim=1)
        self.val_multiclass_acc(preds, y)
        # Set-valued predictions via DM_test.
        set_preds = self.predict_set(x, self.set_params)
        # Convert predictions to tensors for metric updates.
        set_preds_tensor = [torch.tensor(p).to(y.device) for p in set_preds]
        y_one_hot = F.one_hot(y, self.num_classes)
        self.val_acc.update(set_preds_tensor, y_one_hot)
        self.val_set_size.update(set_preds_tensor, y_one_hot)
        for k, metric in self.val_utility_dict.items():
            if metric.device != y_one_hot.device:
                metric.to(y_one_hot.device)
            metric.update(set_preds_tensor, y_one_hot)

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs[:, :self.num_classes], y)
        self.log('test_loss', loss)
        preds = outputs[:, :self.num_classes].argmax(dim=1)
        self.test_multiclass_acc(preds, y)
        time_start = time.time()
        set_preds = self.predict_set(x, self.set_params)
        duration = time.time() - time_start
        self.test_time_logger.update(duration)
        set_preds_tensor = [torch.tensor(p).to(y.device) for p in set_preds]
        y_one_hot = F.one_hot(y, self.num_classes)
        self.test_set_size.update(set_preds_tensor, y_one_hot)
        self.test_acc.update(set_preds_tensor, y_one_hot)
        for k, metric in self.test_utility_dict.items():
            if metric.device != y_one_hot.device:
                metric.to(y_one_hot.device)
            metric.update(set_preds_tensor, y_one_hot)

    def on_train_epoch_end(self) -> None:
        self.log('train_multiclass_acc', self.train_multiclass_acc.compute())

    def on_validation_epoch_end(self) -> None:
        val_acc = self.val_acc.compute()
        val_set_size = self.val_set_size.compute()
        self.log('val_multiclass_acc', val_acc, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_set_size', val_set_size, prog_bar=True)
        wandb.log({"val_set_size": val_set_size}, step=self.current_epoch)
        wandb.log({"val_acc": val_acc}, step=self.current_epoch)
        for k, metric in self.val_utility_dict.items():
            val_util = metric.compute()
            self.log(f'val_{k}', val_util)
            wandb.log({f'val_{k}': val_util}, step=self.current_epoch)

    def on_test_epoch_end(self) -> None:
        test_acc = self.test_acc.compute()
        test_set_size = self.test_set_size.compute()
        self.log('test_multiclass_acc', test_acc)
        self.log('test_acc', test_acc)
        self.log('test_set_size', test_set_size)
        self.log('test_time', self.test_time_logger.compute())
        wandb.log({"test_set_size": test_set_size}, step=self.current_epoch)
        wandb.log({"test_acc": test_acc}, step=self.current_epoch)
        wandb.log({"test_time": self.test_time_logger.compute()}, step=self.current_epoch)
        for k, metric in self.test_utility_dict.items():
            test_util = metric.compute()
            self.log(f'test_{k}', test_util)
            wandb.log({f'test_{k}': test_util}, step=self.current_epoch)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def set_metrics(self):
        self.train_multiclass_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_multiclass_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_multiclass_acc = Accuracy(task='multiclass', num_classes=self.num_classes)

        self.train_acc = HyperAccuracy()
        self.val_acc = HyperAccuracy()
        self.test_acc = HyperAccuracy()

        self.val_utility_dict = {
            'fb_1': AverageUtility(self.num_classes, utility='fb', beta=1),
            'fb_2': AverageUtility(self.num_classes, utility='fb', beta=2),
            'fb_3': AverageUtility(self.num_classes, utility='fb', beta=3),
            'fb_4': AverageUtility(self.num_classes, utility='fb', beta=4),
            'fb_5': AverageUtility(self.num_classes, utility='fb', beta=5),
            'owa_0.5': AverageUtility(self.num_classes, utility='owa', tolerance=0.5),
            'owa_0.6': AverageUtility(self.num_classes, utility='owa', tolerance=0.6),
            'owa_0.7': AverageUtility(self.num_classes, utility='owa', tolerance=0.7),
            'owa_0.8': AverageUtility(self.num_classes, utility='owa', tolerance=0.8),
            'owa_0.9': AverageUtility(self.num_classes, utility='owa', tolerance=0.9)
        }

        self.test_utility_dict = {
            'fb_1': AverageUtility(self.num_classes, utility='fb', beta=1),
            'fb_2': AverageUtility(self.num_classes, utility='fb', beta=2),
            'fb_3': AverageUtility(self.num_classes, utility='fb', beta=3),
            'fb_4': AverageUtility(self.num_classes, utility='fb', beta=4),
            'fb_5': AverageUtility(self.num_classes, utility='fb', beta=5),
            'owa_0.5': AverageUtility(self.num_classes, utility='owa', tolerance=0.5),
            'owa_0.6': AverageUtility(self.num_classes, utility='owa', tolerance=0.6),
            'owa_0.7': AverageUtility(self.num_classes, utility='owa', tolerance=0.7),
            'owa_0.8': AverageUtility(self.num_classes, utility='owa', tolerance=0.8),
            'owa_0.9': AverageUtility(self.num_classes, utility='owa', tolerance=0.9)
        }

        self.val_set_size = SetSize()
        self.test_set_size = SetSize()
        self.test_time_logger = TimeLogger()