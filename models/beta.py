import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import Adam
from torcheval.metrics import MultilabelAccuracy

from losses import ava_edl_criterion
from metrics import BetaEvidenceAccumulator, PredictionSetSize


class BetaModel(pl.LightningModule):
    def __init__(self, model, num_classes=10, learning_rate=1e-3, beta=1, lambda_param=1):
        super(BetaModel, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.beta_param = beta
        self.lambda_param = lambda_param
        self.num_classes = num_classes
        self.alpha = nn.Linear(self.model.linear.in_features, num_classes)
        self.beta = nn.Linear(self.model.linear.in_features, num_classes)
        self.model.linear = nn.Identity()
        self.set_metrics()

    def forward(self, x):
        logits = self.model(x)
        alpha = self.alpha(logits)
        beta = self.beta(logits)
        return alpha, beta

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y, self.num_classes)
        logits_a, logits_b = self(x)
        evidence_a = F.elu(
            logits_a) + 2  # TODO - in the original implementation, they add 2, but it's not clear why, check this later
        evidence_b = F.elu(logits_b) + 2
        loss = ava_edl_criterion(evidence_a, evidence_b, y, fbeta=self.beta_param, lambda_fbeta=self.lambda_param)
        return loss, evidence_a, evidence_b, y

    def training_step(self, batch, batch_idx):
        loss, evidence_a, evidence_b, y = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss)
        probs = evidence_a / (evidence_a + evidence_b)
        y_hat = (probs > 0.5).int()
        self.train_acc.update(y_hat, y)
        self.train_set_size(y_hat)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, evidence_a, evidence_b, y = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss)
        probs = evidence_a / (evidence_a + evidence_b)
        y_hat = (probs > 0.5).int()
        self.val_acc.update(y_hat, y)
        self.val_set_size(y_hat)

    def test_step(self, batch, batch_idx):
        loss, evidence_a, evidence_b, y = self.shared_step(batch, batch_idx)
        self.log('test_loss', loss)
        probs = evidence_a / (evidence_a + evidence_b)
        y_hat = (probs > 0.5).int()
        self.test_acc.update(y_hat, y)
        self.test_set_size(y_hat)
        self.evidence_accumulator.to(self.device)
        self.evidence_accumulator.update(evidence_a, evidence_b, y)

    def on_train_epoch_end(self) -> None:
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        self.log('train_set_size', self.train_set_size.compute(), prog_bar=True)
        wandb.log({"train_set_size": self.train_set_size.compute()}, step=self.current_epoch)
        wandb.log({"train_acc": self.train_acc.compute()}, step=self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_set_size', self.val_set_size.compute(), prog_bar=True)
        wandb.log({"val_set_size": self.val_set_size.compute()}, step=self.current_epoch)
        wandb.log({"val_acc": self.val_acc.compute()}, step=self.current_epoch)

    def on_test_epoch_end(self) -> None:
        self.log('test_acc', self.test_acc.compute())
        self.log('test_set_size', self.test_set_size.compute())
        wandb.log({"test_set_size": self.test_set_size.compute()}, step=self.current_epoch)
        wandb.log({"test_acc": self.test_acc.compute()}, step=self.current_epoch)
        self.evidence_accumulator.save('evidence_accumulator.pth')

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def set_metrics(self):
        self.train_acc = MultilabelAccuracy(criteria='contain')
        self.val_acc = MultilabelAccuracy(criteria='contain')
        self.test_acc = MultilabelAccuracy(criteria='contain')

        self.train_set_size = PredictionSetSize()
        self.val_set_size = PredictionSetSize()
        self.test_set_size = PredictionSetSize()

        self.evidence_accumulator = BetaEvidenceAccumulator()
