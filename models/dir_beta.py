from math import gamma

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import Adam

from losses import ava_edl_criterion, get_equivalence_loss, get_evidential_hyperloss, get_evidential_loss, \
    get_utility_loss
from metrics import AverageUtility, HyperAccuracy, HyperSetSize
from torcheval.metrics import MulticlassAccuracy


class CIFAR10HyperModel(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super(CIFAR10HyperModel, self).__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=False)
        self.num_classes = num_classes
        self.alpha = nn.Linear(self.model.fc.in_features, num_classes)
        self.beta = nn.Linear(self.model.fc.in_features, num_classes)
        self.multinomial_evidence_collector = nn.Linear(self.model.fc.in_features, num_classes)
        self.hyper_evidence_collector = nn.Linear(num_classes * 2, num_classes + 1)
        self.model.fc = nn.Identity()
        self.set_metrics()

    def forward(self, x):
        logits = torch.relu(self.model(x))
        alpha = self.alpha(logits)
        beta = self.beta(logits)
        multinomial_evidence = self.multinomial_evidence_collector(logits)

        ev_diff = torch.relu(alpha - beta)
        # logits_unfused = torch.relu(torch.cat([multinomial_evidence, ev_diff], dim=1))
        logits_unfused = torch.cat([multinomial_evidence, ev_diff], dim=1)
        hyper_evidence = self.hyper_evidence_collector(logits_unfused)
        return alpha, beta, hyper_evidence, multinomial_evidence


    def shared_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y, self.num_classes)
        logits_a, logits_b, logits, multinomial_logits = self(x)
        evidence_a = F.elu(logits_a) + 2 # TODO - in the original implementation, they add 2, but it's not clear why, check this later
        evidence_b = F.elu(logits_b) + 2
        multinomial_evidence = F.elu(multinomial_logits) + 1
        evidence_hyper = F.elu(logits) + 1
        loss_multilabel = ava_edl_criterion(evidence_a, evidence_b, y)
        multilabel_probs = evidence_a / (evidence_a + evidence_b)
        hyperset = (evidence_a > evidence_b).int()

        # gamma to be 0 before some epoch, then gradually go to 1 after some epoch
        gamma = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(self.current_epoch / 10, dtype=torch.float32),
        )
        loss_edl = get_evidential_loss(multinomial_evidence, y, self.current_epoch, self.num_classes, 10, self.device, targets_one_hot=True)
        hyper_loss_edl = get_evidential_hyperloss(evidence_hyper, multilabel_probs, y, self.current_epoch, self.num_classes, 10, self.device)
        eqv_loss = get_equivalence_loss(multinomial_evidence, evidence_hyper)
        utility = get_utility_loss(evidence_hyper, multilabel_probs, y, 1, self.device)

        loss = loss_multilabel + loss_edl + gamma * (hyper_loss_edl + eqv_loss + utility)
        return loss, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, multilabel_probs

    def training_step(self, batch, batch_idx):
        loss, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, hyperset = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        y_hat = evidence_hyper.argmax(dim=1)
        y_hat = F.one_hot(y_hat, self.num_classes + 1)
        self.train_acc.update(y_hat, y, hyperset > 0.5)
        self.train_set_size.update(y_hat, hyperset > 0.5, y)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, hyperset = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        y_hat = evidence_hyper.argmax(dim=1)
        y_hat = F.one_hot(y_hat, self.num_classes + 1)
        self.val_acc.update(y_hat, y, hyperset > 0.5)
        self.val_set_size.update(y_hat, hyperset > 0.5, y)
        self.val_multiclass_acc.update(multinomial_evidence.argmax(dim=1), y.argmax(dim=1))
        y_hat_idx_idx = evidence_hyper.argmax(dim=1)
        pred_sets = []
        for i in range(y_hat.shape[0]):
            if y_hat_idx_idx[i] < self.num_classes:
                pred_set = [y_hat_idx_idx[i].item()]
            else:
                # Assuming hyperset[i] is a tensor of shape (num_classes,) where positive values indicate membership.
                pred_set = torch.nonzero(hyperset[i] > 0.5, as_tuple=False).squeeze(-1).tolist()
                # order the pred set by hyperset values in descending order
                pred_set = sorted(pred_set, key=lambda x: hyperset[i][x], reverse=True)
            pred_sets.append(pred_set)
        self.val_utility.update(pred_sets, y)



    def test_step(self, batch, batch_idx):
        loss, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, hyperset = self.shared_step(batch, batch_idx)
        self.log('test_loss', loss)
        y_hat = evidence_hyper.argmax(dim=1)
        y_hat = F.one_hot(y_hat, self.num_classes + 1)
        self.test_acc.update(y_hat, y, hyperset > 0.5)
        self.test_set_size.update(y_hat, hyperset > 0.5, y)
        self.test_multiclass_acc.update(multinomial_evidence.argmax(dim=1), y.argmax(dim=1))
        y_hat_idx_idx = evidence_hyper.argmax(dim=1)
        pred_sets = []
        for i in range(y_hat.shape[0]):
            if y_hat_idx_idx[i] < self.num_classes:
                pred_set = [y_hat_idx_idx[i].item()]
            else:
                # Assuming hyperset[i] is a tensor of shape (num_classes,) where positive values indicate membership.
                pred_set = torch.nonzero(hyperset[i] > 0.5, as_tuple=False).squeeze(-1).tolist()
                # order the pred set by hyperset values in descending order
                pred_set = sorted(pred_set, key=lambda x: hyperset[i][x], reverse=True)
            pred_sets.append(pred_set)
        self.test_utility.update(pred_sets, y)

    def on_train_epoch_end(self) -> None:
        self.log('train_acc', self.train_acc.compute())
        self.log('train_set_size', self.train_set_size.compute())

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_set_size', self.val_set_size.compute(), prog_bar=True)
        self.log('val_utility', self.val_utility.compute(), prog_bar=True)
        self.log('val_multiclass_acc', self.val_multiclass_acc.compute(), prog_bar=True)

    def on_test_epoch_end(self) -> None:
        self.log('test_acc', self.test_acc.compute())
        self.log('test_set_size', self.test_set_size.compute())
        self.log('test_utility', self.test_utility.compute())
        self.log('test_multiclass_acc', self.test_multiclass_acc.compute())
        self.test_set_size.plot()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def set_metrics(self):
        self.train_acc = HyperAccuracy()
        self.val_acc = HyperAccuracy()
        self.test_acc = HyperAccuracy()

        self.train_utility = AverageUtility(self.num_classes, utility='fb', beta=1)
        self.val_utility = AverageUtility(self.num_classes, utility='fb', beta=1)
        self.test_utility = AverageUtility(self.num_classes, utility='fb', beta=1)
        # self.train_utility = AverageUtility(self.num_classes, tolerance=0.7)
        # self.val_utility = AverageUtility(self.num_classes, tolerance=0.7)
        # self.test_utility = AverageUtility(self.num_classes, tolerance=0.7)

        self.train_set_size = HyperSetSize()
        self.val_set_size = HyperSetSize()
        self.test_set_size = HyperSetSize()

        self.train_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)
