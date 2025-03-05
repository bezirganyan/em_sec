import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import wandb
from torch.optim import Adam
from torcheval.metrics import MulticlassAccuracy

from losses import ava_edl_criterion, get_evidential_hyperloss, get_evidential_loss, get_fbeta_loss
from metrics import AverageUtility, CorrectIncorrectUncertaintyPlotter, HyperAccuracy, HyperEvidenceAccumulator, \
    HyperSetSize, \
    HyperUncertaintyPlotter, TimeLogger
from models.conv_models import BasicBlock, ResNet


class CIFAR10HyperModel(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, beta=1, annealing_start=0, annealing_end=100):
        super(CIFAR10HyperModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.beta_param = beta
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.num_classes = num_classes
        self.alpha = nn.Linear(self.model.linear.in_features, num_classes)
        self.beta = nn.Linear(self.model.linear.in_features, num_classes)
        self.multinomial_evidence_collector = nn.Linear(self.model.linear.in_features, num_classes)
        self.annealing_start = annealing_start
        self.annealing_end = annealing_end
        # self.hyper_evidence_collector = nn.Linear(num_classes * 2, num_classes + 1)
        self.model.linear = nn.Identity()
        self.set_metrics()

    def forward(self, x):
        # Compute logits and apply ReLU.
        logits = torch.relu(self.model(x))

        alpha = self.alpha(logits)
        beta = self.beta(logits)
        evidence_a = F.elu(alpha) + 2  # As in the original implementation.
        evidence_b = F.elu(beta) + 2

        multinomial_evidence = torch.exp(self.multinomial_evidence_collector(logits))

        mask = evidence_a > evidence_b

        b = evidence_a / (evidence_a + evidence_b + 1)

        masked_factors = torch.where(mask, 1 - b, torch.ones_like(b))
        prod_selected = masked_factors.prod(dim=1)
        final_b = 1 - prod_selected

        evidence_sum = (multinomial_evidence + 1).sum(dim=1, keepdim=True)
        multinomial_uncertainty = self.num_classes / evidence_sum
        multiomial_beliefs = multinomial_evidence / evidence_sum

        set_beliefs_scaled = multinomial_uncertainty * final_b.unsqueeze(-1)
        remaining_unc = multinomial_uncertainty * (1 - final_b.unsqueeze(-1))
        hypernomial_beliefs = torch.cat([multiomial_beliefs, set_beliefs_scaled], dim=1)
        evidence_hyper = (self.num_classes + 1) * hypernomial_beliefs / (remaining_unc + 1e-6)

        return evidence_a, evidence_b, multinomial_evidence, evidence_hyper

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y, self.num_classes)
        evidence_a, evidence_b, multinomial_evidence, evidence_hyper = self(x)
        loss_multilabel = ava_edl_criterion(evidence_a, evidence_b, y, self.beta_param, self.current_epoch,
                                            self.annealing_start, self.annealing_end)
        loss_edl = get_evidential_loss(multinomial_evidence, y, self.current_epoch, self.num_classes, 10,
                                       self.device, targets_one_hot=True)
        multilabel_probs = evidence_a / (evidence_a + evidence_b)
        # loss_hyper = get_evidential_hyperloss(evidence_hyper, multilabel_probs, y, self.current_epoch, self.num_classes,
        #                                       10, self.device, beta=self.beta_param)
        # loss_hyper = get_fbeta_loss(evidence_hyper, multilabel_probs, self.beta_param)
        # gamma = torch.relu(torch.tensor(self.current_epoch - 50)) / 50
        loss = loss_multilabel + loss_edl  #+ gamma * loss_hyper
        return loss, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, multilabel_probs

    def training_step(self, batch, batch_idx):
        loss, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, hyperset = self.shared_step(batch,
                                                                                                           batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        y_hat = evidence_hyper.argmax(dim=1)
        y_hat = F.one_hot(y_hat, self.num_classes + 1)
        self.train_acc.update(y_hat, y, hyperset > 0.5)
        self.train_set_size.update(y_hat, hyperset > 0.5, y)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, hyperset = self.shared_step(batch,
                                                                                                           batch_idx)
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
        for utility in self.val_utility_dict.values():
            if utility.device != self.device:
                utility.to(self.device)
            utility.update(pred_sets, y)

    def test_step(self, batch, batch_idx):
        loss, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, hyperset = self.shared_step(batch,
                                                                                                           batch_idx)
        self.log('test_loss', loss)
        y_hat = evidence_hyper.argmax(dim=1)
        y_hat = F.one_hot(y_hat, self.num_classes + 1)
        self.test_acc.update(y_hat, y, hyperset > 0.5)
        self.test_set_size.update(y_hat, hyperset > 0.5, y)
        self.test_multiclass_acc.update(multinomial_evidence.argmax(dim=1), y.argmax(dim=1))
        self.evidence_accumulator.update(multinomial_evidence, evidence_a, evidence_b, y)

        time_start = time.time()
        pred_sets = self.predict_set(batch[0], as_list=True)
        duration = time.time() - time_start
        self.test_time_logger_as_list.update(duration)

        time_start = time.time()
        y_hyper = self.predict_set(batch[0])
        duration = time.time() - time_start
        self.test_time_logger.update(duration)
        self.cor_unc_plot.update(multinomial_evidence, y.argmax(dim=1))
        self.hyper_uncertainty_plot.update(pred_sets, evidence_hyper, y)
        for utility in self.test_utility_dict.values():
            if utility.device != self.device:
                utility.to(self.device)
            utility.update(pred_sets, y)

    def predict_set(self, x, as_list=False):
        evidence_a, evidence_b, multinomial_evidence, evidence_hyper = self(x)
        y_hat = evidence_hyper.argmax(dim=1)
        y_hat = F.one_hot(y_hat, self.num_classes + 1)
        probs = evidence_a / (evidence_a + evidence_b)
        hyperset = (probs > 0.9).float()
        y_hat_idx_idx = evidence_hyper.argmax(dim=1)
        mask = y_hat_idx_idx == self.num_classes
        y_hyper = multinomial_evidence.argmax(dim=1)
        y_hyper = F.one_hot(y_hyper, self.num_classes).long()
        y_hyper[mask] = (hyperset[mask] > 0.5).long()
        if as_list:
            # For each row in y_hyper, get the indices where the element is nonzero.
            y_hyper = [row.nonzero(as_tuple=False).view(-1).tolist() for row in y_hyper]

        return y_hyper

    def on_train_epoch_end(self) -> None:
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        self.log('train_set_size', self.train_set_size.compute())
        wandb.log({'train_set_size': self.train_set_size.compute()}, step=self.current_epoch)
        wandb.log({'train_acc': self.train_acc.compute()}, step=self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_set_size', self.val_set_size.compute(), prog_bar=True)
        self.log('val_multiclass_acc', self.val_multiclass_acc.compute(), prog_bar=True)
        wandb.log({'val_set_size': self.val_set_size.compute()}, step=self.current_epoch)
        wandb.log({'val_acc': self.val_acc.compute()}, step=self.current_epoch)
        wandb.log({'val_multiclass_acc': self.val_multiclass_acc.compute()}, step=self.current_epoch)
        for key, utility in self.val_utility_dict.items():
            progress_bar = True if 'fb' in key else False
            self.log(f'val_{key}', utility.compute(), prog_bar=progress_bar)
            wandb.log({f'val_{key}': utility.compute()}, step=self.current_epoch)

    def on_test_epoch_end(self) -> None:
        self.log('test_acc', self.test_acc.compute())
        self.log('test_set_size', self.test_set_size.compute())
        self.log('test_multiclass_acc', self.test_multiclass_acc.compute())
        self.log('test_time', self.test_time_logger.compute())
        self.log('test_time_as_list', self.test_time_logger_as_list.compute())
        self.evidence_accumulator.save('evidence_with_beta.pt')
        wandb.log({'test_set_size': self.test_set_size.compute()}, step=self.current_epoch)
        wandb.log({'test_acc': self.test_acc.compute()}, step=self.current_epoch)
        wandb.log({'test_time': self.test_time_logger.compute()}, step=self.current_epoch)
        wandb.log({'test_time_as_list': self.test_time_logger_as_list.compute()}, step=self.current_epoch)
        for key, utility in self.test_utility_dict.items():
            self.log(f'test_utility_{key}', utility.compute())
            wandb.log({f'test_{key}': utility.compute()}, step=self.current_epoch)
        self.cor_unc_plot.plot()
        self.hyper_uncertainty_plot.plot()
        self.test_set_size.plot()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def set_metrics(self):
        self.train_acc = HyperAccuracy()
        self.val_acc = HyperAccuracy()
        self.test_acc = HyperAccuracy()

        self.val_utility_dict = {
            'fb_1': AverageUtility(self.num_classes, utility='fb', beta=1, as_list=True),
            'fb_2': AverageUtility(self.num_classes, utility='fb', beta=2, as_list=True),
            'fb_3': AverageUtility(self.num_classes, utility='fb', beta=3, as_list=True),
            'fb_4': AverageUtility(self.num_classes, utility='fb', beta=4, as_list=True),
            'fb_5': AverageUtility(self.num_classes, utility='fb', beta=5, as_list=True),
            'owa_0.5': AverageUtility(self.num_classes, utility='owa', tolerance=0.5, as_list=True),
            'owa_0.6': AverageUtility(self.num_classes, utility='owa', tolerance=0.6, as_list=True),
            'owa_0.7': AverageUtility(self.num_classes, utility='owa', tolerance=0.7, as_list=True),
            'owa_0.8': AverageUtility(self.num_classes, utility='owa', tolerance=0.8, as_list=True),
            'owa_0.9': AverageUtility(self.num_classes, utility='owa', tolerance=0.9, as_list=True)
        }

        self.test_utility_dict = {
            'fb_1': AverageUtility(self.num_classes, utility='fb', beta=1, as_list=True),
            'fb_2': AverageUtility(self.num_classes, utility='fb', beta=2, as_list=True),
            'fb_3': AverageUtility(self.num_classes, utility='fb', beta=3, as_list=True),
            'fb_4': AverageUtility(self.num_classes, utility='fb', beta=4, as_list=True),
            'fb_5': AverageUtility(self.num_classes, utility='fb', beta=5, as_list=True),
            'owa_0.5': AverageUtility(self.num_classes, utility='owa', tolerance=0.5, as_list=True),
            'owa_0.6': AverageUtility(self.num_classes, utility='owa', tolerance=0.6, as_list=True),
            'owa_0.7': AverageUtility(self.num_classes, utility='owa', tolerance=0.7, as_list=True),
            'owa_0.8': AverageUtility(self.num_classes, utility='owa', tolerance=0.8, as_list=True),
            'owa_0.9': AverageUtility(self.num_classes, utility='owa', tolerance=0.9, as_list=True)
        }

        self.train_set_size = HyperSetSize(num_classes=self.num_classes)
        self.val_set_size = HyperSetSize(num_classes=self.num_classes)
        self.test_set_size = HyperSetSize(num_classes=self.num_classes)

        self.train_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)

        self.cor_unc_plot = CorrectIncorrectUncertaintyPlotter()
        self.hyper_uncertainty_plot = HyperUncertaintyPlotter()
        self.test_time_logger = TimeLogger()
        self.test_time_logger_as_list = TimeLogger()

        self.evidence_accumulator = HyperEvidenceAccumulator()
