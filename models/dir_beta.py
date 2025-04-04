import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import Adam
from torcheval.metrics import MulticlassAccuracy, MultilabelAccuracy

from losses import ava_edl_criterion, get_evidential_loss
from metrics import AverageUtility, CorrectIncorrectUncertaintyPlotter, HyperAccuracy, HyperEvidenceAccumulator, \
    HyperSetSize, \
    HyperUncertaintyPlotter, TimeLogger


class EMSECModel(pl.LightningModule):
    def __init__(self, model, num_classes=10, learning_rate=1e-3, beta=1, annealing_start=0, annealing_end=100, lambda_param=1):
        super(EMSECModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.beta_param = beta
        self.lambda_param = lambda_param
        self.model = model
        self.num_classes = num_classes
        self.alpha = nn.Linear(self.model.linear.in_features, num_classes)
        self.beta = nn.Linear(self.model.linear.in_features, num_classes)
        self.multinomial_evidence_collector = nn.Linear(self.model.linear.in_features, num_classes)
        self.annealing_start = annealing_start
        self.annealing_end = annealing_end
        # self.hyper_evidence_collector = nn.Linear(num_classes * 2, num_classes + 1)
        self.model.linear = nn.Identity()
        self.set_metrics()
        self.automatic_optimization = False

    def forward(self, x):
        # Compute logits and apply ReLU.
        logits = torch.relu(self.model(x))

        alpha = self.alpha(logits.detach().clone())
        beta = self.beta(logits.detach().clone())
        evidence_a = F.elu(alpha) + 2
        evidence_b = F.elu(beta) + 2

        logits_evidence = self.multinomial_evidence_collector(logits)
        logits_evidence = torch.clamp(logits_evidence, max=10.0)
        multinomial_evidence = torch.exp(logits_evidence)

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
                                            self.annealing_start, self.annealing_end, self.lambda_param)
        loss_edl = get_evidential_loss(multinomial_evidence, y, self.current_epoch, self.num_classes, 10,
                                       self.device, targets_one_hot=True)
        multilabel_probs = evidence_a / (evidence_a + evidence_b)

        # Return individual losses instead of their sum
        return loss_multilabel, loss_edl, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, multilabel_probs

    def training_step(self, batch, batch_idx):
        # Get optimizers
        opt_multilabel, opt_evidence = self.optimizers()

        # Get individual losses and other outputs
        loss_multilabel, loss_edl, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, multilabel_probs = self.shared_step(
            batch, batch_idx)

        # Optimize multilabel loss
        opt_multilabel.zero_grad()
        self.manual_backward(loss_multilabel)
        opt_multilabel.step()

        # Optimize evidence loss
        opt_evidence.zero_grad()
        self.manual_backward(loss_edl)
        opt_evidence.step()

        # Log the combined loss for tracking purposes
        total_loss = loss_multilabel + loss_edl
        self.log('train_loss', total_loss, prog_bar=True)

        # Update metrics
        y_hat = evidence_hyper.argmax(dim=1)
        y_hat = F.one_hot(y_hat, self.num_classes + 1)
        self.train_multiclass_acc.update(multinomial_evidence.argmax(dim=1), y.argmax(dim=1))
        self.train_set_size.update(y_hat, multilabel_probs > 0.5, y)
        self.train_multilabel_acc.update(multilabel_probs, y)

        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_multilabel, loss_edl, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, hyperset = self.shared_step(
            batch, batch_idx)
        loss = loss_multilabel + loss_edl
        self.log('val_loss', loss, prog_bar=True)
        y_hat = evidence_hyper.argmax(dim=1)
        y_hat = F.one_hot(y_hat, self.num_classes + 1)
        self.val_set_size.update(y_hat, hyperset > 0.5, y)
        self.val_multiclass_acc.update(multinomial_evidence.argmax(dim=1), y.argmax(dim=1))
        y_hat_idx_idx = evidence_hyper.argmax(dim=1)
        pred_sets = self.predict_set(batch[0], as_list=True)
        self.val_acc.update(pred_sets, y)
        self.val_multilabel_acc.update(hyperset, y)
        for utility in self.val_utility_dict.values():
            if utility.device != self.device:
                utility.to(self.device)
            utility.update(pred_sets, y)

    def test_step(self, batch, batch_idx):
        loss_multilabel, loss_edl, evidence_hyper, multinomial_evidence, evidence_a, evidence_b, y, hyperset = self.shared_step(
                batch, batch_idx)
        loss = loss_multilabel + loss_edl
        self.log('test_loss', loss)
        y_hat = evidence_hyper.argmax(dim=1)
        y_hat = F.one_hot(y_hat, self.num_classes + 1)
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
        self.hyper_uncertainty_plot.update(multinomial_evidence, pred_sets, evidence_hyper, y)
        self.test_acc.update(pred_sets, y)
        self.test_multilabel_acc.update(hyperset, y)
        for utility in self.test_utility_dict.values():
            if utility.device != self.device:
                utility.to(self.device)
            utility.update(pred_sets, y)

    def predict_set(self, x, as_list=False):
        evidence_a, evidence_b, multinomial_evidence, evidence_hyper = self(x)
        y_hat = evidence_hyper.argmax(dim=1)
        y_hat = F.one_hot(y_hat, self.num_classes + 1)
        probs = evidence_a / (evidence_a + evidence_b)
        hyperset = (probs > 0.5).float()
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
        self.log('train_multiclass_acc', self.train_multiclass_acc.compute(), prog_bar=True)
        self.log('train_set_size', self.train_set_size.compute())
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        self.log('train_multilabel_acc', self.train_multilabel_acc.compute(), prog_bar=True)
        wandb.log({'train_set_size': self.train_set_size.compute()}, step=self.current_epoch)
        wandb.log({'train_acc': self.train_acc.compute()}, step=self.current_epoch)
        wandb.log({'train_multiclass_acc': self.train_multiclass_acc.compute()}, step=self.current_epoch)
        wandb.log({'train_multilabel_acc': self.train_multilabel_acc.compute()}, step=self.current_epoch)


    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_set_size', self.val_set_size.compute(), prog_bar=True)
        self.log('val_multiclass_acc', self.val_multiclass_acc.compute(), prog_bar=True)
        self.log('val_multilabel_acc', self.val_multilabel_acc.compute(), prog_bar=True)
        wandb.log({'val_set_size': self.val_set_size.compute()}, step=self.current_epoch)
        wandb.log({'val_acc': self.val_acc.compute()}, step=self.current_epoch)
        wandb.log({'val_multiclass_acc': self.val_multiclass_acc.compute()}, step=self.current_epoch)
        wandb.log({'val_multilabel_acc': self.val_multilabel_acc.compute()}, step=self.current_epoch)
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
        self.log('test_multilabel_acc', self.test_multilabel_acc.compute())
        self.evidence_accumulator.save('evidence_with_beta.pt')
        wandb.log({'test_set_size': self.test_set_size.compute()}, step=self.current_epoch)
        wandb.log({'test_acc': self.test_acc.compute()}, step=self.current_epoch)
        wandb.log({'test_time': self.test_time_logger.compute()}, step=self.current_epoch)
        wandb.log({'test_time_as_list': self.test_time_logger_as_list.compute()}, step=self.current_epoch)
        wandb.log({'test_multiclass_acc': self.test_multiclass_acc.compute()}, step=self.current_epoch)
        for key, utility in self.test_utility_dict.items():
            self.log(f'test_utility_{key}', utility.compute())
            wandb.log({f'test_{key}': utility.compute()}, step=self.current_epoch)
        self.cor_unc_plot.plot()
        self.hyper_uncertainty_plot.plot()
        self.test_set_size.plot()

    def configure_optimizers(self):
        # First optimizer for multilabel components (alpha and beta)
        multilabel_params = list(self.alpha.parameters()) + list(self.beta.parameters())
        opt_multilabel = Adam(multilabel_params, lr=self.learning_rate)

        # Second optimizer for multinomial evidence collector and base model
        evidence_params = list(self.multinomial_evidence_collector.parameters()) + list(self.model.parameters())
        opt_evidence = Adam(evidence_params, lr=self.learning_rate)

        return [opt_multilabel, opt_evidence]

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

        self.train_multilabel_acc = MultilabelAccuracy(criteria='contain')
        self.val_multilabel_acc = MultilabelAccuracy(criteria='contain')
        self.test_multilabel_acc = MultilabelAccuracy(criteria='contain')

        self.train_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)

        self.hyper_uncertainty_plot = HyperUncertaintyPlotter()
        self.cor_unc_plot = CorrectIncorrectUncertaintyPlotter()
        self.test_time_logger = TimeLogger()
        self.test_time_logger_as_list = TimeLogger()

        self.evidence_accumulator = HyperEvidenceAccumulator()
