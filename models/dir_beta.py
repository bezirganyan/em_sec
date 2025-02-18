from math import gamma

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import wandb
from torch.optim import Adam

from losses import ava_edl_criterion, get_equivalence_loss, get_evidential_hyperloss, get_evidential_loss, \
    get_utility_loss
from metrics import AverageUtility, CorrectIncorrectUncertaintyPlotter, HyperAccuracy, HyperSetSize, \
    HyperUncertaintyPlotter
from torcheval.metrics import MulticlassAccuracy
import seaborn as sns

class CIFAR10HyperModel(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, beta=1):
        super(CIFAR10HyperModel, self).__init__()
        self.save_hyperparameters()
        self.beta_param = beta
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

    @staticmethod
    def make_binomial_opinion(evidence_a, evidence_b, base_rate=0.5):
        """
        Convert raw evidence (evidence_a, evidence_b) into
        binomial opinions (b, d, u, a) for each entry.
        Shapes of evidence_a, evidence_b: (batch_size, num_classes).
        """
        # We add 1 to the denominator to avoid division by zero
        total = evidence_a + evidence_b + 1.0

        b = evidence_a / total
        d = evidence_b / total
        u = 1.0 - b - d
        a = torch.full_like(b, base_rate)  # broadcast base_rate across all entries

        return b, d, u, a

    @staticmethod
    def binomial_disjunction(bx, dx, ux, ax, by, dy, uy, ay):
        """
        Perform binomial comultiplication (disjunction) on two binomial opinions:
          ωx = (bx, dx, ux, ax)
          ωy = (by, dy, uy, ay)
        Returns: (b_xy, d_xy, u_xy, a_xy)
        following the standard subjective logic disjunction formula.
        """
        a_xy = ax + ay - ax * ay
        b_xy = bx + by - bx * by

        eps = 1e-10
        denom = a_xy + eps  # to avoid division-by-zero

        # d_{x∨y}
        # = d_x d_y + [ (1-ax)*ay*bx*d_y + (1-ay)*ax*d_x*by ] / (ax + ay - ax*ay)
        part_num = (1 - ax) * ay * bx * dy + (1 - ay) * ax * dx * by
        d_xy = dx * dy + part_num / denom

        # u_{x∨y} = (d_x u_y + d_y u_x) / (ax + ay - ax*ay)
        u_xy = (dx * uy + dy * ux) / denom

        return b_xy, d_xy, u_xy, a_xy

    def comultiply_all(self, b_list, d_list, u_list, a_list):
        """
        Reduce a list of binomial opinions [ (b1,d1,u1,a1), (b2,d2,u2,a2), ... ]
        by iteratively applying binomial_disjunction.
        """
        # Start with the first opinion
        b_acc = b_list[0]
        d_acc = d_list[0]
        u_acc = u_list[0]
        a_acc = a_list[0]

        # Iteratively combine
        for i in range(1, len(b_list)):
            b_acc, d_acc, u_acc, a_acc = self.binomial_disjunction(
                b_acc, d_acc, u_acc, a_acc,
                b_list[i], d_list[i], u_list[i], a_list[i]
            )
        return b_acc, d_acc, u_acc, a_acc

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y, self.num_classes)
        logits_a, logits_b, logits, multinomial_logits = self(x)
        evidence_a = F.elu(logits_a) + 2 # TODO - in the original implementation, they add 2, but it's not clear why, check this later
        evidence_b = F.elu(logits_b) + 2
        multinomial_evidence = torch.exp(multinomial_logits)
        # evidence_hyper = torch.exp(logits)
        loss_multilabel = ava_edl_criterion(evidence_a, evidence_b, y)
        multilabel_probs = evidence_a / (evidence_a + evidence_b)
        hyperset = (evidence_a > evidence_b).int()

        b, d, u, a = self.make_binomial_opinion(evidence_a, evidence_b)
        masked_factors = torch.where(hyperset > 0.5, 1 - b, torch.ones_like(b))
        prod_selected = masked_factors.prod(dim=1)
        final_b = 1 - prod_selected

        loss_edl = get_evidential_loss(multinomial_evidence, y, self.current_epoch, self.num_classes, 10, self.device, targets_one_hot=True)

        multinomial_uncertainty = self.num_classes / (multinomial_evidence + 1).sum(dim=1, keepdim=True)
        multiomial_beliefs = multinomial_evidence / (multinomial_evidence + 1).sum(dim=1, keepdim=True)
        set_beliefs_scaled = multinomial_uncertainty * final_b.unsqueeze(-1)
        remaining_unc = multinomial_uncertainty * (1 - final_b.unsqueeze(-1))
        hypernomial_beleifs = torch.cat([multiomial_beliefs, set_beliefs_scaled], dim=1)
        evidence_hyper = (self.num_classes + 1) * hypernomial_beleifs / (remaining_unc + 1e-6)
        loss = loss_multilabel + loss_edl # + gamma * (hyper_loss_edl + eqv_loss + utility)
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
        for utility in self.val_utility_dict.values():
            if utility.device != self.device:
                utility.to(self.device)
            utility.update(pred_sets, y)



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

        self.cor_unc_plot.update(multinomial_evidence, y.argmax(dim=1))
        self.hyper_uncertainty_plot.update(pred_sets, evidence_hyper, y)
        for utility in self.test_utility_dict.values():
            if utility.device != self.device:
                utility.to(self.device)
            utility.update(pred_sets, y)



    def on_train_epoch_end(self) -> None:
        self.log('train_acc', self.train_acc.compute())
        self.log('train_set_size', self.train_set_size.compute())

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_set_size', self.val_set_size.compute(), prog_bar=True)
        self.log('val_multiclass_acc', self.val_multiclass_acc.compute(), prog_bar=True)
        wandb.log({'val_set_size': self.val_set_size.compute()})
        wandb.log({'val_acc': self.val_acc.compute()})
        wandb.log({'val_multiclass_acc': self.val_multiclass_acc.compute()})
        for key, utility in self.val_utility_dict.items():
            progress_bar = True if 'fb' in key else False
            self.log(f'val_{key}', utility.compute(), prog_bar=progress_bar)
            wandb.log({f'val_{key}': utility.compute()})

    def on_test_epoch_end(self) -> None:
        self.log('test_acc', self.test_acc.compute())
        self.log('test_set_size', self.test_set_size.compute())
        self.log('test_multiclass_acc', self.test_multiclass_acc.compute())
        wandb.log({'test_set_size': self.test_set_size.compute()})
        wandb.log({'test_acc': self.test_acc.compute()})
        for key, utility in self.test_utility_dict.items():
            self.log(f'test_utility_{key}', utility.compute())
            wandb.log({f'test_{key}': utility.compute()})
        self.cor_unc_plot.plot()
        self.hyper_uncertainty_plot.plot()
        self.test_set_size.plot()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def set_metrics(self):
        self.train_acc = HyperAccuracy()
        self.val_acc = HyperAccuracy()
        self.test_acc = HyperAccuracy()

        self.val_utility_dict = {
            'fb': AverageUtility(self.num_classes, utility='fb', beta=self.beta_param),
            'owa_0.5': AverageUtility(self.num_classes, utility='owa', tolerance=0.5),
            'owa_0.6': AverageUtility(self.num_classes, utility='owa', tolerance=0.6),
            'owa_0.7': AverageUtility(self.num_classes, utility='owa', tolerance=0.7),
            'owa_0.8': AverageUtility(self.num_classes, utility='owa', tolerance=0.8),
            'owa_0.9': AverageUtility(self.num_classes, utility='owa', tolerance=0.9)
        }

        self.test_utility_dict = {
            'fb': AverageUtility(self.num_classes, utility='fb', beta=self.beta_param),
            'owa_0.5': AverageUtility(self.num_classes, utility='owa', tolerance=0.5),
            'owa_0.6': AverageUtility(self.num_classes, utility='owa', tolerance=0.6),
            'owa_0.7': AverageUtility(self.num_classes, utility='owa', tolerance=0.7),
            'owa_0.8': AverageUtility(self.num_classes, utility='owa', tolerance=0.8),
            'owa_0.9': AverageUtility(self.num_classes, utility='owa', tolerance=0.9)
        }

        self.train_set_size = HyperSetSize()
        self.val_set_size = HyperSetSize()
        self.test_set_size = HyperSetSize()

        self.train_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_multiclass_acc = MulticlassAccuracy(num_classes=self.num_classes)


        self.cor_unc_plot = CorrectIncorrectUncertaintyPlotter()
        self.hyper_uncertainty_plot = HyperUncertaintyPlotter()
