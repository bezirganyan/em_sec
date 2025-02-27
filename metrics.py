import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.optimize import minimize
from torchmetrics import Metric
from tqdm import tqdm


class PredictionSetSize(Metric):
    def __init__(self, **kwargs):
        super(PredictionSetSize, self).__init__(**kwargs)
        self.add_state('counts', default=torch.tensor([]), dist_reduce_fx='cat')

    def update(self, input, target=None):
        set_sizes = input.sum(dim=1)
        self.counts = torch.cat((self.counts, set_sizes))
        return set_sizes

    def merge_state(self, metrics):
        for metric in metrics:
            self.counts = torch.cat((self.counts, metric.counts))
        return self.counts

    def compute(self):
        return self.counts.mean()


class HyperAccuracy(Metric):
    def __init__(self, **kwargs):
        super(HyperAccuracy, self).__init__(**kwargs)
        self.add_state('corrects', default=torch.tensor([0]), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor([0]), dist_reduce_fx='sum')

    def update(self, input, target, hyperset):
        corrects = (input[:, :-1].argmax(dim=1) == target.argmax(dim=1)) & (input[:, :-1].sum(dim=1) > 0)

        set_corrects = ((hyperset & target).sum(dim=1) > 0) & (input[:, -1] > 0)

        corrects = corrects | set_corrects
        num_corrects = corrects.sum()
        self.corrects += num_corrects
        self.total += input.size(0)
        return num_corrects / input.size(0)

    def merge_state(self, metrics):
        for metric in metrics:
            self.corrects += metric.corrects
            self.total += metric.total

    def compute(self):
        return self.corrects / self.total


class HyperSetSize(Metric):
    def __init__(self, num_classes=10, **kwargs):
        super(HyperSetSize, self).__init__(**kwargs)
        self.add_state('counts', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('multinomial_number', default=torch.tensor([0]), dist_reduce_fx='sum')
        self.add_state('hyper_counts', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('corrects_sizes_vector', default=torch.zeros(num_classes + 2), dist_reduce_fx='sum')
        self.add_state('incorrects_sizes_vector', default=torch.zeros(num_classes + 2), dist_reduce_fx='sum')

    def update(self, input, hyper_set, target=None):
        set_sizes = input[:, :-1].sum(dim=1)
        self.multinomial_number += set_sizes.sum()
        hyper_set_sizes = hyper_set.sum(dim=1)
        hs = (1 - set_sizes) * hyper_set_sizes
        # take non-zero hyper set sizes
        hs = hs[hs > 0]
        self.hyper_counts = torch.cat((self.hyper_counts, hs))
        set_sizes = set_sizes + (1 - set_sizes) * hyper_set_sizes
        self.counts = torch.cat((self.counts, set_sizes))

        if target is not None:
            single_corrects = (input[:, :-1].argmax(dim=1) == target.argmax(dim=1)) & (input[:, :-1].sum(dim=1) > 0)
            single_incorrects = (input[:, :-1].argmax(dim=1) != target.argmax(dim=1)) & (input[:, :-1].sum(dim=1) > 0)
            set_corrects = ((hyper_set & target).sum(dim=1) > 0) & (input[:, -1] > 0)
            set_incorrects = ((hyper_set & target).sum(dim=1) == 0) & (input[:, -1] > 0)

            set_corrects_per_set_size = set_sizes[set_corrects].int()
            set_incorrects_per_set_size = set_sizes[set_incorrects].int()
            single_corrects_per_set_size = set_sizes[single_corrects].int()
            single_incorrects_per_set_size = set_sizes[single_incorrects].int()
            n_classes = input.size(1) - 1
            self.corrects_sizes_vector += torch.bincount(set_corrects_per_set_size, minlength=n_classes + 2)
            self.incorrects_sizes_vector += torch.bincount(set_incorrects_per_set_size, minlength=n_classes + 2)
            self.corrects_sizes_vector[-1] += single_corrects_per_set_size.sum()
            self.incorrects_sizes_vector[-1] += single_incorrects_per_set_size.sum()

        return set_sizes

    def merge_state(self, metrics):
        for metric in metrics:
            self.counts = torch.cat((self.counts, metric.counts))
        return self.counts

    def compute(self):
        return self.counts.mean()

    def plot(self):
        sns.barplot(x=range(0, 10), y=[self.counts.eq(i).sum().item() for i in range(0, 10)])
        plt.title('Number of classes in suggeted set')
        plt.show()

        # plot this stacked bar plot with corrects and incorrects
        corrects = self.corrects_sizes_vector.cpu().numpy()
        incorrects = self.incorrects_sizes_vector.cpu().numpy()
        x = range(len(corrects))
        plt.bar(x, corrects, color='g')
        plt.bar(x, incorrects, bottom=corrects, color='r')
        plt.title('Corrects and incorrects per set size')
        plt.show()

        sns.barplot(x=range(0, 10), y=[self.hyper_counts.eq(i).sum().item() for i in range(0, 10)])
        plt.title('Number of classes in Hyperset')
        plt.show()
        sns.barplot(x=['Multinomial', 'Hyper'], y=[self.multinomial_number.item(), self.hyper_counts.shape[0]])
        plt.title('Multinomial vs Hyper predictions')
        plt.show()


def compute_weights(num_class=10):
    """
    Compute weights for each tolerance level and for weight sizes 2 to num_class.

    For each j in range(2, num_class+1):
      - The initial guess for the weights is generated as random values of length j.
      - For each tolerance level tol (0.5, 0.6, 0.7, 0.8, 0.9), SLSQP minimizes `func`
        subject to three constraints:
          1. cons1(x) == 1
          2. cons2(x) == tol
          3. x >= 1e-8 (elementwise)
      - The computed weight vector (of length j) is stored as one row in an array of shape (5, j).

    Parameters:
      num_class (int): Maximum number of classes. The function computes weights for j=2,...,num_class.
      cons1 (callable): A function such that cons1(x) should equal 1.
      cons2 (callable): A function such that cons2(x) should equal tol.
      func (callable): The objective function to be minimized.

    Returns:
      dict: A dictionary mapping keys 'weight2', 'weight3', …, 'weight{num_class}' to a NumPy array of shape (5, j)
            containing the computed weights for each tolerance level.

    Example:
      weights = compute_weights(num_class=10, cons1=my_cons1, cons2=my_cons2, func=my_func)
      # weights['weight2'] is an array of shape (5,2)
      # weights['weight3'] is an array of shape (5,3), etc.
    """

    # Ensure that the required functions have been provided.

    def neg_entropy_criterion(x):
        fun = 0
        for i in range(len(x)):
            fun += x[i] * math.log10(x[i])
        return fun

    # constraint 1: the sum of weights is 1
    def cons1(x):
        return sum(x)

    # constraint 2: define tolerance to imprecision
    def cons2(x):
        tol = 0
        for i in range(len(x)):
            tol += (len(x) - (i + 1)) * x[i] / (len(x) - 1)
        return tol

    weight_dict = {}

    # Loop over j from 2 to num_class (inclusive)
    for j in tqdm(range(2, num_class + 1)):
        num_weights = j
        # Initial guess for the weights
        ini_weights = np.random.rand(num_weights)
        # Create an array to hold the 5 solutions (one per tolerance level)
        weight_array = np.zeros((5, j))

        # For each of the 5 tolerance levels
        for i in range(5):
            tol = 0.5 + i * 0.1  # tol in {0.5, 0.6, 0.7, 0.8, 0.9}

            # Define constraints.
            # Note: we use default arguments in the lambda to "freeze" the current tol value.
            constraints = (
                {'type': 'eq', 'fun': lambda x: cons1(x) - 1},
                {'type': 'eq', 'fun': lambda x: cons2(x) - tol},
                {'type': 'ineq', 'fun': lambda x: x - 1e-8}
            )

            # Minimize the objective using SLSQP.
            res = minimize(neg_entropy_criterion, ini_weights, method='SLSQP', options={'disp': False},
                           constraints=constraints)

            # Store the resulting weights.
            weight_array[i, :] = res.x
            # print(f"Computed weights for j={j}, tol={tol}: {res.x}")

        # Save the computed weight array in the dictionary.
        weight_dict[j] = weight_array

    return weight_dict


def total_utility(inputs, labels, tol, weight_dict):
    """
    Compute the average utility without storing an explicit act_set.

    For each instance (row in `inputs`):
      - The "act-set" is taken to be the indices where the input mass is > threshold.
      - If the true label is not among the active indices, the instance's utility is 0.
      - If the act-set contains exactly one index and that index equals the true label,
        the utility is defined as 1.
      - Otherwise (if more than one index is active and the true label is among them),
        the utility is given by weight_dict[n][tol_i, 0], where n is the number of active indices.

    Args:
      inputs (torch.Tensor): Tensor of shape (batch_size, num_class) with the mass/prediction for each class.
      labels (torch.Tensor or list): Length-`batch_size` containing the true class indices.
      tol_i (int): Tolerance index (e.g., 3 corresponds to tol=0.8) used to select the weight.
      weight_dict (dict): Maps the number of active classes (int) to a weight tensor of shape (num_tol, 1).
                          For example, weight_dict[2] might be a tensor with weights for cases with 2 active classes.
      threshold (float): Threshold to decide whether a mass is “active.”

    Returns:
      avg_utility (float): The average utility over the batch.
    """
    total_utility = 0.0
    batch_size = len(inputs)
    assert 0.5 <= tol <= 0.9, "Tolerance level must be in the range [0.5, 0.9]."
    tol_i = int((tol - 0.5 + 1e-5) * 10)

    # Ensure labels is a tensor on the same device as inputs.
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, device=inputs.device)

    # Process each instance.
    for i in range(batch_size):
        pred_set = inputs[i]
        n_active = len(pred_set)

        # Get the true label for this instance.
        true_label = labels[i].argmax().item() if torch.is_tensor(labels[i]) else labels[i]

        # If no active class or if the true label is not among them, utility is 0.
        if n_active == 0 or true_label not in pred_set:
            util = 0.0
        else:
            if n_active == 1:
                util = 1.0
            else:
                if n_active in weight_dict:
                    # Extract the scalar weight from the weight tensor.
                    util = weight_dict[n_active][tol_i, 0].item()
                else:
                    util = 0.0
        total_utility += util
    return total_utility

def get_fb_measure(inputs, targets, beta=1):
    set_sizes = torch.tensor([len(s) for s in inputs]).to(targets.device)
    utilities = ((1 + beta ** 2) / (set_sizes + beta ** 2)).mean()
    corrects = torch.tensor([targets[i].argmax() in inputs[i] for i in range(len(inputs))]).to(targets.device)
    return (utilities * corrects).sum()

class AverageUtility(Metric):
    def __init__(self, num_classes, utility='owa', tolerance=0.7, beta=1, as_list=True, **kwargs):
        super(AverageUtility, self).__init__(**kwargs)
        # self.utility_matrix = utility_matrix
        self.as_list = as_list
        self.add_state('utility', default=torch.tensor([0.]), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor([0]), dist_reduce_fx='sum')
        self.num_classes = num_classes
        self.tolerance = tolerance
        self.utility_type = utility
        if utility == 'owa':
            # check if cache directory has saved weights for the given number of classes
            # if not, compute the weights and save them, if the cache directory does not exist, create it
            if not os.path.exists('cache'):
                os.makedirs('cache')
            cache_file = f'cache/weights_{num_classes}.pt'
            if os.path.exists(cache_file):
                weight_matrix = torch.load(cache_file, weights_only=False)
            else:
                print(f"Computing weights for {num_classes} classes for OWA utility metric.")
                weight_matrix = compute_weights(num_classes)
                torch.save(weight_matrix, cache_file)
            self.utility_f = lambda x, y: total_utility(x, y, self.tolerance, weight_matrix)
        elif utility == 'fb':
            self.utility_f = lambda x, y: get_fb_measure(x, y, beta)
        else:
            raise ValueError(f"Unknown utility type: {utility}")

    def update(self, inputs, labels):
        if not self.as_list:
            rows, cols = inputs.nonzero(as_tuple=True)
            _, counts = torch.unique_consecutive(rows, return_counts=True)
            groups = torch.split(cols, tuple(counts.tolist()))
            inputs = list(map(lambda x: x.tolist(), groups))
        utility = self.utility_f(inputs, labels)
        self.utility += utility
        n_inputs = len(inputs)
        self.total += n_inputs
        return utility / n_inputs

    def merge_state(self, metrics):
        for metric in metrics:
            self.utility += metric.utility
            self.total += metric.total

    def compute(self):
        return self.utility / self.total


class SetSize(Metric):
    def __init__(self, **kwargs):
        super(SetSize, self).__init__(**kwargs)
        self.add_state('set_sizes', default=torch.tensor([]), dist_reduce_fx='cat')

    def update(self, inputs, labels):
        set_sizes = torch.tensor([len(s) for s in inputs]).to(labels.device)
        self.set_sizes = torch.cat((self.set_sizes, set_sizes))
        return set_sizes.float().mean()

    def merge_state(self, metrics):
        for metric in metrics:
            self.set_sizes = torch.cat((self.set_sizes, metric.set_sizes))
        return self.set_sizes

    def compute(self):
        return self.set_sizes.float().mean()


class CorrectIncorrectUncertaintyPlotter(Metric):
    def __init__(self, **kwargs):
        super(CorrectIncorrectUncertaintyPlotter, self).__init__(**kwargs)
        self.add_state('corrects', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('uncertainties', default=torch.tensor([]), dist_reduce_fx='cat')

    def update(self, inputs, labels):
        corrects = (inputs.argmax(dim=1) == labels).float()
        uncertainties = inputs.shape[1] / (inputs + 1).sum(dim=1)
        self.corrects = torch.cat((self.corrects, corrects))
        self.uncertainties = torch.cat((self.uncertainties, uncertainties))

    def merge_state(self, metrics):
        for metric in metrics:
            self.corrects = torch.cat((self.corrects, metric.corrects))
            self.uncertainties = torch.cat((self.uncertainties, metric.uncertainties))

    def compute(self):
        return self.corrects, self.uncertainties

    def plot(self):
        corrects = self.corrects.cpu().numpy()
        incorrects = 1 - corrects

        corrects_uncertainty = self.uncertainties[self.corrects > 0.5].cpu().numpy()
        incorrects_uncertainty = self.uncertainties[self.corrects < 0.5].cpu().numpy()

        sns.kdeplot(corrects_uncertainty, label='Correct', color='g')
        sns.kdeplot(incorrects_uncertainty, label='Incorrect', color='r')
        plt.xlabel('Uncertainty')
        plt.ylabel('Density')
        plt.title('Correct vs Incorrect Uncertainty')
        plt.legend()
        plt.show()


class HyperUncertaintyPlotter(Metric):
    def __init__(self, **kwargs):
        super(HyperUncertaintyPlotter, self).__init__(**kwargs)
        self.add_state('uncertainties', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('corrects', default=torch.tensor([]), dist_reduce_fx='cat')

    def update(self, inputs, evidences, labels):
        corrects = torch.tensor([labels[i].argmax() in inputs[i] for i in range(len(inputs))]).to(labels.device)
        uncertainties = evidences.shape[1] / (evidences + 1).sum(dim=1)
        self.uncertainties = torch.cat((self.uncertainties, uncertainties))
        self.corrects = torch.cat((self.corrects, corrects))

    def merge_state(self, metrics):
        for metric in metrics:
            self.uncertainties = torch.cat((self.uncertainties, metric.uncertainties))

    def compute(self):
        return self.uncertainties

    def plot(self):
        uncertainties = self.uncertainties.cpu().numpy()
        corrects = self.corrects.cpu().numpy()

        corrects_uncertainty = uncertainties[corrects > 0.5]
        incorrects_uncertainty = uncertainties[corrects < 0.5]

        sns.kdeplot(corrects_uncertainty, label='Correct', color='g')
        sns.kdeplot(incorrects_uncertainty, label='Incorrect', color='r')
        plt.xlabel('Uncertainty')
        plt.ylabel('Density')
        plt.title('Hyper Uncertainty')
        plt.legend()
        plt.show()



class BetaEvidenceAccumulator(Metric):
    def __init__(self, **kwargs):
        super(BetaEvidenceAccumulator, self).__init__(**kwargs)
        self.add_state('alpha', default=[], dist_reduce_fx='cat')
        self.add_state('beta', default=[], dist_reduce_fx='cat')
        self.add_state('labels', default=[], dist_reduce_fx='cat')

    def update(self, alpha, beta, labels):
        self.alpha.append(alpha)
        self.beta.append(beta)
        self.labels.append(labels)

    def merge_state(self, metrics):
        for metric in metrics:
            self.alpha.extend(metric.alpha)
            self.beta.extend(metric.beta)
            self.labels.extend(metric.labels)

    def compute(self):
        alpha = torch.cat(self.alpha).to('cpu')
        beta = torch.cat(self.beta).to('cpu')
        labels = torch.cat(self.labels).to('cpu')
        return alpha, beta, labels

    def save(self, path):
        alpha = torch.cat(self.alpha).to('cpu')
        beta = torch.cat(self.beta).to('cpu')
        labels = torch.cat(self.labels).to('cpu')
        torch.save((alpha, beta, labels), path)


def eval_calibration(predictions, confidences, labels, device, M=15):
    """
    function adapted from: https://github.com/Cogito2012/DEAR/
    M: number of bins for confidence scores
    """
    num_Bm = torch.zeros((M,), dtype=torch.int32, device=device)
    accs = torch.zeros((M,), dtype=torch.float32, device=device)
    confs = torch.zeros((M,), dtype=torch.float32, device=device)
    for m in range(M):
        interval = [m / M, (m + 1) / M]
        Bm = torch.where((confidences > interval[0]) & (confidences <= interval[1]))[0]
        if len(Bm) > 0:
            acc_bin = torch.sum(predictions[Bm] == labels[Bm]).item() / len(Bm)
            conf_bin = torch.mean(confidences[Bm]).item()
            # gather results
            num_Bm[m] = len(Bm)
            accs[m] = acc_bin
            confs[m] = conf_bin
    conf_intervals = torch.arange(0, 1, 1 / M).to(device)
    return accs, confs, num_Bm, conf_intervals

class ExpectedCalibrationError(Metric):
    def __init__(self, num_classes, M=15, **kwargs):
        super(ExpectedCalibrationError, self).__init__(**kwargs)
        self.add_state('accs', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('confs', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('num_Bm', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('conf_intervals', default=torch.tensor([]), dist_reduce_fx='cat')
        self.M = M
        self.num_classes = num_classes

    def update(self, evidences, labels):
        predictions = evidences.argmax(dim=1)
        uncertainties = evidences.shape[1] / (evidences + 1).sum(dim=1)
        confidences = 1 - uncertainties
        accs, confs, num_Bm, conf_intervals = eval_calibration(predictions, confidences, labels, labels.device, M=self.M)
        self.accs = torch.cat((self.accs, accs))
        self.confs = torch.cat((self.confs, confs))
        self.num_Bm = torch.cat((self.num_Bm, num_Bm))
        self.conf_intervals = torch.cat((self.conf_intervals, conf_intervals))

    def merge_state(self, metrics):
        for metric in metrics:
            self.accs = torch.cat((self.accs, metric.accs))
            self.confs = torch.cat((self.confs, metric.confs))
            self.num_Bm = torch.cat((self.num_Bm, metric.num_Bm))
            self.conf_intervals = torch.cat((self.conf_intervals, metric.conf_intervals))

    def compute(self):
        return torch.sum(torch.abs(self.accs - self.confs) * self.num_Bm / torch.sum(self.num_Bm))

    def plot(self):
        accs = self.accs.cpu().numpy()
        confs = self.confs.cpu().numpy()
        num_Bm = self.num_Bm.cpu().numpy()
        conf_intervals = self.conf_intervals.cpu().numpy()

        plt.plot(conf_intervals, accs, label='Accuracy')
        plt.plot(conf_intervals, confs, label='Confidence')
        plt.plot(conf_intervals, num_Bm, label='Number of samples')
        plt.xlabel('Confidence')
        plt.ylabel('Value')
        plt.title('Expected Calibration Error')
        plt.legend()
        plt.show()


class TimeLogger(Metric):
    def __init__(self, reduction='sum', **kwargs):
        super(TimeLogger, self).__init__(**kwargs)
        self.add_state('times', default=torch.tensor([]), dist_reduce_fx='cat')
        self.reduction = reduction

    def update(self, time):
        self.times = torch.cat((self.times, torch.tensor([time], device=self.device)))

    def merge_state(self, metrics):
        for metric in metrics:
            self.times = torch.cat((self.times, metric.times))

    def compute(self):
        if self.reduction == 'sum':
            return self.times.sum()
        elif self.reduction == 'mean':
            return self.times.mean()
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction}")

    def save(self, path):
        torch.save(self.times, path)