import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.optimize import minimize
from torchmetrics import Metric


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
    def __init__(self, n_classes=10, **kwargs):
        super(HyperSetSize, self).__init__(**kwargs)
        self.add_state('counts', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('multinomial_number', default=torch.tensor([0]), dist_reduce_fx='sum')
        self.add_state('hyper_counts', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('corrects_sizes_vector', default=torch.zeros(n_classes + 2), dist_reduce_fx='sum')
        self.add_state('incorrects_sizes_vector', default=torch.zeros(n_classes + 2), dist_reduce_fx='sum')

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
    for j in range(2, num_class + 1):
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
            res = minimize(neg_entropy_criterion, ini_weights, method='SLSQP', options={'disp': True},
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
    def __init__(self, num_classes, utility='owa', tolerance=0.7, beta=1, **kwargs):
        super(AverageUtility, self).__init__(**kwargs)
        # self.utility_matrix = utility_matrix
        self.add_state('utility', default=torch.tensor([0.]), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor([0]), dist_reduce_fx='sum')
        self.num_classes = num_classes
        self.tolerance = tolerance
        self.utility_type = utility
        if utility == 'owa':
            weight_matrix = compute_weights(num_classes)
            self.utility_f = lambda x, y: total_utility(x, y, self.tolerance, weight_matrix)
        elif utility == 'fb':
            self.utility_f = lambda x, y: get_fb_measure(x, y, beta)
        else:
            raise ValueError(f"Unknown utility type: {utility}")

    def update(self, inputs, labels):
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
