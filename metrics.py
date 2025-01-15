import matplotlib.pyplot as plt
import torch
from streamlit import title
from torchmetrics import Metric
import seaborn as sns


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

        set_corrects =  ((hyperset & target).sum(dim=1) > 0) & (input[:, -1] > 0)

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
    def __init__(self, **kwargs):
        super(HyperSetSize, self).__init__(**kwargs)
        self.add_state('counts', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('multinomial_number', default=torch.tensor([0]), dist_reduce_fx='sum')
        self.add_state('hyper_counts', default=torch.tensor([]), dist_reduce_fx='cat')

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

        sns.barplot(x=range(0, 10), y=[self.hyper_counts.eq(i).sum().item() for i in range(0, 10)])
        plt.title('Number of classes in Hyperset')
        plt.show()
        sns.barplot(x=['Multinomial', 'Hyper'], y=[self.multinomial_number.item(), self.hyper_counts.shape[0]])
        plt.title('Multinomial vs Hyper predictions')
        plt.show()
