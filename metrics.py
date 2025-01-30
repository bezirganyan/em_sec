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
    def __init__(self, n_classes=10, **kwargs):
        super(HyperSetSize, self).__init__(**kwargs)
        self.add_state('counts', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('multinomial_number', default=torch.tensor([0]), dist_reduce_fx='sum')
        self.add_state('hyper_counts', default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state('corrects_sizes_vector', default=torch.zeros(n_classes+2), dist_reduce_fx='sum')
        self.add_state('incorrects_sizes_vector', default=torch.zeros(n_classes+2), dist_reduce_fx='sum')

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
            self.corrects_sizes_vector += torch.bincount(set_corrects_per_set_size, minlength=n_classes+2)
            self.incorrects_sizes_vector += torch.bincount(set_incorrects_per_set_size, minlength=n_classes+2)
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
