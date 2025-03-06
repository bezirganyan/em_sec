from torch import nn

class DenseClassifier(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=(256, 256)):
        super(DenseClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_features[0]))
        for i in range(1, len(hidden_features)):
            self.layers.append(nn.Linear(hidden_features[i - 1], hidden_features[i]))
        self.linear = nn.Linear(hidden_features[-1], out_features)

    def forward(self, x):
        for layer in self.layers:
            x = nn.functional.relu(layer(x))
        return self.linear(x)
