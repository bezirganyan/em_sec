from torch import nn

class FitNet4(nn.Module):
    def __init__(self, output_dim=10, dropout=0.5, **kwargs):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout))

        self.block_2 = nn.Sequential(
            nn.Conv2d(48, 80, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(80, 80, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(80, 80, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(80, 80, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(80, 80, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(80),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout))

        self.block_3 = nn.Sequential(
            nn.Conv2d(80, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(8, 8),
            nn.Dropout(dropout))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, output_dim)


    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x