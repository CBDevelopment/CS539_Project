
import torch.nn as nn
import torch.nn.functional as F


class MultiCityClassifier(nn.Module):
    """
    - Convolutional neural network for classifying images from 6 cities: Boston, Amsterdam, Paris, Phoenix, Toronto, Zurich
    """

    def __init__(self):
        super(MultiCityClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        # (250 - 3 + 1) = 248
        self.pool = nn.MaxPool2d(2, 2)
        # 248 / 2 = 124
        self.fc1 = nn.Linear(32 * 124 * 124, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 124 * 124)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
