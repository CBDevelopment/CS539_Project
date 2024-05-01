import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 800)
        self.fc2 = nn.Linear(800, 600)
        self.fc3 = nn.Linear(600, 400)
        self.fc4 = nn.Linear(400, 200)
        self.fc5 = nn.Linear(200, 50)
        self.fc6 = nn.Linear(50, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x