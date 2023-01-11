import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""


class FCN(nn.Module):
    """
    This class implements the fully-connected neural network definition and its corresponding methods
    """

    def __init__(self, hidden_layers, history_length=None, dim_state=8, n_classes=4):
        super(FCN, self).__init__()
        # Define layers of a fully-connected neural network
        self.fc1 = nn.Linear(dim_state, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_layers, n_classes)

    def forward(self, x):
        """
        This method implement the forward pass of the neural network
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    """
    This class implements the conv neural network definition and its corresponding methods
    """

    def __init__(self, history_length, n_classes=4):
        super(CNN, self).__init__()
        # Definition layers of a convolutional neural network
        self.conv1 = nn.Conv2d(history_length, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(224256, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        """
        This method implement the forward pass of the neural network
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        new_input_size = x.size(dim=1)*x.size(dim=2)*x.size(dim=3)
        x = x.view(-1, new_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
