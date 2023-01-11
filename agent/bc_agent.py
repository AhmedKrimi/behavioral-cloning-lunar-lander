import torch
import torch.nn as nn
import torch.optim as optim
from agent.networks import CNN
from agent.networks import FCN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BCAgent:
    """
    This Class implement the neural network type and corresponding methods for behavioral cloning as an Agent
    """
    def __init__(self, network_type: str, lr: float, hidden_layers: int, history_length: int):
        #  Define network, FCN : Fully connect Network, CNN : Convolutional Neural Network
        if network_type == "FCN":
            self.net = FCN(hidden_layers).to(device)
        else:
            self.net = CNN(history_length).to(device)
        # Define loss function
        self.loss_fcn = nn.CrossEntropyLoss()
        # Define optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr)

    def update(self, X_batch, y_batch):
        '''
        This method updates the parameters of the agent
        '''
        # Transform input to tensors
        X_batch = torch.tensor(X_batch).to(device)
        y_batch = torch.FloatTensor(y_batch).to(device)
        self.net.zero_grad()
        # forward pass
        output = self.net(X_batch)
        y_batch = y_batch.view(y_batch.size(0))
        loss = self.loss_fcn(output, y_batch.long())
        # backward pass
        loss.backward()
        # optimizer step
        self.optimizer.step()
        return loss

    def predict(self, X):
        """
        This method implements the forward pass of the neural network
        """
        # forward pass
        X = X.to(device)
        outputs = self.net(X)
        return outputs

    def save(self, file_name):
        """
        This method saves the parameters of the neural network
        """
        torch.save(self.net.state_dict(), file_name)

    def load(self, file_name):
        """
        This method loads the parameters of the neural network
        """
        self.net.load_state_dict(torch.load(file_name, map_location=device))
