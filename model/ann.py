
import torch.nn as nn
import torch.nn.functional as F
# NN architecture
class ANN(nn.Module):
    def __init__(self, D_in, D_out,h=[22,22]):
        super(ANN, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = h[0]
        hidden_2 = h[1]
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(D_in, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, D_out)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = F.relu(self.fc3(x))
        return x