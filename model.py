import torch
from torch.nn import Linear, ReLU, Softmax
from torch_geometric.nn import SAGEConv


class MyGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(MyGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels=-1, out_channels=hidden_channels, normalize=True)
        self.linear = Linear(in_features=hidden_channels, out_features=out_channels)
        self.relu = ReLU()
        self.conv2 = SAGEConv(in_channels=out_channels, out_channels=out_channels, normalize=True)
        self.linear2 = Linear(in_features=out_channels, out_features=2)
        self.softmax = Softmax(dim=1)

    def forward(self, x, edge_index):
        x = x.to(torch.float32)
        x = self.conv1(x, edge_index)
        x = self.linear(x)
        x = self.relu(x)
        y = self.conv2(x, edge_index)
        x = self.linear2(y)
        x = self.softmax(x)
        return x, y
