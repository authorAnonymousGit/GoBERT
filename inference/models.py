from torch.nn import Linear
from torch.nn import functional as Func
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch


class GCN(torch.nn.Module):
    def __init__(self, features_num, hidden_channels, labels_num):
        super(GCN, self).__init__()
        torch.manual_seed(12345)  # 12345
        self.conv1 = GCNConv(features_num, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels//2)
        self.conv5 = GCNConv(hidden_channels//2, hidden_channels//2)
        self.conv6 = GCNConv(hidden_channels//2, hidden_channels//4)
        self.lin = Linear(hidden_channels//4, labels_num)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, edge_index, batch, primary_idxs_in_batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.conv5(x, edge_index)
        x = x.relu()
        x = self.conv6(x, edge_index)

        if primary_idxs_in_batch:
            # x = F.dropout(x, p=0.2, training=self.training)
            x = self.lin(x[primary_idxs_in_batch])
        else:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)

        softmax_vals = Func.softmax(x, dim=1)
        x = self.logsoftmax(x)
        return x, softmax_vals
