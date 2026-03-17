import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


class CustomBayesLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_logvar = torch.nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3))
        self.bias_mu = torch.nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_logvar = torch.nn.Parameter(torch.Tensor(out_features).fill_(-3))

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_loss(self):
        kl = 0.5 * (torch.exp(self.weight_logvar) + self.weight_mu ** 2 - 1 - self.weight_logvar).sum()
        kl += 0.5 * (torch.exp(self.bias_logvar) + self.bias_mu ** 2 - 1 - self.bias_logvar).sum()
        return kl


class BayesianGCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.bayes_fc = CustomBayesLinear(out_channels, out_channels)
        self.bn = BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = self.bayes_fc(x)
        x = self.bn(x)
        return F.elu(x)

    def kl_loss(self):
        return self.bayes_fc.kl_loss()


class AdvancedBayesianGATGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=64, heads=4, dropout=0.3):
        super().__init__()
        self.dropout_rate = dropout
        self.dropout_layer = Dropout(p=dropout)

        # GAT 层
        self.gat = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)

        # 贝叶斯 GCN 层
        self.bayesian_gcn1 = BayesianGCNLayer(hidden_channels * heads, hidden_channels)
        self.bayesian_gcn2 = BayesianGCNLayer(hidden_channels, hidden_channels)

        # 中间全连接层
        self.linear = Linear(hidden_channels, hidden_channels)
        self.bn_linear = BatchNorm1d(hidden_channels)
        self.relu = ReLU()

        # 输出贝叶斯层，使用可调节的输出维度
        self.bayes_fc_out = CustomBayesLinear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.gat(x, edge_index)
        x = self.dropout_layer(x)

        x = self.bayesian_gcn1(x, edge_index)
        x = self.dropout_layer(x)

        x = self.bayesian_gcn2(x, edge_index)
        x = global_mean_pool(x, batch)

        x = self.bn_linear(self.linear(x))
        x = self.relu(x)
        x = self.dropout_layer(x)

        out = self.bayes_fc_out(x)
        return F.log_softmax(out, dim=1), out  # 可以选择使用 out 作为特征

    def kl_loss(self):
        return (
            self.bayesian_gcn1.kl_loss() +
            self.bayesian_gcn2.kl_loss() +
            self.bayes_fc_out.kl_loss()
        )
