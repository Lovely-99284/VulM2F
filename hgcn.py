# import torch
# import torch.nn.functional as F
# from torch.nn import Linear
# from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, global_max_pool, BatchNorm
#
# class TrueHGCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
#         self.bn1 = BatchNorm(hidden_channels)
#
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.pool2 = TopKPooling(hidden_channels, ratio=0.8)
#         self.bn2 = BatchNorm(hidden_channels)
#
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)
#         self.pool3 = TopKPooling(hidden_channels, ratio=0.8)
#         self.bn3 = BatchNorm(hidden_channels)
#
#         self.lin1 = Linear(hidden_channels * 2, hidden_channels)
#         self.lin2 = Linear(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index, batch):
#         # --- 第一层 ---
#         x = F.relu(self.bn1(self.conv1(x, edge_index)))
#         x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
#         x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
#
#         # --- 第二层 ---
#         x = F.relu(self.bn2(self.conv2(x, edge_index)))
#         x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
#         x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
#
#         # --- 第三层 ---
#         x = F.relu(self.bn3(self.conv3(x, edge_index)))
#         x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
#         x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
#
#         # --- 聚合三层输出（跳跃连接）---
#         x = x1 + x2 + x3
#         x = F.relu(self.lin1(x))
#         x = self.lin2(x)
#         return x


#上面的特征维度是71，下面的代码对应特征维度128，可以自己进行补全或者减少

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, global_max_pool, BatchNorm

class TrueHGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # 输入线性映射层，将任意输入特征维度映射到128维
        self.input_proj = Linear(in_channels, 128)

        # 第一层GCN，输入128，输出hidden_channels
        self.conv1 = GCNConv(128, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        self.bn1 = BatchNorm(hidden_channels)

        # 第二层GCN
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.8)
        self.bn2 = BatchNorm(hidden_channels)

        # 第三层GCN
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.pool3 = TopKPooling(hidden_channels, ratio=0.8)
        self.bn3 = BatchNorm(hidden_channels)

        # 全连接层，输入为池化后的特征维度 hidden_channels*2
        self.lin1 = Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 输入特征线性映射到128维
        x = F.relu(self.input_proj(x))

        # --- 第一层 ---
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # --- 第二层 ---
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # --- 第三层 ---
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # --- 跳跃连接三层输出 ---
        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
