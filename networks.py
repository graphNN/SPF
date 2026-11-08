# 导入类型提示相关的模块
from typing import Optional  # 用于定义可选参数类型
from torch_geometric.typing import OptTensor  # PyTorch Geometric的可选张量类型

# 导入数学和深度学习相关的库
import math  # 数学函数库，用于三角函数等计算
import torch  # PyTorch核心库
import torch.nn.functional as F  # PyTorch的函数式API，包含激活函数、损失函数等
import numpy as np  # 数值计算库（本文件中未使用，可以移除）

# 导入PyTorch的张量和神经网络模块
from torch import Tensor  # 张量类型
from torch.nn import Parameter, ModuleList, Linear, init  # 可学习参数、模块列表、线性层、初始化函数
from torch_geometric.nn import MessagePassing  # PyTorch Geometric的消息传递基类
from torch_geometric.utils import add_self_loops, get_laplacian, remove_self_loops  # 图操作工具函数

def poly(i, x):
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        T0 = 1  # T_0(x)
        T1 = x  # T_1(x)
        for _ in range(2, i+1):
            # T_n(x) = 2x·T_{n-1}(x) - T_{n-2}(x)
            T2 = 2 * x * T1 - T0
            # T_{n-2} = T_{n-1}, T_{n-1} = T_n
            T0, T1 = T1, T2
        return T2

class CPF_prop_g(MessagePassing):
    def __init__(self, num_partitions: int, K: int,
                 normalization: Optional[str] = 'sym', bias: bool = True,
                 **kwargs):
        super(CPF_prop_g, self).__init__(aggr='add', **kwargs)
        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.K = K
        self.num_partitions = num_partitions
        self.normalization = normalization
        self.partition_coeffs = Parameter(torch.Tensor(num_partitions, K+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.partition_coeffs.data.fill_(0.5)
        
    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max: OptTensor = None, dtype: Optional[int] = None):
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        lambda_max = 2.0 * edge_weight.max()
        assert lambda_max is not None
        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return 'CPFGNN'

    def forward(self, x: Tensor, edge_index: Tensor, partition_matrix: Tensor,
                edge_weight: OptTensor = None, lambda_max: OptTensor = None):
        coe_tmp = F.relu(self.partition_coeffs)
        coe = coe_tmp.clone()
        for i in range(self.K+1):
            coe[:, i] = coe_tmp[:, 0] * poly(i, math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1, self.K+1):
                x_j = math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[:, i] = coe[:, i] + coe_tmp[:, j] * poly(i, x_j)

            coe[:, i] = 2 * coe[:, i] / (self.K+1)

        node_coe = torch.mm(partition_matrix, coe)  # (num_nodes, K+1)
        edge_index_tilde, norm_tilde = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype)

        Tx_0 = x  # (num_nodes, num_features)
        out = node_coe[:, 0:1] / 2 * Tx_0 + Tx_0  # (num_nodes, num_features)
        Tx_1 = self.propagate(edge_index_tilde, x=Tx_0, norm=norm_tilde, size=None)
        out = out + node_coe[:, 1:2] * Tx_1

        for k in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + node_coe[:, k:k+1] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out  # (num_nodes, num_features)

class CPF_prop_f(torch.nn.Module):
    def __init__(
        self,
        num_partitions: int,
        K: int,
        channels: int,
    ):
        super(CPF_prop_f, self).__init__()
        self.num_partitions = num_partitions
        self.K = K
        self.channels = channels
        self.partition_coeffs = Parameter(torch.Tensor(num_partitions, K+1))
        self.partition_transform = Parameter(torch.Tensor(num_partitions, channels, channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.partition_coeffs.data.fill_(0.5)
        for i in range(self.num_partitions):
            init.xavier_normal_(self.partition_transform[i], gain=0.5)

    def forward(self, x: torch.Tensor, partition_matrix: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for p in range(self.num_partitions):
            weights = partition_matrix[:, p:p+1]  # (num_nodes, 1)
            transformed = torch.mm(x, self.partition_transform[p])  # (num_nodes, channels)
            out = out + weights * transformed

        return out  # (num_nodes, channels)

class CPFGNN(torch.nn.Module):
    def __init__(self, args, data):
        super(CPFGNN, self).__init__()
        self.lin1 = Linear(args.num_features, args.hidden)

        if args.num_classes == 2:
            self.out_classes = 1
        else:
            self.out_classes = args.num_classes
        self.lin2 = Linear(args.hidden, self.out_classes)
        self.prop_g = CPF_prop_g(args.num_partitions, args.K)
        self.prop_f = CPF_prop_f(args.num_partitions, args.K, self.out_classes)
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop_g.reset_parameters()
        self.prop_f.reset_parameters()

    def forward(self, data):
        x, edge_index, partition_struct, partition_freq = data[0], data[1], data[2], data[3]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop_f(x, partition_freq, edge_index)

        return x  # (num_nodes, out_classes)
