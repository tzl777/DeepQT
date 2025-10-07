from typing import Tuple
import time
import torch
from torch import nn
from torch_geometric.utils import degree
import networkx as nx
from .functional import extend_coords_periodic, calculate_voronoi_areas, calculate_voronoi_volumes, create_graph, visualize_2d_graph
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp
from functools import partial

def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x

# class CentralityEncoding(nn.Module):
#     def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
#         super().__init__()
#         self.max_in_degree = max_in_degree
#         self.max_out_degree = max_out_degree
#         self.node_dim = node_dim
#         self.z = nn.Parameter(torch.randn((1, node_dim)))

#     def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, edge_attr: torch.Tensor) -> torch.Tensor:
#         num_nodes = x.shape[0]
#         G = nx.Graph(edge_index)
#         # 计算节点的信息中心性
#         centrality = torch.tensor(list(nx.current_flow_betweenness_centrality(G, weight=edge_attr).values())).view(-1, 1)

#         x += self.z(centrality)

#         return x

#这段代码的核心思想是利用可学习的参数矩阵 self.z_in 和 self.z_out，根据节点的入度和出度为每个节点添加中心性信息，以丰富节点的表示。
class CentralityEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_in_degree: int, max_out_degree: int, node_dim: int, atom_update_net):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree #5
        self.max_out_degree = max_out_degree #5
        self.node_dim = node_dim #64
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim))) #定义了两个可学习的参数矩阵，每行元素分别表示可学习的不同入度和出度的中心性信息。
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))
        self.atom_update_net = atom_update_net
        if atom_update_net == 'TransformerM':
            self.centrality = nn.Linear(1, self.node_dim, dtype=torch.float64)
            self.voronoi = nn.Linear(1, self.node_dim, dtype=torch.float64)
            self.sigmoid = nn.Sigmoid()
            self.sum3Ddistance = nn.Linear(edge_dim, self.node_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, edge_attr: torch.Tensor, voronoi_values: torch.Tensor, centralities: torch.Tensor) -> torch.Tensor:
        """
        :param centralities:
        :param voronoi_values:
        :param edge_attr: graph edge feature matrix
        :param x: node feature matrix, [108,64]
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        num_nodes = x.shape[0]  # 图中节点个数,216

        in_degree = decrease_to_max_value(degree(index=edge_index[1], num_nodes=num_nodes).long(),
                                          self.max_in_degree - 1)#返回大小为216的入度列表，每个元素的值介于0到4之间
        
        #degree是计算给定一维索引张量的(未加权)度，即计算所有节点的入度。
        #decrease_to_max_value是将输入张量 x 中大于指定最大值 max_value 的元素替换为 max_value。用于将入度限制在max_in_degree值的范围，确保不会超过给定的最大值。
        out_degree = decrease_to_max_value(degree(index=edge_index[0], num_nodes=num_nodes).long(),
                                           self.max_out_degree - 1)#返回大小为889的出度列表，每个元素的值介于0到4之间,[72]
        #计算每个节点的入度和出度，并将其限制在 max_in_degree - 1 和 max_out_degree - 1 的范围内。大小为节点个数num_nodes。
        if self.atom_update_net == 'TransformerM':
            #计算sum of 3D distance
            device = next(self.parameters()).device
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            sum3Ddistance = torch.zeros((num_nodes, self.node_dim), device=device)
            for i in range(num_nodes):
                sum3Ddistance[i] = self.sum3Ddistance(edge_attr[edge_index[0] == i, :]).sum(dim=0)
            x += self.z_in[in_degree] + self.z_out[out_degree] + self.sigmoid(self.centrality(centralities)) + self.sigmoid(self.voronoi(voronoi_values)) + sum3Ddistance
        elif self.atom_update_net == 'Graphormer':
            # 如果想要查看每个节点的集群系数
            # all_clustering = torch.tensor(list(nx.clustering(G, weight=edge_attr).values())).view(-1, 1)
            x += self.z_in[in_degree] + self.z_out[out_degree]
        else:
            raise ValueError(f'Unknown atom_update_net: {self.atom_update_net}')
        #"*"是逐元素相乘, "@"是矩阵乘法
        #从self.z_in中选择self.z_in中的对应行，形成一个新的张量。
        #为每个节点的特征向量x添加了入度和出度的中心性信息。具体来说，对于每个节点，从self.z_in和self.z_out中获取对应的中心性表示，然后加到节点的特征向量上。
        #x是(num_nodes, node_dim),self.z_in[in_degree]返回的(num_nodes,node_dim),self.z_out[in_degree]返回的(num_nodes,node_dim)
        return x #根据节点的入度和出度为每个节点添加中心性信息，增加了中心性信息后的大小仍是(num_nodes, node_dim)

#只把节点间的最短路径长度编码进入了，没有考虑角度信息
class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_path_distance = max_path_distance #保存最大路径距离。

        self.b = nn.Parameter(torch.randn(self.max_path_distance + 1)) #定义了一个可学习的参数向量b,其长度为 max_path_distance.用来表示节点间的最大距离长度。

    def forward(self, x: torch.Tensor, paths) -> torch.Tensor: #前向传播方法，接受两个参数：x（节点特征矩阵）和 paths（成对节点路径）。
        """
        :param x: node feature matrix
        :param paths: pairwise node paths, 成对节点路径，每个子图内的最短路径
        :return: torch.Tensor, spatial Encoding matrix
        """
        #初始化一个全零的空间编码矩阵，大小为 (节点数量, 节点数量)，并将其转移到与模型参数所在设备相同的设备上。即可学习的。
        device = next(self.parameters()).device
        x = x.to(device)
        batch = paths.shape[0]
        node_mask = (paths == -1).to(device)
        path_lengths = (~node_mask).sum(dim=-1)
        num_nodes = x.shape[0]
        spatial_matrix = torch.zeros((num_nodes, num_nodes), device=device)
        # 将每个[72, 72]矩阵按主对角线放置
        for i in range(batch):
            start = i * 72
            end = start + 72
            spatial_matrix[start:end, start:end] = self.b[path_lengths[i]]
        return spatial_matrix

class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):

        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_dim = edge_dim #128
        self.max_path_distance = max_path_distance #5
        self.edge_vector = nn.Parameter(torch.randn(self.max_path_distance-1, self.edge_dim)) #[4,128]
        self.eps = 1e-9

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths) -> torch.Tensor:
        """
        :param x: node feature matrix, shape (batch_size * num_node, node_dim)
        :param edge_attr: edge feature matrix, shape (batch_size, num_edges, edge_dim)
        :param edge_paths: pairwise node paths in edge indexes, shape (batch_size, num_nodes, num_nodes, path of edge indexes to traverse from node_i to node_j where len(edge_paths) = max_path_length)
        :return: torch.Tensor, Edge Encoding
        """
        device = next(self.parameters()).device
        x = x.to(device) #[216,64]

        edge_paths = edge_paths.to(device) # torch.Size([3, 72, 72, 4])
        edge_attr = edge_attr.to(device) #[7992,128]

        batch_size = edge_paths.shape[0]
        num_nodes = x.shape[0]

        edge_attr = edge_attr.reshape((batch_size, -1, edge_attr.shape[-1])) #[3,2016,128]

        edge_mask = edge_paths == -1 #torch.Size([3, 72, 72, 4])
        edge_paths_clamped = edge_paths.clamp(min=0)
        batch_indices = torch.arange(batch_size).view(batch_size, 1, 1, 1).expand_as(edge_paths) #这个张量的形状与 data.edge_paths 相同，每个位置的值表示所在批次的索引。torch.Size([3, 72, 72, 4])
        # Get the edge embeddings for each edge in the paths (when defined)
        assert edge_attr is not None
        edge_path_embeddings = edge_attr[batch_indices, edge_paths_clamped, :].to(device) #torch.Size([3, 72, 72, 4, 128])
        edge_path_embeddings[edge_mask] = 0.0 #torch.Size([3, 72, 72, 4, 128])

        path_lengths = (~edge_mask).sum(dim=-1) + self.eps #torch.Size([3, 72, 72])

        # Get sum of embeddings * self.edge_vector for edge in the path,
        # then sum the result for each path
        # b: batch_size
        # n, m: padded num_nodes
        # l: max_path_length
        # d: edge_emb_dim
        # l d 表示 self.edge_vector 的维度。
        # return (batch_size, padded_num_nodes, padded_num_nodes)
        edge_path_encoding = torch.einsum("bnmld,ld->bnm", edge_path_embeddings, self.edge_vector) #->bnm 表示结果的维度为 (batch_size, num_nodes, num_nodes)。
        # 对edge_path_embeddings的最后两个维度(l, d)和self.edge_vector的维度(l, d)进行元素级乘法。
        # 然后沿着这两个维度求和，得到一个形状为(batch_size, num_nodes, num_nodes)的张量edge_path_encoding。

        edge_attr = edge_path_encoding.div(path_lengths) #torch.Size([3, 72, 72])

        edge_encoding_matrix = torch.zeros((num_nodes, num_nodes), device=device)
        num_node = edge_paths.shape[1]
        # 将每个[72, 72]矩阵按主对角线放置
        for i in range(batch_size):
            start = i * num_node
            end = start + num_node
            edge_encoding_matrix[start:end, start:end] = edge_attr[i]
        return edge_encoding_matrix


class ThreeDimDistanceEncoding(nn.Module):
    def __init__(self, edge_dim: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.edge_dim = edge_dim  # 128
        self.w1 = nn.Linear(self.edge_dim, self.edge_dim)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(self.edge_dim, 1)

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor: #前向传播方法，接受两个参数：x（节点特征矩阵）和 paths（成对节点路径）。
        """
        :param edge_attr: edge feature matrix
        :param edge_idx: edge index matrix
        :param x: node feature matrix
        :return: torch.Tensor, spatial Encoding matrix
        """
        #初始化一个全零的空间编码矩阵，大小为 (节点数量, 节点数量)，并将其转移到与模型参数所在设备相同的设备上。即可学习的。
        device = next(self.parameters()).device
        edge_attr = edge_attr.to(device) #[7992,128]
        edge_idx = edge_idx.to(device) #[2,7992]
        distance = self.w2(self.gelu(self.w1(edge_attr))).squeeze() #torch.Size([7992])，耗时0.1814994

        num_nodes = x.shape[0] #216
        distance_matrix = torch.zeros((num_nodes, num_nodes), device=device)

        # Directly index into the distance matrix using edge indices
        distance_matrix[edge_idx[0], edge_idx[1]] = distance
        # distance_matrix = distance_matrix.div(2)
        return distance_matrix

class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int, atom_update_net):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.atom_update_net = atom_update_net
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)  # 输入(128,5)
        if self.atom_update_net == 'TransformerM':
            self.ThreeDimDistance_encoding = ThreeDimDistanceEncoding(edge_dim)  # 输入(128,5)
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                x: torch.Tensor,
                edge_idx: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr=None) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        batch_mask_neg_inf = torch.full(size=(x.shape[0], x.shape[0]), fill_value=-1e6).to(next(self.parameters()).device)#[108,108]，全-1e6
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)#[108,108]，全0

        # OPTIMIZE: get rid of slices: rewrite to torch
        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)#[108,108],全1
            batch_mask_zeros += 1 #变为全1的矩阵张量
        else:
            for i in range(len(ptr) - 1): #每个batch3个图,tensor([0,36,72,108]);每个batch一个图,tensor([0,36])
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        # start = time.perf_counter()
        c = self.edge_encoding(x, edge_attr, edge_paths) #[108,108]
        # print("edge_encoding计算运行时间：", time.perf_counter() - start)
        a = self.compute_a(key, query, ptr) #[216,216]
        if self.atom_update_net == 'TransformerM':
            # start = time.perf_counter()
            d = self.ThreeDimDistance_encoding(x, edge_idx, edge_attr)
            # print("ThreeDimDistance计算运行时间：", time.perf_counter() - start)
            a = (a + b + c + d) * batch_mask_neg_inf
        elif self.atom_update_net == 'Graphormer':
            a = (a + b + c) * batch_mask_neg_inf #一个batch内不同图中的数据不交互
        else:
            raise ValueError(f'Unknown atom_update_net: {self.atom_update_net}')
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value) #每个子图内所有节点之间的attention
        return x #[108,64]

    def compute_a(self, key, query, ptr=None):
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else: #每个batch3个图,tensor([0,36,72,108]);每个batch一个图,tensor([0,36])
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device) #全0的[108,108]
            for i in range(len(ptr) - 1):
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = query[ptr[i]:ptr[i + 1]].mm(key[ptr[i]:ptr[i + 1]].transpose(0, 1)) / query.size(-1) ** 0.5
        return a

    # def compute_a(self, key, query, ptr=None):
    #     scaling_factor = query.size(-1) ** 0.5
    #     if ptr is None:
    #         return query.mm(key.transpose(0, 1)) / scaling_factor
    #     else:
    #         a = torch.block_diag(*[
    #             query[start:end].mm(key[start:end].transpose(0, 1)) / scaling_factor
    #             for start, end in zip(ptr[:-1], ptr[1:])
    #         ])
    #         return a


# FIX: PyG attention instead of regular attention, due to specificity of GNNs
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int, atom_update_net):
        """
        :param num_heads: number of attention heads,4
        :param dim_in: node feature matrix input number of dimension,128
        :param dim_q: query node feature matrix input number of dimension,128
        :param dim_k: key node feature matrix input number of dimension,128
        :param edge_dim: edge feature matrix number of dimension,128
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance, atom_update_net) for _ in range(num_heads)]
        ) #[108,64]*4
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,#(108,64)
                edge_idx: torch.Tensor,#torch.Size([2, 6048])
                edge_attr: torch.Tensor,#[6048, 128]
                b: torch.Tensor,#[108,108]
                edge_paths,#[{}:{(),(),()},...]
                ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes.
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat([
                attention_head(x, edge_idx, edge_attr, b, edge_paths, ptr) for attention_head in self.heads
            ], dim=-1) #[108,64*4]
        )#[108,64]

class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_edge_dim, n_heads, max_path_distance, atom_update_net, if_edge_update, output_layer=False):
        """
        :param node_dim: node feature matrix input number of dimension / hidden dimensions of node features
        :param edge_dim: edge feature matrix input number of dimension / hidden dimensions of edge features
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim#64
        self.edge_dim = edge_dim#128
        self.n_heads = n_heads#8
        self.if_edge_update = if_edge_update
        self.out_edge_dim = out_edge_dim

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,#64
            dim_k=node_dim,#64
            dim_q=node_dim,#64
            num_heads=n_heads,#8
            edge_dim=edge_dim,#128
            max_path_distance=max_path_distance,#5
            atom_update_net=atom_update_net
        )
        
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)

        if if_edge_update:
            if output_layer: #False
                self.e_lin = nn.Sequential(nn.Linear(edge_dim + node_dim * 2, 128),
                                           nn.SiLU(),
                                           nn.Linear(128, self.out_edge_dim),
                                           )
            else:
                self.e_lin = nn.Sequential(nn.Linear(edge_dim + node_dim * 2, 128),
                                           nn.SiLU(),
                                           nn.Linear(128, self.out_edge_dim),
                                           nn.SiLU(),
                                           )#非LCMP层时out_edge_fea_len=128,MP的最后一层（输入到LCMP层）时out_edge_fea_len=103

    def forward(self,
                x: torch.Tensor, #torch.Size([108, 64])
                edge_idx: torch.Tensor, #torch.Size([2, 6048])
                edge_attr: torch.Tensor, #torch.Size([6048, 128])
                b: torch, #节点之间的最短路径空间信息编码,(108,108)
                edge_paths, #最短路径的边列表[(),(),()]形式；ptr是批图数据索引
                ptr) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        #LayerNorm在残差内部，层归一化是在自注意力层或前馈网络层的输入前进行的。
        #这种配置有助于在模型训练初期保持梯度稳定，从而可以使用更大的学习率，加速模型的收敛速度。
        x_prime = self.attention(self.ln_1(x), edge_idx, edge_attr, b, edge_paths, ptr) + x #[108,64]，计算最耗时的部分
        x_new = self.ff(self.ln_2(x_prime)) + x_prime #[108,64]
        # #LayerNorm在残差外部，层归一化是在自注意力输出或前馈网络输出之后、加上残差连接之前进行的。参考论文：https://arxiv.org/abs/2002.04745
        # #这种设置虽然在模型训练初期可能导致梯度不稳定，从而需要更谨慎地选择学习率，但研究表明它对于大规模量子化学属性预测这类任务上能够展现出更好的泛化能力。
        # x_prime = self.ln_1(self.attention(x, edge_attr, b, edge_paths, ptr) + x)
        # x_new = self.ln_2(self.ff(x_prime) + x_prime)

        if self.if_edge_update: #True
            row, col = edge_idx #[6048], [6048]
            edge_fea = self.e_lin(torch.cat([x_new[row], x_new[col], edge_attr], dim=-1)) #[6048,256]->[6048,128/103]
            return x_new, edge_fea #[108,64], [6048,128/103/112]
        else:
            return x_new
