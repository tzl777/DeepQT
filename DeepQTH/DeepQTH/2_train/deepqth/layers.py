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
#         centrality = torch.tensor(list(nx.current_flow_betweenness_centrality(G, weight=edge_attr).values())).view(-1, 1)

#         x += self.z(centrality)

#         return x


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
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim))) 
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
        num_nodes = x.shape[0]  

        in_degree = decrease_to_max_value(degree(index=edge_index[1], num_nodes=num_nodes).long(),
                                          self.max_in_degree - 1)
        
        
        out_degree = decrease_to_max_value(degree(index=edge_index[0], num_nodes=num_nodes).long(),
                                           self.max_out_degree - 1)
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
            # all_clustering = torch.tensor(list(nx.clustering(G, weight=edge_attr).values())).view(-1, 1)
            x += self.z_in[in_degree] + self.z_out[out_degree]
        else:
            raise ValueError(f'Unknown atom_update_net: {self.atom_update_net}')
        
        return x


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_path_distance = max_path_distance 

        self.b = nn.Parameter(torch.randn(self.max_path_distance + 1)) 

    def forward(self, x: torch.Tensor, paths) -> torch.Tensor:

        device = next(self.parameters()).device
        x = x.to(device)
        batch = paths.shape[0]
        node_mask = (paths == -1).to(device)
        path_lengths = (~node_mask).sum(dim=-1)
        num_nodes = x.shape[0]
        spatial_matrix = torch.zeros((num_nodes, num_nodes), device=device)

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
        batch_indices = torch.arange(batch_size).view(batch_size, 1, 1, 1).expand_as(edge_paths) 
        assert edge_attr is not None
        edge_path_embeddings = edge_attr[batch_indices, edge_paths_clamped, :].to(device) #torch.Size([3, 72, 72, 4, 128])
        edge_path_embeddings[edge_mask] = 0.0 #torch.Size([3, 72, 72, 4, 128])

        path_lengths = (~edge_mask).sum(dim=-1) + self.eps #torch.Size([3, 72, 72])

        
        edge_path_encoding = torch.einsum("bnmld,ld->bnm", edge_path_embeddings, self.edge_vector) 
        edge_attr = edge_path_encoding.div(path_lengths) #torch.Size([3, 72, 72])

        edge_encoding_matrix = torch.zeros((num_nodes, num_nodes), device=device)
        num_node = edge_paths.shape[1]
       
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

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor: 
        """
        :param edge_attr: edge feature matrix
        :param edge_idx: edge index matrix
        :param x: node feature matrix
        :return: torch.Tensor, spatial Encoding matrix
        """
       
        device = next(self.parameters()).device
        edge_attr = edge_attr.to(device) #[7992,128]
        edge_idx = edge_idx.to(device) #[2,7992]
        distance = self.w2(self.gelu(self.w1(edge_attr))).squeeze()

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
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance) 
        if self.atom_update_net == 'TransformerM':
            self.ThreeDimDistance_encoding = ThreeDimDistanceEncoding(edge_dim) 
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
        batch_mask_neg_inf = torch.full(size=(x.shape[0], x.shape[0]), fill_value=-1e6).to(next(self.parameters()).device)
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        # OPTIMIZE: get rid of slices: rewrite to torch
        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1 
        else:
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        # start = time.perf_counter()
        c = self.edge_encoding(x, edge_attr, edge_paths) #[108,108]
        # print("edge_encoding cal time：", time.perf_counter() - start)
        a = self.compute_a(key, query, ptr) #[216,216]
        if self.atom_update_net == 'TransformerM':
            # start = time.perf_counter()
            d = self.ThreeDimDistance_encoding(x, edge_idx, edge_attr)
            # print("ThreeDimDistance cal time：", time.perf_counter() - start)
            a = (a + b + c + d) * batch_mask_neg_inf
        elif self.atom_update_net == 'Graphormer':
            a = (a + b + c) * batch_mask_neg_inf
        else:
            raise ValueError(f'Unknown atom_update_net: {self.atom_update_net}')
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x #[108,64]

    def compute_a(self, key, query, ptr=None):
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else: 
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device)
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
        ) 
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
                                           )

    def forward(self,
                x: torch.Tensor, #torch.Size([108, 64])
                edge_idx: torch.Tensor, #torch.Size([2, 6048])
                edge_attr: torch.Tensor, #torch.Size([6048, 128])
                b: torch, #(108,108)
                edge_paths, 
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
        
        x_prime = self.attention(self.ln_1(x), edge_idx, edge_attr, b, edge_paths, ptr) + x 
        x_new = self.ff(self.ln_2(x_prime)) + x_prime #[108,64]
        
        # x_prime = self.ln_1(self.attention(x, edge_attr, b, edge_paths, ptr) + x)
        # x_new = self.ln_2(self.ff(x_prime) + x_prime)

        if self.if_edge_update: #True
            row, col = edge_idx #[6048], [6048]
            edge_fea = self.e_lin(torch.cat([x_new[row], x_new[col], edge_attr], dim=-1)) #[6048,256]->[6048,128/103]
            return x_new, edge_fea #[108,64], [6048,128/103/112]
        else:
            return x_new
