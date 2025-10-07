import os
from typing import Union, Tuple
from math import ceil, sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import LayerNorm, PairNorm, InstanceNorm
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax


from torch_geometric.nn.models.dimenet import BesselBasisLayer
from torch_scatter import scatter_add, scatter
import numpy as np
from scipy.special import comb

from .from_schnetpack import GaussianBasis
from .from_PyG_future import GraphNorm, DiffGroupNorm
from .from_HermNet import RBF, cosine_cutoff, ShiftedSoftplus, _eps

from .from_graphormer.functional import split_batch_data, batched_shortest_path_distance, create_ptr_from_batch
from .from_graphormer.layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding

from torch_geometric.data import Data
import time



#64,128,81,5
class LCMPLayer(nn.Module):
    def __init__(self, in_atom_fea_len, in_edge_fea_len, out_edge_fea_len, num_l,
                 normalization: str = None, bias: bool = True, if_exp: bool = False):
        super(LCMPLayer, self).__init__()
        self.in_atom_fea_len = in_atom_fea_len#64
        self.normalization = normalization#None
        self.if_exp = if_exp #False

        self.lin_f = nn.Linear(in_atom_fea_len * 2 + in_edge_fea_len, in_atom_fea_len, bias=bias)#256,64
        self.lin_s = nn.Linear(in_atom_fea_len * 2 + in_edge_fea_len, in_atom_fea_len, bias=bias)#256,64
        self.bn = nn.BatchNorm1d(in_atom_fea_len, track_running_stats=True)#64, track_running_stats为True时，该模块跟踪运行平均值和方差

        self.e_lin = nn.Sequential(nn.Linear(in_edge_fea_len + in_atom_fea_len * 2 - num_l ** 2, 128),#256-25=231,128; 256-16=240,128
                                   nn.SiLU(),
                                   nn.Linear(128, out_edge_fea_len)#128,81
                                   )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.normalization == 'BatchNorm':
            self.bn.reset_parameters()

    def forward(self, atom_fea, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                huge_structure, output_final_layer_neuron):
        if huge_structure:#训练的时候False,这块不执行；预测的时候True，这块执行
            sub_graph_batch_num = 8

            sub_graph_num = sub_atom_idx.shape[0]
            sub_graph_batch_size = ceil(sub_graph_num / sub_graph_batch_num)#向上取整

            num_edge = edge_fea.shape[0]
            vf_update = torch.zeros((num_edge * 2, self.in_atom_fea_len)).type(torch.get_default_dtype()).to(atom_fea.device)
            for sub_graph_batch_index in range(sub_graph_batch_num):
                if sub_graph_batch_index == sub_graph_batch_num - 1:
                    sub_graph_idx = slice(sub_graph_batch_size * sub_graph_batch_index, sub_graph_num)
                else:
                    sub_graph_idx = slice(sub_graph_batch_size * sub_graph_batch_index,
                                          sub_graph_batch_size * (sub_graph_batch_index + 1))

                sub_atom_idx_batch = sub_atom_idx[sub_graph_idx]
                sub_edge_idx_batch = sub_edge_idx[sub_graph_idx]
                sub_edge_ang_batch = sub_edge_ang[sub_graph_idx]
                sub_index_batch = sub_index[sub_graph_idx]

                z = torch.cat([atom_fea[sub_atom_idx_batch][:, 0, :], atom_fea[sub_atom_idx_batch][:, 1, :],
                               edge_fea[sub_edge_idx_batch], sub_edge_ang_batch], dim=-1)
                out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

                if self.if_exp:
                    sigma = 3
                    n = 2
                    out = out * torch.exp(-distance[sub_edge_idx_batch] ** n / sigma ** n / 2).view(-1, 1)

                vf_update += scatter_add(out, sub_index_batch, dim=0, dim_size=num_edge * 2)

            if self.normalization == 'BatchNorm':
                vf_update = self.bn(vf_update)
            vf_update = vf_update.reshape(num_edge, 2, -1)
            if output_final_layer_neuron != '':
                final_layer_neuron = torch.cat([vf_update[:, 0, :], vf_update[:, 1, :], edge_fea],
                                               dim=-1).detach().cpu().numpy()
                np.save(os.path.join(output_final_layer_neuron, 'final_layer_neuron.npy'), final_layer_neuron)
            out = self.e_lin(torch.cat([vf_update[:, 0, :], vf_update[:, 1, :], edge_fea], dim=-1))

            return out
        # Ensure data is on GPU
        device = next(self.parameters()).device
        atom_fea, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance = (
            atom_fea.to(device),
            edge_fea.to(device),
            sub_atom_idx.to(device),
            sub_edge_idx.to(device),
            sub_edge_ang.to(device),
            sub_index.to(device),
            distance.to(device)
        )

        num_edge = edge_fea.shape[0]#6048,这里的edge_fea.shape=[6048,112]
        z = torch.cat(
            [atom_fea[sub_atom_idx][:, 0, :], atom_fea[sub_atom_idx][:, 1, :], edge_fea[sub_edge_idx], sub_edge_ang],
            dim=-1)#子图中的所有起始节点、终端节点和边的cat,[679320,64]+[679320,64]+[679320,112]+[679320,16]=[679320,256]
        out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))#CGConv卷积函数，[679320,64]

        if self.if_exp:#False，out根据原子间距离衰减
            sigma = 3
            n = 2
            out = out * torch.exp(-distance[sub_edge_idx] ** n / sigma ** n / 2).view(-1, 1) #[679320, 64]
        #sub_index是679320，每一个原子j的截断半径内的邻居原子个数的索引游标3*[53个2017,53个2018,...,53个4032]相拼接在一个列表中，共3*4032*53个子图的列表。参考：https://blog.csdn.net/TYJ00/article/details/131024079
        #https://www.cnblogs.com/X1OO/articles/17153102.html
        out = scatter_add(out, sub_index, dim=0)#[12096, 64]。679320/56=12096,这里的56指每个原子的邻居原子数，将输入张量out的值，按照给定的邻居原子索引添加到一个输出张量的指定位置上，用于按索引散布（scatter）相加的操作，下划线(_)表示该函数是一个原地操作，会修改原始张量而不创建新的张量。
        if self.normalization == 'BatchNorm':
            out = self.bn(out)
        out = out.reshape(num_edge, 2, -1) #[6048,2,64]
        if output_final_layer_neuron != '':
            final_layer_neuron = torch.cat([out[:, 0, :], out[:, 1, :], edge_fea], dim=-1).detach().cpu().numpy()
            np.save(os.path.join(output_final_layer_neuron, 'final_layer_neuron.npy'), final_layer_neuron)
        out = self.e_lin(torch.cat([out[:, 0, :], out[:, 1, :], edge_fea], dim=-1))#输入[6048,64]+[6048,64]+[6048,112]=[6048,240]；输出[6048,81]
        return out


class MultipleLinear(nn.Module):
    def __init__(self, num_linear: int, in_fea_len: int, out_fea_len: int, bias: bool = True) -> None:
        super(MultipleLinear, self).__init__()
        self.num_linear = num_linear #81，要实现的线性层的数量。
        self.out_fea_len = out_fea_len #16、1，每个线性层的输出特征长度。
        self.weight = nn.Parameter(torch.Tensor(num_linear, in_fea_len, out_fea_len))#定义一个形状为 (num_linear, in_fea_len, out_fea_len) 的张量作为权重，这里每个线性层都有一组独立的权重。
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, out_fea_len)) #定义一个形状为 (num_linear, out_fea_len) 的张量作为偏置。
        else:
            self.register_parameter('bias', None) #如果 bias 为 False，则注册 self.bias 为 None。
        # self.ln = LayerNorm(num_linear * out_fea_len)
        # self.gn = GraphNorm(out_fea_len)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5)) #使用 Kaiming 初始化方法（也称为 He 初始化）设置权重，这种方法常用于ReLU激活函数之后的权重初始化。
        if self.bias is not None: #如果存在偏置，则将其初始化为一个小的均匀分布，范围由权重的fan_in（权重矩阵的输入单位数）决定。
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, batch_edge: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(input, self.weight) #input：输入数据，期望的形状是 (N, in_fea_len)，其中 N 是样本数。
        #计算输入数据和每组权重的矩阵乘法。由于 self.weight 的形状是 (num_linear, in_fea_len, out_fea_len)，
        # 结果 output 的形状将会是 (N, num_linear, out_fea_len)。
        if self.bias is not None:
            output += self.bias[:, None, :] #如果有偏置，则将偏置加到输出上。注意，偏置被广播以匹配输出的维度。
        return output


class DeepQTH(nn.Module):
    def __init__(self, num_species, in_atom_fea_len, in_edge_fea_len, num_orbital,
                 distance_expansion, gauss_stop, if_exp, if_MultipleLinear, if_edge_update, if_lcmp,
                 normalization, atom_update_net, separate_onsite,
                 trainable_gaussians, type_affine, n_heads=4, num_l=4, max_path_length=5):
        super(DeepQTH, self).__init__()
        self.num_species = num_species #1
        self.max_path_length = max_path_length
        self.embed = nn.Embedding(num_species + 5, in_atom_fea_len) #创建一个嵌入层，相当于一个查找表，其中每个整数索引都有一个对应的嵌入向量。具有不同类型原子时，外加 5 个可能的附加索引。

        # pair-type aware affine，可能会创建一个用于类型感知的仿射变换的嵌入层。
        if type_affine: #False
            self.type_affine = nn.Embedding(
                num_species ** 2, 2,
                _weight=torch.stack([torch.ones(num_species ** 2), torch.zeros(num_species ** 2)], dim=-1)
            )
        else:
            self.type_affine = None

        if if_edge_update or (if_edge_update is False and if_lcmp is False):
            distance_expansion_len = in_edge_fea_len #128
        else:
            distance_expansion_len = in_edge_fea_len - num_l ** 2
        if distance_expansion == 'GaussianBasis':
            self.distance_expansion = GaussianBasis(
                0.0, gauss_stop, distance_expansion_len, trainable=trainable_gaussians
            ) #0.0, 6.0, 128, False，返回3*2016*128
        elif distance_expansion == 'BesselBasis':
            self.distance_expansion = BesselBasisLayer(distance_expansion_len, gauss_stop, envelope_exponent=5)
        elif distance_expansion == 'ExpBernsteinBasis':
            self.distance_expansion = ExpBernsteinBasis(K=distance_expansion_len, gamma=0.5, cutoff=gauss_stop,
                                                        trainable=True)
        else:
            raise ValueError('Unknown distance expansion function: {}'.format(distance_expansion))

        self.if_MultipleLinear = if_MultipleLinear #False
        self.if_edge_update = if_edge_update #True
        self.if_lcmp = if_lcmp #True
        self.atom_update_net = atom_update_net #CGConv
        self.separate_onsite = separate_onsite #False

        if if_lcmp == True:
            mp_output_edge_fea_len = in_edge_fea_len - num_l ** 2 #128-25=103; 128-16=112
        else:
            assert if_MultipleLinear == False
            mp_output_edge_fea_len = in_edge_fea_len #128

        self.centrality_encoding = CentralityEncoding(
            edge_dim=in_edge_fea_len,
            max_in_degree=5,
            max_out_degree=5,
            node_dim=in_atom_fea_len,
            atom_update_net=self.atom_update_net,
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_length,
        )  # 最大路径长度，可以起到一定的截断作用

        if if_edge_update == True:
            self.layers = nn.ModuleList([
                GraphormerEncoderLayer(
                    node_dim=in_atom_fea_len,
                    edge_dim=in_edge_fea_len,
                    out_edge_dim=in_edge_fea_len,
                    n_heads=n_heads,
                    max_path_distance=max_path_length,
                    atom_update_net=self.atom_update_net,
                    if_edge_update=self.if_edge_update,
                    output_layer=False,
                ) for _ in range(4)
            ])
            if if_lcmp == True:  # True
                self.hidden_layer = GraphormerEncoderLayer(
                    node_dim=in_atom_fea_len,
                    edge_dim=in_edge_fea_len,
                    out_edge_dim=mp_output_edge_fea_len,
                    n_heads=n_heads,
                    max_path_distance=max_path_length,
                    atom_update_net=self.atom_update_net,
                    if_edge_update=self.if_edge_update,
                    output_layer=True,
                )
                if self.if_MultipleLinear == True:  # False
                    self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, 32, num_l, if_exp=if_exp)
                    self.multiple_linear1 = MultipleLinear(num_orbital, 32, 16)
                    self.multiple_linear2 = MultipleLinear(num_orbital, 16, 1)
                else:
                    self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, num_l,
                                          if_exp=if_exp)  # 64，128，81， 5, True，输出2016*81
            else:
                self.output_layer = GraphormerEncoderLayer(
                    node_dim=in_atom_fea_len,
                    edge_dim=in_edge_fea_len,
                    out_edge_dim=num_orbital,
                    n_heads=n_heads,
                    max_path_distance=max_path_length,
                    atom_update_net=self.atom_update_net,
                    if_edge_update=True,
                    output_layer=True
                )
        else:
            self.layers = nn.ModuleList([
                GraphormerEncoderLayer(
                    node_dim=in_atom_fea_len,
                    edge_dim=in_edge_fea_len,
                    out_edge_dim=None,
                    n_heads=n_heads,
                    max_path_distance=max_path_length,
                    atom_update_net=self.atom_update_net,
                    if_edge_update=self.if_edge_update,
                    output_layer=False,
                ) for _ in range(4)
            ])
            if if_lcmp == True:  # True
                self.hidden_layer = GraphormerEncoderLayer(
                    node_dim=in_atom_fea_len,
                    edge_dim=in_edge_fea_len,
                    out_edge_dim=None,
                    n_heads=n_heads,
                    max_path_distance=max_path_length,
                    atom_update_net=self.atom_update_net,
                    if_edge_update=self.if_edge_update,
                    output_layer=True,
                )
                if self.if_MultipleLinear == True:  # False
                    self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, 32, num_l, if_exp=if_exp)
                    self.multiple_linear1 = MultipleLinear(num_orbital, 32, 16)
                    self.multiple_linear2 = MultipleLinear(num_orbital, 16, 1)
                else:
                    self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, num_l,
                                          if_exp=if_exp)  # 64，128，81， 5, True，输出2016*81
            else:
                self.output_layer = GraphormerEncoderLayer(
                    node_dim=in_atom_fea_len,
                    edge_dim=in_edge_fea_len,
                    out_edge_dim=num_orbital,
                    n_heads=n_heads,
                    max_path_distance=max_path_length,
                    atom_update_net=self.atom_update_net,
                    if_edge_update=True,
                    output_layer=True,
                )


    #atom_attr是原子序数列表[36*3]；edge_idx是边索引[2,2016*3];edge_attr是节点特征[3*2016,10];batch是[3*36]；sub_atom_idx是[3*226440,2]；#sub_edge_idx是[3*226440]，sub_edge_ang为[3*226440,25],sub_index是[3*226440]
    def forward(self, atom_attr, edge_idx, edge_attr, node_paths, edge_paths, voronoi_values, centralities, batch,
                sub_atom_idx=None, sub_edge_idx=None, sub_edge_ang=None, sub_index=None,
                huge_structure=False, output_final_layer_neuron=''):
        # print("batch = ",batch) #tensor([0,..., 0, 1,..., 1, 2,..., 2])3*72
        batch_edge = batch[edge_idx[0]] #edge_idx[0]=[0,...,0,...,35,...,35]共2016*3个，[2016个0,2016个1,2016个2]
        # print("batch_edge = ",batch_edge)
        atom_fea0 = self.embed(atom_attr) #[108,64]
        distance = edge_attr[:, 0] #(6048,)
        edge_vec = edge_attr[:, 1:4] - edge_attr[:, 4:7]#6048*3
        if self.type_affine is None: #True
            edge_fea0 = self.distance_expansion(distance) #输出是一个形状为(6048, 128)的二维张量。其中，第一个维度对应于原子距离值的数量，第二个维度对应于高斯基的数量。
        else:
            affine_coeff = self.type_affine(self.num_species * atom_attr[edge_idx[0]] + atom_attr[edge_idx[1]])
            edge_fea0 = self.distance_expansion(distance * affine_coeff[:, 0] + affine_coeff[:, 1])

        ptr = create_ptr_from_batch(batch)
        # start = time.perf_counter()
        atom_fea0 = self.centrality_encoding(atom_fea0, edge_idx, edge_fea0, voronoi_values, centralities)  # 把节点在图中的重要性编码到x特征中，torch.Size([104, 64])
        # print("计算centrality_encoding运行时间：", time.perf_counter() - start)
        # start = time.perf_counter()
        b = self.spatial_encoding(atom_fea0, node_paths)  # 节点之间的最短路径空间信息编码到了attention值中,torch.Size([108, 108])
        # print("计算spatial_encoding运行时间：", time.perf_counter() - start)
        # start = time.perf_counter()

        if self.if_edge_update == True:
            for layer in self.layers:  # 4层，每层处理后的数据传入到下一层
                atom_fea0, edge_fea0 = layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
            # print("前4层graphormer运行时间：", time.perf_counter() - start)
            # start = time.perf_counter()
            if self.if_lcmp == True:  # True
                atom_fea, edge_fea = self.hidden_layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
                # print("隐藏层运行时间：", time.perf_counter() - start)
                # start = time.perf_counter()
                out = self.lcmp(atom_fea, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                                huge_structure, output_final_layer_neuron) #这个层很耗时，占35%，是否有必要？
                # print("输出层运行时间：", time.perf_counter() - start)
                # print("===============================")
                # 输入[108,64],[6048,112],[679320,2],[679320],[679320,25],[679320],[6048],False,''；输出[6048,81]
            else:
                # start = time.perf_counter()
                atom_fea, edge_fea = self.output_layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
                # print("输出层运行时间：", time.perf_counter() - start)
                # print("===============================")
                out = edge_fea
        else:
            for layer in self.layers:  # 4层，每层处理后的数据传入到下一层
                atom_fea0 = layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
            if self.if_lcmp == True:  # True
                atom_fea = self.hidden_layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
                out = self.lcmp(atom_fea, edge_fea0, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                                huge_structure, output_final_layer_neuron)
                # 输入[108,64],[6048,112],[679320,2],[679320],[679320,25],[679320],[6048],False,''；输出[6048,81]
            else:
                atom_fea, edge_fea = self.output_layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
                out = edge_fea
        if self.if_MultipleLinear == True: #False
            out = self.multiple_linear1(F.silu(out), batch_edge) #输入[6048,32]，输出的形状将会是 (81, 6048, 16)。
            out = self.multiple_linear2(F.silu(out), batch_edge) # (81, 6048, 1)
            out = out.T #1*6048*81
        
        return out #输出[6048,81]
