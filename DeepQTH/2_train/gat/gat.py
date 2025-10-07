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



class GAT_Crystal(MessagePassing):
    def __init__(self, in_features, out_features, edge_dim, heads, concat=False, normalization: str = None,
                 dropout=0, bias=True, **kwargs):
        super(GAT_Crystal, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.in_features = in_features #64
        self.out_features = out_features #64
        self.heads = heads #3
        self.concat = concat #False,是否在多头注意力中连接输出。
        self.dropout = dropout #0
        self.neg_slope = 0.2
        self.prelu = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(heads)
        self.W = nn.Parameter(torch.Tensor(in_features + edge_dim, heads * out_features)) #64+128,3*64,用于节点特征转换的权重参数。
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_features)) #1,3,128,用于计算注意力系数的参数。

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_features))#3*64
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_features)) #64
        else:
            self.register_parameter('bias', None)

        self.normalization = normalization
        if self.normalization == 'BatchNorm':
            self.bn = nn.BatchNorm1d(out_features, track_running_stats=True)
        elif self.normalization == 'LayerNorm': #True
            self.ln = LayerNorm(out_features)
        elif self.normalization == 'PairNorm':
            self.pn = PairNorm(out_features)
        elif self.normalization == 'InstanceNorm':
            self.instance_norm = InstanceNorm(out_features)
        elif self.normalization == 'GraphNorm':
            self.gn = GraphNorm(out_features)
        elif self.normalization == 'DiffGroupNorm':
            self.group_norm = DiffGroupNorm(out_features, 128)
        elif self.normalization is None:
            pass
        else:
            raise ValueError('Unknown normalization function: {}'.format(normalization))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W) #也称 Xavier初始化,来初始化 W 和 att。
        glorot(self.att)
        zeros(self.bias) #初始化偏置 bias。

    def forward(self, x, edge_index, edge_attr, batch, distance):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr) #propagate 方法调用 message 函数来计算消息传递。返回108*64
        if self.normalization == 'BatchNorm':
            out = self.bn(out)
        elif self.normalization == 'LayerNorm': #应用相应的标准化层。
            out = self.ln(out, batch)
        elif self.normalization == 'PairNorm':
            out = self.pn(out, batch)
        elif self.normalization == 'InstanceNorm':
            out = self.instance_norm(out, batch)
        elif self.normalization == 'GraphNorm':
            out = self.gn(out, batch)
        elif self.normalization == 'DiffGroupNorm':
            out = self.group_norm(out)
        return out

    # 根据边索引和节点特征来计算每条边的注意力系数和消息值。
    def message(self, edge_index_i, x_i, x_j, size_i, index, ptr: OptTensor, edge_attr):
        x_i = torch.cat([x_i, edge_attr], dim=-1)#6048*128+64，将节点特征与边特征拼接。
        x_j = torch.cat([x_j, edge_attr], dim=-1)

        x_i = F.softplus(torch.matmul(x_i, self.W))#6048*192，将拼接的特征通过转换矩阵 W。
        x_j = F.softplus(torch.matmul(x_j, self.W))

        x_i = x_i.view(-1, self.heads, self.out_features)#6048*3*64，分为3个头，每个头的特征为64
        x_j = x_j.view(-1, self.heads, self.out_features)

        alpha = F.softplus((torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1))#使用多头注意力的权重 att 来计算注意力系数。通过在最后一个维度上求和，得到每一对节点间的原始注意力分数。再次通过 softplus 函数处理，目的是确保注意力分数非负。
        # print(alpha.shape) #6048*3
        alpha = F.softplus(self.bn1(alpha)) #注意力分数在被 softmax 归一化前，应用了批量归一化（通过 self.bn1）。批量归一化有助于减少内部协变量偏移，提高训练稳定性。6048*3
        alpha = softmax(alpha, index, ptr, size_i) #应用 softmax 函数标准化注意力系数。index通常代表目标节点的索引，即每个注意力分数对应的节点索引。6048*3
        #ptr: 如果使用的是稀疏实现方式，ptr 可能是用于优化的额外参数，指向每个节点邻居开始的位置，适用于高效的消息传递。
        #size_i: 这个参数用于确定 softmax 归一化的范围，即确定了多少个邻居节点的特征应被考虑在内。
        alpha = F.dropout(alpha, p=self.dropout, training=True) #应用 dropout 于注意力系数。6048*3
        #在训练过程中，输入张量的一些元素以概率为p被随机归零。
        return x_j * alpha.view(-1, self.heads, 1) #通过 alpha.view(-1, self.heads, 1) 调整了注意力权重 alpha 的形状以匹配特征矩阵 x_j 的维度。将调整后的注意力权重应用于每个邻居节点的特征 x_j。6048*3*64

    def update(self, aggr_out, x): #用来更新聚合后的节点特征。
        if self.concat is True: #根据 concat 参数决定是连接多头输出还是取均值。
            aggr_out = aggr_out.view(-1, self.heads * self.out_features)#108,3*64
        else:
            aggr_out = aggr_out.mean(dim=1)#108*64
        if self.bias is not None:  aggr_out = aggr_out + self.bias #添加偏置（如果有）。
        return aggr_out


#64，128，128/103，True, True, LayerNorm, CGConv, 6.0, False
class MPLayer(nn.Module):
    def __init__(self, in_atom_fea_len, in_edge_fea_len, out_edge_fea_len, if_exp, if_edge_update, normalization,
                 atom_update_net, gauss_stop, output_layer=False):
        super(MPLayer, self).__init__()

        self.cgconv = GAT_Crystal(
            in_features=in_atom_fea_len,
            out_features=in_atom_fea_len,
            edge_dim=in_edge_fea_len,
            heads=4,
            normalization=normalization
        )#输出108*64

        self.if_edge_update = if_edge_update  # True
        self.atom_update_net = atom_update_net  # CGConv，在这里替换其它网络模型

        if if_edge_update:
            if output_layer: #False
                self.e_lin = nn.Sequential(nn.Linear(in_edge_fea_len + in_atom_fea_len * 2, 128),
                                           nn.SiLU(),
                                           nn.Linear(128, out_edge_fea_len),
                                           )
            else:
                self.e_lin = nn.Sequential(nn.Linear(in_edge_fea_len + in_atom_fea_len * 2, 128),
                                           nn.SiLU(),
                                           nn.Linear(128, out_edge_fea_len),
                                           nn.SiLU(),
                                           )#非LCMP层时out_edge_fea_len=128,MP的最后一层（输入到LCMP层）时out_edge_fea_len=103
    #
    def forward(self, atom_fea, edge_idx, edge_fea, batch, distance, edge_vec):

        atom_fea = self.cgconv(atom_fea, edge_idx, edge_fea, batch, distance) #[108,64]
        atom_fea_s = atom_fea #[108,64]
        if self.if_edge_update: #True
            row, col = edge_idx #[6048], [6048]
            edge_fea = self.e_lin(torch.cat([atom_fea_s[row], atom_fea_s[col], edge_fea], dim=-1)) #[6048,256]->[6048,128/103]
            return atom_fea, edge_fea #[108,64], [6048,128/103]
        else:
            return atom_fea

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


class GAT(nn.Module):
    def __init__(self, num_species, in_atom_fea_len, in_edge_fea_len, num_orbital,
                 distance_expansion, gauss_stop, if_exp, if_MultipleLinear, if_edge_update, if_lcmp,
                 normalization, atom_update_net, separate_onsite,
                 trainable_gaussians, type_affine, n_heads=4, num_l=4, max_path_length=5):
        super(GAT, self).__init__()
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

        
        #在MPLayer中更换网络模型
        if if_edge_update == True:
            self.mp1 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop) #64，128，128，True, True, LayerNorm, CGConv, 6.0
            self.mp2 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp3 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp4 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp5 = MPLayer(in_atom_fea_len, in_edge_fea_len, mp_output_edge_fea_len, if_exp, if_edge_update,
                               normalization, atom_update_net, gauss_stop)#64，128，103，True, True, LayerNorm, CGConv, 6.0
        else:
            self.mp1 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp2 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp3 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp4 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp5 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)

        if if_lcmp == True: #True
            if self.if_MultipleLinear == True: #False
                self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, 32, num_l, if_exp=if_exp)
                self.multiple_linear1 = MultipleLinear(num_orbital, 32, 16)
                self.multiple_linear2 = MultipleLinear(num_orbital, 16, 1)
            else:
                self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, num_l, if_exp=if_exp)#64，128，81， 5, True，输出2016*81
        else:
            self.mp_output = MPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, if_exp, if_edge_update=True,
                                     normalization=normalization, atom_update_net=atom_update_net,
                                     gauss_stop=gauss_stop, output_layer=True)


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
        
        if self.if_edge_update == True: #True
            #输入[108,64],[2,6048],[6048,128],[108],[6048],[6048,3];
            atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec) #输出[108,64], [6048,128]
            atom_fea, edge_fea = self.mp2(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)#输出[108,64], [6048,128]
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea#残差
            atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)#输出[108,64], [6048,128]
            atom_fea, edge_fea = self.mp4(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)#输出[108,64], [6048,128]
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea#残差
            atom_fea, edge_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)#输出[108,64], [6048,103]

            if self.if_lcmp == True:#True

                atom_fea_s = atom_fea#[108,64]
                out = self.lcmp(atom_fea_s, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                                huge_structure, output_final_layer_neuron)#输入[108,64],[6048,103],[679320,2],[679320],[679320,25],[679320],[6048],False,''；输出[6048,81]
            else:
                atom_fea, edge_fea = self.mp_output(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
                out = edge_fea
        else:
            atom_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea = self.mp2(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea0 = atom_fea0 + atom_fea
            atom_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea = self.mp4(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea0 = atom_fea0 + atom_fea
            atom_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)


            atom_fea_s = atom_fea
            if self.if_lcmp == True:
                out = self.lcmp(atom_fea_s, edge_fea0, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                                huge_structure, output_final_layer_neuron)
            else:
                atom_fea, edge_fea = self.mp_output(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
                out = edge_fea

        if self.if_MultipleLinear == True: #False
            out = self.multiple_linear1(F.silu(out), batch_edge) #输入[6048,32]，输出的形状将会是 (81, 6048, 16)。
            out = self.multiple_linear2(F.silu(out), batch_edge) # (81, 6048, 1)
            out = out.T #1*6048*81
        return out #输出[6048,81]
