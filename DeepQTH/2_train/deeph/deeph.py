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

from utils import GaussianBasis
from normalization import GraphNorm, DiffGroupNorm
import time




"""
The class CGConv below is extended from "https://github.com/rusty1s/pytorch_geometric", which has the MIT License below

---------------------------------------------------------------------------
Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

class CGConv(MessagePassing): #64，128，add, LayerNorm, True
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                 aggr: str = 'add', normalization: str = None,
                 bias: bool = True, if_exp: bool = False, **kwargs):
        super(CGConv, self).__init__(aggr=aggr, flow="source_to_target", **kwargs)
        self.channels = channels #64
        self.dim = dim #128
        self.normalization = normalization #LayerNorm
        self.if_exp = if_exp #True

        if isinstance(channels, int):
            channels = (channels, channels) #(64,64)

        self.lin_f = nn.Linear(sum(channels) + dim, channels[1], bias=bias) #(256,64)
        self.lin_s = nn.Linear(sum(channels) + dim, channels[1], bias=bias) #(256,64)
        if self.normalization == 'BatchNorm':
            self.bn = nn.BatchNorm1d(channels[1], track_running_stats=True)
        elif self.normalization == 'LayerNorm':
            self.ln = LayerNorm(channels[1]) #LayerNorm(64)
        elif self.normalization == 'PairNorm':
            self.pn = PairNorm(channels[1])
        elif self.normalization == 'InstanceNorm':
            self.instance_norm = InstanceNorm(channels[1])
        elif self.normalization == 'GraphNorm':
            self.gn = GraphNorm(channels[1])
        elif self.normalization == 'DiffGroupNorm':
            self.group_norm = DiffGroupNorm(channels[1], 128)
        elif self.normalization is None:
            pass
        else:
            raise ValueError('Unknown normalization function: {}'.format(normalization))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.normalization == 'BatchNorm':
            self.bn.reset_parameters()

    def forward(self, x: Union[torch.Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor, batch, distance, size: Size = None) -> torch.Tensor:
        """"""
        if isinstance(x, torch.Tensor):
            x: PairTensor = (x, x)

        # 将 distance 并入 edge_attr（如果 edge_attr 已含 distance 列，这步可以省）
        # 假设 distance 形状为 [num_edges], edge_attr 形状为 [num_edges, E]
        if edge_attr is None:
            edge_attr = distance.view(-1, 1)
        else:
            # 若 distance 已是 edge_attr 的第一列，则可不拼接
            if not torch.allclose(edge_attr[:, 0], distance):  # 仅作示例判断，可按需删除
                edge_attr = torch.cat([edge_attr, distance.view(-1, 1)], dim=-1)
        
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        #self.propagate 方法是 torch_geometric 库中 MessagePassing 类的核心方法之一，用于在图形数据上实现信息的传递过程。
        #负责将定义的消息（通过 message 方法计算得到）从一个节点传递到另一个节点，并可能对这些消息进行聚合（根据初始化时指定的 aggr 参数，比如 'add'、'mean' 或 'max'）。propagate 方法的实现细节被封装在 PyTorch Geometric 的 MessagePassing 基类中，使得用户可以通过定义 message、aggregate（如果需要）和 update（如果需要）方法来定制自己的图卷积层。size=None用于指定源节点和目标节点的维度大小，主要在使用稀疏张量时有用。edge_index形状需要定义为[2,num_messages]，其中来自edge_index[0]中的节点的消息被发送到edge_index[1]中的节点。
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size) #输出大小108*64
        if self.normalization == 'BatchNorm':
            out = self.bn(out)
        elif self.normalization == 'LayerNorm':
            out = self.ln(out, batch)
        elif self.normalization == 'PairNorm':
            out = self.pn(out, batch)
        elif self.normalization == 'InstanceNorm':
            out = self.instance_norm(out, batch)
        elif self.normalization == 'GraphNorm':
            out = self.gn(out, batch)
        elif self.normalization == 'DiffGroupNorm':
            out = self.group_norm(out)
        out += x[1]
        return out #[108,64]
    #message卷积函数,此函数可以接受最初传递给propagate()的任何参数作为输入。此外，传递给propagate()的张量可以映射到相应的节点i和j,通过将_i或_j附加到变量名后，生成例如x_i和x_j。
    def message(self, x_i, x_j, edge_attr: OptTensor) -> torch.Tensor:
        distance = edge_attr[:, -1]
        edge_feat = edge_attr[:, :-1]
        z = torch.cat([x_i, x_j, edge_feat], dim=-1) #[6048,256]
        out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z)) #逐元素相乘，[6048,64]
        
        if self.if_exp: #d是节点间的距离，σ是一个正的缩放参数，n 是控制衰减速率的指数，并且这个权重被应用于通过线性层和激活函数处理后的消息上。
            sigma = 3
            n = 2 #n越大，权重随距离增加而衰减得越快，这意味着只有距离非常近的节点之间的交互才是重要的。
            out = out * torch.exp(-distance ** n / sigma ** n / 2).view(-1, 1)
        return out #[6048,64]

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels, self.dim)


#64，128，128/103，True, True, LayerNorm, CGConv, 6.0, False
class MPLayer(nn.Module):
    def __init__(self, in_atom_fea_len, in_edge_fea_len, out_edge_fea_len, if_exp, normalization, gauss_stop):
        super(MPLayer, self).__init__()

        self.cgconv = CGConv(channels=in_atom_fea_len,
                             dim=in_edge_fea_len,
                             aggr='add',
                             normalization=normalization,
                             if_exp=if_exp) #64，128，add, LayerNorm, True


        self.e_lin = nn.Sequential(nn.Linear(in_edge_fea_len + in_atom_fea_len * 2, 128),
                                   nn.SiLU(),
                                   nn.Linear(128, out_edge_fea_len),
                                   )#非LCMP层时out_edge_fea_len=128,MP的最后一层（输入到LCMP层）时out_edge_fea_len=103
    #
    def forward(self, atom_fea, edge_idx, edge_fea, batch, distance):
        
        atom_fea = self.cgconv(atom_fea, edge_idx, edge_fea, batch, distance) #[108,64]
        atom_fea_s = atom_fea #[108,64]

        row, col = edge_idx #[6048], [6048]
        edge_fea = self.e_lin(torch.cat([atom_fea_s[row], atom_fea_s[col], edge_fea], dim=-1)) #[6048,256]->[6048,128/103]
        return atom_fea, edge_fea #[108,64], [6048,128/103]


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
        # print(z.shape) #torch.Size([591408, 128])
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


class DeepH(nn.Module):
    def __init__(self, in_atom_fea_len, in_edge_fea_len, num_orbital,
                 distance_expansion, gauss_stop, if_exp, if_MultipleLinear, normalization, trainable_gaussians, type_affine, num_l=4):
        super(DeepH, self).__init__()

        self.embed = nn.Embedding(50, in_atom_fea_len) #创建一个嵌入层，相当于一个查找表，其中每个整数索引都有一个对应的嵌入向量。具有不同类型原子时，外加 5 个可能的附加索引。

        distance_expansion_len = in_edge_fea_len #64

        if distance_expansion == 'GaussianBasis':
            self.distance_expansion = GaussianBasis(
                0.0, gauss_stop, distance_expansion_len, trainable=trainable_gaussians
            ) #0.0, 6.0, 128, False，返回3*2016*64
        elif distance_expansion == 'BesselBasis':
            self.distance_expansion = BesselBasisLayer(distance_expansion_len, gauss_stop, envelope_exponent=5)
        elif distance_expansion == 'ExpBernsteinBasis':
            self.distance_expansion = ExpBernsteinBasis(K=distance_expansion_len, gamma=0.5, cutoff=gauss_stop,
                                                        trainable=True)
        else:
            raise ValueError('Unknown distance expansion function: {}'.format(distance_expansion))

        self.if_MultipleLinear = if_MultipleLinear #False
              
        mp_output_edge_fea_len = in_edge_fea_len - num_l ** 2 #128-25=103; 128-16=112; 64-16=48
        
        #64，128，128，True, True, LayerNorm, 6.0
        self.mp1 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, normalization, gauss_stop) 
        self.mp2 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, normalization, gauss_stop)
        self.mp3 = MPLayer(in_atom_fea_len, in_edge_fea_len, mp_output_edge_fea_len, if_exp, normalization, gauss_stop)
        
        if self.if_MultipleLinear == True: #False
            self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, 32, num_l, if_exp=if_exp)
            self.multiple_linear1 = MultipleLinear(num_orbital, 32, 16)
            self.multiple_linear2 = MultipleLinear(num_orbital, 16, 1)
        else:
            self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, num_l, if_exp=if_exp)#64，128，81， 5, True，输出2016*81


    #atom_attr是原子序数列表[36*3]；edge_idx是边索引[2,2016*3];edge_attr是节点特征[3*2016,10];batch是[3*36]；sub_atom_idx是[3*226440,2]；#sub_edge_idx是[3*226440]，sub_edge_ang为[3*226440,25],sub_index是[3*226440]
    def forward(self, atom_attr, edge_idx, edge_attr, batch,
                sub_atom_idx=None, sub_edge_idx=None, sub_edge_ang=None, sub_index=None,
                huge_structure=False, output_final_layer_neuron=''):
        # output_final_layer_neuron 是输出倒数第二层的神经元，用于PCA可视化
        # print("batch = ",batch) #tensor([0,..., 0, 1,..., 1, 2,..., 2])3*72
        batch_edge = batch[edge_idx[0]] #edge_idx[0]=[0,...,0,...,35,...,35]共2016*3个，[2016个0,2016个1,2016个2]
        # print("batch_edge = ",batch_edge)
        atom_fea0 = self.embed(atom_attr) #[108,64]
        distance = edge_attr[:, 0] #(6048,)

        edge_fea0 = self.distance_expansion(distance) #输出是一个形状为(6048, 128)的二维张量。其中，第一个维度对应于原子距离值的数量，第二个维度对应于高斯基的数量。

        #输入[108,64],[2,6048],[6048,128],[108],[6048],[6048,3];
        atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance) #输出[108,64], [6048,128]
        atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea#残差
        atom_fea, edge_fea = self.mp2(atom_fea0, edge_idx, edge_fea0, batch, distance)#输出[108,64], [6048,128]
        atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea#残差
        atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance)#输出[108,64], [6048,103]
        # print(atom_fea.shape) #torch.Size([216, 32])
        # print(edge_fea.shape) #torch.Size([7992, 48])
        if self.if_MultipleLinear == True: #False
            out = self.lcmp(atom_fea, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                            huge_structure, output_final_layer_neuron)
            out = self.multiple_linear1(F.silu(out), batch_edge) #输入[6048,32]，输出的形状将会是 (81, 6048, 16)。
            out = self.multiple_linear2(F.silu(out), batch_edge) # (81, 6048, 1)
            out = out.T #1*6048*81
        else:
            #输入[108,64],[6048,103],[679320,2],[679320],[679320,25],[679320],[6048],False,''；输出[6048,81]
            out = self.lcmp(atom_fea, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                            huge_structure, output_final_layer_neuron)

        return out #输出[6048,81]
