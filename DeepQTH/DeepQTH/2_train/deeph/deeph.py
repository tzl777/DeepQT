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


        if edge_attr is None:
            edge_attr = distance.view(-1, 1)
        else:
           
            if not torch.allclose(edge_attr[:, 0], distance): 
                edge_attr = torch.cat([edge_attr, distance.view(-1, 1)], dim=-1)
        
       
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
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
  
    def message(self, x_i, x_j, edge_attr: OptTensor) -> torch.Tensor:
        distance = edge_attr[:, -1]
        edge_feat = edge_attr[:, :-1]
        z = torch.cat([x_i, x_j, edge_feat], dim=-1) #[6048,256]
        out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z)) 
        if self.if_exp: 
            sigma = 3
            n = 2 
            out = out * torch.exp(-distance ** n / sigma ** n / 2).view(-1, 1)
        return out 

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
                                   )
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
        self.bn = nn.BatchNorm1d(in_atom_fea_len, track_running_stats=True)

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
        if huge_structure:
            sub_graph_batch_num = 8
            sub_graph_num = sub_atom_idx.shape[0]
            sub_graph_batch_size = ceil(sub_graph_num / sub_graph_batch_num)

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

        num_edge = edge_fea.shape[0]
        z = torch.cat(
            [atom_fea[sub_atom_idx][:, 0, :], atom_fea[sub_atom_idx][:, 1, :], edge_fea[sub_edge_idx], sub_edge_ang],
            dim=-1)
        # print(z.shape) #torch.Size([591408, 128])
        out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

        if self.if_exp:
            sigma = 3
            n = 2
            out = out * torch.exp(-distance[sub_edge_idx] ** n / sigma ** n / 2).view(-1, 1) #[679320, 64]
       
        out = scatter_add(out, sub_index, dim=0)
        if self.normalization == 'BatchNorm':
            out = self.bn(out)
        out = out.reshape(num_edge, 2, -1) #[6048,2,64]
        if output_final_layer_neuron != '':
            final_layer_neuron = torch.cat([out[:, 0, :], out[:, 1, :], edge_fea], dim=-1).detach().cpu().numpy()
            np.save(os.path.join(output_final_layer_neuron, 'final_layer_neuron.npy'), final_layer_neuron)
        out = self.e_lin(torch.cat([out[:, 0, :], out[:, 1, :], edge_fea], dim=-1))
        return out


class MultipleLinear(nn.Module):
    def __init__(self, num_linear: int, in_fea_len: int, out_fea_len: int, bias: bool = True) -> None:
        super(MultipleLinear, self).__init__()
        self.num_linear = num_linear
        self.out_fea_len = out_fea_len
        self.weight = nn.Parameter(torch.Tensor(num_linear, in_fea_len, out_fea_len))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, out_fea_len))
        else:
            self.register_parameter('bias', None) 
        # self.ln = LayerNorm(num_linear * out_fea_len)
        # self.gn = GraphNorm(out_fea_len)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5)) 
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, batch_edge: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(input, self.weight) 
        if self.bias is not None:
            output += self.bias[:, None, :] 
        return output


class DeepH(nn.Module):
    def __init__(self, in_atom_fea_len, in_edge_fea_len, num_orbital,
                 distance_expansion, gauss_stop, if_exp, if_MultipleLinear, normalization, trainable_gaussians, type_affine, num_l=4):
        super(DeepH, self).__init__()

        self.embed = nn.Embedding(50, in_atom_fea_len)

        distance_expansion_len = in_edge_fea_len #64

        if distance_expansion == 'GaussianBasis':
            self.distance_expansion = GaussianBasis(
                0.0, gauss_stop, distance_expansion_len, trainable=trainable_gaussians
            )
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
        self.mp3 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, normalization, gauss_stop)
        self.mp4 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, normalization, gauss_stop)
        self.mp5 = MPLayer(in_atom_fea_len, in_edge_fea_len, mp_output_edge_fea_len, if_exp, normalization, gauss_stop)
        
        if self.if_MultipleLinear == True: #False
            self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, 32, num_l, if_exp=if_exp)
            self.multiple_linear1 = MultipleLinear(num_orbital, 32, 16)
            self.multiple_linear2 = MultipleLinear(num_orbital, 16, 1)
        else:
            self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, num_l, if_exp=if_exp)

    
    def forward(self, atom_attr, edge_idx, edge_attr, batch,
                sub_atom_idx=None, sub_edge_idx=None, sub_edge_ang=None, sub_index=None,
                huge_structure=False, output_final_layer_neuron=''):

        batch_edge = batch[edge_idx[0]]
        # print("batch_edge = ",batch_edge)
        atom_fea0 = self.embed(atom_attr) #[108,64]
        distance = edge_attr[:, 0] #(6048,)

        edge_fea0 = self.distance_expansion(distance)

        atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance)
        atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
        atom_fea, edge_fea = self.mp2(atom_fea0, edge_idx, edge_fea0, batch, distance)
        atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
        atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance)
        atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
        atom_fea, edge_fea = self.mp4(atom_fea0, edge_idx, edge_fea0, batch, distance)
        atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
        atom_fea, edge_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance)

        if self.if_MultipleLinear == True: #False
            out = self.lcmp(atom_fea, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                            huge_structure, output_final_layer_neuron)
            out = self.multiple_linear1(F.silu(out), batch_edge) 
            out = self.multiple_linear2(F.silu(out), batch_edge)
            out = out.T
        else:
            out = self.lcmp(atom_fea, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                            huge_structure, output_final_layer_neuron)

        return out
