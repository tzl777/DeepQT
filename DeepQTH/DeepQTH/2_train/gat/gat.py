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
import numpy as np   导入numpy为np
from scipy.special import comb

import GaussianBasis
import GraphNorm, DiffGroupNorm
import RBF, cosine_cutoff, ShiftedSoftplus, _eps

from torch_geometric.data import Data
import time



class GAT_Crystal(MessagePassing):
    def __init__(self, in_features, out_features, edge_dim, heads, concat=False, normalization: str = None,
                 dropout=0, bias=True, **kwargs):
        super(GAT_Crystal, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.in_features = in_features #64
        self.out_features = out_features #64
        self.heads = heads #3
        self.concat = concat 
        self.dropout = dropout #0
        self.neg_slope = 0.2
        self.prelu = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(heads)
        self.W = nn.Parameter(torch.Tensor(in_features + edge_dim, heads * out_features))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_features)) 
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
        glorot(self.W) 
        glorot(self.att)
        zeros(self.bias) 

    def forward(self, x, edge_index, edge_attr, batch, distance):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr) 
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
        return out


    def message(self, edge_index_i, x_i, x_j, size_i, index, ptr: OptTensor, edge_attr):
        x_i = torch.cat([x_i, edge_attr], dim=-1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)

        x_i = F.softplus(torch.matmul(x_i, self.W))
        x_j = F.softplus(torch.matmul(x_j, self.W))

        x_i = x_i.view(-1, self.heads, self.out_features)
        x_j = x_j.view(-1, self.heads, self.out_features)

        alpha = F.softplus((torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1))

        alpha = F.softplus(self.bn1(alpha)) 
        alpha = softmax(alpha, index, ptr, size_i) 
        alpha = F.dropout(alpha, p=self.dropout, training=True) 
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out, x): 
        if self.concat is True: 
            aggr_out = aggr_out.view(-1, self.heads * self.out_features)
        else:
            aggr_out = aggr_out.mean(dim=1)
        if self.bias is not None:  aggr_out = aggr_out + self.bias 
        return aggr_out


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
        )

        self.if_edge_update = if_edge_update  # True
        self.atom_update_net = atom_update_net  

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
                                           )
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
        self.bn = nn.BatchNorm1d(in_atom_fea_len, track_running_stats=True)

        self.e_lin = nn.Sequential(nn.Linear(in_edge_fea_len + in_atom_fea_len * 2 - num_l ** 2, 128),
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


class GAT(nn.Module):
    def __init__(self, num_species, in_atom_fea_len, in_edge_fea_len, num_orbital,
                 distance_expansion, gauss_stop, if_exp, if_MultipleLinear, if_edge_update, if_lcmp,
                 normalization, atom_update_net, separate_onsite,
                 trainable_gaussians, type_affine, n_heads=4, num_l=4, max_path_length=5):
        super(GAT, self).__init__()
        self.num_species = num_species #1
        self.max_path_length = max_path_length
        self.embed = nn.Embedding(num_species + 5, in_atom_fea_len) 

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
                self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, num_l, if_exp=if_exp)#64，128，81， 5, True，
        else:
            self.mp_output = MPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, if_exp, if_edge_update=True,
                                     normalization=normalization, atom_update_net=atom_update_net,
                                     gauss_stop=gauss_stop, output_layer=True)


   
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
            edge_fea0 = self.distance_expansion(distance)
        else:
            affine_coeff = self.type_affine(self.num_species * atom_attr[edge_idx[0]] + atom_attr[edge_idx[1]])
            edge_fea0 = self.distance_expansion(distance * affine_coeff[:, 0] + affine_coeff[:, 1])
        
        if self.if_edge_update == True: #True
            atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec) 
            atom_fea, edge_fea = self.mp2(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea, edge_fea = self.mp4(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            atom_fea, edge_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)

            if self.if_lcmp == True:

                atom_fea_s = atom_fea#[108,64]
                out = self.lcmp(atom_fea_s, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                                huge_structure, output_final_layer_neuron)
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
            out = self.multiple_linear1(F.silu(out), batch_edge) 
            out = self.multiple_linear2(F.silu(out), batch_edge) 
            out = out.T 
        return out
