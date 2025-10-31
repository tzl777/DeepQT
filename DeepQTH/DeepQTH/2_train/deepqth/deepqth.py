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

import GaussianBasis
import GraphNorm, DiffGroupNorm
import RBF, cosine_cutoff, ShiftedSoftplus, _eps

from functional import split_batch_data, batched_shortest_path_distance, create_ptr_from_batch
from layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding

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
        out = out.reshape(num_edge, 2, -1) 
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


class DeepQTH(nn.Module):
    def __init__(self, num_species, in_atom_fea_len, in_edge_fea_len, num_orbital,
                 distance_expansion, gauss_stop, if_exp, if_MultipleLinear, if_edge_update, if_lcmp,
                 normalization, atom_update_net, separate_onsite,
                 trainable_gaussians, type_affine, n_heads=4, num_l=4, max_path_length=5):
        super(DeepQTH, self).__init__()
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
            ) #0.0, 6.0, 128, False
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
        )

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
                                          if_exp=if_exp)  # 64，128，81， 5, True
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
                                          if_exp=if_exp)  # 64，128，81， 5, True
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


    
    def forward(self, atom_attr, edge_idx, edge_attr, node_paths, edge_paths, voronoi_values, centralities, batch,
                sub_atom_idx=None, sub_edge_idx=None, sub_edge_ang=None, sub_index=None,
                huge_structure=False, output_final_layer_neuron=''):
        # print("batch = ",batch) #tensor([0,..., 0, 1,..., 1, 2,..., 2])3*72
        batch_edge = batch[edge_idx[0]] 
        # print("batch_edge = ",batch_edge)
        atom_fea0 = self.embed(atom_attr) #[108,64]
        distance = edge_attr[:, 0] #(6048,)
        edge_vec = edge_attr[:, 1:4] - edge_attr[:, 4:7]#6048*3
        if self.type_affine is None: #True
            edge_fea0 = self.distance_expansion(distance) 
        else:
            affine_coeff = self.type_affine(self.num_species * atom_attr[edge_idx[0]] + atom_attr[edge_idx[1]])
            edge_fea0 = self.distance_expansion(distance * affine_coeff[:, 0] + affine_coeff[:, 1])

        ptr = create_ptr_from_batch(batch)
        # start = time.perf_counter()
        atom_fea0 = self.centrality_encoding(atom_fea0, edge_idx, edge_fea0, voronoi_values, centralities) 
       
        b = self.spatial_encoding(atom_fea0, node_paths) 

        if self.if_edge_update == True:
            for layer in self.layers: 
                atom_fea0, edge_fea0 = layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
            
            if self.if_lcmp == True:  # True
                atom_fea, edge_fea = self.hidden_layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
                out = self.lcmp(atom_fea, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                                huge_structure, output_final_layer_neuron) 
            else:
                atom_fea, edge_fea = self.output_layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
                out = edge_fea
        else:
            for layer in self.layers: 
                atom_fea0 = layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
            if self.if_lcmp == True:  # True
                atom_fea = self.hidden_layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
                out = self.lcmp(atom_fea, edge_fea0, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                                huge_structure, output_final_layer_neuron)
            else:
                atom_fea, edge_fea = self.output_layer(atom_fea0, edge_idx, edge_fea0, b, edge_paths, ptr)
                out = edge_fea
        if self.if_MultipleLinear == True: #False
            out = self.multiple_linear1(F.silu(out), batch_edge)
            out = self.multiple_linear2(F.silu(out), batch_edge)
            out = out.T #1*6048*81
        
        return out
