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

from torch_geometric.data import Data
import time


class PaninnNodeFea():
    def __init__(self, node_fea_s, node_fea_v=None):
        self.node_fea_s = node_fea_s
        if node_fea_v == None:
            self.node_fea_v = torch.zeros(node_fea_s.shape[0], node_fea_s.shape[1], 3, dtype=node_fea_s.dtype,
                                          device=node_fea_s.device)
        else:
            self.node_fea_v = node_fea_v 

    def __add__(self, other):
        return PaninnNodeFea(self.node_fea_s + other.node_fea_s, self.node_fea_v + other.node_fea_v)


class PAINN(nn.Module):
    def __init__(self, in_features, edge_dim, rc: float, l: int, normalization):
        super(PAINN, self).__init__()
        self.ms1 = nn.Linear(in_features, in_features)#64,64
        self.ssp = ShiftedSoftplus()
        self.ms2 = nn.Linear(in_features, in_features * 3)#64,192

        self.rbf = RBF(rc, l)#6.0,64
        self.mv = nn.Linear(l, in_features * 3)#64,192
        self.fc = cosine_cutoff(rc)#6.0，

        self.us1 = nn.Linear(in_features * 2, in_features)#128,64
        self.us2 = nn.Linear(in_features, in_features * 3)#64,192

        self.normalization = normalization
        if self.normalization == 'BatchNorm':
            self.bn = nn.BatchNorm1d(in_features, track_running_stats=True)
        elif self.normalization == 'LayerNorm': #True
            self.ln = LayerNorm(in_features)
        elif self.normalization == 'PairNorm':
            self.pn = PairNorm(in_features)
        elif self.normalization == 'InstanceNorm':
            self.instance_norm = InstanceNorm(in_features)
        elif self.normalization == 'GraphNorm':
            self.gn = GraphNorm(in_features)
        elif self.normalization == 'DiffGroupNorm':
            self.group_norm = DiffGroupNorm(in_features, 128)
        elif self.normalization is None or self.normalization == 'None':
            pass
        else:
            raise ValueError('Unknown normalization function: {}'.format(normalization))

    def forward(self, x: Union[torch.Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor, batch, edge_vec) -> torch.Tensor:
        r = torch.sqrt((edge_vec ** 2).sum(dim=-1) + _eps).unsqueeze(-1)
        sj = x.node_fea_s[edge_index[1, :]]
        vj = x.node_fea_v[edge_index[1, :]]
        phi = self.ms2(self.ssp(self.ms1(sj)))
        w = self.fc(r) * self.mv(self.rbf(r))

        v_, s_, r_ = torch.chunk(phi * w, 3, dim=-1)
        ds_update = s_
        dv_update = vj * v_.unsqueeze(-1) + r_.unsqueeze(-1) * (edge_vec / r).unsqueeze(1) 

        ds = scatter(ds_update, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce='mean')
        dv = scatter(dv_update, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce='mean')

        x = x + PaninnNodeFea(ds, dv)

        sj = x.node_fea_s[edge_index[1, :]]#(6048,64)
        vj = x.node_fea_v[edge_index[1, :]]#(6048,64,3)

        norm = torch.sqrt((vj ** 2).sum(dim=-1) + _eps)#(6048,64)
        s = torch.cat([norm, sj], dim=-1) #(6048,128)
        sj = self.us2(self.ssp(self.us1(s)))#(6048,128)->#(6048,64)->#(6048,192)
        uv = scatter(vj, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce='mean')#(108,64,3)
        norm = torch.sqrt((uv ** 2).sum(dim=-1) + _eps).unsqueeze(-1)#(108,64,1)
        s_ = scatter(sj, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce='mean')#(108,192)
        avv, asv, ass = torch.chunk(s_, 3, dim=-1)#3*(108,64)
        ds = ((uv / norm) ** 2).sum(dim=-1) * asv + ass #(108,64)
        dv = uv * avv.unsqueeze(-1)#(108,64,3)

        if self.normalization == 'BatchNorm':
            ds = self.bn(ds)
        elif self.normalization == 'LayerNorm':
            ds = self.ln(ds, batch)
        elif self.normalization == 'PairNorm':
            ds = self.pn(ds, batch)
        elif self.normalization == 'InstanceNorm':
            ds = self.instance_norm(ds, batch)
        elif self.normalization == 'GraphNorm':
            ds = self.gn(ds, batch)
        elif self.normalization == 'DiffGroupNorm':
            ds = self.group_norm(ds)

        x = x + PaninnNodeFea(ds, dv)

        return x

class MPLayer(nn.Module):
    def __init__(self, in_atom_fea_len, in_edge_fea_len, out_edge_fea_len, if_exp, if_edge_update, normalization,
                 atom_update_net, gauss_stop, output_layer=False):
        super(MPLayer, self).__init__()

        self.cgconv = PAINN(
            in_features=in_atom_fea_len,
            edge_dim=in_edge_fea_len,
            rc=gauss_stop,
            l=64,
            normalization=normalization
        )
        self.vs1 = nn.Linear(3, 5)  # 3,5
        self.vv = nn.Linear(3, 3)  # 3,3
        self.vs2 = nn.Linear(in_atom_fea_len * 2, in_atom_fea_len)  # 128,64
        self.vs3 = nn.Linear(in_atom_fea_len, in_atom_fea_len * 2)  # 64,64
        self.ssp = ShiftedSoftplus()

        self.if_edge_update = if_edge_update 
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

        atom_fea = self.cgconv(atom_fea, edge_idx, edge_fea, batch, edge_vec)
        atom_fea_s = atom_fea.node_fea_s #(108,64)
        atom_fea_v = atom_fea.node_fea_v #(108,64,3)
        vs = self.vs1(atom_fea_v)
        norm = torch.sqrt((vs ** 2).sum(dim=-1) + _eps)  # (108,64)
        s = torch.cat([norm, atom_fea_s], dim=-1)  # (108,128)
        sj = self.vs3(self.ssp(self.vs2(s))) 
        sv, sj = torch.chunk(sj, 2, dim=-1) 
        vv = self.vv(atom_fea_v) #(108,64,3)
        vj = vv * sv.unsqueeze(-1)  # (108,64,3)

        atom_fea = atom_fea + PaninnNodeFea(sj, vj)
        atom_fea_s = atom_fea.node_fea_s  # (108,64)

        if self.if_edge_update: #True
            row, col = edge_idx #[6048], [6048]
            edge_fea = self.e_lin(torch.cat([atom_fea_s[row], atom_fea_s[col], edge_fea], dim=-1)) #[6048,256]->[6048,128/103]
            return atom_fea, edge_fea #[108,64], [6048,128/103]
        else:
            return atom_fea

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


class Painn(nn.Module):
    def __init__(self, num_species, in_atom_fea_len, in_edge_fea_len, num_orbital,
                 distance_expansion, gauss_stop, if_exp, if_MultipleLinear, if_edge_update, if_lcmp,
                 normalization, atom_update_net, separate_onsite,
                 trainable_gaussians, type_affine, n_heads=4, num_l=4, max_path_length=5):
        super(Painn, self).__init__()
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
        self.atom_update_net = atom_update_net 
        self.separate_onsite = separate_onsite

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
                               normalization, atom_update_net, gauss_stop)
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


    def forward(self, atom_attr, edge_idx, edge_attr, node_paths, edge_paths, voronoi_values, centralities, batch,
                sub_atom_idx=None, sub_edge_idx=None, sub_edge_ang=None, sub_index=None,
                huge_structure=False, output_final_layer_neuron=''):

        batch_edge = batch[edge_idx[0]]

        atom_fea0 = self.embed(atom_attr)
        distance = edge_attr[:, 0] #(6048,)
        edge_vec = edge_attr[:, 1:4] - edge_attr[:, 4:7]
        if self.type_affine is None: #True
            edge_fea0 = self.distance_expansion(distance) 
        else:
            affine_coeff = self.type_affine(self.num_species * atom_attr[edge_idx[0]] + atom_attr[edge_idx[1]])
            edge_fea0 = self.distance_expansion(distance * affine_coeff[:, 0] + affine_coeff[:, 1])
        
        atom_fea0 = PaninnNodeFea(atom_fea0)
        
        if self.if_edge_update == True: #True

            atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea, edge_fea = self.mp2(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea, edge_fea = self.mp4(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            atom_fea, edge_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            if self.if_lcmp == True:#True
                atom_fea_s = atom_fea.node_fea_s

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


            atom_fea_s = atom_fea.node_fea_s

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
