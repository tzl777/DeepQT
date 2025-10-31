import os
import shutil
import sys
from configparser import ConfigParser
from inspect import signature

import numpy as np
import scipy
import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch import nn, package
import h5py
import matplotlib.pyplot as plt
import sisl
from sisl.io import *
import torch_geometric
from torch.package import PackageExporter

def draw_sub_H(a_i, b_j, H_ab_csr):
    plt.spy(H_ab_csr)
    cmap = plt.colormaps.get_cmap('coolwarm')
    fig, ax = plt.subplots(dpi=200)
    cax = ax.imshow(H_ab_csr, cmap=cmap)
    # fig.colorbar(cax)
    ax.set_title(rf"$H_{{{a_i},{b_j}}}$")
    ax.axvline(x=8.5, color='black', linewidth=1)
    ax.axhline(y=8.5, color='black', linewidth=1)
    for i in range(H_ab_csr.shape[0]):
        for j in range(H_ab_csr.shape[1]):
            text = f"{H_ab_csr[i, j]:.3f}" 
            ax.text(j, i, text, ha="center", va="center", fontsize=4, color="black")
    plt.tight_layout()
    plt.show()

def print_args(args):
    for k, v in args._get_kwargs():
        print('{} = {}'.format(k, v))
    print('')


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1) 

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class MaskMSELoss(nn.Module):
    def __init__(self) -> None:
        super(MaskMSELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert input.shape == target.shape == mask.shape
        mse = torch.pow(input - target, 2)
        mse = torch.masked_select(mse, mask).mean()

        return mse


class MaskMAELoss(nn.Module):
    def __init__(self) -> None:
        super(MaskMAELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert input.shape == target.shape == mask.shape
        mae = torch.abs(input - target)
        mae = torch.masked_select(mae, mask).mean()

        return mae


class LossRecord:
    def __init__(self):
        self.reset() 

    def reset(self):
        self.last_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        self.last_val = val 
        self.sum += val * num 
        self.count += num 
        self.avg = self.sum / self.count 


def collate_fn(graph_list):
    return Collater(if_lcmp=True)(graph_list)

class Collater:
    def __init__(self, if_lcmp):
        self.if_lcmp = if_lcmp #True
        self.flag_pyg2 = (torch_geometric.__version__[0] == '2') 

    def __call__(self, graph_list):
        if self.if_lcmp: #True
            flag_dict = hasattr(graph_list[0], 'subgraph_dict')
            if self.flag_pyg2:
                assert flag_dict, 'Please generate the graph file with the current version of PyG'
            batch = Batch.from_data_list(graph_list) 

            subgraph_atom_idx_batch = []
            subgraph_edge_idx_batch = []
            subgraph_edge_ang_batch = []
            subgraph_index_batch = []
            if flag_dict: #True
                for index_batch in range(len(graph_list)):
                    (subgraph_atom_idx, subgraph_edge_idx, subgraph_edge_ang,
                     subgraph_index) = graph_list[index_batch].subgraph_dict.values()
                    if self.flag_pyg2: 
                        subgraph_atom_idx_batch.append(subgraph_atom_idx + batch._slice_dict['x'][index_batch])
                        subgraph_edge_idx_batch.append(subgraph_edge_idx + batch._slice_dict['edge_attr'][index_batch])
                        subgraph_index_batch.append(subgraph_index + batch._slice_dict['edge_attr'][index_batch] * 2)
                    else:
                        subgraph_atom_idx_batch.append(subgraph_atom_idx + batch.__slices__['x'][index_batch])
                        subgraph_edge_idx_batch.append(subgraph_edge_idx + batch.__slices__['edge_attr'][index_batch])
                        subgraph_index_batch.append(subgraph_index + batch.__slices__['edge_attr'][index_batch] * 2)
                    subgraph_edge_ang_batch.append(subgraph_edge_ang)#[226440,25]
            else:
                for index_batch, (subgraph_atom_idx, subgraph_edge_idx,
                                  subgraph_edge_ang, subgraph_index) in enumerate(batch.subgraph):
                    subgraph_atom_idx_batch.append(subgraph_atom_idx + batch.__slices__['x'][index_batch])
                    subgraph_edge_idx_batch.append(subgraph_edge_idx + batch.__slices__['edge_attr'][index_batch])
                    subgraph_edge_ang_batch.append(subgraph_edge_ang)
                    subgraph_index_batch.append(subgraph_index + batch.__slices__['edge_attr'][index_batch] * 2)

            subgraph_atom_idx_batch = torch.cat(subgraph_atom_idx_batch, dim=0)
            subgraph_edge_idx_batch = torch.cat(subgraph_edge_idx_batch, dim=0)
            subgraph_edge_ang_batch = torch.cat(subgraph_edge_ang_batch, dim=0)
            subgraph_index_batch = torch.cat(subgraph_index_batch, dim=0)

            subgraph = (subgraph_atom_idx_batch, subgraph_edge_idx_batch, subgraph_edge_ang_batch, subgraph_index_batch)

            return batch, subgraph
        else:
            return Batch.from_data_list(graph_list)


def load_orbital_types(path, return_orbital_types=False):
    orbital_types = []
    with open(path) as f:
        line = f.readline()
        while line:
            orbital_types.append(list(map(int, line.split())))
            line = f.readline()
    atom_num_orbital = [sum(map(lambda x: 2 * x + 1,atom_orbital_types)) for atom_orbital_types in orbital_types] 
    if return_orbital_types:
        return atom_num_orbital, orbital_types
    else:
        return atom_num_orbital

class Transform:
    def __init__(self, tensor=None, mask=None, normalizer=False, boxcox=False):
        self.normalizer = normalizer
        self.boxcox = boxcox
        if normalizer:
            raise NotImplementedError
            self.mean = abs(tensor).sum(dim=0) / mask.sum(dim=0)
            self.std = None
            print(f'[normalizer] mean: {self.mean}, std: {self.std}')
        if boxcox:
            raise NotImplementedError
            _, self.opt_lambda = scipy.stats.boxcox(tensor.double())
            print('[boxcox] optimal lambda value:', self.opt_lambda)

    def tran(self, tensor):
        if self.boxcox:
            tensor = scipy.special.boxcox(tensor, self.opt_lambda)
        if self.normalizer:
            tensor = (tensor - self.mean) / self.std
        return tensor

    def inv_tran(self, tensor):
        if self.normalizer:
            tensor = tensor * self.std + self.mean
        if self.boxcox:
            tensor = scipy.special.inv_boxcox(tensor, self.opt_lambda)
        return tensor

    def state_dict(self):
        result = {'normalizer': self.normalizer,
                  'boxcox': self.boxcox}
        if self.normalizer:
            result['mean'] = self.mean
            result['std'] = self.std
        if self.boxcox:
            result['opt_lambda'] = self.opt_lambda
        return result

    def load_state_dict(self, state_dict):
        self.normalizer = state_dict['normalizer']
        self.boxcox = state_dict['boxcox']
        if self.normalizer:
            self.mean = state_dict['mean']
            self.std = state_dict['std']
            print(f'Load state dict, mean: {self.mean}, std: {self.std}')
        if self.boxcox:
            self.opt_lambda = state_dict['opt_lambda']
            print('Load state dict, optimal lambda value:', self.opt_lambda)


import os
import shutil
import torch
from typing import Dict, Any

def save_model(state: Dict[str, Any],
               model_dict: Dict[str, Any],
               model_state_dict: Dict[str, Any],
               path: str,
               is_best: bool):

    os.makedirs(path, exist_ok=True)

    checkpoint0 = {
        'state': state,
        'model_dict': model_dict, 
    }
    ckpt_path = os.path.join(path, 'model.pkl')
    torch.save(checkpoint0, ckpt_path)

    checkpoint1 = {
        'state': state,
        'model_state_dict': model_state_dict, 
    }
    state_dict_path = os.path.join(path, 'state_dict.pkl')
    torch.save(checkpoint1, state_dict_path)

    if is_best:
        try:
            shutil.copyfile(os.path.join(path, 'model.pkl'), os.path.join(path, 'best_model.pt'))
            shutil.copyfile(os.path.join(path, 'state_dict.pkl'), os.path.join(path, 'best_state_dict.pkl'))
        except Exception as e:
            print("Warning copying best files:", e)


# def save_model(state, model_dict, model_state_dict, path, is_best): 
#     model_dir = os.path.join(path, 'model.pt')
#     package_dict = {}
#     if 'verbose' in list(signature(package.PackageExporter.__init__).parameters.keys()):
#         package_dict['verbose'] = False
#     with package.PackageExporter(model_dir, **package_dict) as exp: 
#         exp.intern('DeepQT.**') 
#         exp.extern([
#             'scipy.**', 'numpy.**', 'torch_geometric.**', 'sklearn.**',
#             'torch_scatter.**', 'torch_sparse.**', 'torch_sparse.**', 'torch_cluster.**', 'torch_spline_conv.**',
#             'pyparsing', 'jinja2', 'sys', 'mkl', 'io', 'setuptools.**', 'rdkit.Chem', 'tqdm',
#             '__future__', '_operator', '_ctypes', 'six.moves.urllib', 'ase', 'matplotlib.pyplot', 'sympy', 'networkx',
#         ]) 
#         exp.save_pickle('checkpoint', 'model.pkl', state | model_dict) 
#     torch.save(state | model_state_dict, os.path.join(path, 'state_dict.pkl')) 
#     if is_best:
#         shutil.copyfile(os.path.join(path, 'model.pt'), os.path.join(path, 'best_model.pt'))
#         shutil.copyfile(os.path.join(path, 'state_dict.pkl'), os.path.join(path, 'best_state_dict.pkl'))


def write_ham_h5(hoppings_dict, path):
    fid = h5py.File(path, "w")
    for k, v in hoppings_dict.items():
        fid[k] = v
    fid.close()


def write_ham_npz(hoppings_dict, path):
    np.savez(path, **hoppings_dict)


def write_ham(hoppings_dict, path):
    os.makedirs(path, exist_ok=True)
    for key_term, matrix in hoppings_dict.items():
        np.savetxt(os.path.join(path, f'{key_term}_real.dat'), matrix)

def get_config(args):
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'default.ini'))
    for config_file in args:
        assert os.path.exists(config_file)
        config.read(config_file)
    if config['basic']['target'] == 'O_ij':
        assert config['basic']['O_component'] in ['H_minimum', 'H_minimum_withNA', 'H', 'Rho']
    if config['basic']['target'] == 'E_ij':
        assert config['basic']['energy_component'] in ['xc', 'delta_ee', 'both', 'summation', 'E_ij']
    else:
        assert config['hyperparameter']['criterion'] in ['MaskMSELoss']
    assert config['basic']['target'] in ['hamiltonian']
    assert config['basic']['interface'] in ['h5', 'h5_rc_only', 'h5_Eij', 'npz', 'npz_rc_only']
    assert config['network']['aggr'] in ['add', 'mean', 'max']
    assert config['network']['distance_expansion'] in ['GaussianBasis', 'BesselBasis', 'ExpBernsteinBasis']
    assert config['network']['normalization'] in ['BatchNorm', 'LayerNorm', 'PairNorm', 'InstanceNorm', 'GraphNorm',
                                                  'DiffGroupNorm', 'None']
    assert config['network']['atom_update_net'] in ['CGConv', 'GAT', 'PAINN', 'Graphormer', 'TransformerM']
    assert config['hyperparameter']['optimizer'] in ['sgd', 'sgdm', 'adam', 'adamW', 'adagrad', 'RMSprop', 'lbfgs']
    assert config['hyperparameter']['lr_scheduler'] in ['', 'MultiStepLR', 'ReduceLROnPlateau', 'CyclicLR']

    return config


def get_inference_config(*args):
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'inference', 'inference_default.ini'))
    for config_file in args:
        config.read(config_file)
    assert config['basic']['interface'] in ['openmx', 'abacus', 'siesta', 'transiesta']

    return config


import configparser
import ast
import re
from typing import Any

_num_re = re.compile(r'^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$') 
_list_like_re = re.compile(r'^\s*[\[\(\{].*[\]\)\}]\s*$')   # [..] (..) {..}

def _safe_parse_value(s: str) -> Any:

    if s is None:
        return s
    s = s.strip()
    if s == '':
        return '' 

    low = s.lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('none', 'null'):
        return None


    if _num_re.match(s):
        if '.' not in s and 'e' not in s.lower():
            try:
                return int(s)
            except Exception:
                pass
        try:
            return float(s)
        except Exception:
            pass

    if _list_like_re.match(s):
        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    try:
        val = ast.literal_eval(s)
        return val
    except Exception:
        return s

def read_config_to_dict(path: str, preserve_case: bool = True) -> dict:
    cfg = configparser.ConfigParser()
    if preserve_case:
        cfg.optionxform = str
    cfg.read(path, encoding='utf-8')

    out = {}
    for section in cfg.sections():
        secd = {}
        for opt, raw_val in cfg.items(section):
            parsed = _safe_parse_value(raw_val)
            secd[opt] = parsed
        out[section] = secd
    return out


"""
The code in this folder was obtained from "https://github.com/atomistic-machine-learning/schnetpack", which has the following license:


COPYRIGHT

Copyright (c) 2018 Kristof Schütt, Michael Gastegger, Pan Kessel, Kim Nicoli

All other contributions:
Copyright (c) 2018, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
def gaussian_smearing(distances, offset, widths, centered=False):
    if not centered: #True
        coeff = -0.5 / torch.pow(widths, 2) 
        # Use advanced indexing to compute the individual components
        diff = distances[..., None] - offset
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[..., None]
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss 

class GaussianBasis(nn.Module):
    def __init__(
            self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False
    ):
       
        super(GaussianBasis, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians) 
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset)) 
        
        if trainable:
            self.width = nn.Parameter(widths) #(128,)
            self.offsets = nn.Parameter(offset) #(128,)
        else: 
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered #False

    def forward(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        
        return gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )



