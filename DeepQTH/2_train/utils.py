import os
import shutil
import sys
from configparser import ConfigParser #读ini配置文件的包,https://www.cnblogs.com/hls-code/p/14690275.html
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
    # 添加分割线：x=8.5 的竖线，y=8.5 的横线
    ax.axvline(x=8.5, color='black', linewidth=1)
    ax.axhline(y=8.5, color='black', linewidth=1)
    # 在每个格子里添加数值
    for i in range(H_ab_csr.shape[0]):
        for j in range(H_ab_csr.shape[1]):
            text = f"{H_ab_csr[i, j]:.3f}"  # 保留3位小数
            ax.text(j, i, text, ha="center", va="center", fontsize=4, color="black")
    plt.tight_layout()
    plt.show()

def print_args(args):
    for k, v in args._get_kwargs():
        print('{} = {}'.format(k, v))
    print('')


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout #python中的标准输出流，默认是映射到控制台的，即将信息打印到控制台。
        self.log = open(filename, "a", buffering=1) #打开供追加的文件，如果不存在则创建该文件。buffering文件所需的缓冲区大小, 0 表示无缓冲, 1 表示线路缓冲。

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
        self.reset() #将last_val、avg、sum和count都初始化为0

    def reset(self):
        self.last_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        self.last_val = val #更新最后一次记录的损失值。
        self.sum += val * num #将当前的损失值（考虑其数量）累加到总和上。
        self.count += num #更新总数量，即加上当前数量。
        self.avg = self.sum / self.count #计算平均损失值。


def collate_fn(graph_list):
    return Collater(if_lcmp=True)(graph_list)

class Collater:
    def __init__(self, if_lcmp):
        self.if_lcmp = if_lcmp #True
        self.flag_pyg2 = (torch_geometric.__version__[0] == '2') #2.3.1。True,2.x版本需要有子图

    def __call__(self, graph_list): #用于使对象实例可以像函数一样被调用，即在实例被调用时会执行 __call__ 方法内的代码。传入的graph_list为采样的train_set或val_set或test_set
        if self.if_lcmp: #True
            flag_dict = hasattr(graph_list[0], 'subgraph_dict') #Python中的一个内置函数，用于检查对象(一个图数据)是否具有属性或者方法subgraph_dict，True。subgraph_dict=subgraph
            if self.flag_pyg2:
                assert flag_dict, 'Please generate the graph file with the current version of PyG' #如果条件flag_dict为False，会触发AssertionError，并且输出指定的错误信息'Please ... PyG'。
            batch = Batch.from_data_list(graph_list) #将一个图数据列表 graph_list 转换成一个 Batch 对象。它可以同时包含多个图数据，并提供一些方法来进行批次操作，如合并、转换等,这样可以方便地对整个数据批次进行处理。将graph_list（一个包含多个图数据对象的列表）合并成一个Batch对象。这个Batch对象允许将多个图数据作为一个整体在图神经网络中进行处理，其中每个图仍然可以被单独识别和处理。

            subgraph_atom_idx_batch = []
            subgraph_edge_idx_batch = []
            subgraph_edge_ang_batch = []
            subgraph_index_batch = []
            if flag_dict: #True
                for index_batch in range(len(graph_list)):#训练时：0-575，预测时：0
                    (subgraph_atom_idx, subgraph_edge_idx, subgraph_edge_ang,
                     subgraph_index) = graph_list[index_batch].subgraph_dict.values()#某一个晶体结构图数据的子图属性
                    if self.flag_pyg2: #2.0以上版本的torch_geometric，True
                        subgraph_atom_idx_batch.append(subgraph_atom_idx + batch._slice_dict['x'][index_batch])#[226440, 2]，其中'x'键对应的值是一个切片对象（slice）的列表，用于指示批处理中每个图的节点特征在合并后的特征矩阵中的位置。更新子图的索引信息，以便它们反映在批处理后数据结构中的实际位置。使用batch._slice_dict来获取合并后的特征矩阵中各个图的位置。
                        subgraph_edge_idx_batch.append(subgraph_edge_idx + batch._slice_dict['edge_attr'][index_batch])#[226440]
                        subgraph_index_batch.append(subgraph_index + batch._slice_dict['edge_attr'][index_batch] * 2)#[226440]，这里为什么要乘2？
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
    orbital_types = []#存放orbital_types.dat中的轨道类型
    with open(path) as f:
        line = f.readline()
        while line:
            orbital_types.append(list(map(int, line.split())))#map作用是把int函数依次作用在list中的每一个元素上，得到一个新的list并返回。注意，map不改变原list，而是返回一个新list。
            line = f.readline()
    atom_num_orbital = [sum(map(lambda x: 2 * x + 1,atom_orbital_types)) for atom_orbital_types in orbital_types] #[1+1+3+3+5=13，36个13的列表]，计算每个原子的使用的总轨道数。
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
    """
    简单可靠地保存：保存 checkpoint（meta + state_dict）和单独的 state_dict 文件。
    加载时在目标环境用相同源码构造模型并 model.load_state_dict(...)
    """
    os.makedirs(path, exist_ok=True)

    # 保存 checkpoint（包含 meta 信息和模型权重）
    checkpoint = {
        'meta': state,
        'model_state_dict': model_state_dict  # 已经是 dict，如 {'state_dict': model.state_dict()}
    }
    ckpt_path = os.path.join(path, 'checkpoint.pt')
    torch.save(checkpoint, ckpt_path)

    # 也单独保存 state_dict 以便按常规方式加载
    state_dict_path = os.path.join(path, 'state_dict.pt')
    torch.save(model_state_dict, state_dict_path)

    # 仍然保留复制最佳模型的逻辑
    if is_best:
        try:
            shutil.copyfile(ckpt_path, os.path.join(path, 'best_checkpoint.pt'))
            shutil.copyfile(state_dict_path, os.path.join(path, 'best_state_dict.pt'))
        except Exception as e:
            print("Warning copying best files:", e)


# def save_model(state, model_dict, model_state_dict, path, is_best): #这些步骤将模型和相关信息打包并保存为一个文件，这个文件可以用于在其他环境中加载模型和依赖的库，使得模型的部署和共享更加方便。
#     model_dir = os.path.join(path, 'model.pt') #创建了一个保存模型的文件路径
#     package_dict = {} #初始化了一个空字典，用于存储包的相关信息。
#     if 'verbose' in list(signature(package.PackageExporter.__init__).parameters.keys()): #检查了一个名为 PackageExporter 的类的构造函数中是否包含了一个名为 verbose 的参数。
#         package_dict['verbose'] = False #如果 PackageExporter 的构造函数中有 verbose 这个参数，就将 package_dict 字典中的键值对 'verbose': False 添加进去，将 verbose 设置为 False。这样可以在创建 PackageExporter 对象时设置 verbose 参数为 False。
#     #torch.package是一种将PyTorch模型打包成独立格式的新方法。打包后的文件包含模型参数和元数据及模型的结构，换句话说，我们使用时只要load就可以了。使用 PackageExporter 来创建一个存档文件，这个存档就包含了在另一台机器上运行模型所需的所有东西。
#     with package.PackageExporter(model_dir, **package_dict) as exp: #创建了一个 PackageExporter 对象，用于导出模型相关信息。
#         exp.intern('DeepQT.**') #指定哪些模块或包应该被导出
#         exp.extern([
#             'scipy.**', 'numpy.**', 'torch_geometric.**', 'sklearn.**',
#             'torch_scatter.**', 'torch_sparse.**', 'torch_sparse.**', 'torch_cluster.**', 'torch_spline_conv.**',
#             'pyparsing', 'jinja2', 'sys', 'mkl', 'io', 'setuptools.**', 'rdkit.Chem', 'tqdm',
#             '__future__', '_operator', '_ctypes', 'six.moves.urllib', 'ase', 'matplotlib.pyplot', 'sympy', 'networkx',
#         ]) #哪些不应该被导出,而是在导出文件被加载时从环境中动态导入。
#         exp.save_pickle('checkpoint', 'model.pkl', state | model_dict) #将模型的状态和字典信息保存为一个 .pkl 文件。
#     torch.save(state | model_state_dict, os.path.join(path, 'state_dict.pkl')) #使用 PyTorch 的 torch.save() 函数将模型的状态和状态字典信息保存为 .pkl 文件。
#     if is_best: #如果 is_best 为真，则将保存的模型文件和状态字典文件复制保存为最佳模型文件和状态字典文件。
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
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(widths, 2) #128个224.0139
        # Use advanced indexing to compute the individual components
        diff = distances[..., None] - offset #二维张量，其形状为(2016, 128)，表示每个距离值与每个高斯中心（offset）的差异。这个差异用于计算每个距离如何分布在不同的高斯基函数上。
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[..., None]
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))#二维张量，其形状也为(2016, 128)，包含了经过高斯展宽后的值。这些值表示了每个原子间距离在128个不同高斯基函数上的映射结果，从而提供了一种连续的方式来表示原子间的距离信息。
    return gauss #通过将原子间的距离信息转换成一个连续的高维表示，有助于图神经网络更好地理解和处理结构信息。

class GaussianBasis(nn.Module):
    def __init__(
            self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False
    ):
        #输入参数：start=0.0, stop=6.0, n_gaussians=128, centered=False, trainable=False
        super(GaussianBasis, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians) #生成从0.0到6.0均匀分布的128个元素的张量，每个元素之间的间隔大约是0.0472。均值(128,)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))  #128个方差(128,)
        #widths是一个与offset形状相同的张量，每个元素的值也都是0.0472，表示高斯函数之间的宽度是均匀的，并且每个高斯函数的宽度相等。这样的配置可以用来在指定的范围内（在这个例子中是0到6）均匀地覆盖距离，使得任何一个具体的距离值都能被一系列的高斯函数所覆盖。
        if trainable: #把偏移和宽度参数作为可训练
            self.width = nn.Parameter(widths) #(128,)
            self.offsets = nn.Parameter(offset) #(128,)
        else: #不可训练，在模型训练时不会更新（即调用optimizer.step()后widths和offset参数不会变化，只可人为地改变它们的值），但参数又作为模型参数不可或缺的一部分。
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
        #distances是两个原子间的距离，共2016个。对于给定的每个原子间距离将被映射到一个高斯基上，每个高斯函数都会产生一个展宽值，从而形成一个连续的特征表示。这种表示能够捕捉距离信息的不同方面，有助于图神经网络更准确地理解和预测分子或材料的性质。
        return gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )



