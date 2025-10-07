import collections
import itertools
import os
import json
import warnings
import math
import sys
import torch
import torch_geometric
from torch_geometric.data import Data, Batch #用于将多个图数据合并成一个批处理，以便在图神经网络中进行有效的并行处理。
import numpy as np
import h5py
import networkx as nx
from spherical_harmonics_basis import get_spherical_from_cartesian, SphericalHarmonics, _spherical_harmonics

from pymatgen.core.structure import Structure

"""
The function get_graph below is extended from "https://github.com/materialsproject/pymatgen", which has the MIT License below

---------------------------------------------------------------------------
The MIT License (MIT)
Copyright (c) 2011-2012 MIT & The Regents of the University of California, through Lawrence Berkeley National Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
def get_graph(cart_coords, frac_coords, numbers, stru_id, radius, material_dimension, numerical_tol, lattice,
              default_dtype_torch, tb_folder, data_format, num_l, shortest_path_length, if_lcmp_graph,
              separate_onsite, target='hamiltonian', huge_structure=False, only_get_R_list=False, if_new_sp=False,
              if_require_grad=False, fid_rc=None, **kwargs):
    #cart_coords, frac_coords, [72*6], 1-600, 7.0, 2, 1e-8, lattice, default_dtype_torch, tb_folder当前文件路径, h5, 4, 5, True, False, hamiltonian, False, False, False, False, None
    assert target in ['hamiltonian', 'phiVdphi', 'density_matrix', 'O_ij', 'E_ij', 'E_i']
    if target == 'density_matrix' or target == 'O_ij':
        assert data_format == 'h5' or data_format == 'h5_rc_only'
    if target == 'E_ij':
        assert data_format == 'h5'
        assert separate_onsite is True
    if target == 'E_i':
        assert data_format == 'h5'
        assert if_lcmp_graph is False
        assert separate_onsite is True

    assert tb_folder is not None
    if data_format == 'h5_rc_only' and target == 'E_ij':
        raise NotImplementedError
    elif data_format == 'h5' or (data_format == 'h5_rc_only' and target != 'E_ij'): #True
        key_atom_list = [[] for _ in range(len(numbers))] #创建第stru_id个结构的原子序数的空列表，长度为len(numbers)=72
        edge_idx_target, edge_fea, edge_idx_source = [], [], []
        if if_lcmp_graph: #True
            atom_idx_connect, edge_idx_connect = [], []
            edge_idx_connect_cursor = 0
        if target == 'E_ij':
            fid = h5py.File(os.path.join(tb_folder, 'E_delta_ee_ij.h5'), 'r')
        else:
            if if_require_grad: #False
                fid = fid_rc
            else:
                fid = h5py.File(os.path.join(tb_folder, 'rc.h5'), 'r') #读取第stru_id个结构的所有原子的截断半径内的局域旋转坐标单位向量。
        for k in fid.keys():
            key = json.loads(k) #返回的是python的字典对象。
            key_tensor = torch.tensor([key[0], key[1], key[2], key[3], key[4]]) # (R, i, j) i and j is 0-based index
            if separate_onsite: #False
                if key[0] == 0 and key[1] == 0 and key[2] == 0 and key[3] == key[4]:
                    continue
            key_atom_list[key[3]].append(key_tensor) #将第stru_id结构的每个原子i的截断半径内的所有原子key_tensor存入key_atom_list中，键代表72个原子，值代表每个原子i的所有邻居原子key_tensor列表
        if target != 'E_ij' and not if_require_grad:
            fid.close()
        # print("cart_coords.shape, key_atom_list.shape = ", len(cart_coords), len(key_atom_list), len(key_atom_list[40])) #有36个原子坐标，第36号原子有44个邻居原子
    
        for index_first, (cart_coord, keys_tensor) in enumerate(zip(cart_coords, key_atom_list)): #第stru_id结构中的每个原子的笛卡尔坐标和该原子的邻居原子key_tensor列表，打包成一个可迭代对象，可迭代72次（原子个数次）。
            keys_tensor = torch.stack(keys_tensor) #把张量列表凑成一个2维的张量；也就是在增加新的维度进行堆叠。stru_id结构的每个原子的邻居原子堆叠。44*5
            cart_coords_j = cart_coords[keys_tensor[:, 4]] + keys_tensor[:, :3].type(default_dtype_torch).to(cart_coords.device) @ lattice.to(cart_coords.device)#指定希望在cart_coords.device设备上执行张量和模型操作。找出超胞内相邻原子j的实际笛卡尔坐标。53*3
            dist = torch.norm(cart_coords_j - cart_coord[None, :], dim=1)#指定在哪个维度上求L2范数，默认Frobenius范数计算方法是将矩阵中所有元素的平方和开平方。即计算所有邻居原子j到中心原子i的距离。(53,)
            len_nn = keys_tensor.shape[0] #44，中心原子的邻居原子数
            # print(index_first, len_nn)
            edge_idx_source.extend([index_first] * len_nn) #在edge_idx_source列表末尾扩展序列元素，每个序列元素是每个中心原子的邻居列表[0,...,0,...,35,...,35]，(2016,)
            edge_idx_target.extend(keys_tensor[:, 4].tolist()) #将每个中心原子的邻居原子j扩展存入edge_idx_target中，大小是(2016,)的一维列表。
            #extend方法是将传入的可迭代对象中的元素逐个添加到列表的末尾。如果传入的是一个列表，则列表中的每个元素都会被添加到原列表中。
            edge_fea_single = torch.cat([dist.view(-1, 1), cart_coord.view(1, 3).expand(len_nn, 3)], dim=-1) #53*1 cat #将cart_coord复制扩展为：53*3，得到53*4，单个中心原子的边特征是每个邻居原子到中心原子的距离+该中心原子的坐标
            edge_fea_single = torch.cat([edge_fea_single, cart_coords_j, cart_coords[keys_tensor[:, 4]]], dim=-1) #把所有邻居原子的坐标及其在第一晶胞内的等价原子坐标，都拼接在一起，53*10，每个邻居原子到中心原子的距离+该中心原子的坐标（相同的中心原子）+邻居原子的实际坐标+邻居原子在第一晶胞内的等效坐标
            edge_fea.append(edge_fea_single) #作为其中一个中心原子与其所有邻居原子的边特征，添加到总的边特征列表edge_fea中。36*53（邻居原子数可能不同）*10

            if if_lcmp_graph: #True
                #把每个原子i的截断半径内的邻居原子j的列表存入原子索引连接列表中。共72个原子，每个原子的所有邻居原子索引的tensor列表
                atom_idx_connect.append(keys_tensor[:, 4]) 
                #每一个原子及其截断半径内所有邻居原子的边索引列表[range(0, 37), range(37, 74), range(74, 111), ...,]
                edge_idx_connect.append(range(edge_idx_connect_cursor, edge_idx_connect_cursor + len_nn)) 
                edge_idx_connect_cursor += len_nn #统计晶胞中所有原子一共有多少个邻居原子和连接的边（包含超胞中相邻晶胞的原子）

        edge_fea = torch.cat(edge_fea).type(default_dtype_torch) #torch.Size([2664, 10])，即这个晶体结构中一共有2016条边
        edge_idx = torch.stack([torch.LongTensor(edge_idx_source), torch.LongTensor(edge_idx_target)]) #torch.Size([2, 2664])，每列是1个中心原子到其某个邻居原子

    else:
        raise NotImplemented

    if tb_folder is not None: #训练时：1-600，每次处理一个文件夹；预测时只有一个文件0。inference/C_nanotube_140
        if target == 'E_ij':
            read_file_list = ['E_ij.h5', 'E_delta_ee_ij.h5', 'E_xc_ij.h5']
            graph_key_list = ['E_ij', 'E_delta_ee_ij', 'E_xc_ij']
            read_terms_dict = {}
            for read_file, graph_key in zip(read_file_list, graph_key_list):
                read_terms = {}
                fid = h5py.File(os.path.join(tb_folder, read_file), 'r')
                for k, v in fid.items():
                    key = json.loads(k)
                    key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)
                    read_terms[key] = torch.tensor(v[...], dtype=default_dtype_torch)
                read_terms_dict[graph_key] = read_terms
                fid.close()

            local_rotation_dict = {}
            if if_require_grad:
                fid = fid_rc
            else:
                fid = h5py.File(os.path.join(tb_folder, 'rc.h5'), 'r')
            for k, v in fid.items():
                key = json.loads(k)
                key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)  # (R, i, j) i and j is 0-based index
                if if_require_grad:
                    local_rotation_dict[key] = v
                else:
                    local_rotation_dict[key] = torch.tensor(v, dtype=default_dtype_torch)
            if not if_require_grad:
                fid.close()
        elif target == 'E_i':
            read_file_list = ['E_i.h5']
            graph_key_list = ['E_i']
            read_terms_dict = {}
            for read_file, graph_key in zip(read_file_list, graph_key_list):
                read_terms = {}
                fid = h5py.File(os.path.join(tb_folder, read_file), 'r')
                for k, v in fid.items():
                    index_i = int(k)  # index_i is 0-based index
                    read_terms[index_i] = torch.tensor(v[...], dtype=default_dtype_torch)
                fid.close()
                read_terms_dict[graph_key] = read_terms
        else:
            if data_format == 'h5' or data_format == 'h5_rc_only': #训练时是h5，预测时是h5_rc_only
                atom_num_orbital = np.loadtxt(os.path.join(tb_folder, 'num_orbital_per_atom.dat')).astype(int)
                if data_format == 'h5':
                    with open(os.path.join(tb_folder, 'info.json'), 'r') as info_f:
                        info_dict = json.load(info_f)
                        spinful = info_dict["isspinful"] #False

                if data_format == 'h5':
                    if target == 'hamiltonian':
                        read_file_list = ['rh.h5']
                        graph_key_list = ['term_real']
                    elif target == 'phiVdphi':
                        read_file_list = ['rphiVdphi.h5']
                        graph_key_list = ['term_real']
                    elif target == 'density_matrix':
                        read_file_list = ['rdm.h5']
                        graph_key_list = ['term_real']
                    elif target == 'O_ij':
                        read_file_list = ['rh.h5', 'rdm.h5', 'rvna.h5', 'rvdee.h5', 'rvxc.h5']
                        graph_key_list = ['rh', 'rdm', 'rvna', 'rvdee', 'rvxc']
                    else:
                        raise ValueError('Unknown prediction target: {}'.format(target))
                    read_terms_dict = {}
                    for read_file, graph_key in zip(read_file_list, graph_key_list): #这个for循环只执行了一次，分别为rh.h5和term_real
                        read_terms = {}
                        fid = h5py.File(os.path.join(tb_folder, read_file), 'r') #读取截断半径内局域坐标下旋转后的哈密顿量矩阵,<HDF5 file "rh.h5" (mode r)>
                        for k, v in fid.items():
                            key = json.loads(k)
                            key = (key[0], key[1], key[2], key[3], key[4]) #(-1, -1, 0, 0, 14)。v[...]是numpy类型的9*9的局域坐标下旋转后的哈密顿量矩阵
                            #v是<HDF5 dataset "[-1, -1, 0, 1, 15]": shape (9, 9), type "<f8">
                            if spinful:
                                num_orbital_row = atom_num_orbital[key[3]]
                                num_orbital_column = atom_num_orbital[key[4]]
                                # soc block order:
                                # 1 3
                                # 4 2
                                if target == 'phiVdphi':
                                    raise NotImplementedError
                                else:
                                    read_value = torch.stack([
                                        torch.tensor(v[:num_orbital_row, :num_orbital_column].real, dtype=default_dtype_torch),
                                        torch.tensor(v[:num_orbital_row, :num_orbital_column].imag, dtype=default_dtype_torch),
                                        torch.tensor(v[num_orbital_row:, num_orbital_column:].real, dtype=default_dtype_torch),
                                        torch.tensor(v[num_orbital_row:, num_orbital_column:].imag, dtype=default_dtype_torch),
                                        torch.tensor(v[:num_orbital_row, num_orbital_column:].real, dtype=default_dtype_torch),
                                        torch.tensor(v[:num_orbital_row, num_orbital_column:].imag, dtype=default_dtype_torch),
                                        torch.tensor(v[num_orbital_row:, :num_orbital_column].real, dtype=default_dtype_torch),
                                        torch.tensor(v[num_orbital_row:, :num_orbital_column].imag, dtype=default_dtype_torch)
                                    ], dim=-1)
                                read_terms[key] = read_value
                            else:
                                read_terms[key] = torch.tensor(v[...], dtype=default_dtype_torch)#将截断半径内局域坐标下旋转后的哈密顿量key和value存入read_terms字典中。在HDF5中，v[...]表示对数据集或数组 v 进行全体索引或切片操作。这种语法意味着选择所有元素或对整个数据集执行操作。
                        read_terms_dict[graph_key] = read_terms #某个晶体结构图数据的键为term_real，值为read_terms，read_terms是包含key和value为旋转后的哈密顿量矩阵的字典，即read_terms是图中所有截断半径内局域坐标下的原子对key和旋转后的哈密顿量矩阵v
                        fid.close()
               
                local_rotation_dict = {}
                if if_require_grad: #False
                    fid = fid_rc
                else:
                    fid = h5py.File(os.path.join(tb_folder, 'rc.h5'), 'r') #得到截断半径内的排序后的原子对的3*3的单位局域坐标
                for k, v in fid.items():
                    key = json.loads(k)
                    key = (key[0], key[1], key[2], key[3], key[4])  # (R, i, j) i and j is 0-based index
                    if if_require_grad: #False
                        local_rotation_dict[key] = v
                    else:
                        local_rotation_dict[key] = torch.tensor(v[...], dtype=default_dtype_torch) #存入截断半径内的排序后的原子对的key和3*3的单位局域坐标value到local_rotation_dict中
                if not if_require_grad:
                    fid.close()
                #read_terms_dict存放的是旋转后的原子间的哈密顿量矩阵，local_rotation_dict存放的是截断半径内的单位坐标系
                max_num_orbital = max(atom_num_orbital) #返回指定轨道列表中最大值的元素9/13.

            elif data_format == 'npz' or data_format == 'npz_rc_only':
                spinful = False
                
                atom_num_orbital = np.loadtxt(os.path.join(tb_folder, 'num_orbital_per_atom.dat')).astype(int)
                if data_format == 'npz':
                    graph_key_list = ['term_real']
                    read_terms_dict = {'term_real': {}}
                    hopping_dict_read = np.load(os.path.join(tb_folder, 'rh.npz'))
                    for k, v in hopping_dict_read.items():
                        key = json.loads(k)
                        key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)  # (R, i, j) i and j is 0-based index
                        read_terms_dict['term_real'][key] = torch.tensor(v, dtype=default_dtype_torch)

                local_rotation_dict = {}
                local_rotation_dict_read = np.load(os.path.join(tb_folder, 'rc.npz'))
                for k, v in local_rotation_dict_read.items():
                    key = json.loads(k)
                    key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)
                    local_rotation_dict[key] = torch.tensor(v, dtype=default_dtype_torch)

                max_num_orbital = max(atom_num_orbital)
            else:
                raise ValueError(f'Unknown data format: {data_format}')

        if target == 'E_i':
            term_dict = {}
            onsite_term_dict = {}
            for graph_key in graph_key_list:
                term_dict[graph_key] = torch.full([numbers.shape[0], 1], np.nan, dtype=default_dtype_torch)
            for index_atom in range(numbers.shape[0]):
                assert index_atom in read_terms_dict[graph_key_list[0]]
                for graph_key in graph_key_list:
                    term_dict[graph_key][index_atom] = read_terms_dict[graph_key][index_atom]
            subgraph = None
        else:
            if data_format == 'h5_rc_only' or data_format == 'npz_rc_only':
                local_rotation = []
            else:
                term_dict = {}
                onsite_term_dict = {}
                if target == 'E_ij':
                    for graph_key in graph_key_list: #['term_real']
                        term_dict[graph_key] = torch.full([edge_fea.shape[0], 1], np.nan, dtype=default_dtype_torch) #edge_fea的size为[2016, 10]，创建了一个给定形状[2016,1]和类型的张量（Tensor），其中所有元素都被初始化为 NaN（Not a Number，非数值）。
                    local_rotation = []
                    if separate_onsite is True:
                        for graph_key in graph_key_list:
                            onsite_term_dict['onsite_' + graph_key] = torch.full([numbers.shape[0], 1], np.nan, dtype=default_dtype_torch)
                else:
                    term_mask = torch.zeros(edge_fea.shape[0], dtype=torch.bool) #edge_fea：torch.Size([2664, 10])，生成[2664个False]一维张量
                    for graph_key in graph_key_list:  #['term_real']，次for循环只执行一次
                        if spinful:
                            term_dict[graph_key] = torch.full([edge_fea.shape[0], max_num_orbital, max_num_orbital, 8],
                                                              np.nan, dtype=default_dtype_torch)
                        else:
                            if target == 'phiVdphi':
                                term_dict[graph_key] = torch.full([edge_fea.shape[0], max_num_orbital, max_num_orbital, 3],
                                                                  np.nan, dtype=default_dtype_torch)
                            else:
                                term_dict[graph_key] = torch.full([edge_fea.shape[0], max_num_orbital, max_num_orbital],
                                                                  np.nan, dtype=default_dtype_torch) #用np.nan填充size形状的张量，数据类型为default_dtype_torch，即生成用于存放第一晶胞内所有中心原子与其邻居原子的所有2016个边的9*9哈密顿量矩阵，其中所有元素都被初始化为 NaN（Not a Number，非数值）。
                        # print(term_dict[graph_key].shape) #torch.Size([2664, 9, 9])

                    local_rotation = []
                    if separate_onsite is True:
                        for graph_key in graph_key_list:
                            if spinful:
                                onsite_term_dict['onsite_' + graph_key] = torch.full(
                                    [numbers.shape[0], max_num_orbital, max_num_orbital, 8],
                                    np.nan, dtype=default_dtype_torch)
                            else:
                                if target == 'phiVdphi':
                                    onsite_term_dict['onsite_' + graph_key] = torch.full(
                                        [numbers.shape[0], max_num_orbital, max_num_orbital, 3],
                                        np.nan, dtype=default_dtype_torch)
                                else:
                                    onsite_term_dict['onsite_' + graph_key] = torch.full(
                                        [numbers.shape[0], max_num_orbital, max_num_orbital],
                                        np.nan, dtype=default_dtype_torch)

            # read_terms_dict存放的是旋转后的原子间的哈密顿量矩阵，local_rotation_dict存放的是截断半径内的边的旋转矩阵
            inv_lattice = torch.inverse(lattice).type(default_dtype_torch) #晶格矢量求逆
            for index_edge in range(edge_fea.shape[0]):  #edge_fea.shape=2016*10, edge_idx.shape=2*2016, index_edge从0-2015
                # h_{i0, jR} i and j is 0-based index，用于将笛卡尔坐标系中的原子位置转换到倒格矢空间中。torch.round把输入张量的每个元素舍入到最近的整数。
                R = torch.round(edge_fea[index_edge, 4:7].cpu() @ inv_lattice - edge_fea[index_edge, 7:10].cpu() @ inv_lattice).int().tolist() #求该实际邻居原子j的晶胞索引，R*inv(R)=E
                #把每个邻居原子的实际坐标*晶格矢量的逆 - 等效的第一晶胞内的邻居原子坐标*晶格矢量的逆，R表示了在晶格逆变换下，两个给定向量位置差的整数近似值。
                i, j = edge_idx[:, index_edge] #每一个边对应的原子i及其邻居原子j

                key_term = (*R, i.item(), j.item()) #每一条边对应的中心原子i和实际邻居原子j的哈密顿量的key
                
                if data_format == 'h5_rc_only' or data_format == 'npz_rc_only':
                    local_rotation.append(local_rotation_dict[key_term]) #取出key_term对应的中心原子的单位局域坐标系，即取出所有中心原子i的局域坐标系
                else:
                    if key_term in read_terms_dict[graph_key_list[0]]: #读取晶体图数据中对应'term_real'键下的k,v，判断key_term是否在k中。
                        for graph_key in graph_key_list: #'term_real'
                            if target == 'E_ij':
                                term_dict[graph_key][index_edge] = read_terms_dict[graph_key][key_term]
                            else:
                                term_mask[index_edge] = True #标志着第index_edge个边的哈密顿量矩阵被转存到了term_dict中。[2664个False依次变为True]
                                if spinful:
                                    term_dict[graph_key][index_edge, :atom_num_orbital[i], :atom_num_orbital[j], :] = read_terms_dict[graph_key][key_term]
                                else:
                                    term_dict[graph_key][index_edge, :atom_num_orbital[i], :atom_num_orbital[j]] = read_terms_dict[graph_key][key_term] 
                                    #将read_terms_dict键下key_term键下的9*9哈密顿量矩阵保存到term_dict键下的1*9*9的哈密顿量矩阵，遍历所有index_edge等使得term_dict中key为term_real，value为2016*9*9，2016条边，每条边对应一个9*9的小哈密顿量矩阵
                        local_rotation.append(local_rotation_dict[key_term]) #读取每个原子的截断半径内的排序后的3*3的单位局域坐标，存放到局域旋转列表中。
                    else:
                        print("key 2:", key_term, type(key_term), type(key_term[0], type(key_term[3])))
                        key_ham = read_terms_dict[graph_key_list[0]]
                        print(type(key_ham), key_ham[0], type(key_ham[0].key[0]), type(key_ham[0].key[3]))
                        
                        raise NotImplementedError(
                            "Not yet have support for graph radius including hopping without calculation")
            # term_dict存放的是不同key_term下旋转后的原子间的哈密顿量矩阵，local_rotation存放的是不同key_term下截断半径内的旋转矩阵
            # term_mask 为2664个True

            if separate_onsite is True and data_format != 'h5_rc_only' and data_format != 'npz_rc_only':
                for index_atom in range(numbers.shape[0]):
                    key_term = (0, 0, 0, index_atom, index_atom)
                    assert key_term in read_terms_dict[graph_key_list[0]]
                    for graph_key in graph_key_list:
                        if target == 'E_ij':
                            onsite_term_dict['onsite_' + graph_key][index_atom] = read_terms_dict[graph_key][key_term]
                        else:
                            if spinful:
                                onsite_term_dict['onsite_' + graph_key][index_atom, :atom_num_orbital[i], :atom_num_orbital[j], :] = \
                                read_terms_dict[graph_key][key_term]
                            else:
                                onsite_term_dict['onsite_' + graph_key][index_atom, :atom_num_orbital[i], :atom_num_orbital[j]] = \
                                read_terms_dict[graph_key][key_term]

            if if_lcmp_graph: #True
                local_rotation = torch.stack(local_rotation, dim=0) #2664*3*3
                assert local_rotation.shape[0] == edge_fea.shape[0] #都是2664
                r_vec = edge_fea[:, 1:4] - edge_fea[:, 4:7] #edge_fea[:, 1:4]是中心原子i的坐标，edge_fea[:, 4:7]是相邻原子j的实际坐标，2016*3原子之间的向量差
                r_vec = r_vec.unsqueeze(1) #在输入张量的特定维度上增加一个维度。torch.Size([2664, 1, 3])，可以用来改变张量的形状和结构。2016*1*3，即原子间向量的集合。

                if huge_structure is False:
                    #torch.Size([2664, 1, 1, 3])，torch.Size([1, 2664, 3, 3]) = torch.Size([2664, 2664, 1, 3])=torch.Size([7096896, 3])
                    r_vec = torch.matmul(r_vec[:, None, :, :], local_rotation[None, :, :, :].to(r_vec.device)).reshape(-1, 3) 
                    #r_vec为所有不同局域坐标下的原子间距离向量的集合。把所有边都转为不同局域坐标系下的向量。
                    
                    if if_new_sp: #False
                        r_vec = torch.nn.functional.normalize(r_vec, dim=-1)
                        angular_expansion = _spherical_harmonics(num_l - 1, -r_vec[..., 2], r_vec[..., 0],
                                                                 r_vec[..., 1])
                        angular_expansion.mul_(torch.cat([
                            (math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)) * torch.ones(2 * l + 1,
                                                                                         dtype=angular_expansion.dtype,
                                                                                         device=angular_expansion.device)
                            for l in range(num_l)
                        ]))
                        angular_expansion = angular_expansion.reshape(edge_fea.shape[0], edge_fea.shape[0], -1)
                    else:
                        r_vec_sp = get_spherical_from_cartesian(r_vec) #将所有不同局域坐标下的原子间距离向量（笛卡尔坐标）的集合，计算出球坐标系下的theta角和phi角。torch.Size([4064256, 2])
                        sph_harm_func = SphericalHarmonics()
                        angular_expansion = []
                        for l in range(num_l): #num_l=0-4
                            angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1])) #sph_harm_func.get返回tensor of shape [*theta.shape, 2*l+1]，涉及球谐函数，这里没搞懂。这个方法返回的结果是针对每个角度值(θ, φ)对应的球谐基函数展开，随后这些结果被追加到angular_expansion数组中。
                        angular_expansion = torch.cat(angular_expansion, dim=-1).reshape(edge_fea.shape[0], edge_fea.shape[0], -1) #torch.Size([2664, 2664, 16]，计算每个边在不同局域坐标系下的角度(θ, φ)的球谐基函数展开，这对于许多模拟和计算任务来说是非常有用的。这样的角度展开允许你在后续的计算中方便地使用球面函数的性质，比如在分子动力学模拟、光照计算或声场模拟中。
                subgraph_atom_idx_list = []
                subgraph_edge_idx_list = []
                subgraph_edge_ang_list = []
                subgraph_index = []
                index_cursor = 0

                for index in range(edge_fea.shape[0]):#0-2664
                    # h_{i0, jR}
                    i, j = edge_idx[:, index] #edge_idx尺寸torch.Size([2, 2016])，遍历每一个原子对
                    subgraph_atom_idx = torch.stack([i.repeat(len(atom_idx_connect[i])), atom_idx_connect[i]]).T #atom_idx_connect[i]#是第i个原子的截断半径内的所有邻居原子j的列表。torch.tensor.repeat()函数可以对张量进行重复扩充。原子i和每一个邻居原子j的对的转置，转置后是torch.Size([53, 2])。
                    subgraph_edge_idx = torch.LongTensor(edge_idx_connect[i]) #torch.Size([53])#取出第i个中心原子i的截断半径内的所有邻居原子的边索引
                    # print(subgraph_atom_idx.shape) #torch.Size([37, 2])
                    # print(subgraph_edge_idx.shape) #torch.Size([37])

                    if huge_structure:
                        r_vec_tmp = torch.matmul(r_vec[subgraph_edge_idx, :, :], local_rotation[index, :, :].to(r_vec.device)).reshape(-1, 3) #将全局坐标下的边向量r_vec，转为在各自单位局域坐标系下的边向量r_vec_tmp，53×3。
                        if if_new_sp:
                            r_vec_tmp = torch.nn.functional.normalize(r_vec_tmp, dim=-1)
                            subgraph_edge_ang = _spherical_harmonics(num_l - 1, -r_vec_tmp[..., 2], r_vec_tmp[..., 0], r_vec_tmp[..., 1])
                            subgraph_edge_ang.mul_(torch.cat([
                                (math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)) * torch.ones(2 * l + 1,
                                                                                             dtype=subgraph_edge_ang.dtype,
                                                                                             device=subgraph_edge_ang.device)
                                for l in range(num_l)
                            ]))
                        else:
                            r_vec_sp = get_spherical_from_cartesian(r_vec_tmp) #获得邻居原子局域坐标下的alpha和beta角度，53*2
                            sph_harm_func = SphericalHarmonics()
                            angular_expansion = []
                            for l in range(num_l):
                                angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1])) #25*53*1
                            subgraph_edge_ang = torch.cat(angular_expansion, dim=-1).reshape(-1, num_l ** 2) #53*25
                    else:
                        subgraph_edge_ang = angular_expansion[subgraph_edge_idx, index, :] #取出第i个原子的截断半径内的所有邻居原子的边的球谐基函数展开，torch.Size([53, 1, 25])
                    subgraph_atom_idx_list.append(subgraph_atom_idx) #每一条边的一个原子i及其截断半径内的所有邻居原子j的索引作为一个子图节点集合，存入subgraph_atom_idx_list中，2016*53*2。
                    subgraph_edge_idx_list.append(subgraph_edge_idx) #每一条边的一个原子i及其与所有邻居原子j的边索引作为一个子图的边集合，存入subgraph_edge_idx_list中，2016*53。
                    subgraph_edge_ang_list.append(subgraph_edge_ang) #每一条边的一个原子i及其所有邻居原子j的边的球谐函数展开作为一个子图的边角度集合，存入subgraph_edge_ang_list中，2016*53*25。
                    subgraph_index += [index_cursor] * len(atom_idx_connect[i]) #每一条边的一个原子i的邻居原子个数的索引游标[53个0,53个1,...,53个2016]相拼接在一个列表中，共2016*53个子图的索引列表。
                    index_cursor += 1

                    subgraph_atom_idx = torch.stack([j.repeat(len(atom_idx_connect[j])), atom_idx_connect[j]]).T #atom_idx_connect[j]是第j个原子的邻居原子的列表。原子j和每一个邻居原子的对的转置，torch.Size([53, 2])
                    subgraph_edge_idx = torch.LongTensor(edge_idx_connect[j]) #torch.Size([53])#取出原子j的所有邻居原子的边
                    if huge_structure:
                        r_vec_tmp = torch.matmul(r_vec[subgraph_edge_idx, :, :], local_rotation[index, :, :].to(r_vec.device)).reshape(-1, 3)
                        if if_new_sp:
                            r_vec_tmp = torch.nn.functional.normalize(r_vec_tmp, dim=-1)
                            subgraph_edge_ang = _spherical_harmonics(num_l - 1, -r_vec_tmp[..., 2], r_vec_tmp[..., 0], r_vec_tmp[..., 1])
                            subgraph_edge_ang.mul_(torch.cat([
                                (math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)) * torch.ones(2 * l + 1,
                                                                                             dtype=subgraph_edge_ang.dtype,
                                                                                             device=subgraph_edge_ang.device)
                                for l in range(num_l)
                            ]))
                        else:
                            r_vec_sp = get_spherical_from_cartesian(r_vec_tmp)
                            sph_harm_func = SphericalHarmonics()
                            angular_expansion = []
                            for l in range(num_l):
                                angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1]))
                            subgraph_edge_ang = torch.cat(angular_expansion, dim=-1).reshape(-1, num_l ** 2)#取出原子j的所有邻居原子的边的角度球谐函数，torch.Size([53,25])
                    else:
                        subgraph_edge_ang = angular_expansion[subgraph_edge_idx, index, :] #53*1*25
                    subgraph_atom_idx_list.append(subgraph_atom_idx) #每一条边的另一个原子j及其截断半径内的所有邻居原子的索引作为一个子图节点集合，存入subgraph_atom_idx_list中。里面已经包含了之前原子i的所有邻居原子索引，2*2016*53*2
                    subgraph_edge_idx_list.append(subgraph_edge_idx) #每一条边的另一个原子j及其截断半径内的所有邻居原子的边作为一个子图的边集合，存入subgraph_edge_idx_list中，2*2016*53。
                    subgraph_edge_ang_list.append(subgraph_edge_ang) #每一条边的另一个原子j及其截断半径内的所有邻居原子的边的球谐函数作为一个子图的边角度集合，存入subgraph_edge_ang_list中2*2016*53*25。
                    subgraph_index += [index_cursor] * len(atom_idx_connect[j]) #每一条边的另一个原子j的截断半径内的邻居原子个数的索引游标[53个2017,53个2018,...,53个4032]相拼接在一个列表中，共2*2016*53个子图的列表。
                    index_cursor += 1
                #这个subgraph是由所有边的各自两个原子的各自邻居原子和边构成的子图
                subgraph =  {"subgraph_atom_idx":torch.cat(subgraph_atom_idx_list, dim=0), #torch.Size([197136, 2])
                             "subgraph_edge_idx":torch.cat(subgraph_edge_idx_list, dim=0), #torch.Size([197136])
                             "subgraph_edge_ang":torch.cat(subgraph_edge_ang_list, dim=0), #torch.Size([197136, 16])
                             "subgraph_index":torch.LongTensor(subgraph_index)} #torch.Size([197136])
                # print("\n")
                # print("subgraph_atom_idx.shape = ", subgraph["subgraph_atom_idx"].shape)
                # print("subgraph_edge_idx.shape = ", subgraph["subgraph_edge_idx"].shape)
                # print("subgraph_edge_ang.shape = ", subgraph["subgraph_edge_ang"].shape)
                # print("subgraph_index.shape = ", subgraph["subgraph_index"].shape)
            else:
                subgraph = None

        if data_format == 'h5_rc_only' or data_format == 'npz_rc_only':
            # 创建一个新的空图
            import networkx as nx
            G = nx.Graph()
            # 添加节点及其属性：原子特征和位置坐标
            for i, (x, coords) in enumerate(zip(numbers, cart_coords)):
                G.add_node(i, x=x.item(), position=coords.numpy())
            # 添加边及其属性：边的特征
            for i, (src, dst) in enumerate(edge_idx.t()):
                G.add_edge(src.item(), dst.item(), attr=edge_fea[i].numpy())
            # 添加图的整体属性：晶格矢量
            G.graph['lattice'] = lattice.numpy()
            # 打印图的信息以确认转换成功
            # print("number_of_nodes = ", G.number_of_nodes()) #72
            # print("number_of_edges = ", G.number_of_edges()) #1368
            current_sys_path = "/fs2/home/ndsim10/DeepQT/2_train"
            if current_sys_path not in sys.path:
                sys.path.insert(0, current_sys_path)
            from graphormer.functional import shortest_path_distance, cal_voronoi_and_centrality
            node_paths, edge_paths = shortest_path_distance(G, shortest_path_length)
            # print(node_paths.shape) #torch.Size([1, 72, 72, 5])
            # print(edge_paths.shape) #torch.Size([1, 72, 72, 4])
            voronoi_values, centralities = cal_voronoi_and_centrality(cart_coords, lattice, material_dimension)

            data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id, cart_coords=cart_coords, lattice=lattice,
                        voronoi_values=voronoi_values, centralities=centralities, node_paths=node_paths, edge_paths=edge_paths,
                        term_mask=None, term_real=None, onsite_term_real=None,
                        atom_num_orbital=torch.tensor(atom_num_orbital),
                        subgraph_dict=subgraph,
                        **kwargs)
            #numbers是[原子个数*0]；edge_idx是边索引[2,2016];edge_fea是节点特征[2016,10];stru_id是0；term_mask=None；#term_dict=None；onsite_term_dict=None；atom_num_orbital是[预测的结构C原子个数*13]；subgraph是截断半径内的子图,包含了子图中原子对、边、球谐函数展开和子图索引的信息。
        else:
            if target == 'E_ij' or target == 'E_i':
                data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id, cart_coords=cart_coords, lattice=lattice,
                            **term_dict, **onsite_term_dict,
                            subgraph_dict=subgraph,
                            spinful=False,
                            **kwargs)
            else:
                # 创建一个新的空图
                import networkx as nx
                G = nx.Graph()
                # 添加节点及其属性：原子特征和位置坐标
                for i, (x, coords) in enumerate(zip(numbers, cart_coords)):
                    G.add_node(i, x=x.item(), position=coords.numpy())
                # 添加边及其属性：边的特征
                for i, (src, dst) in enumerate(edge_idx.t()):
                    G.add_edge(src.item(), dst.item(), attr=edge_fea[i].numpy())
                # 添加图的整体属性：晶格矢量
                G.graph['lattice'] = lattice.numpy()
                # 打印图的信息以确认转换成功
                # print("number_of_nodes = ", G.number_of_nodes()) #72
                # print("number_of_edges = ", G.number_of_edges()) #1368
                current_sys_path = "/fs2/home/ndsim10/DeepQT/2_train"
                if current_sys_path not in sys.path:
                    sys.path.insert(0, current_sys_path)
                from graphormer.functional import shortest_path_distance, cal_voronoi_and_centrality
                node_paths, edge_paths = shortest_path_distance(G, shortest_path_length)
                # print(node_paths.shape) #torch.Size([1, 72, 72, 5])
                # print(edge_paths.shape) #torch.Size([1, 72, 72, 4])

                voronoi_values, centralities = cal_voronoi_and_centrality(cart_coords, lattice, material_dimension)
                # print(voronoi_values.shape)
                # print(centralities.shape)
                # numbers是原子序数72个6列表；edge_idx是边索引[2,2016];edge_fea是节点特征[2016,10];stru_id是晶体结构图id；term_mask是[2016个True]张量；#term_dict是key为term_real，value为2016*9*9旋转后的哈密顿量；
                # onsite_term_dict是{}；atom_num_orbital是[72个13的列表]即每个原子使用的总轨道数；subgraph是由截断半径内的所有边的各自两个原子的各自邻居原子和边构成的子图,包含了子图中原子对、边、球谐函数展开和子图索引的信息。
                data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id, voronoi_values=voronoi_values, centralities=centralities,
                            cart_coords=cart_coords, lattice=lattice, term_mask=term_mask, node_paths=node_paths, edge_paths=edge_paths,
                            **term_dict, **onsite_term_dict,
                            atom_num_orbital=atom_num_orbital,
                            subgraph_dict=subgraph,
                            spinful=spinful,
                            **kwargs)

    else:
        data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id, cart_coords=cart_coords, lattice=lattice, **kwargs)
    return data

"""
###以下为调试用，可删除
def process_worker(folder, **kwargs):
    default_dtype_torch = torch.get_default_dtype()
    stru_id = os.path.split(folder)[-1]  # 如果给出的是一个目录和文件名，则输出路径和文件名，如果给出的是一个目录名，则输出路径和为空文件名。0-575

    structure = Structure(np.loadtxt(os.path.join(folder, 'lat.dat')),  # 加载第i个结构的晶格向量、原子序数和笛卡尔坐标
                          np.loadtxt(os.path.join(folder, 'element.dat')),
                          np.loadtxt(os.path.join(folder, 'site_positions.dat')),
                          coords_are_cartesian=True,
                          to_unit_cell=False)  # 用pymatgen创建结构

    cart_coords = torch.tensor(structure.cart_coords, dtype=default_dtype_torch)  # 原子的笛卡尔坐标，当前的默认浮点torch.dtype
    frac_coords = torch.tensor(structure.frac_coords, dtype=default_dtype_torch)
    numbers = torch.tensor(structure.atomic_numbers)  # List of atomic numbers.
    structure.lattice.matrix.setflags(write=True)  # 描述structure.lattice.matrix是否可以写入。
    lattice = torch.tensor(structure.lattice.matrix,
                           dtype=default_dtype_torch)  # 读取structure.lattice.matrix，并转为default_dtype_torch类型
    huge_structure = False  ## r=self.radius==-1。numerical_tol？

    return get_graph(cart_coords, frac_coords, numbers, stru_id, radius=7.0, material_dimension=2,
                         numerical_tol=1e-8, lattice=lattice, default_dtype_torch=default_dtype_torch,
                         tb_folder=folder, data_format="h5", num_l=4, shortest_path_length=5,
                         if_lcmp_graph=True, separate_onsite=False, target="hamiltonian", 
                         huge_structure=huge_structure, if_new_sp=False, **kwargs) #获得图类型的数据集return data

if __name__ == '__main__':
    folder = "/fs2/home/ndsim10/DeepQT/0_generate_dataset/expand_dataset/processed/1"
    data = process_worker(folder)
    print(data)
"""