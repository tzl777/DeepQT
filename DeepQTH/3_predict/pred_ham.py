import json
import os
import time
import warnings
from typing import Union, List
import sys

import tqdm
from configparser import ConfigParser
import numpy as np
from pymatgen.core.structure import Structure
import torch
import torch.autograd.forward_ad as fwAD
import h5py

from deeph import get_graph, DeepHKernel, collate_fn, write_ham_h5, load_orbital_types, Rotate, dtype_dict, get_rc

#inference/C_nanotube_140,inference/C_nanotube_140,False,cpu,True,True,trained_model_dirs
def predict(input_dir: str, output_dir: str, disable_cuda: bool, device: str,
            huge_structure: bool, restore_blocks_py: bool, trained_model_dirs: Union[str, List[str]]):
    atom_num_orbital = load_orbital_types(os.path.join(input_dir, 'orbital_types.dat')) ##[1+1+3+3+5=13，36个13的列表]，即计算每个原子使用的总轨道数。
    if isinstance(trained_model_dirs, str):
        trained_model_dirs = [trained_model_dirs]
    assert isinstance(trained_model_dirs, list)
    os.makedirs(output_dir, exist_ok=True)
    predict_spinful = None

    with torch.no_grad(): #预测，不用梯度更新
        read_structure_flag = False
        if restore_blocks_py: #True
            hoppings_pred = {}
        else:
            index_model = 0
            block_without_restoration = {}
            os.makedirs(os.path.join(output_dir, 'block_without_restoration'), exist_ok=True)
        for trained_model_dir in tqdm.tqdm(trained_model_dirs):
            old_version = False
            assert os.path.exists(os.path.join(trained_model_dir, 'config.ini'))
            if os.path.exists(os.path.join(trained_model_dir, 'best_model.pt')) is False: #True is False = False
                old_version = True
                assert os.path.exists(os.path.join(trained_model_dir, 'best_model.pkl'))
                assert os.path.exists(os.path.join(trained_model_dir, 'src'))

            config = ConfigParser()
            config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'default.ini'))
            #返回当前文件所在的目录,得到当前目录的父目录,将得到的父目录路径与字符串 'default.ini' 结合起来，形成一个新的路径。/fs2/home/ndsim10/DeepH-pack/deeph/default.ini
            config.read(os.path.join(trained_model_dir, 'config.ini'))
            #/fs2/home/ndsim10/deeph/example/work_dir/trained_model/2024-01-03_14-11-23/config.ini，训练和预测可能是用不同的服务器，所以config.ini里的graph_dir、save_dir和raw_dir可能不同，但没关系
            config.set('basic', 'save_dir', os.path.join(output_dir, 'pred_ham_std')) #将预测的哈密顿量文件保存路径更改为inference/pred_ham_std
            config.set('basic', 'disable_cuda', str(disable_cuda)) #False
            config.set('basic', 'device', str(device)) #cpu
            config.set('basic', 'save_to_time_folder', 'False') #False，不需要新建时间文件夹到inference/pred_ham_std
            config.set('basic', 'tb_writer', 'False') #False，不需要tensorboard显示
            config.set('train', 'pretrained', '') #''
            config.set('train', 'resume', '') #''

            kernel = DeepHKernel(config)
            print("old_version = ",old_version)
            if old_version is False:
                checkpoint = kernel.build_model(trained_model_dir, old_version)
            else:
                warnings.warn('You are using the trained model with an old version')
                checkpoint = torch.load(
                    os.path.join(trained_model_dir, 'best_model.pkl'),
                    map_location=kernel.device
                )
                for key in ['index_to_Z', 'Z_to_index', 'spinful']:
                    if key in checkpoint:
                        setattr(kernel, key, checkpoint[key])
                if hasattr(kernel, 'index_to_Z') is False:
                    kernel.index_to_Z = torch.arange(config.getint('basic', 'max_element') + 1)
                if hasattr(kernel, 'Z_to_index') is False:
                    kernel.Z_to_index = torch.arange(config.getint('basic', 'max_element') + 1)
                if hasattr(kernel, 'spinful') is False:
                    kernel.spinful = False
                kernel.num_species = len(kernel.index_to_Z)
                print("=> load best checkpoint (epoch {})".format(checkpoint['epoch']))
                print(f"=> Atomic types: {kernel.index_to_Z.tolist()}, "
                      f"spinful: {kernel.spinful}, the number of atomic types: {len(kernel.index_to_Z)}.")
                kernel.build_model(trained_model_dir, old_version)
                kernel.model.load_state_dict(checkpoint['state_dict'])

            if predict_spinful is None:
                predict_spinful = kernel.spinful #False
            else:
                assert predict_spinful == kernel.spinful, "Different models' spinful are not compatible"

            if read_structure_flag is False: #True
                read_structure_flag = True
                structure = Structure(np.loadtxt(os.path.join(input_dir, 'lat.dat')).T,
                                      np.loadtxt(os.path.join(input_dir, 'element.dat')),
                                      np.loadtxt(os.path.join(input_dir, 'site_positions.dat')).T,
                                      coords_are_cartesian=True,
                                      to_unit_cell=False)
                cart_coords = torch.tensor(structure.cart_coords, dtype=torch.get_default_dtype())
                frac_coords = torch.tensor(structure.frac_coords, dtype=torch.get_default_dtype())
                numbers = kernel.Z_to_index[torch.tensor(structure.atomic_numbers)] #C的Z_to_index是[100个-1，第6个位置为0],structure.atomic_numbers是[原子个数*6],所以numbers是[要预测的结构的原子数*0]
                structure.lattice.matrix.setflags(write=True)
                lattice = torch.tensor(structure.lattice.matrix, dtype=torch.get_default_dtype())
                inv_lattice = torch.inverse(lattice)#晶格的逆

                if os.path.exists(os.path.join(input_dir, 'graph.pkl')): #一开始是没有graph.pkl的，需要生成，所以是False;如有已经存在处理后的图数据，则是True，加载图数据就行。
                    # data = torch.load(os.path.join(input_dir, 'graph.pkl'))
                    data = torch.load(os.path.join(input_dir, 'graph.pkl'), map_location=torch.device('cpu'))
                    print(f"Load processed graph from {os.path.join(input_dir, 'graph.pkl')}")
                else:
                    begin = time.time()
                    data = get_graph(cart_coords, frac_coords, numbers, 0,
                                     r=kernel.config.getfloat('graph', 'radius'),
                                     max_num_nbr=kernel.config.getint('graph', 'max_num_nbr'),
                                     numerical_tol=1e-8, lattice=lattice, default_dtype_torch=torch.get_default_dtype(),
                                     tb_folder=input_dir, interface="h5_rc_only",
                                     num_l=kernel.config.getint('network', 'num_l'),
                                     material_dimension=kernel.config.getint('basic', 'material_dimension'),
                                     max_path_length=kernel.config.getint('network', 'max_path_length'),
                                     create_from_DFT=kernel.config.getboolean('graph', 'create_from_DFT',
                                                                              fallback=True),
                                     if_lcmp_graph=kernel.config.getboolean('graph', 'if_lcmp_graph', fallback=True),
                                     separate_onsite=kernel.separate_onsite,
                                     target=kernel.config.get('basic', 'target'), huge_structure=huge_structure,
                                     if_new_sp=kernel.config.getboolean('graph', 'new_sp', fallback=False),
                                     )
                    print("data = {}\n".format(data))
                    # 输入：cart_coords, frac_coords, numbers,0,9.0,0,1e-8,lattice,default_dtype_torch,inference/C_nanotube_140,h5_rc_only,5,True,True,False,hamiltanian,True,False;
                    # 输出：获得图类型的数据集graph.pkl
                    torch.save(data, os.path.join(input_dir, 'graph.pkl')) #保存处理后的图数据graph.pkl，这是关键之一。
                    print(
                        f"Save processed graph to {os.path.join(input_dir, 'graph.pkl')}, cost {time.time() - begin} seconds")
                batch, subgraph = collate_fn([data])
                print("batch = ", batch)
                '''
                batch =  DataBatch(
                            x=[140], #[140个0]
                            edge_index=[2, 5180],
                            edge_attr=[5180, 10],
                            stru_id=[1],
                            atom_num_orbital=[140],
                            subgraph_dict={
                            subgraph_atom_idx=[383320, 2],
                            subgraph_edge_idx=[383320],
                            subgraph_edge_ang=[383320, 25],
                            subgraph_index=[383320]
                            },
                            batch=[140], #[140个0]
                            ptr=[2], #计算批次中每个图形数据的节点数量的辅助方法。它返回一个指针列表，ptr 指明了每个 batch 的节点的起始索引号。[0,140]
                        )
                '''
                print("subgraph_atom_idx.shape = ", subgraph[0].shape) #torch.Size([383320, 2])
                print("subgraph_edge_idx.shape = ", subgraph[1].shape) #torch.Size([383320])
                print("subgraph_edge_ang.shape = ", subgraph[2].shape) #torch.Size([383320, 25])
                print("subgraph_index.shape = ", subgraph[3].shape) #torch.Size([383320])
                # 将一个图数据列表[data]转换成一个 Batch 对象。它可以同时包含多个图数据，并提供一些方法来进行批次操作，如合并、转换等,这样可以方便地对整个数据批次进行处理。将graph_list（一个包含多个图数据对象的列表）合并成一个Batch对象。这个Batch对象允许将多个图数据作为一个整体在图神经网络中进行处理，其中每个图仍然可以被单独识别和处理。
                sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index = subgraph

            output = kernel.model(batch.x.to(kernel.device),
                                  batch.edge_index.to(kernel.device),
                                  batch.edge_attr.to(kernel.device),
                                  batch.node_paths.to(kernel.device),
                                  batch.edge_paths.to(kernel.device),
                                  batch.voronoi_values.to(kernel.device),
                                  batch.centralities.to(kernel.device),
                                  batch.batch.to(kernel.device),
                                  sub_atom_idx.to(kernel.device),
                                  sub_edge_idx.to(kernel.device),
                                  sub_edge_ang.to(kernel.device),
                                  sub_index.to(kernel.device),
                                  huge_structure=huge_structure) #输出(2016,169)
            output = output.detach().cpu()
            if restore_blocks_py:
                for index in range(batch.edge_attr.shape[0]): #1*2016
                    R = torch.round(batch.edge_attr[index, 4:7] @ inv_lattice - batch.edge_attr[index, 7:10] @ inv_lattice).int().tolist()
                    i, j = batch.edge_index[:, index]
                    key_term = (*R, i.item() + 1, j.item() + 1) #得到每一个边的原子i的key_term,2016
                    key_term = str(list(key_term))
                    for index_orbital, orbital_dict in enumerate(kernel.orbital): #遍历所有169个轨道
                        if f'{kernel.index_to_Z[numbers[i]].item()} {kernel.index_to_Z[numbers[j]].item()}' not in orbital_dict:
                            continue
                        orbital_i, orbital_j = orbital_dict[f'{kernel.index_to_Z[numbers[i]].item()} {kernel.index_to_Z[numbers[j]].item()}'] #预测的碳原子序数6，6

                        if not key_term in hoppings_pred: #一开始这里hoppings_pred={}
                            if kernel.spinful:
                                hoppings_pred[key_term] = np.full((2 * atom_num_orbital[i], 2 * atom_num_orbital[j]), np.nan + np.nan * (1j))
                            else:
                                hoppings_pred[key_term] = np.full((atom_num_orbital[i], atom_num_orbital[j]), np.nan) #生成存放原子i和原子j的哈密顿量矩阵13*13，元素为nan
                        if kernel.spinful:
                            hoppings_pred[key_term][orbital_i, orbital_j] = output[index][index_orbital * 8 + 0] + output[index][index_orbital * 8 + 1] * 1j
                            hoppings_pred[key_term][atom_num_orbital[i] + orbital_i, atom_num_orbital[j] + orbital_j] = output[index][index_orbital * 8 + 2] + output[index][index_orbital * 8 + 3] * 1j
                            hoppings_pred[key_term][orbital_i, atom_num_orbital[j] + orbital_j] = output[index][index_orbital * 8 + 4] + output[index][index_orbital * 8 + 5] * 1j
                            hoppings_pred[key_term][atom_num_orbital[i] + orbital_i, orbital_j] = output[index][index_orbital * 8 + 6] + output[index][index_orbital * 8 + 7] * 1j
                        else:
                            hoppings_pred[key_term][orbital_i, orbital_j] = output[index][index_orbital]  # about output shape w/ or w/o soc, see graph.py line 164, and kernel.py line 281.
                            #保存预测的哈密顿量的跳跃项，可能包含在位项？
            else:
                if 'edge_index' not in block_without_restoration:
                    assert index_model == 0
                    block_without_restoration['edge_index'] = batch.edge_index
                    block_without_restoration['edge_attr'] = batch.edge_attr
                block_without_restoration[f'output_{index_model}'] = output.numpy()
                with open(os.path.join(output_dir, 'block_without_restoration', f'orbital_{index_model}.json'), 'w') as orbital_f:
                    json.dump(kernel.orbital, orbital_f, indent=4)
                index_model += 1
            sys.stdout = sys.stdout.terminal #用于重定向标准输出和标准错误流。重新设置为原始的终端输出，使它们再次输出到标准输出和标准错误流。之前在kernel中输出被打印到控制台并存入到log文件中了。
            sys.stderr = sys.stderr.terminal

        if restore_blocks_py:
            for hamiltonian in hoppings_pred.values():
                assert np.all(np.isnan(hamiltonian) == False) #np.isnan判断hamiltonian中的每个元素是否为NaN。np.all() 则检查整个矩阵是否全部为 True。这个断言语句的意思是确保预测的hamiltonian中没有NaN元素。如果存在任何一个元素为NaN，断言将会失败并引发AssertionError。
            write_ham_h5(hoppings_pred, path=os.path.join(output_dir, 'rh_pred.h5')) #保存预测的哈密顿量
        else:
            block_without_restoration['num_model'] = index_model
            write_ham_h5(block_without_restoration, path=os.path.join(output_dir, 'block_without_restoration', 'block_without_restoration.h5'))
        with open(os.path.join(output_dir, "info.json"), 'w') as info_f:
            json.dump({
                "isspinful": predict_spinful
            }, info_f) #使用json.dump()函数将字典转换为JSON格式并写入到info_f文件流中。


def predict_with_grad(input_dir: str, output_dir: str, disable_cuda: bool, device: str,
                      huge_structure: bool, trained_model_dirs: Union[str, List[str]]): #Union[str, List[str]]这个声明表明trained_model_dirs是一个变量，它可以是一个字符串，也可以是字符串组成的列表。
    atom_num_orbital, orbital_types = load_orbital_types(os.path.join(input_dir, 'orbital_types.dat'), return_orbital_types=True) #返回[450个13的列表]和[[0,0,1,1,2],...[0,0,1,1,2]]，即每个原子的使用的总轨道数和轨道类型；

    if isinstance(trained_model_dirs, str):
        trained_model_dirs = [trained_model_dirs]
    assert isinstance(trained_model_dirs, list)
    os.makedirs(output_dir, exist_ok=True)
    predict_spinful = None

    read_structure_flag = False
    rh_dict = {}
    hamiltonians_pred = {}
    hamiltonians_grad_pred = {}

    for trained_model_dir in tqdm.tqdm(trained_model_dirs):
        old_version = False
        assert os.path.exists(os.path.join(trained_model_dir, 'config.ini'))
        if os.path.exists(os.path.join(trained_model_dir, 'best_model.pt')) is False:
            old_version = True
            assert os.path.exists(os.path.join(trained_model_dir, 'best_model.pkl'))
            assert os.path.exists(os.path.join(trained_model_dir, 'src'))

        config = ConfigParser()
        config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'default.ini'))
        config.read(os.path.join(trained_model_dir, 'config.ini'))
        config.set('basic', 'save_dir', os.path.join(output_dir, 'pred_ham_std')) #预测的哈密顿量输出文件夹
        config.set('basic', 'disable_cuda', str(disable_cuda)) #False
        config.set('basic', 'device', str(device)) #cpu
        config.set('basic', 'save_to_time_folder', 'False') #
        config.set('basic', 'tb_writer', 'False') #True
        config.set('train', 'pretrained', '') #[]
        config.set('train', 'resume', '') #[]

        kernel = DeepHKernel(config)
        if old_version is False:
            checkpoint = kernel.build_model(trained_model_dir, old_version)
        else:
            warnings.warn('You are using the trained model with an old version')
            checkpoint = torch.load(
                os.path.join(trained_model_dir, 'best_model.pkl'),
                map_location=kernel.device
            )
            for key in ['index_to_Z', 'Z_to_index', 'spinful']:
                if key in checkpoint:
                    setattr(kernel, key, checkpoint[key])
            if hasattr(kernel, 'index_to_Z') is False:
                kernel.index_to_Z = torch.arange(config.getint('basic', 'max_element') + 1)
            if hasattr(kernel, 'Z_to_index') is False:
                kernel.Z_to_index = torch.arange(config.getint('basic', 'max_element') + 1)
            if hasattr(kernel, 'spinful') is False:
                kernel.spinful = False
            kernel.num_species = len(kernel.index_to_Z)
            print("=> load best checkpoint (epoch {})".format(checkpoint['epoch']))
            print(f"=> Atomic types: {kernel.index_to_Z.tolist()}, "
                  f"spinful: {kernel.spinful}, the number of atomic types: {len(kernel.index_to_Z)}.")
            kernel.build_model(trained_model_dir, old_version)
            kernel.model.load_state_dict(checkpoint['state_dict'])

        if predict_spinful is None:
            predict_spinful = kernel.spinful #False
        else:
            assert predict_spinful == kernel.spinful, "Different models' spinful are not compatible"

        if read_structure_flag is False:
            read_structure_flag = True
            structure = Structure(np.loadtxt(os.path.join(input_dir, 'lat.dat')).T,
                                  np.loadtxt(os.path.join(input_dir, 'element.dat')),
                                  np.loadtxt(os.path.join(input_dir, 'site_positions.dat')).T,
                                  coords_are_cartesian=True,
                                  to_unit_cell=False)
            cart_coords = torch.tensor(structure.cart_coords, dtype=torch.get_default_dtype(), requires_grad=True, device=kernel.device)
            num_atom = cart_coords.shape[0]
            frac_coords = torch.tensor(structure.frac_coords, dtype=torch.get_default_dtype())
            numbers = kernel.Z_to_index[torch.tensor(structure.atomic_numbers)]
            structure.lattice.matrix.setflags(write=True)
            lattice = torch.tensor(structure.lattice.matrix, dtype=torch.get_default_dtype())
            inv_lattice = torch.inverse(lattice)

            fid_rc = get_rc(input_dir, None, radius=-1, create_from_DFT=True, if_require_grad=True, cart_coords=cart_coords) #得到截断半径内的排序后的原子对的3*3的单位局域坐标，存入rc.h5文件

            assert kernel.config.getboolean('graph', 'new_sp', fallback=False)
            data = get_graph(cart_coords.to(kernel.device), frac_coords, numbers, 0,
                             r=kernel.config.getfloat('graph', 'radius'),
                             max_num_nbr=kernel.config.getint('graph', 'max_num_nbr'),
                             numerical_tol=1e-8, lattice=lattice, default_dtype_torch=torch.get_default_dtype(),
                             tb_folder=input_dir, interface="h5_rc_only",
                             num_l=kernel.config.getint('network', 'num_l'),
                             create_from_DFT=kernel.config.getboolean('graph', 'create_from_DFT', fallback=True),
                             if_lcmp_graph=kernel.config.getboolean('graph', 'if_lcmp_graph', fallback=True),
                             separate_onsite=kernel.separate_onsite,
                             target=kernel.config.get('basic', 'target'), huge_structure=huge_structure,
                             if_new_sp=True, if_require_grad=True, fid_rc=fid_rc) #获得图类型的数据集
            batch, subgraph = collate_fn([data]) #collate_fn参数是合并一个样本列表，形成一个小批量的张量。当从图样式数据集批量加载时使用。用于对每个batch_size的数据进行处理和组合。返回batch和batch对应的子图。
            sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index = subgraph

            torch_dtype, torch_dtype_real, torch_dtype_complex = dtype_dict[torch.get_default_dtype()]
            rotate_kernel = Rotate(torch_dtype, torch_dtype_real=torch_dtype_real,
                                   torch_dtype_complex=torch_dtype_complex,
                                   device=kernel.device, spinful=kernel.spinful) #将哈密顿量旋转，并转为与key[0, 0, 0, 1, 1]的9*9的小哈密顿量相同的数据类型

        output = kernel.model(batch.x, batch.edge_index.to(kernel.device),
                              batch.edge_attr,
                              batch.batch.to(kernel.device),
                              sub_atom_idx.to(kernel.device), sub_edge_idx.to(kernel.device),
                              sub_edge_ang, sub_index.to(kernel.device),
                              huge_structure=huge_structure)

        index_for_matrix_block_real_dict = {}  # key is atomic number pair
        if kernel.spinful:
            index_for_matrix_block_imag_dict = {}  # key is atomic number pair

        for index in range(batch.edge_attr.shape[0]):
            R = torch.round(batch.edge_attr[index, 4:7].cpu() @ inv_lattice - batch.edge_attr[index, 7:10].cpu() @ inv_lattice).int().tolist()
            i, j = batch.edge_index[:, index]
            key_tensor = torch.tensor([*R, i, j])
            numbers_pair = (kernel.index_to_Z[numbers[i]].item(), kernel.index_to_Z[numbers[j]].item())
            if numbers_pair not in index_for_matrix_block_real_dict:
                if not kernel.spinful:
                    index_for_matrix_block_real = torch.full((atom_num_orbital[i], atom_num_orbital[j]), -1) #用-1填充形状为(atom_num_orbital[i], atom_num_orbital[j])的矩阵
                else:
                    index_for_matrix_block_real = torch.full((2 * atom_num_orbital[i], 2 * atom_num_orbital[j]), -1)
                    index_for_matrix_block_imag = torch.full((2 * atom_num_orbital[i], 2 * atom_num_orbital[j]), -1)
                for index_orbital, orbital_dict in enumerate(kernel.orbital):#遍历轨道的字典列表
                    if f'{kernel.index_to_Z[numbers[i]].item()} {kernel.index_to_Z[numbers[j]].item()}' not in orbital_dict:
                        continue
                    orbital_i, orbital_j = orbital_dict[f'{kernel.index_to_Z[numbers[i]].item()} {kernel.index_to_Z[numbers[j]].item()}']
                    if not kernel.spinful:
                        index_for_matrix_block_real[orbital_i, orbital_j] = index_orbital
                    else:
                        index_for_matrix_block_real[orbital_i, orbital_j] = index_orbital * 8 + 0
                        index_for_matrix_block_imag[orbital_i, orbital_j] = index_orbital * 8 + 1
                        index_for_matrix_block_real[atom_num_orbital[i] + orbital_i, atom_num_orbital[j] + orbital_j] = index_orbital * 8 + 2
                        index_for_matrix_block_imag[atom_num_orbital[i] + orbital_i, atom_num_orbital[j] + orbital_j] = index_orbital * 8 + 3
                        index_for_matrix_block_real[orbital_i, atom_num_orbital[j] + orbital_j] = index_orbital * 8 + 4
                        index_for_matrix_block_imag[orbital_i, atom_num_orbital[j] + orbital_j] = index_orbital * 8 + 5
                        index_for_matrix_block_real[atom_num_orbital[i] + orbital_i, orbital_j] = index_orbital * 8 + 6
                        index_for_matrix_block_imag[atom_num_orbital[i] + orbital_i, orbital_j] = index_orbital * 8 + 7
                assert torch.all(index_for_matrix_block_real != -1), 'json string "orbital" should be complete for Hamiltonian grad'
                if kernel.spinful:
                    assert torch.all(index_for_matrix_block_imag != -1), 'json string "orbital" should be complete for Hamiltonian grad' #测试 input 中的所有元素是否评估为 True 。

                index_for_matrix_block_real_dict[numbers_pair] = index_for_matrix_block_real
                if kernel.spinful:
                    index_for_matrix_block_imag_dict[numbers_pair] = index_for_matrix_block_imag
            else:
                index_for_matrix_block_real = index_for_matrix_block_real_dict[numbers_pair]
                if kernel.spinful:
                    index_for_matrix_block_imag = index_for_matrix_block_imag_dict[numbers_pair]

            if not kernel.spinful:
                rh_dict[key_tensor] = output[index][index_for_matrix_block_real]
            else:
                rh_dict[key_tensor] = output[index][index_for_matrix_block_real] + 1j * output[index][index_for_matrix_block_imag]

        sys.stdout = sys.stdout.terminal
        sys.stderr = sys.stderr.terminal

    print("=> Hamiltonian has been predicted, calculate the grad...")
    for key_tensor, rotated_hamiltonian in tqdm.tqdm(rh_dict.items()):
        atom_i = key_tensor[3]
        atom_j = key_tensor[4]
        assert atom_i >= 0
        assert atom_i < num_atom
        assert atom_j >= 0
        assert atom_j < num_atom
        key_str = str(list([key_tensor[0].item(), key_tensor[1].item(), key_tensor[2].item(), atom_i.item() + 1, atom_j.item() + 1]))
        assert key_str in fid_rc, f'Can not found the key "{key_str}" in rc.h5'
        # rotation_matrix = torch.tensor(fid_rc[key_str], dtype=torch_dtype_real, device=kernel.device).T
        rotation_matrix = fid_rc[key_str].T
        hamiltonian = rotate_kernel.rotate_openmx_H(rotated_hamiltonian, rotation_matrix, orbital_types[atom_i], orbital_types[atom_j])
        hamiltonians_pred[key_str] = hamiltonian.detach().cpu()
        assert kernel.spinful is False  # 检查soc时是否正确
        assert len(hamiltonian.shape) == 2
        dim_1, dim_2 = hamiltonian.shape[:]
        assert key_str not in hamiltonians_grad_pred
        if not kernel.spinful:
            hamiltonians_grad_pred[key_str] = np.full((dim_1, dim_2, num_atom, 3), np.nan) #创建一个给定形状和类型的数组，填充值为给定的标量值。
        else:
            hamiltonians_grad_pred[key_str] = np.full((2 * dim_1, 2 * dim_2, num_atom, 3), np.nan + 1j * np.nan)

    write_ham_h5(hamiltonians_pred, path=os.path.join(output_dir, 'hamiltonians_pred.h5'))
    write_ham_h5(hamiltonians_grad_pred, path=os.path.join(output_dir, 'hamiltonians_grad_pred.h5'))
    with open(os.path.join(output_dir, "info.json"), 'w') as info_f:
        json.dump({
            "isspinful": predict_spinful
        }, info_f)
    fid_rc.close()

