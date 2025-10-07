import csv
import os
import argparse
import time
import warnings
from configparser import ConfigParser

import numpy as np
import torch
from pymatgen.core.structure import Structure

from ..train import get_graph, DeepHKernel, collate_fn


def main():
    parser = argparse.ArgumentParser(description='Predict Hamiltonian') #创建对象
    parser.add_argument('--trained_model_dir', type=str,
                        help='path of trained model')  #/fs2/home/ndsim10/example/work_dir/trained_model，增加参数
    parser.add_argument('--input_dir', type=str,
                        help='') #类似这样的数据的文件夹：/fs2/home/ndsim10/example/work_dir/dataset/processed/0/
    parser.add_argument('--output_dir', type=str,
                        help='') #类似这样的数据的文件夹：/fs2/home/ndsim10/example/work_dir/dataset/processed/0/
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA') #False
    parser.add_argument('--save_csv', action='store_true', help='Save the result for each edge in csv format') #False
    parser.add_argument(
        '--interface',
        type=str,
        default='h5',
        choices=['h5', 'npz']) #h5
    parser.add_argument('--huge_structure', type=bool, default=False, help='') #False
    args = parser.parse_args() #解析命令行参数

    old_version = False
    assert os.path.exists(os.path.join(args.trained_model_dir, 'config.ini')) #用于检查'config.ini' 文件是否存在于 args.trained_model_dir 目录下。os.path.exists接受一个路径作为参数，如果该路径存在并指向一个有效的文件或目录，就返回 True，否则返回 False。
    if os.path.exists(os.path.join(args.trained_model_dir, 'best_model.pt')) is False: #存在best_model.pt，所以为False
        old_version = True
        assert os.path.exists(os.path.join(args.trained_model_dir, 'best_model.pkl')) #在kernel.py中写入了best_model.pt
        assert os.path.exists(os.path.join(args.trained_model_dir, 'src')) #在kernel.py中复制了

    os.makedirs(args.output_dir, exist_ok=True) #递归创建目录。
    config = ConfigParser() #创建了 ConfigParser 对象，你可以使用它的方法来读取、写入和操作配置文件，比如读取配置文件中的选项值、添加新的选项、写入配置文件等。
    config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'default.ini')) #os.path.dirname(__file__) 返回当前文件的目录路径。os.path.dirname(os.path.dirname(__file__)) 返回当前文件目录的上一级目录路径，读取default.ini
    config.read(os.path.join(args.trained_model_dir, 'config.ini')) #并将上面的default.ini和config.ini配置文件的内容合并到 config 对象中，后面读取的配置会覆盖先前读取的配置。
    config.set('basic', 'save_dir', os.path.join(args.output_dir)) #将args.output_dir的值作为路径，设置到配置文件中'basic' 部分下的 'save_dir' 键对应的值中。
    config.set('basic', 'disable_cuda', str(args.disable_cuda)) #False
    config.set('basic', 'save_to_time_folder', 'False') #True，如果为True，则将每次训练的结果存入到一个时间戳文件夹中
    config.set('basic', 'tb_writer', 'False') #False
    config.set('train', 'pretrained', '') #[]
    config.set('train', 'resume', '') #[]

    # print(config)

    kernel = DeepHKernel(config)
    if old_version is False:
        checkpoint = kernel.build_model(args.trained_model_dir, old_version) #/fs2/home/ndsim10/example/work_dir/trained_model, False
    else:
        warnings.warn('You are using the trained model with an old version')
        checkpoint = torch.load(
            os.path.join(args.trained_model_dir, 'best_model.pkl'),
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
        kernel.build_model(args.trained_model_dir, old_version)
        kernel.model.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        input_dir = args.input_dir #类似/fs2/home/ndsim10/example/work_dir/dataset/processed/0这样的文件夹
        structure = Structure(np.loadtxt(os.path.join(args.input_dir, 'lat.dat')).T,
                              np.loadtxt(os.path.join(args.input_dir, 'element.dat')),
                              np.loadtxt(os.path.join(args.input_dir, 'site_positions.dat')).T,
                              coords_are_cartesian=True,
                              to_unit_cell=False)
        cart_coords = torch.tensor(structure.cart_coords, dtype=torch.get_default_dtype())
        frac_coords = torch.tensor(structure.frac_coords, dtype=torch.get_default_dtype())
        numbers = kernel.Z_to_index[torch.tensor(structure.atomic_numbers)] #Z_to_index是[第83个位置为0，其余为-1的(100,)张量]，0
        structure.lattice.matrix.setflags(write=True)
        lattice = torch.tensor(structure.lattice.matrix, dtype=torch.get_default_dtype()) #正格矢
        inv_lattice = torch.inverse(lattice) #倒格矢

        if os.path.exists(os.path.join(input_dir, 'graph.pkl')): #False
            data = torch.load(os.path.join(input_dir, 'graph.pkl'))
            print(f"Load processed graph from {os.path.join(input_dir, 'graph.pkl')}")
        else:
            begin = time.time()
            data = get_graph(cart_coords, frac_coords, numbers, 0,
                             r=kernel.config.getfloat('graph', 'radius'),
                             max_num_nbr=kernel.config.getint('graph', 'max_num_nbr'),
                             numerical_tol=1e-8, lattice=lattice, default_dtype_torch=torch.get_default_dtype(),
                             tb_folder=args.input_dir, interface=args.interface,
                             num_l=kernel.config.getint('network', 'num_l'),
                             create_from_DFT=kernel.config.getboolean('graph', 'create_from_DFT', fallback=True),
                             if_lcmp_graph=kernel.config.getboolean('graph', 'if_lcmp_graph', fallback=True),
                             separate_onsite=kernel.separate_onsite,
                             target=kernel.config.get('basic', 'target'), huge_structure=args.huge_structure) #返回每一个晶体结构的图类型的数据，即一个晶体结构对应一个图。
            torch.save(data, os.path.join(input_dir, 'graph.pkl'))
            print(f"Save processed graph to {os.path.join(input_dir, 'graph.pkl')}, cost {time.time() - begin} seconds")

        dataset_mask = kernel.make_mask([data])
        batch, subgraph = collate_fn(dataset_mask)
        sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index = subgraph

        output = kernel.model(batch.x.to(kernel.device), batch.edge_index.to(kernel.device),
                              batch.edge_attr.to(kernel.device),
                              batch.batch.to(kernel.device),
                              sub_atom_idx.to(kernel.device), sub_edge_idx.to(kernel.device),
                              sub_edge_ang.to(kernel.device), sub_index.to(kernel.device),
                              huge_structure=args.huge_structure)

        label = batch.label
        mask = batch.mask
        output = output.cpu().reshape(label.shape)

        assert label.shape == output.shape == mask.shape
        mse = torch.pow(label - output, 2)
        mae = torch.abs(label - output)

        print()
        for index_orb, orbital_single in enumerate(kernel.orbital): #字典的列表，Bi原子的轨道是：9*9=81
            if index_orb != 0:
                print('================================================================')
            print('orbital:', orbital_single)
            if kernel.spinful == False:
                print(f'mse: {torch.masked_select(mse[:, index_orb], mask[:, index_orb]).mean().item()}, '
                      f'mae: {torch.masked_select(mae[:, index_orb], mask[:, index_orb]).mean().item()}')
            else:
                for index_soc, str_soc in enumerate([
                    'left_up_real', 'left_up_imag', 'right_down_real', 'right_down_imag',
                    'right_up_real', 'right_up_imag', 'left_down_real', 'left_down_imag',
                ]):
                    if index_soc != 0:
                        print('----------------------------------------------------------------')
                    print(str_soc, ':')
                    index_out = index_orb * 8 + index_soc
                    print(f'mse: {torch.masked_select(mse[:, index_out], mask[:, index_out]).mean().item()}, '
                          f'mae: {torch.masked_select(mae[:, index_out], mask[:, index_out]).mean().item()}')

        if args.save_csv:
            edge_stru_index = torch.squeeze(batch.batch[batch.edge_index[0]]).numpy()
            edge_slices = torch.tensor(batch.__slices__['x'])[edge_stru_index].view(-1, 1)
            atom_ids = torch.squeeze(batch.edge_index.T - edge_slices).tolist()
            atomic_numbers = torch.squeeze(kernel.index_to_Z[batch.x[batch.edge_index.T]]).tolist()
            edge_infos = torch.squeeze(batch.edge_attr[:, :7].detach().cpu()).tolist()

            with open(os.path.join(kernel.config.get('basic', 'save_dir'), 'error_distance.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['index', 'atom_id', 'atomic_number', 'dist', 'atom1_x', 'atom1_y', 'atom1_z',
                                 'atom2_x', 'atom2_y', 'atom2_z']
                                + ['target'] * kernel.out_fea_len + ['pred'] * kernel.out_fea_len + [
                                    'mask'] * kernel.out_fea_len)
                for index_edge in range(batch.edge_attr.shape[0]):
                    writer.writerow([
                        index_edge,
                        atom_ids[index_edge],
                        atomic_numbers[index_edge],
                        *(edge_infos[index_edge]),
                        *(label[index_edge].tolist()),
                        *(output[index_edge].tolist()),
                        *(mask[index_edge].tolist()),
                    ])


if __name__ == '__main__':
    main()
