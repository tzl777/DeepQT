import collections
import itertools
import os
import json
import warnings
import math
import sys
import torch
import torch_geometric
from torch_geometric.data import Data, Batch
import numpy as np
import h5py
import networkx as nx
from spherical_harmonics_basis import get_spherical_from_cartesian, SphericalHarmonics, _spherical_harmonics

from pymatgen.core.structure import Structure

def get_graph(cart_coords, frac_coords, numbers, stru_id, radius, material_dimension, numerical_tol, lattice,
              default_dtype_torch, tb_folder, data_format, num_l, shortest_path_length, if_lcmp_graph,
              target='hamiltonian', huge_structure=False, **kwargs):

    assert target in ['hamiltonian', 'phiVdphi']
    assert tb_folder is not None
    if data_format == 'h5': #True
        key_atom_list = [[] for _ in range(len(numbers))]
        edge_idx_target, edge_fea, edge_idx_source = [], [], []
        if if_lcmp_graph: #True
            atom_idx_connect, edge_idx_connect = [], []
            edge_idx_connect_cursor = 0
        fid = h5py.File(os.path.join(tb_folder, 'rc.h5'), 'r')
        for k in fid.keys():
            key = json.loads(k)
            key_tensor = torch.tensor([key[0], key[1], key[2], key[3], key[4]])
            key_atom_list[key[3]].append(key_tensor)
        fid.close()
    
        for index_first, (cart_coord, keys_tensor) in enumerate(zip(cart_coords, key_atom_list)): 
            keys_tensor = torch.stack(keys_tensor)
            cart_coords_j = cart_coords[keys_tensor[:, 4]] + keys_tensor[:, :3].type(default_dtype_torch).to(cart_coords.device) @ lattice.to(cart_coords.device)
            dist = torch.norm(cart_coords_j - cart_coord[None, :], dim=1)
            len_nn = keys_tensor.shape[0]
            # print(index_first, len_nn)
            edge_idx_source.extend([index_first] * len_nn)
            edge_idx_target.extend(keys_tensor[:, 4].tolist())
            edge_fea_single = torch.cat([dist.view(-1, 1), cart_coord.view(1, 3).expand(len_nn, 3)], dim=-1)
            edge_fea_single = torch.cat([edge_fea_single, cart_coords_j, cart_coords[keys_tensor[:, 4]]], dim=-1) 
            edge_fea.append(edge_fea_single)

            if if_lcmp_graph: #True
                atom_idx_connect.append(keys_tensor[:, 4])
                edge_idx_connect.append(range(edge_idx_connect_cursor, edge_idx_connect_cursor + len_nn))
                edge_idx_connect_cursor += len_nn

        edge_fea = torch.cat(edge_fea).type(default_dtype_torch)
        edge_idx = torch.stack([torch.LongTensor(edge_idx_source), torch.LongTensor(edge_idx_target)])

    else:
        raise NotImplemented

    if tb_folder is not None:

        if data_format == 'h5' or data_format == 'h5_rc_only': #train h5，predict h5_rc_only
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
                else:
                    raise ValueError('Unknown prediction target: {}'.format(target))
                read_terms_dict = {}
                for read_file, graph_key in zip(read_file_list, graph_key_list):
                    read_terms = {}
                    fid = h5py.File(os.path.join(tb_folder, read_file), 'r')
                    for k, v in fid.items():
                        key = json.loads(k)
                        key = (key[0], key[1], key[2], key[3], key[4])
                        if spinful:
                            num_orbital_row = atom_num_orbital[key[3]]
                            num_orbital_column = atom_num_orbital[key[4]]
                            # soc block order:
                            # 1 3
                            # 4 2
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
                            read_terms[key] = torch.tensor(v[...], dtype=default_dtype_torch)
                    read_terms_dict[graph_key] = read_terms
                    fid.close()
           
            local_rotation_dict = {}
            fid = h5py.File(os.path.join(tb_folder, 'rc.h5'), 'r')
            for k, v in fid.items():
                key = json.loads(k)
                key = (key[0], key[1], key[2], key[3], key[4])
                local_rotation_dict[key] = torch.tensor(v[...], dtype=default_dtype_torch)
            fid.close()

            max_num_orbital = max(atom_num_orbital)
        else:
            raise ValueError(f'Unknown data format: {data_format}')

        if data_format == 'h5_rc_only':
            local_rotation = []
        else:
            term_dict = {}
            onsite_term_dict = {}
            term_mask = torch.zeros(edge_fea.shape[0], dtype=torch.bool)
            for graph_key in graph_key_list:
                if spinful:
                    term_dict[graph_key] = torch.full([edge_fea.shape[0], max_num_orbital, max_num_orbital, 8],
                                                      np.nan, dtype=default_dtype_torch)
                else:
                    term_dict[graph_key] = torch.full([edge_fea.shape[0], max_num_orbital, max_num_orbital],
                                                          np.nan, dtype=default_dtype_torch)

            local_rotation = []
            inv_lattice = torch.inverse(lattice).type(default_dtype_torch)
            for index_edge in range(edge_fea.shape[0]):
                R = torch.round(edge_fea[index_edge, 4:7].cpu() @ inv_lattice - edge_fea[index_edge, 7:10].cpu() @ inv_lattice).int().tolist()
                i, j = edge_idx[:, index_edge]
                key_term = (*R, i.item(), j.item())
                if data_format == 'h5_rc_only':
                    local_rotation.append(local_rotation_dict[key_term])
                else:
                    if key_term in read_terms_dict[graph_key_list[0]]:
                        for graph_key in graph_key_list:
                            term_mask[index_edge] = True
                            if spinful:
                                term_dict[graph_key][index_edge, :atom_num_orbital[i], :atom_num_orbital[j], :] = read_terms_dict[graph_key][key_term]
                            else:
                                term_dict[graph_key][index_edge, :atom_num_orbital[i], :atom_num_orbital[j]] = read_terms_dict[graph_key][key_term] 
                        local_rotation.append(local_rotation_dict[key_term])
                    else:
                        print("key 2:", key_term, type(key_term), type(key_term[0], type(key_term[3])))
                        key_ham = read_terms_dict[graph_key_list[0]]
                        print(type(key_ham), key_ham[0], type(key_ham[0].key[0]), type(key_ham[0].key[3]))
                        
                        raise NotImplementedError(
                            "Not yet have support for graph radius including hopping without calculation")

            if if_lcmp_graph: #True
                local_rotation = torch.stack(local_rotation, dim=0)
                assert local_rotation.shape[0] == edge_fea.shape[0]
                r_vec = edge_fea[:, 1:4] - edge_fea[:, 4:7]
                r_vec = r_vec.unsqueeze(1)

                if huge_structure is False:
                    #torch.Size([2664, 1, 1, 3])，torch.Size([1, 2664, 3, 3]) = torch.Size([2664, 2664, 1, 3])=torch.Size([7096896, 3])
                    r_vec = torch.matmul(r_vec[:, None, :, :], local_rotation[None, :, :, :].to(r_vec.device)).reshape(-1, 3) 
                    r_vec_sp = get_spherical_from_cartesian(r_vec) 
                    sph_harm_func = SphericalHarmonics()
                    angular_expansion = []
                    for l in range(num_l):
                        angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1]))
                    angular_expansion = torch.cat(angular_expansion, dim=-1).reshape(edge_fea.shape[0], edge_fea.shape[0], -1) 
                subgraph_atom_idx_list = []
                subgraph_edge_idx_list = []
                subgraph_edge_ang_list = []
                subgraph_index = []
                index_cursor = 0

                for index in range(edge_fea.shape[0]):
                    i, j = edge_idx[:, index]
                    subgraph_atom_idx = torch.stack([i.repeat(len(atom_idx_connect[i])), atom_idx_connect[i]]).T
                    subgraph_edge_idx = torch.LongTensor(edge_idx_connect[i])
                    # print(subgraph_atom_idx.shape) #torch.Size([37, 2])
                    # print(subgraph_edge_idx.shape) #torch.Size([37])

                    if huge_structure:
                        r_vec_tmp = torch.matmul(r_vec[subgraph_edge_idx, :, :], local_rotation[index, :, :].to(r_vec.device)).reshape(-1, 3)
                        r_vec_sp = get_spherical_from_cartesian(r_vec_tmp)
                        sph_harm_func = SphericalHarmonics()
                        angular_expansion = []
                        for l in range(num_l):
                            angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1])) #25*53*1
                        subgraph_edge_ang = torch.cat(angular_expansion, dim=-1).reshape(-1, num_l ** 2) #53*25
                    else:
                        subgraph_edge_ang = angular_expansion[subgraph_edge_idx, index, :]
                    subgraph_atom_idx_list.append(subgraph_atom_idx)
                    subgraph_edge_idx_list.append(subgraph_edge_idx)
                    subgraph_edge_ang_list.append(subgraph_edge_ang)
                    subgraph_index += [index_cursor] * len(atom_idx_connect[i])
                    index_cursor += 1

                    subgraph_atom_idx = torch.stack([j.repeat(len(atom_idx_connect[j])), atom_idx_connect[j]]).T
                    subgraph_edge_idx = torch.LongTensor(edge_idx_connect[j])
                    if huge_structure:
                        r_vec_tmp = torch.matmul(r_vec[subgraph_edge_idx, :, :], local_rotation[index, :, :].to(r_vec.device)).reshape(-1, 3)
                        r_vec_sp = get_spherical_from_cartesian(r_vec_tmp)
                        sph_harm_func = SphericalHarmonics()
                        angular_expansion = []
                        for l in range(num_l):
                            angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1]))
                        subgraph_edge_ang = torch.cat(angular_expansion, dim=-1).reshape(-1, num_l ** 2)
                    else:
                        subgraph_edge_ang = angular_expansion[subgraph_edge_idx, index, :]
                    subgraph_atom_idx_list.append(subgraph_atom_idx)
                    subgraph_edge_idx_list.append(subgraph_edge_idx)
                    subgraph_edge_ang_list.append(subgraph_edge_ang)
                    subgraph_index += [index_cursor] * len(atom_idx_connect[j])
                    index_cursor += 1

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

        if data_format == 'h5_rc_only':
            import networkx as nx
            G = nx.Graph()
            for i, (x, coords) in enumerate(zip(numbers, cart_coords)):
                G.add_node(i, x=x.item(), position=coords.numpy())
            for i, (src, dst) in enumerate(edge_idx.t()):
                G.add_edge(src.item(), dst.item(), attr=edge_fea[i].numpy())
            G.graph['lattice'] = lattice.numpy()
            # print("number_of_nodes = ", G.number_of_nodes()) #72
            # print("number_of_edges = ", G.number_of_edges()) #1368
            current_sys_path = "/fs2/home/ndsim10/DeepQT/DeepQTH/2_train/deepqth"
            if current_sys_path not in sys.path:
                sys.path.insert(0, current_sys_path)
            from functional import shortest_path_distance, cal_voronoi_and_centrality
            node_paths, edge_paths = shortest_path_distance(G, shortest_path_length)
            # print(node_paths.shape) #torch.Size([1, 72, 72, 5])
            # print(edge_paths.shape) #torch.Size([1, 72, 72, 4])
            voronoi_values, centralities = cal_voronoi_and_centrality(cart_coords, lattice, material_dimension)

            data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id, cart_coords=cart_coords, lattice=lattice,
                        voronoi_values=voronoi_values, centralities=centralities, node_paths=node_paths, edge_paths=edge_paths,
                        term_mask=None, term_real=None, onsite_term_real=None,
                        atom_num_orbital=atom_num_orbital,
                        subgraph_dict=subgraph,
                        **kwargs)
        else:
            import networkx as nx
            G = nx.Graph()
            for i, (x, coords) in enumerate(zip(numbers, cart_coords)):
                G.add_node(i, x=x.item(), position=coords.numpy())
            for i, (src, dst) in enumerate(edge_idx.t()):
                G.add_edge(src.item(), dst.item(), attr=edge_fea[i].numpy())
            G.graph['lattice'] = lattice.numpy()
            # print("number_of_nodes = ", G.number_of_nodes()) #72
            # print("number_of_edges = ", G.number_of_edges()) #1368
            current_sys_path = "/fs2/home/ndsim10/DeepQT/DeepQTH/2_train/deepqth"
            if current_sys_path not in sys.path:
                sys.path.insert(0, current_sys_path)
            from functional import shortest_path_distance, cal_voronoi_and_centrality
            node_paths, edge_paths = shortest_path_distance(G, shortest_path_length)
            # print(node_paths.shape) #torch.Size([1, 72, 72, 5])
            # print(edge_paths.shape) #torch.Size([1, 72, 72, 4])
            voronoi_values, centralities = cal_voronoi_and_centrality(cart_coords, lattice, material_dimension)
            # print(voronoi_values.shape)
            # print(centralities.shape)
            data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id, cart_coords=cart_coords, lattice=lattice,
                        voronoi_values=voronoi_values, centralities=centralities, term_mask=term_mask, node_paths=node_paths,
                        edge_paths=edge_paths,
                        **term_dict, **onsite_term_dict,
                        atom_num_orbital=atom_num_orbital,
                        subgraph_dict=subgraph,
                        spinful=spinful,
                        **kwargs)

    else:
        data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id, cart_coords=cart_coords, lattice=lattice, **kwargs)
    return data

"""
### Debugging
def process_worker(folder, **kwargs):
    default_dtype_torch = torch.get_default_dtype()
    stru_id = os.path.split(folder)[-1]

    structure = Structure(np.loadtxt(os.path.join(folder, 'lat.dat')),
                          np.loadtxt(os.path.join(folder, 'element.dat')),
                          np.loadtxt(os.path.join(folder, 'site_positions.dat')),
                          coords_are_cartesian=True,
                          to_unit_cell=False)

    cart_coords = torch.tensor(structure.cart_coords, dtype=default_dtype_torch)
    frac_coords = torch.tensor(structure.frac_coords, dtype=default_dtype_torch)
    numbers = torch.tensor(structure.atomic_numbers) 
    structure.lattice.matrix.setflags(write=True)
    lattice = torch.tensor(structure.lattice.matrix,dtype=default_dtype_torch)
    huge_structure = False
    return get_graph(cart_coords, frac_coords, numbers, stru_id, radius=7.0, material_dimension=2,
                         numerical_tol=1e-8, lattice=lattice, default_dtype_torch=default_dtype_torch,
                         tb_folder=folder, data_format="h5", num_l=4, shortest_path_length=5,
                         if_lcmp_graph=True, target="hamiltonian", 
                         huge_structure=huge_structure, **kwargs)

if __name__ == '__main__':
    folder = "/fs2/home/ndsim10/DeepQT/0_generate_dataset/expand_dataset/processed/1"
    data = process_worker(folder)
    print(data)
"""