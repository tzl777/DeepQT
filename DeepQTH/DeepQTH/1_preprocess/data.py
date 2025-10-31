import warnings
import os
import time
import tqdm
import json
from pymatgen.core.structure import Structure
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from pathos.multiprocessing import ProcessingPool as Pool

from graph import get_graph

def generate_orbital(atom_list, unique_orbital_basis_number):
    # atom_list = dataset.info['index_to_Z']
    # unique_orbital_basis_number = np.unique(dataset[0].atom_num_orbital)
    num_element = len(atom_list)
    assert num_element == len(unique_orbital_basis_number), "Unique orbital bases must be the same as unique atomic numbers."
    
    atomic_number = []
    num_orbitals = []
    assert num_element > 0, "Number of atomic types should be greater than 0."
    for index_element in range(num_element):
        atom_number = int(atom_list[index_element])
        print(f"Atomic number {index_element}: {atom_number}")
        assert atom_number > 0, "Atomic number should be greater than 0."
        orbital_number = int(unique_orbital_basis_number[index_element])
        print(f"Orbital basis number {index_element}: {orbital_number}")
        assert orbital_number > 0, "Orbital basis number should be greater than 0."
        atomic_number.append(atom_number)
        num_orbitals.append(orbital_number)
    
    orbital_str = '['
    first_flag = True
    for ele_i, ele_j in ((ele_i, ele_j) for ele_i in range(num_element) for ele_j in range(num_element)):
        for orb_i, orb_j in ((orb_i, orb_j) for orb_i in range(num_orbitals[ele_i]) for orb_j in range(num_orbitals[ele_j])):
            if first_flag:
                orbital_str += '{'
                first_flag = False
            else:
                orbital_str += ', {'
            orbital_str += f'"{atomic_number[ele_i]} {atomic_number[ele_j]}": [{orb_i}, {orb_j}]}}'
    orbital_str += ']'
    return orbital_str

class HData(InMemoryDataset):
    def __init__(self, config: dict, default_dtype_torch, transform=None, pre_transform=None, pre_filter=None): 

        if config is None:
            raise ValueError("config must be provided")
        if default_dtype_torch is None:
            raise ValueError("default_dtype_torch must be provided")

        self.processed_data_dir = config['basic']['processed_data_dir']
        self.graph_dir = config['basic']['graph_dir']
        self.data_format = config['basic']['data_format']
        self.target = config['basic']['target']
        self.material_dimension = config['basic']['material_dimension']
        # self.multiprocessing = config['basic']['multiprocessing']
        self.multiprocessing = 0  # Single-process processing of graph data
        self.radius = config['graph']['radius']
        self.num_l = config['graph']['num_l']
        self.if_lcmp_graph = config['graph']['if_lcmp_graph']
        self.shortest_path_length = config['graph']['shortest_path_length']
        self.default_dtype_torch = default_dtype_torch
        self.transform = transform #None
        self.pre_transform = pre_transform #None
        self.pre_filter = pre_filter #None
        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None

        if self.if_lcmp_graph: #True
            lcmp_str = f'{self.num_l}l'
        else:
            lcmp_str = 'WithoutLCMP'

        if self.target == 'hamiltonian':
            title = 'HGraph'
        else:
            raise ValueError('Unknown prediction target: {}'.format(self.target))
            
        self.graph_file_name = f'{title}-{self.data_format}-{lcmp_str}.pkl'
        self.data_file = os.path.join(self.graph_dir, self.graph_file_name)
        os.makedirs(self.graph_dir, exist_ok=True)
        
        print(f'Graph data file: {self.graph_file_name}')
        if os.path.exists(self.data_file):
            print("Use existing graph data file")
            self.load()
        else:
            super(HData, self).__init__(self.processed_data_dir, transform, pre_transform, pre_filter)

        
        begin = time.time()
        try:
            loaded_data = torch.load(self.data_file)
        except AttributeError:
            raise RuntimeError('Error in loading graph data file, try to delete it and generate the graph file with the current version of PyG')
        max_element = -1
        if len(loaded_data) == 2:
            warnings.warn('You are using the graph data file with an old version')
            self.data, self.slices = loaded_data
            self.info = {
                "spinful": False,
                "index_to_Z": torch.arange(max_element + 1),
                "Z_to_index": torch.arange(max_element + 1),
            }
        elif len(loaded_data) == 3:
            self.data, self.slices, tmp = loaded_data 
            if isinstance(tmp, dict):
                self.info = tmp
                print(f"Atomic types: {self.info['index_to_Z'].tolist()}") 
                print(f"Atomic types vectors: {self.info['Z_to_index'].tolist()}")
            else:
                warnings.warn('You are using an old version of the graph data file')
                self.info = {
                    "spinful": tmp,
                    "index_to_Z": torch.arange(max_element + 1),
                    "Z_to_index": torch.arange(max_element + 1),
                }
        else:
            raise RuntimeError(f'Unexpected format in saved graph file: found {len(loaded_data)} elements')
            
        print(f'Finish loading the processed {len(self)} structures (spinful: {self.info["spinful"]}, '
              f'the number of atomic types: {len(self.info["index_to_Z"])}), cost {time.time() - begin:.0f} seconds')
        
    @property
    def processed_file_names(self):
        pass
        return []

    def load(self):
        if os.path.exists(self.data_file):
            print("Loading existing dataset:", self.data_file)
            self.loaded_data = torch.load(self.data_file)
        else:
            raise RuntimeError(f"Processed file {self.data_file} not found, please run process()")
    
    def process(self):
        print('Process new data file......')
        
        begin = time.time()
        folder_list = []
        for root, dirs, files in os.walk(self.processed_data_dir):
            if self.data_format == 'h5' and 'rc.h5' in files:
                folder_list.append(root)
        folder_list = sorted(folder_list)
        assert len(folder_list) != 0, "Can not find any structure"


        if self.multiprocessing == 0:
            print(f'Use multiprocessing (nodes = num_processors x num_threads = 1 x {torch.get_num_threads()})')
            data_list = [self.process_worker(folder) for folder in tqdm.tqdm(folder_list, ncols=80, leave=False, position=0)]
        else:
            pool_dict = {} if self.multiprocessing < 0 else {'nodes': self.multiprocessing}
            torch_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            with Pool(**pool_dict) as pool:
                nodes = pool.nodes
                print(f'Use multiprocessing (nodes = num_processors x num_threads = {nodes} x {torch.get_num_threads()})')
                data_list = list(tqdm.tqdm(pool.imap(self.process_worker, folder_list), total=len(folder_list), ncols=80, leave=False, position=0))
            torch.set_num_threads(torch_num_threads)

        if self.pre_filter is not None: #pre_filter=None
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None: #pre_transform=None
            data_list = [self.pre_transform(d) for d in data_list]
        
        index_to_Z, Z_to_index = self.element_statistics(data_list)
        self.spinful = data_list[0].spinful #False
        for d in data_list:
            assert self.spinful == d.spinful

        data_list = self.make_mask(data_list) 
        
        data, slices = self.collate(data_list)
        torch.save((data, slices, dict(spinful=self.spinful, index_to_Z=index_to_Z, Z_to_index=Z_to_index)), self.data_file) 
        print('Finish saving %d structures to %s, have cost %d seconds' % (len(data_list), self.data_file, time.time() - begin))
        

    def process_worker(self, folder, **kwargs):
        stru_id = os.path.split(folder)[-1]
        print("process dir:", stru_id)
        structure = Structure(np.loadtxt(os.path.join(folder, 'lat.dat')),
                              np.loadtxt(os.path.join(folder, 'element.dat')),
                              np.loadtxt(os.path.join(folder, 'site_positions.dat')),
                              coords_are_cartesian=True,
                              to_unit_cell=False)

        cart_coords = torch.tensor(structure.cart_coords, dtype=self.default_dtype_torch) 
        frac_coords = torch.tensor(structure.frac_coords, dtype=self.default_dtype_torch)
        numbers = torch.tensor(structure.atomic_numbers)
        structure.lattice.matrix.setflags(write=True)
        lattice = torch.tensor(structure.lattice.matrix, dtype=self.default_dtype_torch)
        huge_structure = False
        return get_graph(cart_coords, frac_coords, numbers, stru_id, radius=self.radius, material_dimension=self.material_dimension,
                         numerical_tol=1e-8, lattice=lattice, default_dtype_torch=self.default_dtype_torch,
                         tb_folder=folder, data_format=self.data_format, num_l=self.num_l, shortest_path_length=self.shortest_path_length,
                         if_lcmp_graph=self.if_lcmp_graph, target=self.target, 
                         huge_structure=huge_structure, **kwargs)
        

    def element_statistics(self, data_list):
        index_to_Z, inverse_indices = torch.unique(data_list[0].x, sorted=True, return_inverse=True)
        Z_to_index = torch.full((100,), -1, dtype=torch.int64)
        Z_to_index[index_to_Z] = torch.arange(len(index_to_Z))
        # for data in data_list:
        #     data.x = Z_to_index[data.x]
        return index_to_Z, Z_to_index
    
    def make_mask(self, dataset):
        dataset_mask = []
        for data in dataset:

            unique_atom_number = list(dict.fromkeys([x.numpy().tolist() for x in data.x]))
            unique_orbital_basis_number = list(dict.fromkeys([n.tolist() for n in data.atom_num_orbital]))

            self.orbital = json.loads(generate_orbital(unique_atom_number, unique_orbital_basis_number))
            self.num_orbital = len(self.orbital)
            
            if self.target == 'hamiltonian' or self.target == 'phiVdphi':
                Oij_value = data.term_real  #旋转后的哈密顿量2016*9*9
                if data.term_real is not None:
                    if_only_rc = False
                else:
                    if_only_rc = True
            else:
                raise ValueError(f'Unknown target: {self.target}')
            if if_only_rc == False:
                if not torch.all(data.term_mask):
                    raise NotImplementedError("Not yet have support for graph radius including hopping without calculation")

            if self.spinful: #False
                if self.target == 'phiVdphi':
                    raise NotImplementedError("Not yet have support for phiVdphi")
                else:
                    out_fea_len = self.num_orbital * 8
            else:
                if self.target == 'phiVdphi':
                    out_fea_len = self.num_orbital * 3
                else:
                    out_fea_len = self.num_orbital 
            mask = torch.zeros(data.edge_attr.shape[0], out_fea_len, dtype=torch.int8)
            label = torch.zeros(data.edge_attr.shape[0], out_fea_len, dtype=torch.get_default_dtype())

            atomic_number_edge_i = data.x[data.edge_index[0]]
            atomic_number_edge_j = data.x[data.edge_index[1]]
            for index_out, orbital_dict in enumerate(self.orbital):
                for N_M_str, a_b in orbital_dict.items():
                    condition_atomic_number_i, condition_atomic_number_j = map(lambda x: int(x), N_M_str.split())
                    condition_orbital_i, condition_orbital_j = a_b
                    if self.spinful:
                        if self.target == 'phiVdphi':
                            raise NotImplementedError("Not yet have support for phiVdphi")
                        else:
                            mask[:, 8 * index_out:8 * (index_out + 1)] = torch.where(
                                (atomic_number_edge_i == condition_atomic_number_i)
                                & (atomic_number_edge_j == condition_atomic_number_j),
                                1,
                                0
                            )[:, None].repeat(1, 8)
                    else:
                        if self.target == 'phiVdphi':
                            mask[:, 3 * index_out:3 * (index_out + 1)] += torch.where(
                                (atomic_number_edge_i == condition_atomic_number_i)
                                & (atomic_number_edge_j == condition_atomic_number_j),
                                1,
                                0
                            )[:, None].repeat(1, 3)
                        else:
                            mask[:, index_out] += torch.where(
                                (atomic_number_edge_i == condition_atomic_number_i)
                                & (atomic_number_edge_j == condition_atomic_number_j), #同上，都是[2016个True]
                                1,
                                0
                            )
                            
                    if if_only_rc == False:
                        if self.spinful:
                            if self.target == 'phiVdphi':
                                raise NotImplementedError
                            else:
                                label[:, 8 * index_out:8 * (index_out + 1)] = torch.where(
                                    (atomic_number_edge_i == condition_atomic_number_i)
                                    & (atomic_number_edge_j == condition_atomic_number_j),
                                    Oij_value[:, condition_orbital_i, condition_orbital_j].t(),
                                    torch.zeros(8, data.edge_attr.shape[0], dtype=torch.get_default_dtype())
                                ).t()
                        else:
                            if self.target == 'phiVdphi':
                                label[:, 3 * index_out:3 * (index_out + 1)] = torch.where(
                                    (atomic_number_edge_i == condition_atomic_number_i)
                                    & (atomic_number_edge_j == condition_atomic_number_j),
                                    Oij_value[:, condition_orbital_i, condition_orbital_j].t(),
                                    torch.zeros(3, data.edge_attr.shape[0], dtype=torch.get_default_dtype())
                                ).t()
                            else:
                                label[:, index_out] += torch.where(
                                    (atomic_number_edge_i == condition_atomic_number_i)
                                    & (atomic_number_edge_j == condition_atomic_number_j),
                                    Oij_value[:, condition_orbital_i, condition_orbital_j],
                                    torch.zeros(data.edge_attr.shape[0], dtype=torch.get_default_dtype())
                                )
            assert len(torch.where((mask != 1) & (mask != 0))[0]) == 0
            mask = mask.bool()
            # print("mask.all = ", mask.all())
            data.mask = mask
            del data.term_mask
            if if_only_rc == False:
                data.label = label
                if self.target == 'hamiltonian' or self.target == 'phiVdphi':
                    del data.term_real

            dataset_mask.append(data)
        return dataset_mask
