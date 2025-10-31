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

sys.path.insert(0, "/fs2/home/ndsim10/DeepQT/DeepQTH/1_preprocess")
from graph import get_graph
from rotate import Rotate, dtype_dict
from get_rotate_coord import get_rc
from data import generate_orbital
sys.path.insert(0, "/fs2/home/ndsim10/DeepQT/DeepQTH/2_train")
from kernel import DeepHKernel
from utils import collate_fn, write_ham_h5, read_config_to_dict, Logger

#inference/C_nanotube_140,inference/C_nanotube_140,False,cpu,True,True,trained_model_dirs
def predict(input_dir: str, output_dir: str, disable_cuda: bool, device: str,
            huge_structure: bool, restore_blocks_py: bool, trained_model_dirs: Union[str, List[str]]):
    
    atom_num_orbital = np.loadtxt(os.path.join(input_dir, 'num_orbital_per_atom.dat')).astype(int)
    if isinstance(trained_model_dirs, str):
        trained_model_dirs = [trained_model_dirs]
    assert isinstance(trained_model_dirs, list)
    os.makedirs(output_dir, exist_ok=True)
    predict_spinful = None

    with torch.no_grad():
        read_structure_flag = False
        if restore_blocks_py: #True
            hoppings_pred = {}
        else:
            index_model = 0
            block_without_restoration = {}
            os.makedirs(os.path.join(output_dir, 'block_without_restoration'), exist_ok=True)
        for trained_model_dir in tqdm.tqdm(trained_model_dirs):
            old_version = False
            assert os.path.exists(os.path.join(trained_model_dir, 'config.txt'))
            if os.path.exists(os.path.join(trained_model_dir, 'best_model.pt')) is False: #True is False = False
                old_version = True
                assert os.path.exists(os.path.join(trained_model_dir, 'best_model.pkl'))
                assert os.path.exists(os.path.join(trained_model_dir, 'src'))

            # config = ConfigParser()
            # config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'default.ini'))
           
            # config.read(os.path.join(trained_model_dir, 'config.txt'))
            config = read_config_to_dict(os.path.join(trained_model_dir, 'config.txt'))
            
            config['basic']['save_dir'] = output_dir 
            config['basic']['disable_cuda'] =  disable_cuda #False
            config['basic']['device'] = device #cpu
            config['basic']['save_to_time_folder'] = False #False，
            config['train']['pretrained'] = '' #''
            config['train']['resume'] = '' #''

            kernel = DeepHKernel(config, is_predict=True)
            print("old_version = ", old_version)
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
                    kernel.index_to_Z = torch.arange(config['basic']['max_element'] + 1)
                if hasattr(kernel, 'Z_to_index') is False:
                    kernel.Z_to_index = torch.arange(config['basic']['max_element'] + 1)
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
                structure = Structure(np.loadtxt(os.path.join(input_dir, 'lat.dat')),
                                      np.loadtxt(os.path.join(input_dir, 'element.dat')),
                                      np.loadtxt(os.path.join(input_dir, 'site_positions.dat')),
                                      coords_are_cartesian=True,
                                      to_unit_cell=False)
                cart_coords = torch.tensor(structure.cart_coords, dtype=torch.get_default_dtype())
                frac_coords = torch.tensor(structure.frac_coords, dtype=torch.get_default_dtype())
                
                numbers = torch.tensor(structure.atomic_numbers)
                structure.lattice.matrix.setflags(write=True)
                lattice = torch.tensor(structure.lattice.matrix, dtype=torch.get_default_dtype())
                inv_lattice = torch.inverse(lattice)#晶格的逆

                if os.path.exists(os.path.join(input_dir, 'graph.pkl')):
                    # data = torch.load(os.path.join(input_dir, 'graph.pkl'))
                    data = torch.load(os.path.join(input_dir, 'graph.pkl'), map_location=torch.device('cpu'))
                    print(f"Load processed graph from {os.path.join(input_dir, 'graph.pkl')}")
                else:
                    begin = time.time()
                    data = get_graph(cart_coords, frac_coords, numbers, 0,
                                     radius=kernel.config['graph']['radius'],
                                     material_dimension=kernel.config['basic']['material_dimension'],
                                     numerical_tol=1e-8, lattice=lattice, default_dtype_torch=torch.get_default_dtype(),
                                     tb_folder=input_dir, data_format="h5_rc_only",
                                     num_l=kernel.config['network']['num_l'],
                                     shortest_path_length=kernel.config['graph']['shortest_path_length'],                      
                                     max_path_length=kernel.config['network']['max_path_length'],
                                     if_lcmp_graph=kernel.config['graph']['if_lcmp_graph'],
                                     separate_onsite=kernel.separate_onsite,
                                     target=kernel.config['basic']['target'], huge_structure=huge_structure,
                                     if_new_sp=kernel.config['graph']['new_sp'],
                                     )
                    print("data = {}\n".format(data))
                   
                    torch.save(data, os.path.join(input_dir, 'graph.pkl'))
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
                            ptr=[2], 
                        )
                '''
                print("subgraph_atom_idx.shape = ", subgraph[0].shape) 
                print("subgraph_edge_idx.shape = ", subgraph[1].shape) 
                print("subgraph_edge_ang.shape = ", subgraph[2].shape) 
                print("subgraph_index.shape = ", subgraph[3].shape) 
               
                sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index = subgraph

            output = kernel.model(batch.x.to(kernel.device),
                                  batch.edge_index.to(kernel.device),
                                  batch.edge_attr.to(kernel.device),
                                  # batch.node_paths.to(kernel.device),
                                  # batch.edge_paths.to(kernel.device),
                                  # batch.voronoi_values.to(kernel.device),
                                  # batch.centralities.to(kernel.device),
                                  batch.batch.to(kernel.device),
                                  sub_atom_idx.to(kernel.device),
                                  sub_edge_idx.to(kernel.device),
                                  sub_edge_ang.to(kernel.device),
                                  sub_index.to(kernel.device),
                                  huge_structure=huge_structure) 
            output = output.detach().cpu()
            if restore_blocks_py:
                unique_atom_number = list(dict.fromkeys([x.numpy().tolist() for x in data.x]))
                unique_orbital_basis_number = list(dict.fromkeys([n.tolist() for n in data.atom_num_orbital]))
                orbital = json.loads(generate_orbital(unique_atom_number, unique_orbital_basis_number))
                
                for index in range(batch.edge_attr.shape[0]): #1*2016
                    R = torch.round(batch.edge_attr[index, 4:7] @ inv_lattice - batch.edge_attr[index, 7:10] @ inv_lattice).int().tolist()
                    i, j = batch.edge_index[:, index]
                    key_term = (*R, i.item(), j.item())
                    key_term = str(list(key_term))
                    for index_orbital, orbital_dict in enumerate(orbital): 
                        if f'{numbers[i].item()} {numbers[j].item()}' not in orbital_dict:
                            continue
                        orbital_i, orbital_j = orbital_dict[f'{numbers[i].item()} {numbers[j].item()}'] 

                        if not key_term in hoppings_pred: 
                            if kernel.spinful:
                                hoppings_pred[key_term] = np.full((2 * atom_num_orbital[i], 2 * atom_num_orbital[j]), np.nan + np.nan * (1j))
                            else:
                                hoppings_pred[key_term] = np.full((atom_num_orbital[i], atom_num_orbital[j]), np.nan) 
                        if kernel.spinful:
                            hoppings_pred[key_term][orbital_i, orbital_j] = output[index][index_orbital * 8 + 0] + output[index][index_orbital * 8 + 1] * 1j
                            hoppings_pred[key_term][atom_num_orbital[i] + orbital_i, atom_num_orbital[j] + orbital_j] = output[index][index_orbital * 8 + 2] + output[index][index_orbital * 8 + 3] * 1j
                            hoppings_pred[key_term][orbital_i, atom_num_orbital[j] + orbital_j] = output[index][index_orbital * 8 + 4] + output[index][index_orbital * 8 + 5] * 1j
                            hoppings_pred[key_term][atom_num_orbital[i] + orbital_i, orbital_j] = output[index][index_orbital * 8 + 6] + output[index][index_orbital * 8 + 7] * 1j
                        else:
                            hoppings_pred[key_term][orbital_i, orbital_j] = output[index][index_orbital] 
            else:
                if 'edge_index' not in block_without_restoration:
                    assert index_model == 0
                    block_without_restoration['edge_index'] = batch.edge_index
                    block_without_restoration['edge_attr'] = batch.edge_attr
                block_without_restoration[f'output_{index_model}'] = output.numpy()
                with open(os.path.join(output_dir, 'block_without_restoration', f'orbital_{index_model}.json'), 'w') as orbital_f:
                    json.dump(kernel.orbital, orbital_f, indent=4)
                index_model += 1

            # sys.stdout = Logger(os.path.join(config["basic"]["save_dir"], "result.txt")) 
            # sys.stderr = Logger(os.path.join(config["basic"]["save_dir"], "stderr.txt"))
            # sys.stdout = sys.stdout.terminal 
            # sys.stderr = sys.stderr.terminal

        if restore_blocks_py:
            for hamiltonian in hoppings_pred.values():
                assert np.all(np.isnan(hamiltonian) == False) 
            write_ham_h5(hoppings_pred, path=os.path.join(output_dir, 'rh_pred.h5')) 
        else:
            block_without_restoration['num_model'] = index_model
            write_ham_h5(block_without_restoration, path=os.path.join(output_dir, 'block_without_restoration', 'block_without_restoration.h5'))
        with open(os.path.join(output_dir, "info.json"), 'w') as info_f:
            json.dump({
                "isspinful": predict_spinful
            }, info_f) 

