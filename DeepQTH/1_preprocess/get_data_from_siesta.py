import os
import numpy as np
import h5py
import json

import matplotlib.pyplot as plt
import sisl
from sisl.io import *

def get_data_from_siesta(input_path, output_path, interface="siesta", input_file="input.fdf"):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    # finds system name
    f_list = os.listdir(input_path)
    system_name = [element.split(".")[0] for element in f_list if (".TSHS" in element) or (".HSX" in element)][0]

    #read structure file
    # hsx = None
    if interface == 'siesta':
        geom_str = input_path + "/" + input_file
        ham_str = input_path + "/" + f"{system_name}.HSX"
        geom = sisl.get_sile(geom_str).read_geometry()
        hsx = hsxSileSiesta(ham_str)
    elif interface == 'transiesta':
        geom_str = input_path + "/" + input_file
        ham_str = input_path + "/" + f"{system_name}.TSHS"
        geom = sisl.get_sile(geom_str).read_geometry()
        hsx = tshsSileSiesta(ham_str)
    else:
        raise ValueError(f"Unknown interface: {interface}")
    H = hsx.read_hamiltonian(geometry=geom)
    S = hsx.read_overlap(geometry=geom)
    # print(H.shape)

    lattice = geom.lattice.cell
    atomic_numbers = geom.atoms.Z
    # atom_coord_cart = geom.xyz
    rlattice = geom.lattice.rcell

    np.savetxt('{}/rlat.dat'.format(output_path), rlattice, fmt='%.8e') 
    np.savetxt('{}/lat.dat'.format(output_path), lattice, fmt='%.8e')
    np.savetxt('{}/element.dat'.format(output_path), atomic_numbers, fmt='%d')

    atom_coord_cart = np.genfromtxt('{}/{}.XV'.format(input_path, system_name), skip_header = 4)
    atom_coord_cart = atom_coord_cart[:,2:5] * 0.529177249
    np.savetxt('{}/site_positions.dat'.format(output_path), atom_coord_cart)
    
    # Extract the basis set information, including the number of atoms, whether the basis set is orthogonal, whether it is spin-related, and the number of atomic orbitals.
    num_atoms = H.na
    isorthogonal = bool(H.orthogonal)
    isspinful = bool(H.spin.is_polarized)
    norbits = int(H.shape[1])
    # print(norbits)
    info = {'nsites': num_atoms, 'isorthogonal': isorthogonal, 'isspinful': isspinful, 'norbits': norbits}
    with open('{}/info.json'.format(output_path), 'w') as info_f:
        json.dump(info, info_f)

    num_orbital_per_atom = geom.orbitals
    np.savetxt('{}/num_orbital_per_atom.dat'.format(output_path), num_orbital_per_atom, fmt='%d')
    
    orb_indx = np.genfromtxt('{}/{}.ORB_INDX'.format(input_path, system_name), skip_header=3, skip_footer=17)

    isc_list = orb_indx[:, 12:15]
    result = []
    seen = set()
    for i in range(len(isc_list)):
        isc = str(isc_list[i])
        if isc not in seen:
            seen.add(isc)
            result.append(np.array([int(x.strip('.')) for x in isc[1:-1].split()]))
    np.savetxt('{}/unique_isc.dat'.format(output_path), result, fmt='%d')

    l_z = orb_indx[:, [1, 6, 8]]
    orbital_type = []
    seen = set()
    for i in range(1,num_atoms+1):
        result = []
        no_l_z = l_z[l_z[:,0] == i, :]
        for i in range(len(no_l_z)):
            lz = str(no_l_z[i])
            if lz not in seen:
                seen.add(lz)
                result.append(np.array([int(x.strip('.')) for x in lz[1:-1].split()])[1])
        # print(result)
        orbital_type.append(result)
    np.savetxt('{}/orbital_types.dat'.format(output_path), orbital_type, fmt='%d')


    H_block_matrix = dict()
    S_block_matrix = dict()
    seen = set()
    for i, j in H.iter_nnz():
        a_i = H.o2a(orbitals=i, unique=True) # orbit i belongs to atom i
        b_j = H.o2a(orbitals=j, unique=True)
        uc_b_j = H.asc2uc(atoms=b_j)
        isc = H.a2isc(atoms=b_j)
        
        key = '[{}, {}, {}, {}, {}]'.format(isc[0],isc[1],isc[2],a_i,uc_b_j)
        if key not in seen:
            # print(key)
            seen.add(key)
            H_ab = H.sub([a_i, b_j])
            H_ab_array = H_ab.tocsr().toarray()
            # print(H_ab_array.shape)
            H_ab_matrix = H_ab_array[:9,9:18] # No spin and orbital coupling
            S_ab = S.sub([a_i, b_j])
            S_ab_array = S_ab.tocsr().toarray()
            S_ab_matrix = S_ab_array[:9,9:18]
 
            H_block_matrix[key] = H_ab_matrix
            S_block_matrix[key] = S_ab_matrix
            # break
            
    with h5py.File(os.path.join(output_path, "hamiltonians.h5"), "w") as f:
        for key in H_block_matrix.keys():
            f[key] = H_block_matrix[key]

    with h5py.File(os.path.join(output_path, "overlaps.h5"), "w") as f:
        for key in S_block_matrix.keys():
            f[key] = S_block_matrix[key]

if __name__ == '__main__':
    input_path = "/fs2/home/ndsim10/DeepQT/DeepQTH/0_generate_dataset/sample_data/0"
    output_path = "/fs2/home/ndsim10/DeepQT/DeepQTH/0_generate_dataset/sample_data/processed/0"
    interface = "siesta"
    input_file = "input.fdf"
    get_data_from_siesta(input_path, output_path, interface, input_file)