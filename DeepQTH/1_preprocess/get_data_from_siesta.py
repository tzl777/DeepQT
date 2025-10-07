import os
import numpy as np
import h5py
import json

import matplotlib.pyplot as plt
import sisl
from sisl.io import *

# from utils import draw_sub_H

def get_data_from_siesta(interface, input_file, input_path, output_path):
    input_path = os.path.abspath(input_path) #../example/work_dir/dataset/raw/0-575，预测时是olp文件夹
    output_path = os.path.abspath(output_path) #~/example/work_dir/dataset/processed/0-575,预测时是inference文件夹
    os.makedirs(output_path, exist_ok=True) #创建用于放处理后数据的每一个子文件夹

    # finds system name
    f_list = os.listdir(input_path)
    system_name = [element.split(".")[0] for element in f_list if (".TSHS" in element) or (".HSX" in element)][0]

    #read structure file
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
    H = hsx.read_hamiltonian(geometry=geom)
    S = hsx.read_overlap(geometry=geom)
    # print(H.shape)

    # 获取晶格矢量
    lattice = geom.lattice.cell
    # 获取原子序数
    atomic_numbers = geom.atoms.Z
    # 获取原子坐标
    atom_coord_cart = geom.xyz
    #计算倒格矢
    rlattice = geom.lattice.rcell #倒空间的格矢
    # 保存晶胞矢量、倒格矢、原子序数原子坐标位置
    np.savetxt('{}/rlat.dat'.format(output_path), rlattice, fmt='%.8e') #保存倒格矢量   
    np.savetxt('{}/lat.dat'.format(output_path), lattice, fmt='%.8e')#保存晶格矢量的转置
    np.savetxt('{}/element.dat'.format(output_path), atomic_numbers, fmt='%d') #找出所有原子的原子序数
    np.savetxt('{}/site_positions.dat'.format(output_path), atom_coord_cart) #原子位置的转置
    
    #提取基组信息，原子数基组是否正交、是否自旋、基组/轨道数量
    num_atoms = H.na
    isorthogonal = bool(H.orthogonal)
    isspinful = bool(H.spin.is_polarized)
    norbits = int(H.no)
    
    info = {'nsites': num_atoms, 'isorthogonal': isorthogonal, 'isspinful': isspinful, 'norbits': norbits}
    with open('{}/info.json'.format(output_path), 'w') as info_f:
        json.dump(info, info_f)  # 把python字典对象info转换成json对象，生成一个fp的文件流，和文件相关。

    num_orbital_per_atom = geom.orbitals
    np.savetxt('{}/num_orbital_per_atom.dat'.format(output_path), num_orbital_per_atom, fmt='%d')

    
    orb_indx = np.genfromtxt('{}/{}.ORB_INDX'.format(input_path, system_name), skip_header=3, skip_footer=17)

    #保存超胞中的超胞索引
    # all_isc = geom.o2isc(orbitals=list(range(H.shape[1])))
    # unique_isc = np.unique(all_isc, axis=0)
    # np.savetxt('{}/unique_isc.dat'.format(output_path), unique_isc, fmt='%d')
    isc_list = orb_indx[:, 12:15]
    result = []
    seen = set()
    for i in range(len(isc_list)):
        isc = str(isc_list[i])
        if isc not in seen:
            seen.add(isc)
            result.append(np.array([int(x.strip('.')) for x in isc[1:-1].split()]))
    np.savetxt('{}/unique_isc.dat'.format(output_path), result, fmt='%d')


        #提取基组或双ζ基组信息
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
        a_i = H.o2a(orbitals=i, unique=True) # orbit i belongs to atom_1。#表示该轨道在第一个晶胞中的等效原子的索引。
        b_j = H.o2a(orbitals=j, unique=True)
        uc_b_j = H.asc2uc(atoms=b_j)
        isc = H.a2isc(atoms=b_j)
        # print(H.tocsr().toarray()[:9,:9])
        key = '[{}, {}, {}, {}, {}]'.format(isc[0],isc[1],isc[2],a_i,uc_b_j)
        if key not in seen:
            # print(key)
            seen.add(key)
            H_ab = H.sub([a_i, b_j])
            H_ab_array = H_ab.tocsr().toarray()
            H_ab_matrix = H_ab_array[:9,9:18]
            S_ab = S.sub([a_i, b_j])
            S_ab_array = S_ab.tocsr().toarray()
            S_ab_matrix = S_ab_array[:9,9:18]
            #draw_sub_H(a_i, b_j, S_ab_matrix)
            H_block_matrix[key] = H_ab_matrix
            S_block_matrix[key] = S_ab_matrix
            
    with h5py.File(os.path.join(output_path, "hamiltonians.h5"), "w") as f:
        for key in H_block_matrix.keys():
            f[key] = H_block_matrix[key]

    with h5py.File(os.path.join(output_path, "overlaps.h5"), "w") as f:
        for key in S_block_matrix.keys():
            f[key] = S_block_matrix[key]

if __name__ == '__main__':
    input_path = "/fs2/home/ndsim10/DeepQT/0_generate_dataset/expand_dataset/raw/1/"
    output_path = "/fs2/home/ndsim10/DeepQT/0_generate_dataset/expand_dataset/processed/1/"
    input_file = "input.fdf"
    interface = "siesta"
    get_data_from_siesta(interface, input_file, input_path, output_path)