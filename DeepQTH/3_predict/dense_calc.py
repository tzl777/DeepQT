import json
import argparse
import h5py
import numpy as np
import os
from time import time
from scipy.sparse import csc_matrix
from scipy.linalg import eigh, qr
from scipy.sparse.linalg import eigs
from sisl import *

def parse_commandline(): #使用 argparse 模块创建了一个命令行解析器，并定义了三个命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-i", type=str, default="./",
        help="path of rlat.dat, orbital_types.dat, site_positions.dat, hamiltonians_pred.h5, and overlaps.h5"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="./",
        help="path of output pred_bands.dat and pred_dos.dat"
    )
    parser.add_argument(
        "--config", type=str,
        help="config file in the format of JSON"
    )
    return parser.parse_args() ##parse_args() 方法解析命令行参数，并返回一个带有解析结果的命名空间对象。

parsed_args = parse_commandline() #函数 parse_commandline() 被调用并将解析结果保存在 parsed_args 变量中，这样就可以在程序中使用这些参数了。

def _create_dict_h5(filename):
    fid = h5py.File(filename, "r")
    d_out = {}
    for key in fid.keys():
        data = np.array(fid[key])
        nk = tuple(map(int, key[1:-1].split(','))) ##假设键的格式类似于"[0,0,0,2,3]"，key[1:-1]即不读取开头和结尾的括号，然后根据逗号分隔键的内容并转换为整数元组。
        # BS: 
        # the matrix do not need be transposed in Python, 
        # But the transpose should be done in Julia.
        d_out[nk] = data # np.transpose(data)
    fid.close()
    return d_out


ev2Hartree = 0.036749324533634074
Bohr2Ang = 0.529177249


def genlist(x):
    return np.linspace(x[0], x[1], int(x[2]), endpoint=False)


def k_data2num_ks(kdata):
    return int(kdata.split()[0])


def k_data2kpath(kdata):
    return [float(x) for x in kdata.split()[1:7]]


def std_out_array(a):
    return ''.join([str(x) + ' ' for x in a])

def std_out_EKarray(E_K):
    output = ""  # 初始化一个空字符串，用于累加每行的字符串表示
    for ek in E_K:
        line = ''.join(str(x)+" " for x in ek) + "\n"  # 将一行转换为字符串并加上换行符
        output += line  # 将这一行添加到输出字符串中
    return output

def str_to_bool(s):
    tmp = bool()
    if s == "True":
        tmp = True
    elif s == "False":
        tmp = False
    else:
        raise ValueError(f"Unknown calulate job: {s}")
    return tmp

def constructmeshkpts(nkmesh, offset=[0.0, 0.0, 0.0], k1=[0.0, 0.0, 0.0], k2=[1.0, 1.0, 1.0]):
    if len(nkmesh) != 3:
        raise ValueError("nkmesh must be of size 3.")
    nkpts = np.prod(nkmesh)
    kpts = np.zeros((3, nkpts))
    ik = 0
    for ikx in range(nkmesh[0]):
        for iky in range(nkmesh[1]):
            for ikz in range(nkmesh[2]):
                kpts[:, ik] = [
                    (ikx / nkmesh[0]) * (k2[0] - k1[0]) + k1[0],
                    (iky / nkmesh[1]) * (k2[1] - k1[1]) + k1[1],
                    (ikz / nkmesh[2]) * (k2[2] - k1[2]) + k1[2]
                ]
                ik += 1
    return kpts + np.array(offset).reshape(3, 1)


default_dtype = np.complex128 #设置了 NumPy 数组的默认数据类型为复数类型 complex128。在 NumPy 中，数据类型 np.complex128 表示复数，由双精度浮点数构成，即实部和虚部都是双精度浮点数。

print(parsed_args.config) #打印出在命令行中传递的 --config 参数的值，这个值应该是一个 JSON 格式的配置文件的路径。

with open(parsed_args.config) as f:
    config = json.load(f)
print(config)
calc_bands = str_to_bool(config["calc_bands"]) #band，这里可以改为True或False
calc_dos = str_to_bool(config["calc_dos"]) #band，这里可以改为True或False
fermi_level = str_to_bool(config["fermi_level"])
print(calc_bands)
print(calc_dos)
print(fermi_level)

if os.path.isfile(os.path.join(parsed_args.input_dir, "info.json")):
    with open(os.path.join(parsed_args.input_dir, "info.json")) as f:
        spinful = json.load(f)["isspinful"] #False
else:
    spinful = False

site_positions = np.loadtxt(os.path.join(parsed_args.input_dir, "site_positions.dat")) #(3,244)原子坐标

if len(site_positions.shape) == 2:
    nsites = site_positions.shape[1] #244
else:
    nsites = 1
    # in case of single atom


with open(os.path.join(parsed_args.input_dir, "orbital_types.dat")) as f: #(244,3)   [[0,1,2],[0,1,2],...,[0,1,2]]
    site_norbits = np.zeros(nsites, dtype=int) #[244个0]
    orbital_types = []
    for index_site in range(nsites):
        orbital_type = list(map(int, f.readline().split()))
        orbital_types.append(orbital_type)
        site_norbits[index_site] = np.sum(np.array(orbital_type) * 2 + 1) #[244个9]
    norbits = np.sum(site_norbits) #244*9=2196
    site_norbits_cumsum = np.cumsum(site_norbits) ##用于计算数组中元素的累积和。它接受一个数组作为输入，并返回一个新的数组，其中每个元素是原数组中对应位置之前所有元素的累积和。返回一个与原始数组大小相同的数组，其中第一个元素是原数组的第一个元素，第二个元素是原数组的前两个元素之和，以此类推。
    #site_norbits_cumsum:[9   18   27 ... 2196]

rlat = np.loadtxt(os.path.join(parsed_args.input_dir, "rlat.dat")).T #读取倒格矢(3,3)
# require transposition while reading rlat.dat in python


print("read h5")
begin_time = time()
hamiltonians_pred = _create_dict_h5(os.path.join(parsed_args.input_dir, "hamiltonians_pred.h5")) #读取预测的哈密顿量矩阵
overlaps = _create_dict_h5(os.path.join(parsed_args.input_dir, "overlaps.h5")) #读取siesta_get_data预先计算的重叠矩阵
print("Time for reading h5: ", time() - begin_time, "s")


H_R = {}
S_R = {}

print("construct Hamiltonian and overlap matrix in the real space")
begin_time = time()

# BS:
# this is for debug python and julia
# in julia, you can use 'sort(collect(keys(hamiltonians_pred)))'
# for key in dict(sorted(hamiltonians_pred.items())).keys():
for key in hamiltonians_pred.keys():

    hamiltonian_pred = hamiltonians_pred[key] #读取预测的每一个key的哈密顿量矩阵

    if key in overlaps.keys(): #预测哈密顿量矩阵的key，在重叠矩阵的key中，则读取对应key的重叠矩阵，否则生成一个类哈密顿量矩阵的0矩阵
        overlap = overlaps[key]
    else:
        overlap = np.zeros_like(hamiltonian_pred)
    if spinful: #如果存在自旋，构建自旋的重叠矩阵
        overlap = np.vstack((np.hstack((overlap, np.zeros_like(overlap))), np.hstack((np.zeros_like(overlap), overlap))))
    R = key[:3] #读取key中的R，即原子所属的晶胞的索引
    atom_i = key[3] - 1 #读取在晶胞R中存在连接的原子i和原子j
    atom_j = key[4] - 1

    assert (site_norbits[atom_i], site_norbits[atom_j]) == hamiltonian_pred.shape #(9,9)的矩阵
    assert (site_norbits[atom_i], site_norbits[atom_j]) == overlap.shape #(9,9)的矩阵

    if R not in H_R.keys(): #True
        H_R[R] = np.zeros((norbits, norbits), dtype=default_dtype) #(2196,2196)的数据类型为复数类型complex128的零矩阵，存放实空间的哈密顿量矩阵和重叠矩阵
        S_R[R] = np.zeros((norbits, norbits), dtype=default_dtype)

    for block_matrix_i in range(1, site_norbits[atom_i]+1): #1-9
        for block_matrix_j in range(1, site_norbits[atom_j]+1): #1-9
            index_i = site_norbits_cumsum[atom_i] - site_norbits[atom_i] + block_matrix_i - 1 #key中R对应的是轨道所属的晶胞索引，这里是求得原子i所在晶胞内的实际原子轨道
            index_j = site_norbits_cumsum[atom_j] - site_norbits[atom_j] + block_matrix_j - 1
            H_R[R][index_i, index_j] = hamiltonian_pred[block_matrix_i-1, block_matrix_j-1] #将两个原子的轨道对的预测的的哈密顿量值填充到H_R[R]中，H_R[R]是（244*9=2196）*（244*9=2196）
            S_R[R][index_i, index_j] = overlap[block_matrix_i-1, block_matrix_j-1] #将两个原子对的重叠矩阵值填充到S_R[R]中
#到这里就得到了实空间的哈密顿矩阵H_R和重叠矩阵S_R
print("Time for constructing Hamiltonian and overlap matrix in the real space: ", time() - begin_time, " s")

e_fermi = 0.0
if calc_bands == True:
    k_data = config["k_data"] #["15 0 0 0 0.5 0 0 Γ M", "15 0.5 0 0 0.3333333 0.33333333 0 M K", "15 0.33333333 0.33333333 0 0 0 0 K Γ"]
    print("k_data",k_data)

    print("calculate bands")
    num_ks = [k_data2num_ks(k) for k in k_data] #[15,15,15]
    kpaths = [k_data2kpath(k) for k in k_data] #[[0 0 0 0.5 0 0],[0.5 0 0 0.3333333 0.33333333 0],[0.33333333 0.33333333 0 0 0 0]]

    egvals = np.zeros((norbits, sum(num_ks)+1)) #(2196,46)，构建了k点下存放特征值的零矩阵

    begin_time = time()

    k_points = np.empty((1,3))
    for i in range(len(kpaths)): #0-2
        kpath = kpaths[i]
        pnkpts = num_ks[i]
        kxs = np.linspace(kpath[0], kpath[3], pnkpts, endpoint=False) #创建了一个在闭区间 [start, end] 上均匀分布的等差数列，包含了 pnkpts 个元素。
        kys = np.linspace(kpath[1], kpath[4], pnkpts, endpoint=False)
        kzs = np.linspace(kpath[2], kpath[5], pnkpts, endpoint=False)
        matrix = np.column_stack((kxs, kys, kzs))
        k_points = np.vstack((k_points, matrix))
    k_points = np.delete(k_points, 0, axis=0)
    k_points = np.vstack((k_points, np.array(kpaths[len(kpaths)-1][3:])))
    # print(k_points)
    print(k_points.shape) #46*3
    idx_k = 0
    for i in range(len(k_points)): #同时迭代处理这三个列表中对应位置的元素。
        kx, ky, kz = k_points[i,:]
        H_k = np.zeros((norbits, norbits), dtype=default_dtype) #(2196,2196)的数据类型为复数类型complex128的零矩阵，存放实空间的哈密顿量矩阵和重叠矩阵
        S_k = np.zeros((norbits, norbits), dtype=default_dtype)
        for R in H_R.keys(): #下面两个表达式的意思可能是在k（动量）空间中，通过加权计算不同晶格向量位置上的哈密顿量矩阵，并将结果累加到 H_k 中。
            H_k += H_R[R] * np.exp(1j*2*np.pi*np.dot([kx, ky, kz], R)) #np.dot两个向量的点积运算。点积（内积）操作将这两个向量对应位置的元素相乘，然后将所有相乘结果相加得到一个标量值。这个结果表示了两个向量之间的投影关系和相似性。
            S_k += S_R[R] * np.exp(1j*2*np.pi*np.dot([kx, ky, kz], R)) #复数值为np.exp(1j*2*np.pi*np.dot([kx, ky, kz], R))。
        #---------------------------------------------
        # BS: only eigenvalues are needed in this part,
        # the upper matrix is used
        #
        # egval, egvec = linalg.eig(H_k, S_k)
        egval = linalg.eigvalsh(H_k, S_k, lower=False) #使用了SciPy中linalg.eigvalsh()函数来计算一个Hermitean（厄米特）矩阵H_k的特征值。同时，它指定了矩阵S_k作为正交矩阵，用于进行特征值计算。返回一个数组egval，其中包含了H_k的特征值。参数lower=False表示计算所有的特征值，且这些特征值按照升序排列。
        egvals[:, idx_k] = egval #这里只计算了一个路径下一个k点的特征值，每一列是某一个k点下的所有特征值2196个。所以每一行就是一条能带！2196*46

        print("Time for solving No.{} eigenvalues at k = {} : {} s".format(idx_k+1, [kx, ky, kz], time() - begin_time))
        idx_k += 1 #计算其它路径下k点的特征值

    if fermi_level == True:
        total_electrons = config["total_electrons"] # 需用户提供实际电子数值
        # 输入参数
        num_kpoints = egvals.shape[1]         
        k_weights = np.ones(num_kpoints) / num_kpoints      # 均匀k点权重（可选）
        
        # 展平本征值并附加权重
        weighted_energies = []
        for k_idx in range(num_kpoints):
            energies = egvals[:, k_idx]          # 当前k点的所有本征值
            weights = np.full_like(energies, k_weights[k_idx])  # 当前k点的权重
            weighted_energies.extend(zip(energies, weights))
        
        # 转换为数组并按能量排序
        weighted_energies = np.array(weighted_energies, dtype=[('energy', 'f8'), ('weight', 'f8')])
        weighted_energies.sort(order='energy')  # 按能量升序排列
        
        # 累积权重直到达到总电子数
        cumulative_weight = np.cumsum(weighted_energies['weight'])
        fermi_index = np.searchsorted(cumulative_weight, total_electrons / 2)  # 考虑自旋，每个态最多2电子
        e_fermi = round(weighted_energies['energy'][fermi_index],4)
        
        print(f"fermi level: E_F = {e_fermi} eV")
        

    begin_time = time()

    spin=int(spinful)+1
    siesta_rlat = np.array(rlat * Bohr2Ang) #(3,3)倒格矢
    ymin = egvals.min() #-14.065387633674039
    ymax = egvals.max() #17.63961205616035
    # print(ymin, ymax)

    # output in siesta band format
    with open(os.path.join(parsed_args.output_dir, "pred_bands.dat"), "w") as f:
        kpoint = 0.000000
        k_list = [0.000000]
        for i in range(len(k_points)-1):  # 同时迭代处理这三个列表中对应位置的元素。
            kvec0 = k_points[i, :]
            kvec1 = k_points[i + 1, :]
            kpoint = np.linalg.norm(kvec0 @ siesta_rlat - kvec1 @ siesta_rlat) + kpoint
            k_list.append(np.around(kpoint, decimals=6))
        # print("kpoint = ", k_list)
        assert len(k_list) == len(k_points)

        f.write("# ----------------------------------------------------------------\n")
        f.write("# E_F\t=\t{}\n".format(e_fermi))  # -3.82373
        f.write("# k_min, k_max\t=\t{}\t{}\n".format(k_list[0], k_list[-1]))
        f.write("# E_min, E_max\t=\t{}\t{}\n".format(ymin, ymax))
        f.write("# Nbands, Nspin, Nk\t=\t{}\t{}\t{}\n".format(norbits, spin, len(k_list)))  # 2196 某个k点的坐标
        f.write("# k\tE[eV]\n")
        f.write("# ----------------------------------------------------------------\n")

        for i in range(len(egvals)):  # 同时迭代处理这三个列表中对应位置的元素。
            E_K = np.column_stack((np.array(k_list), egvals[i, :])) #写入每个k点的特征值，已校归零到费米能级
            f.write(std_out_EKarray(E_K) + "\n\n")
    print("Time for generate siesta band structure data: {} s".format(time() - begin_time))


begin_time = time()
if calc_dos == True:
    e_min = config["e_min"]
    e_max = config["e_max"]
    e_point = config["e_point"]
    sigma = config["sigma"]
    egvals_flat = egvals.flatten()  # 形状变为 (5850*46, )
    energy_grid = np.linspace(e_min, e_max, e_point)  # 调整范围和点数
    dos = np.zeros_like(energy_grid)
    
    for eig in egvals_flat:
        dos += np.exp(-0.5 * ((energy_grid - eig) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    dos /= len(egvals_flat)  # 归一化（总积分为1）

    # 将DOS数据写入文件
    with open(os.path.join(parsed_args.output_dir, "pred_dos.dat"), "w") as f:
        np.savetxt(f, np.column_stack((energy_grid, dos)))
    print("Time for generate siesta dos data: {} s".format(time() - begin_time))
