import json
import os.path
import warnings

import numpy as np
import h5py
import torch
from e3nn.o3 import Irrep, Irreps, matrix_to_angles
#Irreducible representation(irreps): 不可约表示，既等于一个不可分解的Representation。


dtype_dict = {
    np.float32: (torch.float32, torch.float32, torch.complex64),
    np.float64: (torch.float64, torch.float64, torch.complex128),
    np.complex64: (torch.complex64, torch.float32, torch.complex64),
    np.complex128: (torch.complex128, torch.float64, torch.complex128),
    torch.float32: (torch.float32, torch.float32, torch.complex64),
    torch.float64: (torch.float64, torch.float64, torch.complex128),
    torch.complex64: (torch.complex64, torch.float32, torch.complex64),
    torch.complex128: (torch.complex128, torch.float64, torch.complex128),
}

# self.cdouble() is equivalent to self.to(torch.complex128).
class Rotate:
    def __init__(self, torch_dtype, torch_dtype_real=torch.float64, torch_dtype_complex=torch.cdouble,
                 device=torch.device('cpu'), spinful=False):
        self.dtype = torch_dtype #torch.float32
        self.torch_dtype_real = torch_dtype_real #torch.float32
        self.device = device #cpu
        self.spinful = spinful #False
        sqrt_2 = 1.4142135623730951
        self.Us_openmx = {
            0: torch.tensor([1], dtype=torch_dtype_complex, device=device),
            1: torch.tensor([[-1 / sqrt_2, 1j / sqrt_2, 0], [0, 0, 1], [1 / sqrt_2, 1j / sqrt_2, 0]],
                            dtype=torch_dtype_complex, device=device),
            2: torch.tensor([[0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0]], dtype=torch_dtype_complex, device=device),
            3: torch.tensor([[0, 0, 0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [0, 0, 0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, -1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2, 0, 0],
                             [0, 0, 0, 0, 0, 1 / sqrt_2, 1j / sqrt_2]], dtype=torch_dtype_complex, device=device),
        } #没有使用
        self.Us_openmx2wiki = {
            0: torch.eye(1, dtype=torch_dtype).to(device=device),
            1: torch.eye(3, dtype=torch_dtype)[[1, 2, 0]].to(device=device),
            2: torch.eye(5, dtype=torch_dtype)[[2, 4, 0, 3, 1]].to(device=device),
            3: torch.eye(7, dtype=torch_dtype)[[6, 4, 2, 0, 1, 3, 5]].to(device=device)
        }#生成旋转矩阵，行变换
        self.Us_wiki2openmx = {k: v.T for k, v in self.Us_openmx2wiki.items()} #生成旋转矩阵，列变换

    def rotate_e3nn_v(self, v, R, l, order_xyz=True):
        if self.spinful:
            raise NotImplementedError
        assert len(R.shape) == 2
        if order_xyz:
            R_e3nn = self.rotate_matrix_convert(R)
        else:
            R_e3nn = R
        return v @ Irrep(l, 1).D_from_matrix(R_e3nn)

    def rotate_openmx_H_old(self, H, R, l_lefts, l_rights, order_xyz=True):
        assert len(R.shape) == 2
        if order_xyz:
            R_e3nn = self.rotate_matrix_convert(R)
        else:
            R_e3nn = R

        block_lefts = []
        for l_left in l_lefts:
            block_lefts.append(
                self.Us_openmx2wiki[l_left].T @ Irrep(l_left, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_left])
        rotation_left = torch.block_diag(*block_lefts)

        block_rights = []
        for l_right in l_rights:
            block_rights.append(
                self.Us_openmx2wiki[l_right].T @ Irrep(l_right, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_right])
        rotation_right = torch.block_diag(*block_rights)

        return torch.einsum("cd,ca,db->ab", H, rotation_left, rotation_right)

    def rotate_openmx_H(self, H, R, l_lefts, l_rights, rotate_back=False, order_xyz=False): #9*9张量类型的哈密顿量，3*3的局域坐标系下的单位向量，原子i的轨道类型，原子j的轨道类型，order_xyz=True
        # spin-1/2 is writed by gongxx
        assert len(R.shape) == 2 #R.shape=（3,3）
        if order_xyz:
            R_e3nn = self.rotate_matrix_convert(R)
        else:
            R_e3nn = R
        irreps_left = Irreps([(1, (int(l), 1)) for l in l_lefts]) #输入[(1, (0, 1)), (1, (1, 1)), (1, (2, 1))]，输出1x0e+1x1e+1x2e
        irreps_right = Irreps([(1, (int(l), 1)) for l in l_rights]) #输入[(1, (0, 1)), (1, (1, 1)), (1, (2, 1))]，输出1x0e+1x1e+1x2e
        U_left = irreps_left.D_from_matrix(R_e3nn)
        U_right = irreps_right.D_from_matrix(R_e3nn) #得到9*9的矩阵，对角元是1-3-5直和
        # openmx2wiki_left = torch.block_diag(*[self.Us_openmx2wiki[l] for l in l_lefts]) #从提供的张量创建一个块对角矩阵。对角元是1-3-5直和
        # openmx2wiki_right = torch.block_diag(*[self.Us_openmx2wiki[l] for l in l_rights])
        # if self.spinful:
        #     U_left = torch.kron(self.D_one_half(R_e3nn), U_left)
        #     U_right = torch.kron(self.D_one_half(R_e3nn), U_right)
        #     openmx2wiki_left = torch.block_diag(openmx2wiki_left, openmx2wiki_left)
        #     openmx2wiki_right = torch.block_diag(openmx2wiki_right, openmx2wiki_right)
        # return openmx2wiki_left.T @ U_left.transpose(-1, -2).conj() @ openmx2wiki_left @ H \
        #        @ openmx2wiki_right.T @ U_right @ openmx2wiki_right #.transpose(-1, -2)调整维度位置，即转置；逐元素返回复共轭，复数的复共轭是通过改变其虚部的符号来获得的。
        if rotate_back:
            return U_left.transpose(-1, -2).conj() @ H @ U_right
        else:
            return U_left @ H @ U_right.transpose(-1, -2).conj()

    def rotate_openmx_phiVdphi(self, phiVdphi, R, l_lefts, l_rights, order_xyz=True):
        if self.spinful:
            raise NotImplementedError
        assert phiVdphi.shape[-1] == 3
        assert len(R.shape) == 2
        if order_xyz:
            R_e3nn = self.rotate_matrix_convert(R)
        else:
            R_e3nn = R
        block_lefts = []
        for l_left in l_lefts:
            block_lefts.append(
                self.Us_openmx2wiki[l_left].T @ Irrep(l_left, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_left])
        rotation_left = torch.block_diag(*block_lefts)

        block_rights = []
        for l_right in l_rights:
            block_rights.append(
                self.Us_openmx2wiki[l_right].T @ Irrep(l_right, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_right])
        rotation_right = torch.block_diag(*block_rights)

        rotation_x = self.Us_openmx2wiki[1].T @ Irrep(1, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[1]

        return torch.einsum("def,da,eb,fc->abc", phiVdphi, rotation_left, rotation_right, rotation_x)

    def wiki2openmx_H(self, H, l_left, l_right):
        if self.spinful:
            raise NotImplementedError
        return self.Us_openmx2wiki[l_left].T @ H @ self.Us_openmx2wiki[l_right]

    def openmx2wiki_H(self, H, l_left, l_right):
        if self.spinful:
            raise NotImplementedError
        return self.Us_openmx2wiki[l_left] @ H @ self.Us_openmx2wiki[l_right].T

    def rotate_matrix_convert(self, R): #输入3*3的局域坐标系下的单位矢量，旋转单位矢量
        return R.index_select(0, R.new_tensor([1, 2, 0]).int()).index_select(1, R.new_tensor([1, 2, 0]).int())
        #.new_tensor()能够在深拷贝的同时提供更细致的dtype和device属性的控制。在默认参数下，即tensor.new_tensor(x)等同于x.copy().detach()；R.new_tensor([1, 2, 0]).int()输出tensor([1, 2, 0])
        # tensor.new_tensor(x, requires_grad=True)则等同于x.clone().detach().requires_grad_(True)。
        # .index_select返回的是沿着输入张量的指定维度的指定索引号进行索引的张量子集
        #相当于左乘了一个行旋转矩阵:[[0,1,0],   右乘一个列旋转矩阵：[[0,0,1],
        #                       [0,0,1],                     [1,0,0],
        #                       [1,0,0]]                     [0,1,0]]
    def D_one_half(self, R):
        # writed by gongxx
        assert self.spinful
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2  # parity index
        alpha, beta, gamma = matrix_to_angles(R)
        J = torch.tensor([[1, 1], [1j, -1j]], dtype=self.dtype) / 1.4142135623730951  # <1/2 mz|1/2 my>
        Uz1 = self._sp_z_rot(alpha)
        Uy = J @ self._sp_z_rot(beta) @ J.T.conj()
        Uz2 = self._sp_z_rot(gamma)
        return Uz1 @ Uy @ Uz2

    def _sp_z_rot(self, angle):
        # writed by gongxx
        assert self.spinful
        M = torch.zeros([*angle.shape, 2, 2], dtype=self.dtype)
        inds = torch.tensor([0, 1])
        freqs = torch.tensor([0.5, -0.5], dtype=self.dtype)
        M[..., inds, inds] = torch.exp(- freqs * (1j) * angle[..., None])
        return M


def get_rh(input_dir, output_dir, target='hamiltonian'): #这里的input_dir和output_dir是相同的绝对路径example/work_dir/dataset/processed/0-575
    torch_device = torch.device('cpu')
    assert target in ['hamiltonian', 'phiVdphi']
    file_name = {
        'hamiltonian': 'hamiltonians.h5',
        'phiVdphi': 'phiVdphi.h5',
    }[target]
    prime_file_name = {
        'hamiltonian': 'rh.h5',
        'phiVdphi': 'rphiVdphi.h5',
    }[target]
    assert os.path.exists(os.path.join(input_dir, file_name)) #断言processed/0-575文件夹中存在hamiltonians.h5文件
    assert os.path.exists(os.path.join(input_dir, 'rc.h5'))
    assert os.path.exists(os.path.join(input_dir, 'orbital_types.dat'))
    assert os.path.exists(os.path.join(input_dir, 'info.json'))

    #返回每个原子的总轨道数，相同原子即[72个9的列表]和orbital_types.dat中每个原子的轨道类型[0,1,2]或加了z基的[0,0,1,1,2]
    atom_num_orbital = torch.tensor(np.loadtxt(os.path.join(input_dir, 'num_orbital_per_atom.dat')))
    orbital_types = torch.tensor(np.loadtxt(os.path.join(input_dir, 'orbital_types.dat')))
    
    nsite = len(atom_num_orbital) #36个坐标，总的原子数
    with open(os.path.join(input_dir, 'info.json'), 'r') as info_f:
        info_dict = json.load(info_f) #{"nsites": 72, "isorthogonal": false, "isspinful": false, "norbits": 5832}
        spinful = info_dict["isspinful"] #false
        
    with h5py.File(os.path.join(input_dir, file_name), 'r') as fid_H, \
         h5py.File(os.path.join(input_dir, 'rc.h5'), 'r') as fid_rc, \
         h5py.File(os.path.join(output_dir, prime_file_name), 'w') as fid_rh:
             
        assert '[0, 0, 0, 1, 1]' in fid_H.keys() #断言第一个晶胞内的第1个原子和该原子1的哈密顿量的key存在
        h5_dtype = fid_H['[0, 0, 0, 1, 1]'].dtype #对应key[0, 0, 0, 1, 1]的9*9的小哈密顿量的数据类型
        torch_dtype, torch_dtype_real, torch_dtype_complex = dtype_dict[h5_dtype.type]#siesta是np.float32: (torch.float32, torch.float32, torch.complex64),transiesta是np.float64: (torch.float64, torch.float64, torch.complex128),
        rotate_kernel = Rotate(torch_dtype, torch_dtype_real=torch_dtype_real, torch_dtype_complex=torch_dtype_complex,
                               device=torch_device, spinful=spinful) #将哈密顿量旋转，并转为与key[0, 0, 0, 1, 1]的9*9的小哈密顿量相同的数据类型
        for key_str, hamiltonian in fid_H.items():
            if key_str not in fid_rc:
                warnings.warn(f'Hamiltonian matrix block ({key_str}) do not have local coordinate')
                continue

            rotation_matrix = torch.tensor(
                np.array(fid_rc[key_str]), dtype=torch_dtype_real, device=torch_device
            )
            key = json.loads(key_str)
            atom_i, atom_j = key[3], key[4]

            assert 0 <= atom_i < nsite
            assert 0 <= atom_j < nsite

            if target == 'hamiltonian':
                rotated_hamiltonian = rotate_kernel.rotate_openmx_H(
                    torch.tensor(np.array(hamiltonian)),
                    rotation_matrix,
                    orbital_types[atom_i],
                    orbital_types[atom_j],
                    rotate_back=False,
                )
            elif target == 'phiVdphi':
                rotated_hamiltonian = rotate_kernel.rotate_openmx_phiVdphi(
                    torch.tensor(hamiltonian),
                    rotation_matrix,
                    orbital_types[atom_i],
                    orbital_types[atom_j],
                    rotate_back=False,
                )
            fid_rh[key_str] = rotated_hamiltonian.numpy()
    
        


def rotate_back(input_dir, output_dir, target='hamiltonian'):
    torch_device = torch.device('cpu')
    assert target in ['hamiltonian', 'phiVdphi']
    file_name = {
        'hamiltonian': 'hamiltonians_pred.h5',
        'phiVdphi': 'phiVdphi_pred.h5',
    }[target]
    prime_file_name = {
        'hamiltonian': 'rh_pred.h5',
        'phiVdphi': 'rphiVdphi_pred.h5',
    }[target]
    assert os.path.exists(os.path.join(input_dir, prime_file_name)) #断言文件夹中存在预测的哈密顿量矩阵原始文件rh_pred.h5文件
    assert os.path.exists(os.path.join(input_dir, 'rc.h5'))
    assert os.path.exists(os.path.join(input_dir, 'orbital_types.dat'))
    assert os.path.exists(os.path.join(input_dir, 'info.json'))

    atom_num_orbital = torch.tensor(np.loadtxt(os.path.join(input_dir, 'num_orbital_per_atom.dat')))
    orbital_types = torch.tensor(np.loadtxt(os.path.join(input_dir, 'orbital_types.dat')))
    nsite = len(atom_num_orbital) #预测的结构原子数
    
    with open(os.path.join(input_dir, 'info.json'), 'r') as info_f:
        info_dict = json.load(info_f)
        spinful = info_dict["isspinful"] #True

    with h5py.File(os.path.join(input_dir, 'rc.h5'), 'r') as fid_rc, \
         h5py.File(os.path.join(input_dir, prime_file_name), 'r') as fid_rh, \
         h5py.File(os.path.join(output_dir, file_name), 'w') as fid_H:
    #读取预测得到的局域坐标系下的哈密顿量rh_pred.h5
    #旋转到全局坐标系下的哈密顿量矩阵，保存到hamiltonians_pred.h5中
        assert '[0, 0, 0, 1, 1]' in fid_rh.keys()
        h5_dtype = fid_rh['[0, 0, 0, 1, 1]'].dtype
        torch_dtype, torch_dtype_real, torch_dtype_complex = dtype_dict[h5_dtype.type]
        rotate_kernel = Rotate(torch_dtype, torch_dtype_real=torch_dtype_real, torch_dtype_complex=torch_dtype_complex,
                               device=torch_device, spinful=spinful)
        for key_str, rotated_hamiltonian in fid_rh.items():
            assert key_str in fid_rc
    
            rotation_matrix = torch.tensor(
                np.array(fid_rc[key_str]), dtype=torch_dtype_real, device=torch_device
            )
            key = json.loads(key_str)
            atom_i, atom_j = key[3], key[4]
    
            assert 0 <= atom_i < nsite
            assert 0 <= atom_j < nsite
    
            if target == 'hamiltonian':
                hamiltonian = rotate_kernel.rotate_openmx_H(
                    torch.tensor(np.array(rotated_hamiltonian)),
                    rotation_matrix,
                    orbital_types[atom_i],
                    orbital_types[atom_j],
                    rotate_back=True,
                )
            elif target == 'phiVdphi':
                hamiltonian = rotate_kernel.rotate_openmx_phiVdphi(
                    torch.tensor(rotated_hamiltonian),
                    rotation_matrix,
                    orbital_types[atom_i],
                    orbital_types[atom_j],
                    rotate_back=True,
                )
            fid_H[key_str] = hamiltonian.numpy()
