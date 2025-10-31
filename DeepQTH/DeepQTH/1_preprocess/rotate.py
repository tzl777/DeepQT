import json
import os.path
import warnings

import numpy as np
import h5py
import torch
from e3nn.o3 import Irreps

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

class Rotate:
    def __init__(self, torch_dtype, torch_dtype_real=torch.float64, torch_dtype_complex=torch.cdouble,
                 device=torch.device('cpu'), spinful=False):
        self.dtype = torch_dtype
        self.torch_dtype_real = torch_dtype_real #torch.float32
        self.device = device
        self.spinful = spinful #False

    def rotate_openmx_H(self, H, R, l_lefts, l_rights, rotate_back=False, order_xyz=False):
        assert len(R.shape) == 2
        if order_xyz:
            R_e3nn = self.rotate_matrix_convert(R)
        else:
            R_e3nn = R
        irreps_left = Irreps([(1, (int(l), 1)) for l in l_lefts]) #1x0e+1x1e+1x2e
        irreps_right = Irreps([(1, (int(l), 1)) for l in l_rights]) #1x0e+1x1e+1x2e
        U_left = irreps_left.D_from_matrix(R_e3nn)
        U_right = irreps_right.D_from_matrix(R_e3nn)

        if rotate_back:
            return U_left.transpose(-1, -2).conj() @ H @ U_right
        else:
            return U_left @ H @ U_right.transpose(-1, -2).conj()


    def rotate_matrix_convert(self, R):
        return R.index_select(0, R.new_tensor([1, 2, 0]).int()).index_select(1, R.new_tensor([1, 2, 0]).int())



def get_rh(input_dir, output_dir, target='hamiltonian'): 
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
    assert os.path.exists(os.path.join(input_dir, file_name))
    assert os.path.exists(os.path.join(input_dir, 'rc.h5'))
    assert os.path.exists(os.path.join(input_dir, 'orbital_types.dat'))
    assert os.path.exists(os.path.join(input_dir, 'info.json'))

    atom_num_orbital = torch.tensor(np.loadtxt(os.path.join(input_dir, 'num_orbital_per_atom.dat')))
    orbital_types = torch.tensor(np.loadtxt(os.path.join(input_dir, 'orbital_types.dat')))
    
    nsite = len(atom_num_orbital)
    with open(os.path.join(input_dir, 'info.json'), 'r') as info_f:
        info_dict = json.load(info_f)
        spinful = info_dict["isspinful"] #false
        
    with h5py.File(os.path.join(input_dir, file_name), 'r') as fid_H, \
         h5py.File(os.path.join(input_dir, 'rc.h5'), 'r') as fid_rc, \
         h5py.File(os.path.join(output_dir, prime_file_name), 'w') as fid_rh:
        
        assert '[0, 0, 0, 0, 0]' in fid_H.keys()
        h5_dtype = fid_H['[0, 0, 0, 0, 0]'].dtype
        torch_dtype, torch_dtype_real, torch_dtype_complex = dtype_dict[h5_dtype.type]
        rotate_kernel = Rotate(torch_dtype, torch_dtype_real=torch_dtype_real, torch_dtype_complex=torch_dtype_complex,
                               device=torch_device, spinful=spinful)
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
    assert os.path.exists(os.path.join(input_dir, prime_file_name))
    assert os.path.exists(os.path.join(input_dir, 'rc.h5'))
    assert os.path.exists(os.path.join(input_dir, 'orbital_types.dat'))
    assert os.path.exists(os.path.join(input_dir, 'info.json'))

    atom_num_orbital = torch.tensor(np.loadtxt(os.path.join(input_dir, 'num_orbital_per_atom.dat')))
    orbital_types = torch.tensor(np.loadtxt(os.path.join(input_dir, 'orbital_types.dat')))
    nsite = len(atom_num_orbital)
    
    with open(os.path.join(input_dir, 'info.json'), 'r') as info_f:
        info_dict = json.load(info_f)
        spinful = info_dict["isspinful"] #True

    with h5py.File(os.path.join(input_dir, 'rc.h5'), 'r') as fid_rc, \
         h5py.File(os.path.join(input_dir, prime_file_name), 'r') as fid_rh, \
         h5py.File(os.path.join(output_dir, file_name), 'w') as fid_H:

        # print(fid_rh.keys())
        assert '[0, 0, 0, 0, 0]' in fid_rh.keys()
        h5_dtype = fid_rh['[0, 0, 0, 0, 0]'].dtype
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
