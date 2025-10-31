import os
import json

import h5py
import numpy as np
import torch


class Neighbours:
    def __init__(self):
        self.Rs = []
        self.dists = []
        self.eijs = []
        self.indices = []

    def __str__(self):
        return 'Rs: {}\ndists: {}\neijs: {}\nindices: {}'.format(
            self.Rs, self.dists, self.eijs, self.indices)

def _get_local_coordinate(eij, neighbours_i):
    rc_idx = None
    if not np.allclose(eij.detach(), torch.zeros_like(eij)):
        r1 = eij
    else:
        r1 = neighbours_i.eijs[1]
    r2_flag = None
    for r2, r2_index, r2_R in zip(neighbours_i.eijs[1:], neighbours_i.indices[1:], neighbours_i.Rs[1:]): 
        if torch.norm(torch.cross(r1, r2, dim=-1)) > 1e-6:
            r2_flag = True
            break
    if r2_flag is None:
        x1, y1, z1 = r1
        if not np.isclose(z1, 0):
            r2 = np.array([-y1, x1, 0])
        else:
            r2 = np.array([0, 0, 1])
    
    local_coordinate_1 = r1 / torch.norm(r1)
    r2_proj = r2 - torch.dot(r2, local_coordinate_1) * local_coordinate_1
    local_coordinate_2 = r2_proj / torch.norm(r2_proj)
    local_coordinate_3 = torch.cross(local_coordinate_1, local_coordinate_2, dim=-1)

    # local_coordinate_2 = torch.cross(r1, r2, dim=-1) / torch.norm(torch.cross(r1, r2, dim=-1))
    # local_coordinate_3 = torch.cross(local_coordinate_1, local_coordinate_2, dim=-1)
    return torch.stack([local_coordinate_1, local_coordinate_2, local_coordinate_3], dim=-1), rc_idx


def get_rc(input_dir, output_dir, radius, neighbour_file='overlaps.h5'):

    assert os.path.exists(os.path.join(input_dir, 'site_positions.dat')), 'No site_positions.dat found in {}'.format(input_dir)
    cart_coords = torch.tensor(np.loadtxt(os.path.join(input_dir, 'site_positions.dat')))
    
    assert os.path.exists(os.path.join(input_dir, 'lat.dat')), 'No lat.dat found in {}'.format(input_dir)
    lattice = torch.tensor(np.loadtxt(os.path.join(input_dir, 'lat.dat')), dtype=cart_coords.dtype)

    rc_dict = {}
    neighbours_dict = {}

    assert os.path.exists(os.path.join(input_dir, neighbour_file)), 'No {} found in {}'.format(neighbour_file, input_dir)
    with h5py.File(os.path.join(input_dir, neighbour_file), 'r') as fid_OLP:
        for key_str in fid_OLP.keys(): 
            key = json.loads(key_str)
            R = torch.tensor([key[0], key[1], key[2]])
            atom_i = key[3]
            atom_j = key[4]
            cart_coords_i = cart_coords[atom_i]
            cart_coords_j = cart_coords[atom_j] + R.type(cart_coords.dtype) @ lattice
            eij = cart_coords_j - cart_coords_i
            dist = torch.norm(eij)
            # Filter out the atoms that are larger than the truncation radius. If radius = -1, no truncation is performed, indicating that all neighboring atoms of atom i in the structure are considered.
            if radius > 0 and dist > radius:
                continue
            if atom_i not in neighbours_dict:
                neighbours_dict[atom_i] = Neighbours()
            neighbours_dict[atom_i].Rs.append(R)
            neighbours_dict[atom_i].dists.append(dist)
            neighbours_dict[atom_i].eijs.append(eij)
            neighbours_dict[atom_i].indices.append(atom_j)
        
    for atom_i, neighbours_i in neighbours_dict.items():
        neighbours_i.Rs = torch.stack(neighbours_i.Rs)
        neighbours_i.dists = torch.tensor(neighbours_i.dists, dtype=cart_coords.dtype)
        neighbours_i.eijs = torch.stack(neighbours_i.eijs)
        neighbours_i.indices = torch.tensor(neighbours_i.indices)

        neighbours_i.dists, sorted_index = torch.sort(neighbours_i.dists)
        neighbours_i.Rs = neighbours_i.Rs[sorted_index]
        neighbours_i.eijs = neighbours_i.eijs[sorted_index]
        neighbours_i.indices = neighbours_i.indices[sorted_index]

        assert np.allclose(neighbours_i.eijs[0].detach(), torch.zeros_like(neighbours_i.eijs[0])), 'eijs[0] should be zero'

        for R, eij, atom_j, atom_j_R in zip(neighbours_i.Rs, neighbours_i.eijs, neighbours_i.indices, neighbours_i.Rs):
            key_str = str(list([*R.tolist(), atom_i, atom_j.item()]))
            rc_dict[key_str] = _get_local_coordinate(eij, neighbours_i)[0]
            # print(rc_dict[key_str].shape)
        # print("\n")

    with h5py.File(os.path.join(output_dir, 'rc.h5'), 'w') as fid_rc:
        for k, v in rc_dict.items():
            fid_rc[k] = v
