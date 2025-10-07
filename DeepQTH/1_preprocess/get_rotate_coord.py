import os
import json

import h5py #安装没问题
import numpy as np
import torch #torch包安装没问题


class Neighbours:
    def __init__(self):
        self.Rs = []
        self.dists = []
        self.eijs = []
        self.indices = []

    def __str__(self):
        return 'Rs: {}\ndists: {}\neijs: {}\nindices: {}'.format(
            self.Rs, self.dists, self.eijs, self.indices)

#eij是原子i指向其所有邻居原子的向量列表（坐标差），neighbours_i是原子i的所有邻居原子对象，一个包含R\dists\eijs\indices的属性
def _get_local_coordinate(eij, neighbours_i, gen_rc_idx=False, atom_j=None, atom_j_R=None, r2_rand=False):
    if gen_rc_idx:
        rc_idx = np.full(8, np.nan, dtype=np.int32)
        assert r2_rand is False
        assert atom_j is not None, 'atom_j must be specified when gen_rc_idx is True'
        assert atom_j_R is not None, 'atom_j_R must be specified when gen_rc_idx is True'
    else:
        rc_idx = None
    if r2_rand:
        r2_list = []

    if not np.allclose(eij.detach(), torch.zeros_like(eij)): #该邻居原子j不都是原子i本身时
        r1 = eij #原子i到某个邻近原子j的向量
        if gen_rc_idx:
            rc_idx[0] = atom_j
            rc_idx[1:4] = atom_j_R
    else:#该邻居原子j（第一个邻居）是原子i本身时，邻居原子是它自己
        r1 = neighbours_i.eijs[1] #r1置原子i到第二近的邻居原子j的向量
        if gen_rc_idx:
            rc_idx[0] = neighbours_i.indices[1]
            rc_idx[1:4] = neighbours_i.Rs[1]
    r2_flag = None
    for r2, r2_index, r2_R in zip(neighbours_i.eijs[1:], neighbours_i.indices[1:], neighbours_i.Rs[1:]): #除了自己原子（第一个最近邻原子）之外的不共线的另一个邻居原子
        if torch.norm(torch.cross(r1, r2, dim=-1)) > 1e-6: #找到与向量r1不共线的向量r2，即r1和r2张成的平行四边形的面积要>1e-6，或r1和r2叉积得到的垂直向量不为0，一旦找到次近邻的不共线的r2向量就break。
            if gen_rc_idx:
                rc_idx[4] = r2_index
                rc_idx[5:8] = r2_R
            r2_flag = True #找到了不共线的r2后就break
            if r2_rand:
                if (len(r2_list) == 0) or (torch.norm(r2_list[0]) + 0.5 > torch.norm(r2)):
                    r2_list.append(r2)
                else:
                    break
            else:
                break
    if r2_flag is None:
        x1, y1, z1 = r1
        if not np.isclose(z1, 0):
            r2 = np.array([-y1, x1, 0])
        else:
            r2 = np.array([0, 0, 1])
    
    if r2_rand:
        # print(f"r2 is randomly chosen from {len(r2_list)} candidates")
        r2 = r2_list[np.random.randint(len(r2_list))]
    #获得局域坐标的三个单位向量
    local_coordinate_1 = r1 / torch.norm(r1)
    r2_proj = r2 - torch.dot(r2, local_coordinate_1) * local_coordinate_1
    local_coordinate_2 = r2_proj / torch.norm(r2_proj)
    local_coordinate_3 = torch.cross(local_coordinate_1, local_coordinate_2, dim=-1)
    # local_coordinate_1 = r1 / torch.norm(r1)#计算局域坐标向量 r1 的单位向量,torch.norm(r1) 用于计算向量 r1 的范数（或模），即向量的长度。这是通过计算向量各个分量的平方和然后取平方根的方式来实现的。
    # local_coordinate_2 = torch.cross(r1, r2, dim=-1) / torch.norm(torch.cross(r1, r2, dim=-1)) #计算局域坐标向量2
    # local_coordinate_3 = torch.cross(local_coordinate_1, local_coordinate_2, dim=-1) #计算局域坐标向量3
    return torch.stack([local_coordinate_1, local_coordinate_2, local_coordinate_3], dim=-1), rc_idx #dim=-1,返回3*3的局域坐标下的单位向量，rc_idx=None

#example/work_dir/dataset/processed/0-575,example/work_dir/dataset/processed/0-575,9.0,False,False,"",True,hamiltionians.h5,False,None
def get_rc(input_dir, output_dir, radius, r2_rand=False, gen_rc_idx=False, gen_rc_by_idx="", neighbour_file='overlaps.h5', if_require_grad=False, cart_coords=None):
    if not if_require_grad:
        assert os.path.exists(os.path.join(input_dir, 'site_positions.dat')), 'No site_positions.dat found in {}'.format(input_dir) #断言存在site_positions.dat文件
        cart_coords = torch.tensor(np.loadtxt(os.path.join(input_dir, 'site_positions.dat'))) #36*3的原子位置笛卡尔坐标矩阵
    else:
        assert cart_coords is not None, 'cart_coords must be provided if "if_require_grad" is True'#计算梯度有什么作用？
    assert os.path.exists(os.path.join(input_dir, 'lat.dat')), 'No lat.dat found in {}'.format(input_dir)
    lattice = torch.tensor(np.loadtxt(os.path.join(input_dir, 'lat.dat')), dtype=cart_coords.dtype) #3*3的晶格矢量

    rc_dict = {}
    if gen_rc_idx:
        assert r2_rand is False, 'r2_rand must be False when gen_rc_idx is True'
        assert gen_rc_by_idx == "", 'gen_rc_by_idx must be "" when gen_rc_idx is True'
        rc_idx_dict = {}
    neighbours_dict = {}
    if gen_rc_by_idx != "":
        # print(f'get local coordinate using {os.path.join(gen_rc_by_idx, "rc_idx.h5")} from: {input_dir}')
        assert os.path.exists(os.path.join(gen_rc_by_idx, "rc_idx.h5")), 'Atomic indices for constructing rc rc_idx.h5 is not found in {}'.format(gen_rc_by_idx)
        fid_rc_idx = h5py.File(os.path.join(gen_rc_by_idx, "rc_idx.h5"), 'r')
        for key_str, rc_idx in fid_rc_idx.items():
            key = json.loads(key_str)
            # R = torch.tensor([key[0], key[1], key[2]])
            atom_i = key[3] - 1
            cart_coords_i = cart_coords[atom_i]

            r1 = cart_coords[rc_idx[0]] + torch.tensor(rc_idx[1:4]).type(cart_coords.dtype) @ lattice - cart_coords_i
            r2 = cart_coords[rc_idx[4]] + torch.tensor(rc_idx[5:8]).type(cart_coords.dtype) @ lattice - cart_coords_i
            local_coordinate_1 = r1 / torch.norm(r1)
            local_coordinate_2 = torch.cross(r1, r2) / torch.norm(torch.cross(r1, r2))
            local_coordinate_3 = torch.cross(local_coordinate_1, local_coordinate_2)

            rc_dict[key_str] = torch.stack([local_coordinate_1, local_coordinate_2, local_coordinate_3], dim=-1)
        fid_rc_idx.close()
    else:
        # print("get local coordinate from:", input_dir)
        assert os.path.exists(os.path.join(input_dir, neighbour_file)), 'No {} found in {}'.format(neighbour_file, input_dir)
        with h5py.File(os.path.join(input_dir, neighbour_file), 'r') as fid_OLP: #读取哈密顿量文件/重叠积分矩阵文件
            for key_str in fid_OLP.keys(): #遍历所有矩阵元素不为0的项，然后找出原子间距离小于截断半径的所有相邻的原子对的isc，距离，笛卡尔坐标差（局域向量），近邻原子索引
                key = json.loads(key_str) #key是[3个isc,atom1,atom2],将一个JSON字符串解析为Python对象。
                R = torch.tensor([key[0], key[1], key[2]])
                atom_i = key[3] #中心原子
                atom_j = key[4] #邻居原子
                cart_coords_i = cart_coords[atom_i] #第i个原子的笛卡尔坐标
                cart_coords_j = cart_coords[atom_j] + R.type(cart_coords.dtype) @ lattice #这里@是表示矩阵乘法，求出整个超胞内的某个晶胞内的原子j的实际笛卡尔坐标，R是轨道/原子所属的晶胞（unit cell）的索引，它用于确定轨道/原子在超胞中的位置。
                eij = cart_coords_j - cart_coords_i #两个笛卡尔坐标之差，即为局域坐标下，一个原子指向另一个原子的向量
                dist = torch.norm(eij) #返回所给tensor的矩阵范数或向量范数，即两个原子间的距离
                if radius > 0 and dist > radius: #过滤掉大于截断半径的原子，如果radius=-1，则不截断，表示考虑结构中所有原子为原子i的邻居原子。这里不仅是为了寻找局域坐标，也是为了找出截断半径内的邻居原子
                    continue
                if atom_i not in neighbours_dict:
                    neighbours_dict[atom_i] = Neighbours() #存放每个原子的邻居原子信息的字典。key是每个原子i，值是原子i的邻居原子的类对象。邻居原子对象有四个属性：所属晶胞索引,到原子i的距离,原子i指向该邻居原子的向量,该邻居原子的索引
                neighbours_dict[atom_i].Rs.append(R) #邻近原子/轨道所属的晶胞（unit cell）的索引。它用于确定原子/轨道在超胞中的位置。
                neighbours_dict[atom_i].dists.append(dist) #邻居原子j到原子i的距离
                neighbours_dict[atom_i].eijs.append(eij) #原子i指向邻居原子j的向量
                neighbours_dict[atom_i].indices.append(atom_j) #邻居原子j的索引
            
        #neighbours_dict中每个key（atom_i）对应的值是多个邻居原子，是一个列表
        for atom_i, neighbours_i in neighbours_dict.items():
            neighbours_i.Rs = torch.stack(neighbours_i.Rs)#原子i的所有邻居原子的晶胞索引进行堆叠,如tensor([-1, -1,  0])和([-1, -1,  0])变为tensor([[-1, -1,  0],[1, 1,  0]])
            neighbours_i.dists = torch.tensor(neighbours_i.dists, dtype=cart_coords.dtype)#原子i的所有邻居原子到i的距离，转为张量类型
            neighbours_i.eijs = torch.stack(neighbours_i.eijs)#原子i指向所有邻居原子的向量堆叠，两个原子的笛卡尔坐标差（局域向量）
            neighbours_i.indices = torch.tensor(neighbours_i.indices)#原子i的所有邻近原子j的索引，转为张量的数据类型

            neighbours_i.dists, sorted_index = torch.sort(neighbours_i.dists)#按原子间的距离从小到大排序，返回排序后的dists和排序后的元素在原矩阵中的索引
            neighbours_i.Rs = neighbours_i.Rs[sorted_index]#按距离排序Rs,eijs,indices
            neighbours_i.eijs = neighbours_i.eijs[sorted_index]
            neighbours_i.indices = neighbours_i.indices[sorted_index]
            #如果两个数组在默认的公差范围内按元素方式相等，则返回True。detach()是起别名复制，两个在内存中实际是一个东西。
            assert np.allclose(neighbours_i.eijs[0].detach(), torch.zeros_like(neighbours_i.eijs[0])), 'eijs[0] should be zero' #eijs[0]是因为存在原子自身交互的哈密顿量，相同原子的笛卡尔坐标之差为一个(3,)的零向量。

            for R, eij, atom_j, atom_j_R in zip(neighbours_i.Rs, neighbours_i.eijs, neighbours_i.indices, neighbours_i.Rs):
                key_str = str(list([*R.tolist(), atom_i, atom_j.item()]))
                if gen_rc_idx:
                    rc_dict[key_str], rc_idx_dict[key_str] = _get_local_coordinate(eij, neighbours_i, gen_rc_idx, atom_j, atom_j_R)
                else:
                    rc_dict[key_str] = _get_local_coordinate(eij, neighbours_i, r2_rand=r2_rand)[0] #返回原子i和每个邻居原子j的，以原子i为中心的3*3单位向量局域坐标系，有多少条边就有多少个局域坐标系。
                    # print(rc_dict[key_str].shape)
            # print("\n")

    if if_require_grad:
        return rc_dict
    else:
        if os.path.exists(os.path.join(output_dir, 'rc_julia.h5')):
            rc_old_flag = True
            fid_rc_old = h5py.File(os.path.join(output_dir, 'rc_julia.h5'), 'r')
        else:
            rc_old_flag = False
        with h5py.File(os.path.join(output_dir, 'rc.h5'), 'w') as fid_rc:
            for k, v in rc_dict.items():  # key是原子索引[3个isc,atom1,atom2], value是3x3单位向量
                if rc_old_flag:
                    assert np.allclose(v, fid_rc_old[k][...], atol=1e-4), f"{k}, {v}, {fid_rc_old[k][...]}"
                # 如果要保存矩阵，应该用 create_dataset，而不是直接赋值
                fid_rc[k] = v
        # fid_rc = h5py.File(os.path.join(output_dir, 'rc.h5'), 'w') #执行这里
        # for k, v in rc_dict.items():#key是原子索引[3个isc,atom1,atom2],value是atom1的局域坐标系的3*3的单位向量
        #     if rc_old_flag:
        #         assert np.allclose(v, fid_rc_old[k][...], atol=1e-4), f"{k}, {v}, {fid_rc_old[k][...]}"
        #     fid_rc[k] = v
        # fid_rc.close()
        if gen_rc_idx:
            fid_rc_idx = h5py.File(os.path.join(output_dir, 'rc_idx.h5'), 'w')
            for k, v in rc_idx_dict.items():
                fid_rc_idx[k] = v
            fid_rc_idx.close()