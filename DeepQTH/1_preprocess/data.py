import warnings
import os
import time
import tqdm
import json
from pymatgen.core.structure import Structure
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from pathos.multiprocessing import ProcessingPool as Pool #多参数输入并行计算

from graph import get_graph


class HData(InMemoryDataset):
    def __init__(self, config: dict, default_dtype_torch, transform=None, pre_transform=None, pre_filter=None): 
        # 必须传入 config 和 default_dtype_torch，保持你的原始参数风格
        if config is None:
            raise ValueError("config must be provided")
        if default_dtype_torch is None:
            raise ValueError("default_dtype_torch must be provided")

        self.processed_data_dir = config['basic']['processed_data_dir']
        self.graph_dir = config['basic']['graph_dir']
        self.data_format = config['basic']['data_format']
        self.target = config['basic']['target']
        self.material_dimension = config['basic']['material_dimension']
        # self.multiprocessing = config['basic']['multiprocessing'] #多进程处理转换为图结构数据
        self.multiprocessing = 0 #单进程处理转换为图结构数据
        self.radius = config['graph']['radius'] #这个半径和预处理的截断半径不同
        self.num_l = config['graph']['num_l'] #球谐函数展开的角量子数
        self.if_lcmp_graph = config['graph']['if_lcmp_graph']
        self.separate_onsite = config['graph']['separate_onsite']
        self.new_sp = config['graph']['new_sp']
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
        if self.separate_onsite is True:
            onsite_str = '-SeparateOnsite'
        else:
            onsite_str = ''
        if self.new_sp:
            new_sp_str = '-NewSP'
        else:
            new_sp_str = ''
        if self.target == 'hamiltonian':
            title = 'HGraph'
        else:
            raise ValueError('Unknown prediction target: {}'.format(self.target))
            
        self.graph_file_name = f'{title}-{self.data_format}-{lcmp_str}{onsite_str}{new_sp_str}.pkl'
        self.data_file = os.path.join(self.graph_dir, self.graph_file_name) #创建图数据的文件路径和名字
        os.makedirs(self.graph_dir, exist_ok=True)
        
        print(f'Graph data file: {self.graph_file_name}')
        if os.path.exists(self.data_file):
            print("Use existing graph data file")
            self.load()
        else:
            super(HData, self).__init__(self.processed_data_dir, transform, pre_transform, pre_filter)

        
        begin = time.time()
        try:
            loaded_data = torch.load(self.data_file) #加载torch.save()保存的模型文件。
        except AttributeError:
            raise RuntimeError('Error in loading graph data file, try to delete it and generate the graph file with the current version of PyG')
        max_element = -1
        if len(loaded_data) == 2:
            warnings.warn('You are using the graph data file with an old version')
            self.data, self.slices = loaded_data
            self.info = {
                "spinful": False,
                "index_to_Z": torch.arange(max_element + 1), #max_element=-1，创建一个空的张量，因为没有任何数字满足这样的条件。
                "Z_to_index": torch.arange(max_element + 1),
            }
        elif len(loaded_data) == 3: #加载新版本图数据,data, slices, dict3个文件
            self.data, self.slices, tmp = loaded_data #tmp是dict(spinful=spinful, index_to_Z=index_to_Z, Z_to_index=Z_to_index)。#False, [83], [第83个位置为0，其余为-1的(100,)张量]
            if isinstance(tmp, dict): #如果对象tmp的类型与参数二的类型（dict）相同则返回 True，否则返回 False。
                self.info = tmp
                print(f"Atomic types: {self.info['index_to_Z'].tolist()}") #取出去重和排序后的原子序数列表，即不同的原子类型列表，[83]
                print(f"Atomic types vectors: {self.info['Z_to_index'].tolist()}")#[第83个位置为0，其余为-1的(100,)张量]
            else:
                warnings.warn('You are using an old version of the graph data file')
                self.info = {
                    "spinful": tmp,
                    "index_to_Z": torch.arange(max_element + 1), #创建一个空的张量，因为没有任何数字满足这样的条件。
                    "Z_to_index": torch.arange(max_element + 1),
                }
        else:
            raise RuntimeError(f'Unexpected format in saved graph file: found {len(loaded_data)} elements')
            
        print(f'Finish loading the processed {len(self)} structures (spinful: {self.info["spinful"]}, '
              f'the number of atomic types: {len(self.info["index_to_Z"])}), cost {time.time() - begin:.0f} seconds')
        
    @property
    def processed_file_names(self): #检查data/processed目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        pass
        return []

    def load(self):
        # 如果文件存在，就直接加载，不再走 process
        if os.path.exists(self.data_file):
            print("Loading existing dataset:", self.data_file)
            self.loaded_data = torch.load(self.data_file)
        else:
            raise RuntimeError(f"Processed file {self.data_file} not found, please run process()")
    
    def process(self):
        print('Process new data file......')
        
        begin = time.time()
        folder_list = []
        for root, dirs, files in os.walk(self.processed_data_dir): #example/work_dir/dataset/processed
            if (self.data_format == 'h5' and 'rc.h5' in files) or (
                    self.data_format == 'npz' and 'rc.npz' in files):
                folder_list.append(root) #example/work_dir/dataset/processed/0-575
        folder_list = sorted(folder_list) #按processed中数字文件夹的顺序排序
        assert len(folder_list) != 0, "Can not find any structure"


        if self.multiprocessing == 0:
            print(f'Use multiprocessing (nodes = num_processors x num_threads = 1 x {torch.get_num_threads()})') #获得用于并行化CPU操作的OpenMP线程数，然后使用单进程运行。
            data_list = [self.process_worker(folder) for folder in tqdm.tqdm(folder_list, ncols=80, leave=False, position=0)] #tqdm.tqdm()实现进度条，输入一个可迭代对象folder_list。输入每一个晶体结构文件夹到process_worker函数中。得到每一个晶体结构的图类型的数据，即一个晶体结构对应一个图data。
        else:
            pool_dict = {} if self.multiprocessing < 0 else {'nodes': self.multiprocessing} #如果self.multiprocessing>=0，pool_dict被赋值为包含一个键值对'nodes': self.multiprocessing 的字典。
            torch_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            #with中代码主要目的是基于条件选择是否使用多进程来并行处理 folder_list 中的任务，并在处理过程中提供了进度条的可视化信息。
            with Pool(**pool_dict) as pool: #Pool()创建了一个进程池，参数**pool_dict是将字典内容解包作为关键字参数传递给ProcessingPool。with语句确保在离开其代码块之前正确关闭进程池，这样可以避免资源泄漏。
                nodes = pool.nodes #获取进程池的节点数，下面打印出多进程处理的相关信息。
                print(f'Use multiprocessing (nodes = num_processors x num_threads = {nodes} x {torch.get_num_threads()})') #处理器数量和线程数量
                data_list = list(tqdm.tqdm(pool.imap(self.process_worker, folder_list), total=len(folder_list), ncols=80, leave=False, position=0)) #对folder_list中的每个元素调用self.process_worker函数，使用进程池并行处理，返回各自结构的图数据。tqdm.tqdm则提供了一个可视化的进度条，用于显示处理进度。total=len(folder_list)设置了进度条的总数。
            torch.set_num_threads(torch_num_threads) #设置 PyTorch 运行时所使用的线程数

        if self.pre_filter is not None: #pre_filter=None
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None: #pre_transform=None
            data_list = [self.pre_transform(d) for d in data_list]
        
        index_to_Z, Z_to_index = self.element_statistics(data_list) #[83], [第83个位置为0，其余为-1的张量]。且遍历所有晶体结构，把每个晶体结构的data.x都变为了[36个0或1等]，这里不是显式。
        self.spinful = data_list[0].spinful #False
        for d in data_list:
            assert self.spinful == d.spinful

        data_list = self.make_mask(data_list) 
        
        data, slices = self.collate(data_list) #将Data或HeteroData对象列表整理为InMemoryDataset的内部存储格式。
        torch.save((data, slices, dict(spinful=self.spinful, index_to_Z=index_to_Z, Z_to_index=Z_to_index)), self.data_file) #将这些数据对象保存到指定路径的文件中，存data, slices, dict3个文件，以便之后可以通过 torch.load() 函数重新加载并使用这些数据。已存储，近19GB的数据。
        print('Finish saving %d structures to %s, have cost %d seconds' % (len(data_list), self.data_file, time.time() - begin))
        

    def process_worker(self, folder, **kwargs):
        stru_id = os.path.split(folder)[-1] #用于将路径分割成头部和尾部两个部分。头部是路径中最后一个斜杠之前的部分（即父目录路径），而尾部是最后一个斜杠之后的部分（通常是文件名或最后一个目录名）。这里得到了每个结构的id：0-575
        print("process dir:", stru_id)
        structure = Structure(np.loadtxt(os.path.join(folder, 'lat.dat')), #加载第i个结构的晶格向量、原子序数和笛卡尔坐标
                              np.loadtxt(os.path.join(folder, 'element.dat')),
                              np.loadtxt(os.path.join(folder, 'site_positions.dat')),
                              coords_are_cartesian=True,
                              to_unit_cell=False) #用pymatgen创建结构

        cart_coords = torch.tensor(structure.cart_coords, dtype=self.default_dtype_torch) #晶体的原子笛卡尔坐标，当前的默认浮点torch.dtype，36*3
        frac_coords = torch.tensor(structure.frac_coords, dtype=self.default_dtype_torch)
        numbers = torch.tensor(structure.atomic_numbers) #List of atomic numbers.tensor([36*83])，36个Bi原子的原子序数列表
        # print("numbers = ", numbers)
        # print(cart_coords)
        structure.lattice.matrix.setflags(write=True) #描述structure.lattice.matrix是否可以写入。
        lattice = torch.tensor(structure.lattice.matrix, dtype=self.default_dtype_torch) #读取structure.lattice.matrix，并转为default_dtype_torch类型
        # print(lattice)
        if self.target == 'E_ij':
            huge_structure = True
        else:
            huge_structure = False # r=self.radius==-1。
        return get_graph(cart_coords, frac_coords, numbers, stru_id, radius=self.radius, material_dimension=self.material_dimension,
                         numerical_tol=1e-8, lattice=lattice, default_dtype_torch=self.default_dtype_torch,
                         tb_folder=folder, data_format=self.data_format, num_l=self.num_l, shortest_path_length=self.shortest_path_length,
                         if_lcmp_graph=self.if_lcmp_graph, separate_onsite=self.separate_onsite, target=self.target, 
                         huge_structure=huge_structure, if_new_sp=self.new_sp, **kwargs) #获得图类型的数据集return data
        

    def element_statistics(self, data_list):
        #data_list[0].x是该晶体结构中所有原子的原子序数列表。torch.unique是从张量中提取不重复的值，即去重后排序。index_to_Z是去重排序后的原子序数列表；inverse_indices是原子序数列表中每个元素在去重排序后列表中的位置索引。
        index_to_Z, inverse_indices = torch.unique(data_list[0].x, sorted=True, return_inverse=True)
        #data_list[0].x是[36个83]的张量，index_to_Z将会是一个包含单一元素[83]的张量，因为所有元素都相同，去重后只剩下一个元素。
        #inverse_indices将会是一个长度为36的张量，每个元素都是0，表示原始列表中的每个元素都对应去重后列表的第一个（也是唯一一个）元素。
        Z_to_index = torch.full((100,), -1, dtype=torch.int64)#创建了一个形状为 (100,) 的张量，所有元素的值都是 -1，数据类型为 64 位整型。
        Z_to_index[index_to_Z] = torch.arange(len(index_to_Z)) #torch.arange生成一个包含不同原子数的张量，包含从 0 开始逐步增加到len(index_to_Z)-1的整数序列
        #Z_to_index[6] = 0， 如果有不同原子，则Z_to_index[26] = 1，以此类推
        # for data in data_list:#遍历所有晶体结构，把每个晶体结构的data.x都转为[36个0]
        #     data.x = Z_to_index[data.x] #data.x是[36个83]的张量，Z_to_index[data.x]返回的是data.x作为每个索引下的值，36个0。如果存在不同原子，则把原子序数列表data.x(numbers)转为从0开始的不同元素的标识。

        return index_to_Z, Z_to_index #[83], [第83个位置为0，其余为-1的(100,)张量]，如果存在不同原子，则不同原子的原子序数位置的值为1，2...

    def generate_orbital(self, atom_list, unique_orbital_basis_number):
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
    
    def make_mask(self, dataset):
        dataset_mask = []
        for data in dataset:

            unique_atom_number = list(dict.fromkeys([x.numpy().tolist() for x in data.x]))
            unique_orbital_basis_number = list(dict.fromkeys(data.atom_num_orbital))

            self.orbital = json.loads(self.generate_orbital(unique_atom_number, unique_orbital_basis_number))
            self.num_orbital = len(self.orbital)
            
            if self.target == 'hamiltonian' or self.target == 'phiVdphi' or self.target == 'density_matrix':
                Oij_value = data.term_real  #旋转后的哈密顿量2016*9*9
                if data.term_real is not None:
                    if_only_rc = False
                else:
                    if_only_rc = True
            elif self.target == 'O_ij':
                if self.O_component == 'H_minimum':
                    Oij_value = data.rvdee + data.rvxc
                elif self.O_component == 'H_minimum_withNA':
                    Oij_value = data.rvna + data.rvdee + data.rvxc
                elif self.O_component == 'H':
                    Oij_value = data.rh
                elif self.O_component == 'Rho':
                    Oij_value = data.rdm
                else:
                    raise ValueError(f'Unknown O_component: {self.O_component}')
                if_only_rc = False
            else:
                raise ValueError(f'Unknown target: {self.target}')
            if if_only_rc == False:
                if not torch.all(data.term_mask): #torch.all测试输入的所有元素是否都为 True，只要不全为True，输出则为False，即所有term_mask中所有元素都应为True
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
                    out_fea_len = self.num_orbital #Bi是81/169
            mask = torch.zeros(data.edge_attr.shape[0], out_fea_len, dtype=torch.int8) #创建2016*81大小的全0张量，int类型
            label = torch.zeros(data.edge_attr.shape[0], out_fea_len, dtype=torch.get_default_dtype()) #同上，但为torch.float32类型

            atomic_number_edge_i = data.x[data.edge_index[0]] #data.edge_index是边索引[2,2016]，取出所有2016条边的起始原子的索引[2016*0-35]，得到data.x[2016*0]，再得到原子序数列表[2016个83]，即起始原子i的原子序数列表
            atomic_number_edge_j = data.x[data.edge_index[1]] #data.edge_index是边索引[2,2016]，取出所有2016条边的终点原子的索引[2016*0-35]，得到data.x[2016*0]，再得到原子序数列表[2016个83]，即终端原子j的原子序数列表
            for index_out, orbital_dict in enumerate(self.orbital):# basic-orbital:[{"83 83": [0, 0]}, {"83 83": [0, 1]}, 。。。。]
                for N_M_str, a_b in orbital_dict.items():
                    # N_M, a_b means: H_{ia, jb} when the atomic number of atom i is N and the atomic number of atom j is M
                    condition_atomic_number_i, condition_atomic_number_j = map(lambda x: int(x), N_M_str.split()) #[83,83]
                    condition_orbital_i, condition_orbital_j = a_b #[0,0],[0,1],.....
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
                            # (atomic_number_edge_i == condition_atomic_number_i)输出所有元素都是True,2016*True。
                            # 输出将是一个与atomic_number_edge_i和atomic_number_edge_j同样长度的布尔型张量，其中的每个元素都是True,2016*True。
                            mask[:, index_out] += torch.where(
                                (atomic_number_edge_i == condition_atomic_number_i)
                                & (atomic_number_edge_j == condition_atomic_number_j), #同上，都是[2016个True]
                                1,
                                0
                            ) ##mask是2016*81大小的全1张量，int类型，当torch.where(condition, x, y)condition为真，返回x，否则返回y的值。mask每一列都为1。
                            
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
                                ) #condition为真，读取轨道i和轨道j的哈密顿量存入label，否则存入[2016个0]
            assert len(torch.where((mask != 1) & (mask != 0))[0]) == 0 #(mask != 1) 对mask中的每个元素进行检查，如果元素的值不等于1（即不是1），则对应位置的值为True，否则为False。断言在 mask 中不存在既不是0也不是1的值，即所有值都是0或1。
            mask = mask.bool() #将张量中的所有1变为 True，而所有0变为 False。#如果生成的mask正确，所有元素都为True
            # print("mask.all = ", mask.all()) #检查mask是否是2016*81的全为True的张量
            data.mask = mask #创建图数据的mask属性，为2016*81个mask
            del data.term_mask #删除多余的term_mask属性
            if if_only_rc == False:
                data.label = label #将数据的label定义为2016条边的轨道i和轨道j的哈密顿量元素
                if self.target == 'hamiltonian' or self.target == 'density_matrix':
                    
                    del data.term_real #删除多余的图数据中的term_real属性，因为已经把它变为label属性了。
                elif self.target == 'O_ij':
                    del data.rh
                    del data.rdm
                    del data.rvdee
                    del data.rvxc
                    del data.rvna
            dataset_mask.append(data) #每个晶体结构图数据中就包含了mask和label属性
        return dataset_mask #返回包含mask和label标签的dataset
