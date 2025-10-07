import json
import os
from inspect import signature
import time
import csv
import sys
import shutil
import random
import warnings
from math import sqrt
from itertools import islice
from configparser import ConfigParser
import glob
import torch
import torch.optim as optim
from torch import package
from torch.nn import MSELoss
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CyclicLR
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch_scatter import scatter_add
import numpy as np
from psutil import cpu_count


# from .data import HData
from utils import Logger, save_model, LossRecord, MaskMSELoss, Transform, Collater


class DeepHKernel:
    def __init__(self, config):
        self.config = config
        # basic config
        if config['basic']['save_to_time_folder']: #训练时为True，预测时为False
            config["basic"]["save_dir"] = os.path.join(
                config["basic"]["save_dir"],
                time.strftime("%Y-%m-%d_%H-%M-%S-", time.localtime())+config["network"]["atom_update_net"],
            )
            assert not os.path.exists(config["basic"]["save_dir"])
        os.makedirs(config["basic"]["save_dir"], exist_ok=True) #训练时：/fs2/home/ndsim10/example/work_dir/trained_model/strftime;预测时：inference/pred_ham_std
        
        self.graph_dir = config['basic']['graph_dir']
        
        sys.stdout = Logger(os.path.join(config["basic"]["save_dir"], "result.txt")) #打印到控制台并存入到log文件
        sys.stderr = Logger(os.path.join(config["basic"]["save_dir"], "stderr.txt"))

        # src_dir = os.path.join(config["basic"]["save_dir"], "src") #项目源码拷入
        # os.makedirs(src_dir, exist_ok=True)
        # try:
        #     shutil.copytree(os.path.dirname(__file__), os.path.join(src_dir, 'deeph')) #shutil.copytree（原文件路径，保存路径），拷进去方便迁移到其它设备
        # except:
        #     warnings.warn("Unable to copy scripts")
        if not config["basic"]["disable_cuda"]: #False
            self.device = torch.device(config["basic"]["device"] if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        config["basic"]["device"] = str(self.device) #使用set()方法修改指定配置项device的值
        if config["hyperparameter"]["dtype"] == 'float32': #siesta的哈密顿量是float32
            default_dtype_torch = torch.float32
        elif config["hyperparameter"]["dtype"] == 'float16':
            default_dtype_torch = torch.float16
        elif config["hyperparameter"]["dtype"] == 'float64': #transiesta的哈密顿量是float64
            default_dtype_torch = torch.float64
        else:
            raise ValueError('Unknown dtype: {}'.format(config["hyperparameter"]["dtype"]))
        np.seterr(all='raise') #设置如何处理浮点错误。提高一个FloatingPointError。
        np.seterr(under='warn') #浮点下溢的处理，警告。
        np.set_printoptions(precision=8, linewidth=160) #precision：控制输出结果的精度(即小数点后的位数)，默认值为8；linewidth：每行字符的数目，其余的数值会换到下一行
        torch.set_default_dtype(default_dtype_torch)#将默认浮点数据类型设置为default_dtype_torch
        torch.set_printoptions(precision=8, linewidth=160, threshold=np.inf)#threshold：当数组元素总数大于阈值，控制输出的浮点数值的精度为8，当数组元素小于或者等于设置值得时候，全部显示。这里是全部显示。
        np.random.seed(config["basic"]["seed"])
        torch.manual_seed(config["basic"]["seed"])#设置 CPU 生成随机数的种子，方便下次复现实验结果。
        torch.cuda.manual_seed_all(config["basic"]["seed"])#为所有可见可用GPU设置随机种子
        random.seed(config["basic"]["seed"])#设置随机数生成器的种子
        torch.backends.cudnn.benchmark = False #True时, PyTorch会在每次运行时自动寻找最适合当前硬件的卷积实现算法,并进行优化。这样可以加速模型的训练和推断过程。然而,由于寻找最佳算法需要额外的计算开销,因此在输入大小不变的情况下,首次运行可能会比后续运行慢一些。如果输入大小经常变化，建议将此选项设为True，以获得最佳性能。
        torch.backends.cudnn.deterministic = True #True时, PyTorch的卷积操作将以确定性模式运行，即给定相同的输入和参数,输出将始终相同。这对于确保结果的可重复性很重要,尤其是在进行模型训练和验证时。然而,由于确定性模式可能会带来一些性能损失，因此在不需要结果可重复性的情况下，可以将此选项设置为False。
        torch.cuda.empty_cache()#释放缓存分配器当前持有的且未占用的缓存显存，以便这些显存可以被其他GPU应用程序中使用，并且通过 nvidia-smi命令可见。注意使用此命令不会释放tensors占用的显存。
        
        if config["basic"]["num_threads"] == -1: #得到basic中num_threads的值，返回int类型的结果，当配置文件中没有找到指定的部分或选项时，返回的默认值是 -1。
            if torch.cuda.device_count() == 0: #返回当前系统中可用的CUDA数量
                torch.set_num_threads(cpu_count(logical=False))#获取CPU的逻辑个数，可以设置 PyTorch 进行 CPU 多线程并行计算时所占用的线程数，用来限制 PyTorch 所占用的 CPU 数目；
            else:
                torch.set_num_threads(cpu_count(logical=False) // torch.cuda.device_count())
        else:
            torch.set_num_threads(config["basic"]["num_threads"])

        print('====== CONFIG ======')
        with open(os.path.join(config["basic"]["save_dir"], "config.txt"), "w") as f:
            for section_k, section_v in islice(config.items(), 1, None):  
                f.write(f'[{section_k}]\n')
                for k, v in section_v.items():
                    f.write(f'{k}={v}\n')
                f.write('\n')

        self.atom_update_net = config["network"]["atom_update_net"]
        self.if_lcmp = config["network"]["if_lcmp"] #是否加入LCMP层，True
        self.if_lcmp_graph = config["graph"]["if_lcmp_graph"] #True
        self.new_sp = config["graph"]["new_sp"] #False
        self.separate_onsite = config["graph"]["separate_onsite"] #False
        if self.if_lcmp == True:
            assert self.if_lcmp_graph == True
        self.target = config["basic"]["target"] #hamiltonian
        if self.target == 'O_ij':
            self.O_component = config['basic']['O_component']

          
        if self.target != 'E_ij' and self.target != 'E_i':
            self.num_orbital = config['basic']['num_orbital'] #ao1*ao1 + ao1*ao2 + ao2*ao1 + ao2*ao2
        else:
            self.energy_component = config['basic']['energy_component'] #summation
        # early_stopping
        self.early_stopping_loss_epoch = config['train']['early_stopping_loss_epoch'] #[0.000000, 500] #提前停止检查的起始和结束时期。这表明在第500个epoch之后，如果损失满足提前停止的条件，则可以停止训练。
        
    def build_model(self, model_pack_dir: str = None, old_version=None): #预测时是：trained_model_dir, old_version=False
        if model_pack_dir is not None: #定义了一个参数名为 model_pack_dir 的模型包变量路径，它的类型是字符串 (str)，并且这个参数的默认值是 None。
            assert old_version is not None
            if old_version is True:
                print(f'import HGNN from {model_pack_dir}')
                sys.path.append(model_pack_dir)
                from deeph import HGNN
            else:
                imp = package.PackageImporter(os.path.join(model_pack_dir, 'best_model.pt')) #package.PackageImporter() 接受一个路径参数，该路径指向包含了模型或相关信息的文件，尝试从这个文件中导入模型或相关内容。
                checkpoint = imp.load_pickle('checkpoint', 'model.pkl', map_location=self.device) #调用了 PackageImporter 对象中的 load_pickle() 方法用于从导入的包中加载 .pkl 格式的文件。'checkpoint' 和 'model.pkl' 分别是包中的文件夹和文件名称，用于指定要加载的内容。map_location=self.device用于指定设备位置，告诉加载器将模型放置在特定设备上。从模型包中加载 'model.pkl' 文件，其中可能包含了模型的状态、超参数或其他相关信息，并将其存储在变量 checkpoint 中。
                self.model = checkpoint['model']
                self.model.to(self.device)
                self.index_to_Z = checkpoint["index_to_Z"] #bi是[83]，C是[6]
                self.Z_to_index = checkpoint["Z_to_index"] #bi是[100个-1，第83个位置为0]，C是[100个-1，第6个位置为0]
                self.spinful = checkpoint["spinful"] #False
                print("=> load best checkpoint (epoch {})".format(checkpoint['epoch']))
                print(f"=> Atomic types: {self.index_to_Z.tolist()}, "
                      f"spinful: {self.spinful}, the number of atomic types: {len(self.index_to_Z)}.")
                if self.target != 'E_ij':
                    if self.spinful:
                        self.out_fea_len = self.num_orbital * 8
                    else:
                        self.out_fea_len = self.num_orbital #81/169
                else:
                    if self.energy_component == 'both':
                        self.out_fea_len = 2
                    elif self.energy_component in ['xc', 'delta_ee', 'summation']:
                        self.out_fea_len = 1
                    else:
                        raise ValueError('Unknown energy_component: {}'.format(self.energy_component))
                return checkpoint #预测时，到这里return后就结束了，后面的就不执行了。
        else:
            if self.atom_update_net == "deeph":
                from deeph.deeph import DeepH
                parameter_list = list(signature(DeepH.__init__).parameters.keys()) #获取 DeepH 类的构造函数 __init__ 的参数列表。
            elif self.atom_update_net == "gat":
                pass
                # schnet, phisnet, deeph, gat, painn, graphormer, transformer-m, deepqth
            else:
                pass

        if self.spinful:
            if self.target == 'phiVdphi':
                raise NotImplementedError("Not yet have support for phiVdphi")
            else:
                self.out_fea_len = self.num_orbital * 8
        else:
            if self.target == 'phiVdphi':
                self.out_fea_len = self.num_orbital * 3
            else:
                self.out_fea_len = self.num_orbital  #81/169

        print(f'Output features length of single edge: {self.out_fea_len}')
        model_kwargs = dict(
            n_elements=self.num_species, #1，这个不在HGNN参数中，之后被删除了
            num_species=self.num_species, #1
            in_atom_fea_len=self.config['network']['atom_fea_len'], #64
            in_vfeats=self.config['network']['atom_fea_len'], #64，这个不在HGNN参数中，之后被删除了
            in_edge_fea_len=self.config['network']['edge_fea_len'], #128
            in_efeats=self.config['network']['edge_fea_len'], #128，这个不在HGNN参数中，之后被删除了
            out_edge_fea_len=self.out_fea_len, #81，这个不在HGNN参数中，之后被删除了
            out_efeats=self.out_fea_len, #81，这个不在HGNN参数中，之后被删除了
            num_orbital=self.out_fea_len, #81
            distance_expansion=self.config['network']['distance_expansion'], #GaussianBasis
            gauss_stop=self.config['network']['gauss_stop'], #6.0
            cutoff=self.config['network']['gauss_stop'], #6.0，这个不在HGNN参数中，之后被删除了
            if_exp=self.config['network']['if_exp'], #True
            if_MultipleLinear=self.config['network']['if_MultipleLinear'], #False
            if_edge_update=self.config['network']['if_edge_update'], #True
            if_lcmp=self.if_lcmp, #True
            normalization=self.config['network']['normalization'], #LayerNorm
            atom_update_net=self.config['network']['atom_update_net'], #这个不在HGNN参数中，之后被删除了
            separate_onsite=self.separate_onsite, #False
            n_heads=self.config['network']['n_heads'],  #4,这个不在HGNN参数中，之后被删除了
            num_l=self.config['network']['num_l'], #5
            trainable_gaussians=self.config['network']['trainable_gaussians'], #True
            type_affine=self.config['network']['type_affine'], #False
            if_fc_out=False, #这个不在HGNN参数中，之后被删除了
            max_path_length=self.config['network']['max_path_length'], #5,这个不在HGNN参数中，之后被删除了
        )
        current_parameter_list = list(model_kwargs.keys())
        for k in current_parameter_list:
            if k not in parameter_list:
                model_kwargs.pop(k) #删除model_kwargs参数中在HGNN构造函数中没有的参数，这些行代码用于检查和更新模型参数，确保只传递__init__方法中定义的参数。
        if 'num_elements' in parameter_list: #False
            model_kwargs['num_elements'] = self.config['basic']['max_element'] + 1 #0
        if self.atom_update_net == "deeph":
            self.model = DeepH(**model_kwargs) #建立通用网络模型和参数，具体选择哪个网络模型在HGNN中选择
        elif self.atom_update_net == "gat":
            pass
            # schnet, phisnet, deeph, gat, painn, graphormer, transformer-m, deepqth
        else:
            pass
        
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters()) #获取模型中所有需要进行梯度更新（即 requires_grad 为 True）的参数。
        params = sum([np.prod(p.size()) for p in model_parameters]) #对于 model_parameters 中的每个参数 p，它计算了该参数的大小（尺寸）并将其元素数量相乘。这样就得到了网络的总参数量。
        print("The model you built has: %d parameters" % params)
        print("model = ", self.model)
        self.model.to(self.device)
        self.load_pretrained() #加载预训练模型

    def set_train(self):
        self.criterion_name = self.config['hyperparameter']['criterion']
        if self.target == "E_i":
            self.criterion = MSELoss()
        elif self.target == "E_ij":
            self.criterion = MSELoss()
            self.retain_edge_fea = self.config.getboolean('hyperparameter', 'retain_edge_fea')
            self.lambda_Eij = self.config.getfloat('hyperparameter', 'lambda_Eij')
            self.lambda_Ei = self.config.getfloat('hyperparameter', 'lambda_Ei')
            self.lambda_Etot = self.config.getfloat('hyperparameter', 'lambda_Etot')
            if self.retain_edge_fea is False:
                assert self.lambda_Eij == 0.0
        else:
            if self.criterion_name == 'MaskMSELoss':
                self.criterion = MaskMSELoss()
            else:
                raise ValueError(f'Unknown criterion: {self.criterion_name}')

        learning_rate = self.config['hyperparameter']['learning_rate'] #0.001
        momentum = self.config['hyperparameter']['momentum'] #0.9
        weight_decay = self.config['hyperparameter']['weight_decay'] #0

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters()) #获取模型中所有需要进行梯度更新（即 requires_grad 为 True）的参数。
        if self.config['hyperparameter']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(model_parameters, lr=learning_rate, weight_decay=weight_decay)
        elif self.config['hyperparameter']['optimizer'] == 'sgdm':
            self.optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif self.config['hyperparameter']['optimizer'] == 'adam': #adam
            self.optimizer = optim.Adam(model_parameters, lr=learning_rate, betas=(0.9, 0.999)) #(beta1, beta2)，分别控制‌梯度一阶矩（均值）和二阶矩（方差）的指数衰减率
        elif self.config['hyperparameter']['optimizer'] == 'adamW':
            self.optimizer = optim.AdamW(model_parameters, lr=learning_rate, betas=(0.9, 0.999))
        elif self.config['hyperparameter']['optimizer'] == 'adagrad':
            self.optimizer = optim.Adagrad(model_parameters, lr=learning_rate)
        elif self.config['hyperparameter']['optimizer'] == 'RMSprop':
            self.optimizer = optim.RMSprop(model_parameters, lr=learning_rate)
        elif self.config['hyperparameter']['optimizer'] == 'lbfgs':
            self.optimizer = optim.LBFGS(model_parameters, lr=0.1)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')

        if self.config['hyperparameter']['lr_scheduler'] == '': #True，不使用学习率调度器
            pass
        elif self.config['hyperparameter']['lr_scheduler'] == 'MultiStepLR':
            lr_milestones = json.loads(self.config.get('hyperparameter', 'lr_milestones'))
            self.scheduler = MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=0.2)
        elif self.config['hyperparameter']['lr_scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=10,
                                               verbose=True, threshold=1e-4, threshold_mode='rel', min_lr=0)
        elif self.config['hyperparameter']['lr_scheduler'] == 'CyclicLR':
            self.scheduler = CyclicLR(self.optimizer, base_lr=learning_rate * 0.1, max_lr=learning_rate,
                                      mode='triangular', step_size_up=50, step_size_down=50, cycle_momentum=False)
        else:
            raise ValueError('Unknown lr_scheduler: {}'.format(self.config['hyperparameter']['lr_scheduler']))
        self.load_resume()

    #加载一个预训练的模型并将其状态（权重和偏差）传输到当前的模型中。
    def load_pretrained(self):
        pretrained = self.config['train']['pretrained'] #加载预训练的模型。这里没有预先训练好的模型[]
        if pretrained: #是否非空，即配置文件中是否提供了预训练模型的路径。
            if os.path.isfile(pretrained): #检查该路径的文件是否真实存在。这是为了避免尝试加载不存在的文件，从而引发错误。
                checkpoint = torch.load(pretrained, map_location=self.device) #加载预训练模型。map_location=self.device 参数确保模型加载到指定的设备上（如CPU或GPU）。
                pretrained_dict = checkpoint['state_dict'] #获取保存的模型的状态字典，这包括了模型的所有参数（如权重和偏差）。
                model_dict = self.model.state_dict() #获取当前模型的状态字典。

                transfer_dict = {}
                for k, v in pretrained_dict.items():
                    if v.shape == model_dict[k].shape: #迭代预训练模型的状态字典，并检查每个参数的形状是否与当前模型中对应的参数形状匹配。
                        transfer_dict[k] = v #如果形状匹配，该参数将被加入到 transfer_dict 字典中，随后用于更新当前模型。这一步确保只有形状相同的参数才被传递，避免因形状不匹配导致的错误。
                        print('Use pretrained parameters:', k)

                model_dict.update(transfer_dict) #使用 update 方法将匹配的预训练参数合并到当前模型的状态字典中。
                self.model.load_state_dict(model_dict) #通过 self.model.load_state_dict 将更新后的状态字典加载到模型中，这样模型就具备了预训练的参数。
                print(f'=> loaded pretrained model at "{pretrained}" (epoch {checkpoint["epoch"]})') #确认预训练模型已被加载，并显示模型是在哪个训练周期保存的。
            else:
                print(f'=> no checkpoint found at "{pretrained}"') #通知用户未找到预训练模型。

    #从一个保存的检查点（checkpoint）中恢复整个模型的状态，包括模型的参数和优化器的状态。这是在模型训练过程中常用的技术，用于实现断点续训或从先前的训练状态恢复训练。
    #允许研究人员或开发者在任何时候停止并重新开始训练，而无需从头开始，从而节省大量时间和计算资源。
    def load_resume(self):
        resume = self.config['train']['resume'] #获取存放恢复文件的路径。
        if resume:
            if os.path.isfile(resume): #检查提供的路径指向的文件是否存在。这步是必要的，以避免尝试加载不存在的文件，从而引发错误。
                checkpoint = torch.load(resume, map_location=self.device) #使用 torch.load 函数加载保存的检查点文件。参数 map_location=self.device 确保检查点数据加载到指定的设备上（如CPU或GPU），这对于模型的兼容性和后续操作非常重要。
                self.model.load_state_dict(checkpoint['state_dict']) #从检查点中提取模型状态字典 checkpoint['state_dict'] 并使用 load_state_dict 方法恢复模型的参数。这确保了模型能够精确恢复到保存时的状态。
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #从检查点中提取优化器状态字典 checkpoint['optimizer_state_dict'] 并使用 load_state_dict 方法恢复优化器的状态。这是非常关键的，因为优化器的状态（如动量、学习率等）对于训练过程的连续性非常重要。
                print(f'=> loaded model at "{resume}" (epoch {checkpoint["epoch"]})')
            else:
                print(f'=> no checkpoint found at "{resume}"')

    def get_dataset(self):
        '''
        # Data(
        #     x=[36], #36个0
        #     edge_index=[2, 2016],
        #     edge_attr=[2016, 10],
        #     stru_id='0',
        #     voronoi_values=[72,1],
        #     centralities=[72,1],
        #     cart_coords=[72, 3],
        #     lattice=[3, 3],
        #     node_paths=[1, 72, 72, 5], 
        #     edge_paths=[1, 72, 72, 4],
        #     atom_num_orbital=[72],
        #     subgraph_dict={
        #         subgraph_atom_idx = [226440, 2],
        #         subgraph_edge_idx = [226440],
        #         subgraph_edge_ang = [226440, 25],
        #         subgraph_index = [226440]
        #     },
        #     spinful = [1],
        #     mask=[2016, 169],
        #     label=[2016, 169]
        # )
        '''

        if os.path.exists(self.graph_dir) and os.path.isdir(self.graph_dir):
            pkl_files = glob.glob(os.path.join(self.graph_dir, "*.pkl"))
            if len(pkl_files) > 0:
                print("Loading existing dataset:", pkl_files[0])
                self.data, self.slices, self.info = torch.load(pkl_files[0])
                sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '1_preprocess')))
                from data import HData
                # 创建空实例并设置必要属性
                dataset = HData.__new__(HData)
                dataset.data = self.data
                dataset.slices = self.slices
                dataset.info = self.info
                dataset.transform = None
                dataset.pre_transform = None
                dataset.pre_filter = None
                dataset._indices = None
                dataset._data_list = None
                # 验证数据
                print(f"num graphs: {len(dataset)}")
            else:
                raise RuntimeError(f"No .pkl files found in {self.graph_dir}, please run process.py")
        else:
            raise RuntimeError(f"Directory {self.graph_dir} not found, please run process.py")

        self.spinful = dataset.info["spinful"] #False
        self.index_to_Z = dataset.info["index_to_Z"] #[83],这是dataset的属性之一
        self.Z_to_index = dataset.info["Z_to_index"] #[第83个位置为0，其余为-1的(100,)张量],这是dataset的属性之一
        self.num_species = len(dataset.info["index_to_Z"]) #1，
        
        dataset_size = len(dataset) #576个图结构数据
        train_size = int(self.config["train"]["train_ratio"] * dataset_size) #int(0.6 * 576) = 345
        val_size = int(self.config["train"]["val_ratio"] * dataset_size) #int(0.2 * 576) = 115
        test_size = int(self.config["train"]["test_ratio"] * dataset_size) #int(0.2 *576) = 115
        assert train_size + val_size + test_size <= dataset_size

        indices = list(range(dataset_size)) #生成[0-575]的列表
        np.random.shuffle(indices) #打乱数据集，会直接修改传入的数组indices，而不会返回一个新的数组。因此，它是一个原地操作（in-place operation）。
        print(f'number of train set: {len(indices[:train_size])}') #训练集个数
        print(f'number of val set: {len(indices[train_size:train_size + val_size])}') #验证集个数
        print(f'number of test set: {len(indices[train_size + val_size:train_size + val_size + test_size])}') #测试集个数
        train_sampler = SubsetRandomSampler(indices[:train_size]) #用于创建子集随机采样器类，用于数据集的分割和采样，从给定的索引列表中随机采样元素，不进行替换。
        val_sampler = SubsetRandomSampler(indices[train_size:train_size + val_size])
        test_sampler = SubsetRandomSampler(indices[train_size + val_size:train_size + val_size + test_size])
        train_loader = DataLoader(dataset, batch_size=self.config["hyperparameter"]["batch_size"],
                                  shuffle=False, sampler=train_sampler,
                                  collate_fn=Collater(self.if_lcmp)) #batch_size=3，self.if_lcmp=True。collate_fn=Collater(self.if_lcmp)用于batch划分
        val_loader = DataLoader(dataset, batch_size=self.config["hyperparameter"]["batch_size"],
                                shuffle=False, sampler=val_sampler,
                                collate_fn=Collater(self.if_lcmp))
        test_loader = DataLoader(dataset, batch_size=self.config["hyperparameter"]["batch_size"],
                                 shuffle=False, sampler=test_sampler,
                                 collate_fn=Collater(self.if_lcmp)) #collate_fn参数是合并一个样本列表，形成一个小批量的张量。当从图样式数据集批量加载时使用。用于对每个batch_size的数据进行处理和组合。返回batch和batch对应的子图。

        if self.config["basic"]["statistics"]: #False，对图数据集的哈密顿量矩阵元素的值分布做平均值统计，论文中的图3(a)
            sample_label = torch.cat([dataset[i].label for i in range(len(dataset))]) #[36*2016,81]
            sample_mask = torch.cat([dataset[i].mask for i in range(len(dataset))]) #[36*2016,81]
            mean_value = abs(sample_label).sum(dim=0) / sample_mask.sum(dim=0)#结果是一个1x81的张量，因为dim=0意味着我们沿着第一个维度进行压缩求和。
            import matplotlib.pyplot as plt
            len_matrix = int(sqrt(len(torch.squeeze(mean_value)))) #9
            if len_matrix ** 2 != len(torch.squeeze(mean_value)):
                raise ValueError
            mean_value = mean_value.reshape(len_matrix, len_matrix) #[9,9]
            im = plt.imshow(mean_value, cmap='Blues')
            plt.colorbar(im)
            plt.xticks(range(len_matrix), range(len_matrix))
            plt.yticks(range(len_matrix), range(len_matrix))
            plt.xlabel(r'Orbital $\beta$')
            plt.ylabel(r'Orbital $\alpha$')
            plt.title(r'Mean of abs($H^\prime_{i\alpha, j\beta}$)')
            plt.tight_layout() #tight_layout会自动调整子图参数，使之填充整个图像区域。
            plt.savefig(os.path.join(self.config["basic"]["save_dir"], 'mean.png'), dpi=800)
            np.savetxt(os.path.join(self.config["basic"]["save_dir"], 'mean.dat'), mean_value.numpy())

            print(f"The statistical results are saved to {os.path.join(self.config.get('basic', 'save_dir'), 'mean.dat')}")

        normalizer = self.config["basic"]["normalizer"] #False
        boxcox = self.config["basic"]["boxcox"] #False
        if normalizer == False and boxcox == False:
            transform = Transform()
        else:
            sample_label = torch.cat([dataset[i].label for i in range(len(dataset))])
            sample_mask = torch.cat([dataset[i].mask for i in range(len(dataset))])
            transform = Transform(sample_label, mask=sample_mask, normalizer=normalizer, boxcox=boxcox)
        print(transform.state_dict()) #打印{'normalizer': False,'boxcox': False}，没有用到transform
        #用于对张量数据进行一些变换操作。这个类包括初始化、正向变换、逆向变换、状态保存和状态加载几个主要部分。

        return train_loader, val_loader, test_loader, transform

    def train(self, train_loader, val_loader, test_loader):
        begin_time = time.time()
        self.best_val_loss = 1e10
        if self.config["train"]["revert_then_decay"]: #True，当模型的性能开始下降时，将回退到之前最好的模型状态，并开始应用学习率衰减策略。
            lr_step = 0
        #revert_threshold = 30，回退操作的阈值，指定允许连续下降的epoch次数，在此之后将回退到最优状态。
        revert_decay_epoch = self.config["train"]["revert_decay_epoch"] #[1000, 2000, 3000]，在这些epoch点上应用学习率衰减。
        revert_decay_gamma = self.config["train"]["revert_decay_gamma"] #[0.5, 0.5, 0.5]，在上面指定的epoch点上应用的学习率衰减因子。
        assert len(revert_decay_epoch) == len(revert_decay_gamma)
        lr_step_num = len(revert_decay_epoch) #3

        try:
            for epoch in range(self.config["train"]["epochs"]): #0-2999
                print("==================================================================")
                #switch_sgd是否在某个epoch后切换到SGD优化器。在这里默认为False，不切换。切换到SGD优化器的具体epoch。值为-1意味着不进行切换。
                if self.config["train"]["switch_sgd"] and epoch == self.config["train"]["switch_sgd_epoch"]: #False and 0-3999 == -1，不切换sgd优化器
                    model_parameters = filter(lambda p: p.requires_grad, self.model.parameters()) #获取模型中所有需要进行梯度更新（即 requires_grad 为 True）的参数。
                    self.optimizer = optim.SGD(model_parameters, lr=self.config["train"]["switch_sgd_lr"]) #切换到SGD后的学习率。创建随机梯度下降优化器的类。model_parameters 作为第一个参数传递给了optim.SGD，这个参数是模型中需要进行优化的参数列表。lr=0.0001 是学习率，它指定了在梯度下降过程中更新参数时的步长或者速率。
                    print(f"Switch to sgd (epoch: {epoch})")
                #self.optimizer.param_groups是长度为2的list，其中的元素是2个字典,optimizer.param_groups[0]：长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数
                learning_rate = self.optimizer.param_groups[0]['lr'] #从模型当前使用的优化器 (self.optimizer) 中获取学习率 (learning_rate) 的值。之前初始化的self.optimizer = optim.Adam(model_parameters, lr=learning_rate, betas=(0.9, 0.999))
                
                with open(os.path.join(self.config["basic"]["save_dir"], 'learning_rate.csv'), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, learning_rate])
                
                # start = time.perf_counter()
                # train
                train_losses = self.kernel_fn(train_loader, 'TRAIN')#执行训练步骤，并返回训练集的损失值。
                with open(os.path.join(self.config["basic"]["save_dir"], 'training_loss.csv'), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_losses.avg])
                
                # print("单次epoch训练时间：", time.perf_counter() - start)
                # start = time.perf_counter()
                # val
                with torch.no_grad(): #会在不跟踪梯度的情况下运行。
                    val_losses = self.kernel_fn(val_loader, 'VAL') #计算验证集（val_loader）的损失值，由于使用了 torch.no_grad()，所以这个计算过程不会记录梯度信息，即不会影响模型参数。
                # print("单次epoch验证时间：", time.perf_counter() - start)

                print("val_losses.avg = ", val_losses.avg)
                print("best_val_loss = ", self.config["train"]["revert_threshold"] * self.best_val_loss) #30*1e10, 回退操作的阈值，指定允许连续下降的epoch次数，在此之后将回退到最优状态。
                if val_losses.avg > self.config["train"]["revert_threshold"] * self.best_val_loss: #tensor(0.09981232) > 30 * 1e10(初始)
                    print(f'Epoch #{epoch:01d} \t| '
                          f'Learning rate: {learning_rate:0.2e} \t| '
                          f'Epoch time: {time.time() - begin_time:.2f} \t| '
                          f'Train loss: {train_losses.avg:.8f} \t| '
                          f'Val loss: {val_losses.avg:.8f} \t| '
                          f'Best val loss: {self.best_val_loss:.8f}.'
                          )
                    best_checkpoint = torch.load(os.path.join(self.config["basic"]["save_dir"], 'best_state_dict.pkl'))
                    self.model.load_state_dict(best_checkpoint['state_dict'])
                    self.optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
                    if self.config["train"]["revert_then_decay"]: #True
                        if lr_step < lr_step_num: #降低学习率
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = learning_rate * revert_decay_gamma[lr_step]
                            lr_step += 1
                    with torch.no_grad():
                        val_losses = self.kernel_fn(val_loader, 'VAL')
                    print(f"Revert (threshold: {self.config['train']['revert_threshold']}) to epoch {best_checkpoint['epoch']} \t| Val loss: {val_losses.avg:.8f}")
                    with open(os.path.join(self.config["basic"]["save_dir"], 'validation_loss.csv'), 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, val_losses.avg])

                    if self.config["hyperparameter"]["lr_scheduler"] == 'MultiStepLR':
                        self.scheduler.step()
                    elif self.config["hyperparameter"]["lr_scheduler"] == 'ReduceLROnPlateau':
                        self.scheduler.step(val_losses.avg)
                    elif self.config["hyperparameter"]["lr_scheduler"] == 'CyclicLR':
                        self.scheduler.step()
                    continue #跳过本次计算，下面的就不执行了，继续下一次循环。

                with open(os.path.join(self.config["basic"]["save_dir"], 'validation_loss.csv'), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, val_losses.avg])
                
                if self.config["train"]["revert_then_decay"]: #True
                    if lr_step < lr_step_num and epoch >= revert_decay_epoch[lr_step]: #每隔[500, 2000, 3000]降低一次学习率
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= revert_decay_gamma[lr_step]
                        lr_step += 1

                is_best = val_losses.avg < self.best_val_loss #True
                self.best_val_loss = min(val_losses.avg, self.best_val_loss)

                save_complete = False
                while not save_complete:
                    try:
                        save_model({
                            'epoch': epoch + 1,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_val_loss': self.best_val_loss,
                            'spinful': self.spinful,
                            'Z_to_index': self.Z_to_index,
                            'index_to_Z': self.index_to_Z,
                        }, {'model': self.model}, {'state_dict': self.model.state_dict()},
                            path=self.config["basic"]["save_dir"], is_best=is_best) #用于保存模型和相关信息到磁盘上。它接受多个参数来保存不同的信息。
                        save_complete = True #保持了当前最优模型
                    except KeyboardInterrupt:
                        print('\nKeyboardInterrupt while saving model to disk') #异常处理块，它会捕获键盘中断信号

                if self.config["hyperparameter"]["lr_scheduler"] == 'MultiStepLR':
                    self.scheduler.step()
                elif self.config["hyperparameter"]["lr_scheduler"] == 'ReduceLROnPlateau':
                    self.scheduler.step(val_losses.avg)
                elif self.config["hyperparameter"]["lr_scheduler"] == 'CyclicLR':
                    self.scheduler.step()

                print(f'Epoch #{epoch:01d} \t| '
                      f'Learning rate: {learning_rate:0.2e} \t| '
                      f'Epoch time: {time.time() - begin_time:.2f} \t| '
                      f'Train loss: {train_losses.avg:.8f} \t| '
                      f'Val loss: {val_losses.avg:.8f} \t| '
                      f'Best val loss: {self.best_val_loss:.8f}.'
                      )

                if val_losses.avg < self.config["train"]["early_stopping_loss"]:
                    print(f"Early stopping because the target accuracy (validation loss < {self.config['train']['early_stopping_loss']}) is achieved at eopch #{epoch:01d}")
                    break
                if epoch > self.early_stopping_loss_epoch[1] and val_losses.avg < self.early_stopping_loss_epoch[0]:
                    print(f"Early stopping because the target accuracy (validation loss < {self.early_stopping_loss_epoch[0]} and epoch > {self.early_stopping_loss_epoch[1]}) is achieved at eopch #{epoch:01d}")
                    break

                begin_time = time.time()
        except KeyboardInterrupt:
            print('\nKeyboardInterrupt')

        print('---------Evaluate Model on Test Set---------------')
        best_checkpoint = torch.load(os.path.join(self.config["basic"]["save_dir"], 'best_state_dict.pkl'))
        self.model.load_state_dict(best_checkpoint['state_dict'])
        print("=> load best checkpoint (epoch {})".format(best_checkpoint['epoch']))
        with torch.no_grad():
            test_csv_name = 'test_results.csv'
            train_csv_name = 'train_results.csv'
            val_csv_name = 'val_results.csv'

            if self.config["basic"]["save_csv"]: #False
                tmp = 'TEST'
            else:
                tmp = 'VAL'
            test_losses = self.kernel_fn(test_loader, tmp, test_csv_name, output_E=True) #拿测试集去验证
            print(f'Test loss: {test_losses.avg:.8f}.')
            with open(os.path.join(self.config["basic"]["save_dir"], 'test_loss.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, test_losses.avg])
                
            test_losses = self.kernel_fn(train_loader, tmp, train_csv_name, output_E=True) #拿训练集去验证
            print(f'Train loss: {test_losses.avg:.8f}.')
            test_losses = self.kernel_fn(val_loader, tmp, val_csv_name, output_E=True) #拿验证集去验证
            print(f'Val loss: {test_losses.avg:.8f}.')

    def predict(self, hamiltonian_dirs):
        raise NotImplementedError

    def kernel_fn(self, loader, task: str, save_name=None, output_E=False): #loader是要处理的数据，task="TRAIN"
        assert task in ['TRAIN', 'VAL', 'TEST']
        losses = LossRecord()
        if task == 'TRAIN':
            self.model.train() #分别用于设置模型的训练模式和评估（测试）模式。
        else:
            self.model.eval() #在评估模式下，所有的训练特定的层如Dropout和BatchNorm都会被设置为不活跃状态。
        if task == 'TEST':
            assert save_name != None #保存csv
            if self.target == "E_i" or self.target == "E_ij":
                test_targets = []
                test_preds = []
                test_ids = []
                test_atom_ids = []
                test_atomic_numbers = []
            else:
                test_targets = []
                test_preds = []
                test_ids = []
                test_atom_ids = []
                test_atomic_numbers = []
                test_edge_infos = []

        if task != 'TRAIN' and (self.out_fea_len != 1): #self.out_fea_len = 81
            losses_each_out = [LossRecord() for _ in range(self.out_fea_len)] #创建一个81个LossRecord的实例的列表。用于存储评估时每个哈密顿量值的损失值的列表？
        for step, batch_tuple in enumerate(loader):#训练时step=0-345，验证和测试时step=115，迭代每batch个数据
            # start = time.perf_counter()
            if self.if_lcmp:
                batch, subgraph = batch_tuple #batch=3
                # print(batch.keys) #batch=3个图数据
                sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index = subgraph
                output = self.model(
                    batch.x.to(self.device), #3个[36*6],[108]
                    batch.edge_index.to(self.device), #3个[2,2016],[2,6048]
                    batch.edge_attr.to(self.device), #3个[2016,10],[6048,10]
                    # batch.node_paths.to(self.device),
                    # batch.edge_paths.to(self.device),
                    # batch.voronoi_values.to(self.device),
                    # batch.centralities.to(self.device),
                    batch.batch.to(self.device), #[36个0，36个1，36个2],[108]
                    sub_atom_idx.to(self.device), #3个[226440,2],[679320,2]
                    sub_edge_idx.to(self.device), #[679320]
                    sub_edge_ang.to(self.device), #[679320,25]
                    sub_index.to(self.device) #[679320]
                )#输出[6048,81]或[1,6048,81]，即[3*2016,81]，耗时5.76196
            else:
                batch = batch_tuple
                output = self.model(
                    batch.x.to(self.device),
                    batch.edge_index.to(self.device),
                    batch.edge_attr.to(self.device),
                    # batch.node_paths.to(self.device),
                    # batch.edge_paths.to(self.device),
                    # batch.voronoi_values.to(self.device),
                    # batch.centralities.to(self.device),
                    batch.batch.to(self.device)
                )
            # print("单个batch训练时间：", time.perf_counter() - start)
            if self.target == 'E_ij':
                if self.energy_component == 'E_ij':
                    label_non_onsite = batch.E_ij.to(self.device)
                    label_onsite = batch.onsite_E_ij.to(self.device)
                elif self.energy_component == 'summation':
                    label_non_onsite = batch.E_delta_ee_ij.to(self.device) + batch.E_xc_ij.to(self.device)
                    label_onsite = batch.onsite_E_delta_ee_ij.to(self.device) + batch.onsite_E_xc_ij.to(self.device)
                elif self.energy_component == 'delta_ee':
                    label_non_onsite = batch.E_delta_ee_ij.to(self.device)
                    label_onsite = batch.onsite_E_delta_ee_ij.to(self.device)
                elif self.energy_component == 'xc':
                    label_non_onsite = batch.E_xc_ij.to(self.device)
                    label_onsite = batch.onsite_E_xc_ij.to(self.device)
                elif self.energy_component == 'both':
                    raise NotImplementedError
                output_onsite, output_non_onsite = output
                if self.retain_edge_fea is False:
                    output_non_onsite = output_non_onsite * 0

            elif self.target == 'E_i':
                label = batch.E_i.to(self.device)
                output = output.reshape(label.shape)
            else:
                label = batch.label.to(self.device) #[6048,81]
                output = output.reshape(label.shape) #[6048,81]

            if self.target == 'E_i':
                loss = self.criterion(output, label)
            elif self.target == 'E_ij':
                loss_Eij = self.criterion(torch.cat([output_onsite, output_non_onsite], dim=0),
                                          torch.cat([label_onsite, label_non_onsite], dim=0))
                output_non_onsite_Ei = scatter_add(output_non_onsite, batch.edge_index.to(self.device)[0, :], dim=0)
                label_non_onsite_Ei = scatter_add(label_non_onsite, batch.edge_index.to(self.device)[0, :], dim=0)
                output_Ei = output_non_onsite_Ei + output_onsite
                label_Ei = label_non_onsite_Ei + label_onsite
                loss_Ei = self.criterion(output_Ei, label_Ei)
                loss_Etot = self.criterion(scatter_add(output_Ei, batch.batch.to(self.device), dim=0),
                                           scatter_add(label_Ei, batch.batch.to(self.device), dim=0))
                loss = loss_Eij * self.lambda_Eij + loss_Ei * self.lambda_Ei + loss_Etot * self.lambda_Etot
            else:
                if self.criterion_name == 'MaskMSELoss':
                    mask = batch.mask.to(self.device) #[6048,81],元素都是True
                    loss = self.criterion(output, label, mask) #标量：tensor(0.14334539, grad_fn=<MeanBackward0>)
                else:
                    raise ValueError(f'Unknown criterion: {self.criterion_name}')
            if task == 'TRAIN':
                if self.config["hyperparameter"]["optimizer"] == 'lbfgs':
                    def closure():
                        self.optimizer.zero_grad()
                        if self.if_lcmp:
                            output = self.model(
                                batch.x.to(self.device),
                                batch.edge_index.to(self.device),
                                batch.edge_attr.to(self.device),
                                batch.node_paths.to(self.device),
                                batch.edge_paths.to(self.device),
                                batch.voronoi_values.to(self.device),
                                batch.centralities.to(self.device),
                                batch.batch.to(self.device),
                                sub_atom_idx.to(self.device),
                                sub_edge_idx.to(self.device),
                                sub_edge_ang.to(self.device),
                                sub_index.to(self.device)
                            )
                        else:
                            output = self.model(
                                batch.x.to(self.device),
                                batch.edge_index.to(self.device),
                                batch.edge_attr.to(self.device),
                                batch.node_paths.to(self.device),
                                batch.edge_paths.to(self.device),
                                batch.voronoi_values.to(self.device),
                                batch.centralities.to(self.device),
                                batch.batch.to(self.device)
                            )
                        loss = self.criterion(output, label.to(self.device), mask)
                        loss.backward()
                        return loss

                    self.optimizer.step(closure)
                else: #adam，耗时19.48730
                    self.optimizer.zero_grad()
                    # start = time.perf_counter()
                    loss.backward() #最耗时部分，18.167473
                    # print("反向传播运行时间：", time.perf_counter() - start)
                    #clip_grad是否启用梯度裁剪。这有助于防止训练过程中的梯度爆炸问题。
                    if self.config['train']['clip_grad']: #True。clip_grad_norm_参考：https://blog.csdn.net/m0_46412065/article/details/131396098
                        clip_grad_norm_(self.model.parameters(), self.config['train']['clip_grad_value']) #4.2。梯度裁剪的阈值，超过此值的梯度将被裁剪。限制模型参数列表中参数的梯度的最大范数值为4.2，如果模型梯度的范数超过了这个值，那么梯度将被重新缩放，使得其范数不超过这个指定的最大值，以防止梯度爆炸的问题。直接修改原梯度。
                    self.optimizer.step()
            if self.target == "E_i" or self.target == "E_ij":
                losses.update(loss.item(), batch.num_nodes)
            else:
                if self.criterion_name == 'MaskMSELoss':
                    losses.update(loss.item(), mask.sum())
                    #这一行调用update方法来更新损失记录。这里，loss.item()是当前损失值的标量（假设loss是一个PyTorch的标量张量），而mask.sum()表示当前批次中有效损失计算的元素数量。这种做法在处理变长数据或加权损失时特别有用，mask常用于指示数据的哪些部分应该计入损失计算。
                    #loss.item()返回当前损失值的标量（一个单独的数值），这个值会被losses对象使用update()方法更新。将当前的损失值和mask中所有元素的总和一起传递给losses对象的update()方法，以便记录损失值，计算平均损失值。
                if task != 'TRAIN' and self.out_fea_len != 1:
                    if self.criterion_name == 'MaskMSELoss':
                        se_each_out = torch.pow(output - label.to(self.device), 2) ##[6048,81]，计算了模型每次输出 output 与标签 label 之间的平方差
                        for index_out, losses_each_out_for in enumerate(losses_each_out): #遍历用于存储损失值的列表。losses_each_out是81个损失函数列表
                            count = mask[:, index_out].sum().item() #mask是2016*81，计算了当前维度（index_out）上有效的样本数量。选择了mask张量中的特定列，然后使用 sum() 方法计算了非零值的数量，并将结果存储在 count=2016中。
                            if count == 0: #根据有效样本数量是否为0，对损失值的更新进行不同的处理
                                losses_each_out_for.update(-1, 1) # count 为0，表示在当前维度上没有有效的样本，所以将 losses_each_out_for 对应的损失值更新为 -1。
                            else:
                                losses_each_out_for.update(
                                    torch.masked_select(se_each_out[:, index_out], mask[:, index_out]).mean().item(), #torch.masked_select根据掩码张量mask中的二元值，取输入张量中的指定项(mask为一个ByteTensor)，将取值返回到一个新的1D张量，
                                    count
                                ) #如果 count 不为0，表示有有效的样本，则计算当前维度上非零的损失值，并计算其均值，并更新到 losses_each_out_for 中。
            if task == 'TEST':
                if self.target == "E_ij":
                    test_targets += torch.squeeze(label_Ei.detach().cpu()).tolist()
                    test_preds += torch.squeeze(output_Ei.detach().cpu()).tolist()
                    test_ids += np.array(batch.stru_id)[torch.squeeze(batch.batch).numpy()].tolist()
                    test_atom_ids += torch.squeeze(
                        torch.tensor(range(batch.num_nodes)) - torch.tensor(batch.__slices__['x'])[
                            batch.batch]).tolist()
                    test_atomic_numbers += torch.squeeze(self.index_to_Z[batch.x]).tolist()
                elif self.target == "E_i":
                    test_targets = torch.squeeze(label.detach().cpu()).tolist()
                    test_preds = torch.squeeze(output.detach().cpu()).tolist()
                    test_ids = np.array(batch.stru_id)[torch.squeeze(batch.batch).numpy()].tolist()
                    test_atom_ids += torch.squeeze(torch.tensor(range(batch.num_nodes)) - torch.tensor(batch.__slices__['x'])[batch.batch]).tolist()
                    test_atomic_numbers += torch.squeeze(self.index_to_Z[batch.x]).tolist()
                else:
                    edge_stru_index = torch.squeeze(batch.batch[batch.edge_index[0]]).numpy() #torch.squeeze是维度压缩， input 中大小为1的所有维都被删除。所有边的起始原子。获取所有边的起始节点所属的图的索引。
                    edge_slices = torch.tensor(batch.__slices__['x'])[edge_stru_index].view(-1, 1)#表示节点特征在批处理中每个图的切片位置，通常是每个图中的节点数目。
                    test_preds += torch.squeeze(output.detach().cpu()).tolist()
                    test_targets += torch.squeeze(label.detach().cpu()).tolist()
                    test_ids += np.array(batch.stru_id)[edge_stru_index].tolist()
                    test_atom_ids += torch.squeeze(batch.edge_index.T - edge_slices).tolist()
                    # test_atomic_numbers += torch.squeeze(self.index_to_Z[batch.x[batch.edge_index.T]]).tolist()
                    test_atomic_numbers += torch.squeeze(batch.x[batch.edge_index.T]).tolist()
                    test_edge_infos += torch.squeeze(batch.edge_attr[:, :7].detach().cpu()).tolist()
            if output_E is True: #False
                if self.target == 'E_ij':
                    output_non_onsite_Ei = scatter_add(output_non_onsite, batch.edge_index.to(self.device)[1, :], dim=0)
                    label_non_onsite_Ei = scatter_add(label_non_onsite, batch.edge_index.to(self.device)[1, :], dim=0)
                    output_Ei = output_non_onsite_Ei + output_onsite
                    label_Ei = label_non_onsite_Ei + label_onsite
                    Etot_error = abs(scatter_add(output_Ei, batch.batch.to(self.device), dim=0)
                                     - scatter_add(label_Ei, batch.batch.to(self.device), dim=0)).reshape(-1).tolist()
                    for test_stru_id, test_error in zip(batch.stru_id, Etot_error):
                        print(f'{test_stru_id}: {test_error * 1000:.2f} meV / unit_cell')
                elif self.target == 'E_i':
                    Etot_error = abs(scatter_add(output, batch.batch.to(self.device), dim=0)
                                     - scatter_add(label, batch.batch.to(self.device), dim=0)).reshape(-1).tolist()
                    for test_stru_id, test_error in zip(batch.stru_id, Etot_error):
                        print(f'{test_stru_id}: {test_error * 1000:.2f} meV / unit_cell')

        if task != 'TRAIN' and (self.out_fea_len != 1): #True
            print('%s loss each out:' % task) #VAL
            #loss_list = list(map(lambda x: f'{x.avg:0.1e}', losses_each_out)) #[81个损失值的平均值]
            #print('[' + ', '.join(loss_list) + ']') #', '.join(loss_list)是将一个列表 loss_list 中的元素连接成一个以 ', ' 为分隔符的字符串。
            loss_list = list(map(lambda x: x.avg, losses_each_out))
            print(f'max orbital: {max(loss_list):0.1e} (0-based index: {np.argmax(loss_list)})') #取出损失值的最大值。
        if task == 'TEST':
            with open(os.path.join(self.config["basic"]["save_dir"], save_name), 'w', newline='') as f:
                writer = csv.writer(f)
                if self.target == "E_i" or self.target == "E_ij":
                    writer.writerow(['stru_id', 'atom_id', 'atomic_number'] +
                                    ['target'] * self.out_fea_len + ['pred'] * self.out_fea_len)
                    for stru_id, atom_id, atomic_number, target, pred in zip(test_ids, test_atom_ids,
                                                                             test_atomic_numbers,
                                                                             test_targets, test_preds):
                        if self.out_fea_len == 1:
                            writer.writerow((stru_id, atom_id, atomic_number, target, pred))
                        else:
                            writer.writerow((stru_id, atom_id, atomic_number, *target, *pred))

                else:
                    writer.writerow(['stru_id', 'atom_id', 'atomic_number', 'dist', 'atom1_x', 'atom1_y', 'atom1_z',
                                     'atom2_x', 'atom2_y', 'atom2_z']
                                    + ['target'] * self.out_fea_len + ['pred'] * self.out_fea_len)
                    for stru_id, atom_id, atomic_number, edge_info, target, pred in zip(test_ids, test_atom_ids,
                                                                                        test_atomic_numbers,
                                                                                        test_edge_infos,
                                                                                        test_targets,
                                                                                        test_preds):
                        if self.out_fea_len == 1:
                            writer.writerow((stru_id, atom_id, atomic_number, *edge_info, target, pred))
                        else:
                            writer.writerow((stru_id, atom_id, atomic_number, *edge_info, *target, *pred))
        return losses #返回训练集的损失值
