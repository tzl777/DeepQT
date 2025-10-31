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
    def __init__(self, config, is_predict=False):
        self.config = config
        # basic config
        if is_predict == False:
            if config['basic']['save_to_time_folder']:
                config["basic"]["save_dir"] = os.path.join(
                    config["basic"]["save_dir"],
                    time.strftime("%Y-%m-%d_%H-%M-%S-", time.localtime())+config["network"]["atom_update_net"],
                )
                assert not os.path.exists(config["basic"]["save_dir"])
            os.makedirs(config["basic"]["save_dir"], exist_ok=True) 
            sys.stdout = Logger(os.path.join(config["basic"]["save_dir"], "result.txt")) 
            sys.stderr = Logger(os.path.join(config["basic"]["save_dir"], "stderr.txt"))
            print('====== CONFIG ======')
            with open(os.path.join(config["basic"]["save_dir"], "config.txt"), "w") as f:
                for section_k, section_v in config.items():  
                    f.write(f'[{section_k}]\n')
                    for k, v in section_v.items():
                        f.write(f'{k}={v}\n')
                    f.write('\n')
        self.graph_dir = config['basic']['graph_dir']
        # src_dir = os.path.join(config["basic"]["save_dir"], "src")
        # os.makedirs(src_dir, exist_ok=True)
        # try:
        #     shutil.copytree(os.path.dirname(__file__), os.path.join(src_dir, 'deeph'))
        # except:
        #     warnings.warn("Unable to copy scripts")
        if not config["basic"]["disable_cuda"]: #False
            self.device = torch.device(config["basic"]["device"] if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        config["basic"]["device"] = str(self.device) 
        if config["hyperparameter"]["dtype"] == 'float32': 
            default_dtype_torch = torch.float32
        elif config["hyperparameter"]["dtype"] == 'float16':
            default_dtype_torch = torch.float16
        elif config["hyperparameter"]["dtype"] == 'float64':
            default_dtype_torch = torch.float64
        else:
            raise ValueError('Unknown dtype: {}'.format(config["hyperparameter"]["dtype"]))
        np.seterr(all='raise') 
        np.seterr(under='warn')
        np.set_printoptions(precision=8, linewidth=160)
        torch.set_default_dtype(default_dtype_torch)
        torch.set_printoptions(precision=8, linewidth=160, threshold=np.inf)
        np.random.seed(int(config["basic"]["seed"]))
        torch.manual_seed(int(config["basic"]["seed"]))
        torch.cuda.manual_seed_all(int(config["basic"]["seed"]))
        random.seed(int(config["basic"]["seed"]))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        
        if int(config["basic"]["num_threads"]) == -1:
            if torch.cuda.device_count() == 0:
                torch.set_num_threads(cpu_count(logical=False)
            else:
                torch.set_num_threads(cpu_count(logical=False) // torch.cuda.device_count())
        else:
            torch.set_num_threads(int(config["basic"]["num_threads"]))

        self.atom_update_net = config["network"]["atom_update_net"]
        self.if_lcmp = config["network"]["if_lcmp"] 
        self.if_lcmp_graph = config["graph"]["if_lcmp_graph"] #True
        self.new_sp = config["graph"]["new_sp"] #False
        self.separate_onsite = config["graph"]["separate_onsite"] #False
        # print(self.separate_onsite)
        # print(type(self.separate_onsite))
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
        self.early_stopping_loss_epoch = config['train']['early_stopping_loss_epoch'] 
        
    def build_model(self, model_pack_dir: str = None, old_version=None):
        if model_pack_dir is not None:
            assert old_version is not None
            if old_version is True:
                print(f'import HGNN from {model_pack_dir}')
                sys.path.append(model_pack_dir)
                from deeph import HGNN
            else:
                checkpoint = torch.load(os.path.join(model_pack_dir, 'best_model.pt'))
                # imp = package.PackageImporter(os.path.join(model_pack_dir, 'best_model.pt')) 
                # checkpoint = imp.load_pickle('checkpoint', 'model.pkl', map_location=self.device) 
                self.model = checkpoint['model_dict']['model']
                self.model.to(self.device)
                self.index_to_Z = checkpoint['state']["index_to_Z"] 
                self.Z_to_index = checkpoint['state']["Z_to_index"]
                self.spinful = checkpoint['state']["spinful"] #False
                print("=> load best checkpoint (epoch {})".format(checkpoint['state']['epoch']))
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
                return checkpoint
        else:
            if self.atom_update_net == "deeph":
                from deeph.deeph import DeepH
                parameter_list = list(signature(DeepH.__init__).parameters.keys())
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
            n_elements=self.num_species,
            num_species=self.num_species, #1
            in_atom_fea_len=self.config['network']['atom_fea_len'], #64
            in_vfeats=self.config['network']['atom_fea_len'], #64
            in_edge_fea_len=self.config['network']['edge_fea_len'], #128
            in_efeats=self.config['network']['edge_fea_len'], #128，
            out_edge_fea_len=self.out_fea_len, #81，
            out_efeats=self.out_fea_len, #81，
            num_orbital=self.out_fea_len, #81
            distance_expansion=self.config['network']['distance_expansion'], #GaussianBasis
            gauss_stop=self.config['network']['gauss_stop'], #6.0
            cutoff=self.config['network']['gauss_stop'], #6.0，
            if_exp=self.config['network']['if_exp'], #True
            if_MultipleLinear=self.config['network']['if_MultipleLinear'], #False
            if_edge_update=self.config['network']['if_edge_update'], #True
            if_lcmp=self.if_lcmp, #True
            normalization=self.config['network']['normalization'], #LayerNorm
            atom_update_net=self.config['network']['atom_update_net'], 
            separate_onsite=self.separate_onsite, #False
            n_heads=self.config['network']['n_heads'],  #4,
            num_l=self.config['network']['num_l'], #5
            trainable_gaussians=self.config['network']['trainable_gaussians'], #True
            type_affine=self.config['network']['type_affine'], #False
            if_fc_out=False, 
            max_path_length=self.config['network']['max_path_length'], #5,
        )
        current_parameter_list = list(model_kwargs.keys())
        for k in current_parameter_list:
            if k not in parameter_list:
                model_kwargs.pop(k) 
        if 'num_elements' in parameter_list: #False
            model_kwargs['num_elements'] = self.config['basic']['max_element'] + 1 #0
        if self.atom_update_net == "deeph":
            self.model = DeepH(**model_kwargs) 
        elif self.atom_update_net == "gat":
            pass
            # schnet, phisnet, deeph, gat, painn, graphormer, transformer-m, deepqth
        else:
            pass
        
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters]) 
        print("The model you built has: %d parameters" % params)
        print("model = ", self.model)
        self.model.to(self.device)
        self.load_pretrained()

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

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters()) 
        if self.config['hyperparameter']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(model_parameters, lr=learning_rate, weight_decay=weight_decay)
        elif self.config['hyperparameter']['optimizer'] == 'sgdm':
            self.optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif self.config['hyperparameter']['optimizer'] == 'adam': #adam
            self.optimizer = optim.Adam(model_parameters, lr=learning_rate, betas=(0.9, 0.999)) 
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

        if self.config['hyperparameter']['lr_scheduler'] == '': #True，
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

    def load_pretrained(self):
        pretrained = self.config['train']['pretrained']
        if pretrained:
            if os.path.isfile(pretrained):
                checkpoint = torch.load(pretrained, map_location=self.device)
                pretrained_dict = checkpoint['model_state_dict']['state_dict']
                model_dict = self.model.state_dict() 
                transfer_dict = {}
                for k, v in pretrained_dict.items():
                    if v.shape == model_dict[k].shape:
                        transfer_dict[k] = v
                        print('Use pretrained parameters:', k)

                model_dict.update(transfer_dict) 
                self.model.load_state_dict(model_dict)
                print(f'=> loaded pretrained model at "{pretrained}" (epoch {checkpoint["state"]["epoch"]})') 
            else:
                print(f'=> no checkpoint found at "{pretrained}"') 

    
    def load_resume(self):
        resume = self.config['train']['resume']
        if resume:
            if os.path.isfile(resume):
                checkpoint = torch.load(resume, map_location=self.device) 
                self.model.load_state_dict(checkpoint['model_state_dict']['state_dict'])
                self.optimizer.load_state_dict(checkpoint['state']['optimizer_state_dict']) 
                print(f'=> loaded model at "{resume}" (epoch {checkpoint["state"]["epoch"]})')
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
        self.index_to_Z = dataset.info["index_to_Z"] 
        self.Z_to_index = dataset.info["Z_to_index"] 
        self.num_species = len(dataset.info["index_to_Z"]) 
        
        dataset_size = len(dataset)
        train_size = int(self.config["train"]["train_ratio"] * dataset_size) 
        val_size = int(self.config["train"]["val_ratio"] * dataset_size) 
        test_size = int(self.config["train"]["test_ratio"] * dataset_size) 
        assert train_size + val_size + test_size <= dataset_size

        indices = list(range(dataset_size)) 
        np.random.shuffle(indices) 
        print(f'number of train set: {len(indices[:train_size])}') 
        print(f'number of val set: {len(indices[train_size:train_size + val_size])}') 
        print(f'number of test set: {len(indices[train_size + val_size:train_size + val_size + test_size])}') 
        train_sampler = SubsetRandomSampler(indices[:train_size]) 
        val_sampler = SubsetRandomSampler(indices[train_size:train_size + val_size])
        test_sampler = SubsetRandomSampler(indices[train_size + val_size:train_size + val_size + test_size])
        train_loader = DataLoader(dataset, batch_size=self.config["hyperparameter"]["batch_size"],
                                  shuffle=False, sampler=train_sampler,
                                  collate_fn=Collater(self.if_lcmp)) 
        val_loader = DataLoader(dataset, batch_size=self.config["hyperparameter"]["batch_size"],
                                shuffle=False, sampler=val_sampler,
                                collate_fn=Collater(self.if_lcmp))
        test_loader = DataLoader(dataset, batch_size=self.config["hyperparameter"]["batch_size"],
                                 shuffle=False, sampler=test_sampler,
                                 collate_fn=Collater(self.if_lcmp))

        if self.config["basic"]["statistics"]: #False，
            sample_label = torch.cat([dataset[i].label for i in range(len(dataset))])
            sample_mask = torch.cat([dataset[i].mask for i in range(len(dataset))])
            mean_value = abs(sample_label).sum(dim=0) / sample_mask.sum(dim=0)
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
            plt.tight_layout()
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
        print(transform.state_dict()) 

        return train_loader, val_loader, test_loader, transform

    def train(self, train_loader, val_loader, test_loader):
        begin_time = time.time()
        self.best_val_loss = 1e10
        if self.config["train"]["revert_then_decay"]: #True，
            lr_step = 0
        #revert_threshold = 30，
        revert_decay_epoch = self.config["train"]["revert_decay_epoch"]
        revert_decay_gamma = self.config["train"]["revert_decay_gamma"]
        assert len(revert_decay_epoch) == len(revert_decay_gamma)
        lr_step_num = len(revert_decay_epoch) #3

        try:
            for epoch in range(self.config["train"]["epochs"]):
                print("==================================================================")
                
                if self.config["train"]["switch_sgd"] and epoch == self.config["train"]["switch_sgd_epoch"]: 
                    model_parameters = filter(lambda p: p.requires_grad, self.model.parameters()) 
                    self.optimizer = optim.SGD(model_parameters, lr=self.config["train"]["switch_sgd_lr"])
                    print(f"Switch to sgd (epoch: {epoch})")
                learning_rate = self.optimizer.param_groups[0]['lr'] 
                
                with open(os.path.join(self.config["basic"]["save_dir"], 'learning_rate.csv'), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, learning_rate])
                
                # start = time.perf_counter()
                # train
                train_losses = self.kernel_fn(train_loader, 'TRAIN')
                with open(os.path.join(self.config["basic"]["save_dir"], 'training_loss.csv'), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_losses.avg])
                
                with torch.no_grad():
                    val_losses = self.kernel_fn(val_loader, 'VAL')

                print("val_losses.avg = ", val_losses.avg)
                print("best_val_loss = ", self.config["train"]["revert_threshold"] * self.best_val_loss)
                if val_losses.avg > self.config["train"]["revert_threshold"] * self.best_val_loss:
                    print(f'Epoch #{epoch:01d} \t| '
                          f'Learning rate: {learning_rate:0.2e} \t| '
                          f'Epoch time: {time.time() - begin_time:.2f} \t| '
                          f'Train loss: {train_losses.avg:.8f} \t| '
                          f'Val loss: {val_losses.avg:.8f} \t| '
                          f'Best val loss: {self.best_val_loss:.8f}.'
                          )
                    best_checkpoint = torch.load(os.path.join(self.config["basic"]["save_dir"], 'best_state_dict.pkl'))
                    self.model.load_state_dict(best_checkpoint['model_state_dict']['state_dict'])
                    self.optimizer.load_state_dict(best_checkpoint['state']['optimizer_state_dict'])
                    if self.config["train"]["revert_then_decay"]: #True
                        if lr_step < lr_step_num:
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = learning_rate * revert_decay_gamma[lr_step]
                            lr_step += 1
                    with torch.no_grad():
                        val_losses = self.kernel_fn(val_loader, 'VAL')
                    print(f"Revert (threshold: {self.config['train']['revert_threshold']}) to epoch {best_checkpoint['state']['epoch']} \t| Val loss: {val_losses.avg:.8f}")
                    with open(os.path.join(self.config["basic"]["save_dir"], 'validation_loss.csv'), 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, val_losses.avg])

                    if self.config["hyperparameter"]["lr_scheduler"] == 'MultiStepLR':
                        self.scheduler.step()
                    elif self.config["hyperparameter"]["lr_scheduler"] == 'ReduceLROnPlateau':
                        self.scheduler.step(val_losses.avg)
                    elif self.config["hyperparameter"]["lr_scheduler"] == 'CyclicLR':
                        self.scheduler.step()
                    continue

                with open(os.path.join(self.config["basic"]["save_dir"], 'validation_loss.csv'), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, val_losses.avg])
                
                if self.config["train"]["revert_then_decay"]: #True
                    if lr_step < lr_step_num and epoch >= revert_decay_epoch[lr_step]:
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
                            path=self.config["basic"]["save_dir"], is_best=is_best)
                        save_complete = True
                    except KeyboardInterrupt:
                        print('\nKeyboardInterrupt while saving model to disk') 

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
        self.model.load_state_dict(best_checkpoint['model_state_dict']['state_dict'])
        print("=> load best checkpoint (epoch {})".format(best_checkpoint['state']['epoch']))
        with torch.no_grad():
            test_csv_name = 'test_results.csv'
            train_csv_name = 'train_results.csv'
            val_csv_name = 'val_results.csv'

            if self.config["basic"]["save_csv"]: #False
                tmp = 'TEST'
            else:
                tmp = 'VAL'
            test_losses = self.kernel_fn(test_loader, tmp, test_csv_name, output_E=True)
            print(f'Test loss: {test_losses.avg:.8f}.')
            with open(os.path.join(self.config["basic"]["save_dir"], 'test_loss.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, test_losses.avg])
                
            test_losses = self.kernel_fn(train_loader, tmp, train_csv_name, output_E=True)
            print(f'Train loss: {test_losses.avg:.8f}.')
            test_losses = self.kernel_fn(val_loader, tmp, val_csv_name, output_E=True)
            print(f'Val loss: {test_losses.avg:.8f}.')

    def predict(self, hamiltonian_dirs):
        raise NotImplementedError

    def kernel_fn(self, loader, task: str, save_name=None, output_E=False):
        assert task in ['TRAIN', 'VAL', 'TEST']
        losses = LossRecord()
        if task == 'TRAIN':
            self.model.train() 
        else:
            self.model.eval()
        if task == 'TEST':
            assert save_name != None
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
            losses_each_out = [LossRecord() for _ in range(self.out_fea_len)]
        for step, batch_tuple in enumerate(loader):
            # start = time.perf_counter()
            if self.if_lcmp:
                batch, subgraph = batch_tuple #batch=3
                # print(batch.keys)
                sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index = subgraph
                output = self.model(
                    batch.x.to(self.device), 
                    batch.edge_index.to(self.device), 
                    batch.edge_attr.to(self.device), 
                    # batch.node_paths.to(self.device),
                    # batch.edge_paths.to(self.device),
                    # batch.voronoi_values.to(self.device),
                    # batch.centralities.to(self.device),
                    batch.batch.to(self.device), 
                    sub_atom_idx.to(self.device),
                    sub_edge_idx.to(self.device), 
                    sub_edge_ang.to(self.device),
                    sub_index.to(self.device) 
                )
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
                    mask = batch.mask.to(self.device) #[6048,81]
                    loss = self.criterion(output, label, mask)
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
                else:
                    self.optimizer.zero_grad()
                    # start = time.perf_counter()
                    loss.backward() 
                    if self.config['train']['clip_grad']:
                        clip_grad_norm_(self.model.parameters(), self.config['train']['clip_grad_value']) 
                    self.optimizer.step()
            if self.target == "E_i" or self.target == "E_ij":
                losses.update(loss.item(), batch.num_nodes)
            else:
                if self.criterion_name == 'MaskMSELoss':
                    losses.update(loss.item(), mask.sum())
                    
                if task != 'TRAIN' and self.out_fea_len != 1:
                    if self.criterion_name == 'MaskMSELoss':
                        se_each_out = torch.pow(output - label.to(self.device), 2) 
                        for index_out, losses_each_out_for in enumerate(losses_each_out): 
                            count = mask[:, index_out].sum().item()
                            if count == 0:
                                losses_each_out_for.update(-1, 1) 
                            else:
                                losses_each_out_for.update(
                                    torch.masked_select(se_each_out[:, index_out], mask[:, index_out]).mean().item(), 
                                    count
                                ) 
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
                    edge_stru_index = torch.squeeze(batch.batch[batch.edge_index[0]]).numpy()
                    edge_slices = torch.tensor(batch.__slices__['x'])[edge_stru_index].view(-1, 1)
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
            #loss_list = list(map(lambda x: f'{x.avg:0.1e}', losses_each_out))
            #print('[' + ', '.join(loss_list) + ']') #', '.join(loss_list)
            loss_list = list(map(lambda x: x.avg, losses_each_out))
            print(f'max orbital: {max(loss_list):0.1e} (0-based index: {np.argmax(loss_list)})') 
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
        return losses
