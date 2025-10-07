import os
import time
import subprocess as sp
import json

import argparse

import sys
import os
sys.path.append('/fs2/home/ndsim10/DeepQT/DeepQTH/3_predict')


from  import get_inference_config, rotate_back, siesta_parse
from preprocess import openmx_parse_overlap, get_rc
from predict import predict, predict_with_grad

def main():
    parser = argparse.ArgumentParser(description='Deep Hamiltonian')
    parser.add_argument('--config', default=[], nargs='+', type=str, metavar='N')
    args = parser.parse_args()

    print(f'User config name: {args.config}')
    config = get_inference_config(args.config)

    work_dir = os.path.abspath(config.get('basic', 'work_dir')) #要预测的大结构的数据的目录../example/work_dir/inference/5_4/
    OLP_dir = os.path.abspath(config.get('basic', 'OLP_dir')) #大结构的重叠矩阵的文件路径../example/work_dir/olp/5_4/
    system_name = os.path.abspath(config.get('basic', 'system_name')).split('/')[-1]
    interface = config.get('basic', 'interface') #siesta
    abacus_suffix = str(config.get('basic', 'abacus_suffix', fallback='ABACUS')) # 从配置对象config中获取了'basic'部分下的键'abacus_suffix'对应的值。如果找不到这个键，就使用'ABACUS'作为默认值
    task = json.loads(config.get('basic', 'task')) #[1, 2, 3, 4, 5]
    assert isinstance(task, list) #检查变量 task 是否是列表类型的实例。如果task不是列表类型，则会引发AssertionError，程序执行会停止。
    eigen_solver = config.get('basic', 'eigen_solver') #dense_py
    disable_cuda = config.getboolean('basic', 'disable_cuda') #False
    device = config.get('basic', 'device') #cpu
    huge_structure = config.getboolean('basic', 'huge_structure') #True
    restore_blocks_py = config.getboolean('basic', 'restore_blocks_py') #True
    gen_rc_idx = config.getboolean('basic', 'gen_rc_idx') #False
    gen_rc_by_idx = config.get('basic', 'gen_rc_by_idx') #[]
    with_grad = config.getboolean('basic', 'with_grad') #False
    julia_interpreter = config.get('interpreter', 'julia_interpreter', fallback='') #julia
    python_interpreter = config.get('interpreter', 'python_interpreter', fallback='') #python
    radius = config.getfloat('graph', 'radius') #9.0
    print("system_name = ",system_name)

    if 5 in task:
        if eigen_solver in ['sparse_jl', 'dense_jl']: #预测出的哈密顿量矩阵是稀疏的，即预测的哈密顿量矩阵中元素不为零的原子对的轨道哈密顿量小矩阵。
            assert julia_interpreter, "Please specify julia_interpreter to use Julia code to calculate eigenpairs"
        elif eigen_solver in ['dense_py']:
            assert python_interpreter, "Please specify python_interpreter to use Python code to calculate eigenpairs"
        else:
            raise ValueError(f"Unknown eigen_solver: {eigen_solver}")
    if 3 in task and not restore_blocks_py:
        assert julia_interpreter, "Please specify julia_interpreter to use Julia code to rearrange matrix blocks"

    if with_grad:
        assert restore_blocks_py is True
        assert 4 not in task
        assert 5 not in task

    os.makedirs(work_dir, exist_ok=True)
    config.write(open(os.path.join(work_dir, 'config.ini'), "w")) #要预测的大结构的配置文件


    if not restore_blocks_py:
        cmd3_post = f"{julia_interpreter} " \
                    f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inference', 'restore_blocks.jl')} " \
                    f"--input_dir {work_dir} --output_dir {work_dir}"
    #生成计算稀疏矩阵和能带图的command命令
    if eigen_solver == 'sparse_jl':
        cmd5 = f"{julia_interpreter} " \
               f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inference', 'sparse_calc.jl')} " \
               f"--input_dir {work_dir} --output_dir {work_dir} --config {config.get('basic', 'sparse_calc_config')}" #julia sparse_calc.jl --input_dir work_dir --output_dir work_dir --config "~/example/work_dir/inference/5_4/band.json"
    elif eigen_solver == 'dense_jl':
        cmd5 = f"{julia_interpreter} " \
               f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inference', 'dense_calc.jl')} " \
               f"--input_dir {work_dir} --output_dir {work_dir} --config {config.get('basic', 'sparse_calc_config')}"
    elif eigen_solver == 'dense_py':
        cmd5 = f"{python_interpreter} " \
               f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inference', 'dense_calc.py')} " \
               f"--input_dir {work_dir} --output_dir {work_dir} --config {config.get('basic', 'sparse_calc_config')}"
    else:
        raise ValueError(f"Unknown eigen_solver: {eigen_solver}")


    print(f"\n~~~~~~~ 1.parse_Overlap\n")
    print(f"\n~~~~~~~ 2.get_local_coordinate\n")
    print(f"\n~~~~~~~ 3.get_pred_Hamiltonian\n")
    if not restore_blocks_py:
        print(f"\n~~~~~~~ 3_post.restore_blocks, command: \n{cmd3_post}\n")
    print(f"\n~~~~~~~ 4.rotate_back\n")
    print(f"\n~~~~~~~ 5.sparse_calc, command: \n{cmd5}\n")

    
    #1方法要改成siesta的数据，如何获取要预测的大结构的overlaps.h5是关键，直接单步计算就行
    if 1 in task:
        begin = time.time()
        print(f"\n####### Begin 1.parse_Overlap")
        if interface == 'openmx':
            assert os.path.exists(os.path.join(OLP_dir, 'openmx.out')), "Necessary files could not be found in OLP_dir"
            assert os.path.exists(os.path.join(OLP_dir, 'output')), "Necessary files could not be found in OLP_dir"
            openmx_parse_overlap(OLP_dir, work_dir) #生成大结构的overlaps.h5、site_positions.dat、lat.dat、rlat.dat、element.dat、orbital_types.dat文件
        elif interface == 'abacus':
            print("Output subdirectories:", "OUT." + abacus_suffix)
            assert os.path.exists(os.path.join(OLP_dir, 'SR.csr')), "Necessary files could not be found in OLP_dir"
            assert os.path.exists(os.path.join(OLP_dir, f'OUT.{abacus_suffix}')), "Necessary files could not be found in OLP_dir"
            abacus_parse(OLP_dir, work_dir, data_name=f'OUT.{abacus_suffix}', only_S=True)
        elif interface == 'siesta':
            assert os.path.exists(os.path.join(OLP_dir, '{}.HSX'.format(system_name))), "Necessary files could not be found in OLP_dir"
            siesta_parse(OLP_dir, work_dir)
        assert os.path.exists(os.path.join(work_dir, "overlaps.h5"))
        assert os.path.exists(os.path.join(work_dir, "lat.dat"))
        assert os.path.exists(os.path.join(work_dir, "rlat.dat"))
        assert os.path.exists(os.path.join(work_dir, "site_positions.dat"))
        assert os.path.exists(os.path.join(work_dir, "orbital_types.dat"))
        assert os.path.exists(os.path.join(work_dir, "element.dat"))
        print('\n******* Finish 1.parse_Overlap, cost %d seconds\n' % (time.time() - begin))

    if not with_grad and 2 in task: #True
        begin = time.time()
        print(f"\n####### Begin 2.get_local_coordinate")
        get_rc(work_dir, work_dir, radius=radius, gen_rc_idx=gen_rc_idx, gen_rc_by_idx=gen_rc_by_idx,
               create_from_DFT=config.getboolean('graph', 'create_from_DFT')) #得到截断半径内的排序后的原子对的3*3的单位局域坐标，存入rc.h5文件
        assert os.path.exists(os.path.join(work_dir, "rc.h5"))
        print('\n******* Finish 2.get_local_coordinate, cost %d seconds\n' % (time.time() - begin))

    if 3 in task:
        begin = time.time()
        print(f"\n####### Begin 3.get_pred_Hamiltonian")
        trained_model_dir = config.get('basic', 'trained_model_dir')
        if trained_model_dir[0] == '[' and trained_model_dir[-1] == ']':
            trained_model_dir = json.loads(trained_model_dir)
        print(trained_model_dir) #['/fs2/home/ndsim10/deeph/example/work_dir/trained_model/2024-01-03_14-11-23']

        if with_grad: #False
            predict_with_grad(input_dir=work_dir, output_dir=work_dir, disable_cuda=disable_cuda, device=device,
                              huge_structure=huge_structure, trained_model_dirs=trained_model_dir) #预测出大结构的哈密顿量存入到了hamiltonians_pred.h5文件中
        else:
            predict(input_dir=work_dir, output_dir=work_dir, disable_cuda=disable_cuda, device=device,
                    huge_structure=huge_structure, restore_blocks_py=restore_blocks_py,
                    trained_model_dirs=trained_model_dir) #预测的大结构哈密顿量存入到了rh_pred.h5文件中
            #inference/5_4/, inference/5_4/, False, cpu, True, True, trained_model/2024-01-03_14-11-23
        if restore_blocks_py: #True
            if with_grad: #False
                assert os.path.exists(os.path.join(work_dir, "hamiltonians_grad_pred.h5"))
                assert os.path.exists(os.path.join(work_dir, "hamiltonians_pred.h5"))
            else:
                assert os.path.exists(os.path.join(work_dir, "rh_pred.h5")) #断言成功预测的哈密顿量矩阵
        else:
            capture_output = sp.run(cmd3_post, shell=True, capture_output=False, encoding="utf-8")
            assert capture_output.returncode == 0
            assert os.path.exists(os.path.join(work_dir, "rh_pred.h5"))
        print('\n******* Finish 3.get_pred_Hamiltonian, cost %d seconds\n' % (time.time() - begin))


    if 4 in task:
        begin = time.time()
        print(f"\n####### Begin 4.rotate_back")
        rotate_back(input_dir=work_dir, output_dir=work_dir) #把预测的局域坐标下的哈密顿量矩阵旋转回全局坐标,得到最终预测的哈密顿量hamiltonians_pred.h5
        assert os.path.exists(os.path.join(work_dir, "hamiltonians_pred.h5"))
        print('\n******* Finish 4.rotate_back, cost %d seconds\n' % (time.time() - begin))

    
    if 5 in task:
        begin = time.time()
        print(f"\n####### Begin 5.sparse_calc")
        capture_output = sp.run(cmd5, shell=True, capture_output=False, encoding="utf-8") #求稀疏矩阵的特征值和特征向量，得到能带结构文件openmx.Band和态密度文件dos.dat。
        assert capture_output.returncode == 0
        if eigen_solver in ['sparse_jl']:
            assert os.path.exists(os.path.join(work_dir, "sparse_matrix.jld"))
        print('\n******* Finish 5.sparse_calc, cost %d seconds\n' % (time.time() - begin))


if __name__ == '__main__':
    main()
