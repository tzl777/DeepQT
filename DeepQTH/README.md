**DeepQTH**(Deep learning of Quantum Transport Hamiltonian) is one of the submodules of **DeepQT**, designed to predict the intermediate variable in NEGF-DFT—the equilibrium Hamiltonian under zero-bias—and thereby enable the prediction of various electronic structure properties.

**DeepQTH** supports both DFT results and NEGF-DFT results from [SIESTA/TranSIESTA](https://siesta-project.org/siesta/CodeAccess/index.html).

# Contents
1. [Installation](#Installation)
2. [Use DeepQTH](#Use-DeepQTH)
3. [Demo](#demo-deeph-study-on-twisted-bilayer-bismuthene)
4. [How to cite](#how-to-cite)


# Installation

## System Requirements
- **Python Version**: 3.9 or higher  
- **Environment Management**: Conda is recommended  

## Installation Steps

### Method 1: Using Conda Environment (Recommended)

1. **Create and activate the Conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate deepqt
   ```

### Method 2: Manual Installation

1. **Create a Python virtual environment**

   ```bash
   conda create --name deepqt python=3.9
   conda activate deepqt
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Clone DeepQT and Navigate to the root directory:

   ```bash
   git clone https://github.com/tzl777/DeepQT.git
   cd DeepQT
   pip install .
   ```

### Install siesta
Install [SIESTA package](https://gitlab.com/siesta-project/siesta) for NEGF-DFT Hamiltonian matrix calculation to construct datasets. DeepQTH requires SIESTA version >= 4.1.5.

# Use DeepQTH

### Prepare the dataset
DeepQTH is designed to predict the intermediate variable in DFT/NEGF-DFT—the Hamiltonian—and thereby enable efficient first-principles electronic structure prediction for large-scale systems. To achieve this, an appropriate small-scale structural dataset that reflects the bonding environments of the target large-scale system must be constructed, followed by DFT/NEGF-DFT calculations to obtain localized-basis Hamiltonian matrices.

### Preprocess the dataset
`Preprocess` is a part of DeepH-pack. Through `Preprocess`, DeepH-pack will
convert the unit of physical quantity, store the data files in the format
of text and *HDF5* for each structure in a separate folder, generate local
coordinates, and perform basis transformation for DFT Hamiltonian matrices.
We use the following convention of units:


You need to edit a configuration in the format of *ini*, setting up the
file referring to the default file `DeepH-pack/deeph/preprocess/preprocess_default.ini`.
The meaning of the keywords can be found in the
[documentation](https://deeph-pack.readthedocs.io/en/latest/keyword/preprocess.html).
For a quick start, you must set up *raw_dir*, *processed_dir* and *interface*.

With the configuration file prepared, run 
```bash
deeph-preprocess --config ${config_path}
```
with `${config_path}` replaced by the path of your configuration file.

### Train your model
`Train` is a part of DeepH-pack, which is used to train a deep learning model using the processed dataset.

Prepare a configuration in the format of *ini*, setting up the file referring to the default `DeepH-pack/deeph/default.ini`. The meaning of the keywords can be found in the [documentation](https://deeph-pack.readthedocs.io/en/latest/keyword/train.html). For a quick start, you must set up *graph_dir*, *save_dir*, *raw_dir* and *orbital*, other keywords can stay default and be adjusted later.

With the configuration file prepared, run 
```bash
deeph-train --config ${config_path}
```
with `${config_path}` replaced by the path of your configuration file.

Tips:
- **Name your dataset**. Use *dataset_name* to name your dataset, the same names may overwrite each other.

- **Hyperparameters of the neural network**. The neural network here contains some hyperparameters. For a specific problem your should try adjusting the hyperparameters to obtain better results.

- **The keyword *orbital***. The keyword *orbital* states which orbitals or matrix elements are predicted. It is a little complicated to understand its data structure. To figure out it, you can refer to the [documentation](https://deeph-pack.readthedocs.io/en/latest/keyword/train.html#:~:text=generate%20crystal%20graphs.-,orbital,-%3A%20A%20JSON%20format) or the method [make_mask](https://github.com/mzjb/DeepH-pack/blob/main/deeph/kernel.py#:~:text=def%20make_mask(self%2C%20dataset)%3A) in class `DeepHKernel` defined in `DeepH-pack/deeph/kernel.py`.

    Alternatively, a Python script at `DeepH-pack/tools/get_all_orbital_str.py` can be used to generate a default configuration to predict all orbitals with one model.

- **Use TensorBoard for visualizations**. You can track and visualize the training process through TensorBoard by running
  ```bash
  tensorboard --logdir=./tensorboard
  ```
  in the output directory (*save_dir*):

### Inference with your model
`Inference` is a part of DeepH-pack, which is used to predict the 
DFT Hamiltonian for large-scale material structures and perform 
sparse calculation of physical properties.

Firstly, one should prepare the structure file of large-scale material 
and calculate the overlap matrix. Overlap matrix calculation does not
require `SCF`. Even if the material system is large, only a small calculation
time and memory consumption are required. Following are the steps to
calculate the overlap matrix using different supported DFT packages:
1. **ABACUS**: Set the following parameters in the input file of ABACUS `INPUT`:
    ```
    calculation   get_S
    ```
    and run ABACUS like a normal `SCF` calculation.
    [ABACUS version >= 2.3.2](https://github.com/deepmodeling/abacus-develop/releases/tag/v2.3.2) is required.
2. **OpenMX**: See this [repository](https://github.com/mzjb/overlap-only-OpenMX#usage).

For overlap matrix calculation, you need to use the same basis set and DFT
software when preparing the dataset.

Then, prepare a configuration in the format of *ini*, setting up the 
file referring to the default `DeepH-pack/deeph/inference/inference_default.ini`. 
The meaning of the keywords can be found in the
[INPUT KEYWORDS section](https://deeph-pack.readthedocs.io/en/latest/keyword/inference.html). 
For a quick start, you must set up *OLP_dir*, *work_dir*, *interface*,
*trained_model_dir* and *sparse_calc_config*, as well as a `JSON` 
configuration file located at *sparse_calc_config* for sparse calculation.

With the configuration files prepared, run 
```bash
deeph-inference --config ${config_path}
```
with `${config_path}` replaced by the path of your configuration file.

# Demo: DeepH study on twisted bilayer bismuthene
When the directory structure of the code folder is not modified, the scripts in it can be used to generate a dataset of non-twisted structures, train a DeepH model, make predictions on the DFT Hamiltonian matrix of twisted structure, and perform sparse diagonalization to compute the band structure for the example study of bismuthene.

Firstly, generate example input files according to your environment path by running the following command:
```bash
cd DeepH-pack
python gen_example.py ${openmx_path} ${openmx_overlap_path} ${pot_path} ${python_interpreter} ${julia_interpreter}
```
with `${openmx_path}`, `${openmx_overlap_path}`, `${pot_path}`, `${python_interpreter}`, and `${julia_interpreter}` replaced by the path of original OpenMX executable program, modified 'overlap only' OpenMX executable program, VPS and PAO directories of OpenMX, Python interpreter, and Julia interpreter, respectively. For example, 
```bash
cd DeepH-pack
python gen_example.py /home/user/openmx/source/openmx /home/user/openmx_overlap/source/openmx /home/user/openmx/DFT_DATA19 python /home/user/julia-1.5.4/bin/julia
```

Secondly, enter the generated `example/` folder and run `run.sh` in each folder one-by-one from 1 to 5. Please note that `run.sh` should be run in the directory where the `run.sh` file is located.
```bash
cd example/1_DFT_calculation
bash run.sh
cd ../2_preprocess
bash run.sh
cd ../3_train
bash run.sh
cd ../4_compute_overlap
bash run.sh
cd ../5_inference
bash run.sh
```
The third step, the neural network training process, is recommended to be carried out on the GPU. In addition, in order to get the energy band faster, it is recommended to calculate the eigenvalues ​​of different k points in parallel in the fifth step by *which_k* interface.

After completing the calculation, you can find the band structure data in OpenMX Band format of twisted bilayer bismuthene with 244 atoms per supercell computed by the predicted DFT Hamiltonian in the file below:
```
example/work_dir/inference/5_4/openmx.Band
```
The plotted band structure will be consistent with the right pannel of figure 6c in our paper.

# How to cite

