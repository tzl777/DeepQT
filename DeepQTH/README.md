**DeepQTH**(Deep learning of Quantum Transport Hamiltonian) is one of the submodules of **DeepQT**, designed to predict the intermediate variable in NEGF-DFT—the equilibrium Hamiltonian under zero-bias—and thereby enable the prediction of various electronic structure properties.

**DeepQTH** supports both DFT results and NEGF-DFT results from [SIESTA/TranSIESTA](https://siesta-project.org/siesta/CodeAccess/index.html).

# Contents
1. [Installation](#Installation)
2. [Use DeepQTH](#Use-DeepQTH)
3. [Demo](#Demo-Prediction-of-defect-free-graphene)


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

### Generate the dataset
DeepQTH is designed to predict the intermediate variable in DFT/NEGF-DFT—the Hamiltonian—and thereby enable efficient first-principles electronic structure prediction for large-scale systems. To achieve this, an appropriate small-scale structural dataset that reflects the bonding environments of the target large-scale system must be constructed, followed by DFT/NEGF-DFT calculations to obtain localized-basis Hamiltonian matrices.

We use the script `Read_sample_data_and_expand_the_dataset.ipynb` located in the `./0_generate_dataset` directory to generate the dataset. First, an appropriate small-scale structural sample that reflects the bonding environment of the target large-scale system is selected and placed in `./0_generate_dataset/sample_data`. For each sample, molecular dynamics (MD) simulations are performed (with fixed left and right electrodes for device sample, and MD applied only to the scattering region). The last 600 relatively stable MD trajectories are extracted as training samples and saved in `./0_generate_dataset/expand_dataset/raw`. Each training sample is then computed using SIESTA/TranSIESTA for DFT/NEGF-DFT calculations.

### Preprocess the dataset
`Preprocess` is an part of DeepQTH. Using the script `preprocess.ipynb` in the `./1_preprocess`, DeepQTH extracts structural parameters, local coordinate systems, and Hamiltonian blocks rotated into the local coordinate frame from each SIESTA/TranSIESTA-calculated sample, saving them in `./0_expand_dataset/processed`. Each sample and its local substructures are then converted into graph-type data and stored in `./0_expand_dataset/graph`, with the preprocessed data for each sample saved in its own subfolder.

In the `preprocess.ipynb` file, users need to define a dictionary-type configuration parameter, `config`, specifying the output path for the preprocessed files, selecting whether the data come from SIESTA or TranSIESTA calculations, and defining the local cutoff (nearsightedness) radius.

By running the `main()` function in `preprocess.ipynb`, the preprocessing procedure is completed, and all preprocessed samples are generalized into a graph-type dataset.

### Train your model
`Train` is one of the most important parts of DeepQTH, used to train the deep learning model with the prepared dataset. In the `train.ipynb` script, users need to define a dictionary-type configuration parameter, `config`, specifying the path to the preprocessed dataset, the directory for saving the trained model, as well as the training parameters, hyperparameters, and network architecture settings.

By executing the `main()` function in `train.ipynb`, the program partitions the dataset into training, validation, and test sets, and loads the neural network model for training. The training outputs and the trained model are saved in the `./2_train/trained_model` directory.

### Predict with your model
`Predict` is another key component of DeepQTH. It preprocesses the target large-scale system, loads the trained model, and predicts the Hamiltonian blocks under equilibrium. The predicted Hamiltonian blocks are then assembled into the real-space Hamiltonian and saved as `.HSX` or `.TSHS` files. In the `predict.ipynb` script, users need to define a configuration parameter, `config`, specifying the path to the target large-scale system, the path to the trained neural network model, and the local nearsightedness radius (which must be consistent for the same material type). 

By executing the `main()` function in `predict.ipynb`, the program preprocesses the target large-scale system, loads the trained model to predict local Hamiltonian blocks, and transforms and assembles them into the full real-space Hamiltonian. Finally, using the `sisl` package to read the predicted full Hamiltonian, multiple electronic structure properties can be predicted.


# Demo: Prediction of defect-free graphene

We provide a defect-free graphene [dataset](https://doi.org/10.5281/zenodo.17490788) for predicting the equilibrium-state Hamiltonian of large-scale graphene systems. Copy the raw folder from the dataset into the `./0_generate_dataset/expand_dataset` directory. 

Then, open the `preprocess.ipynb` script located in the `./1_preprocess` directory, set the corresponding paths and computational parameters, and execute the code. The preprocessed `h5` data and `graph` data will be generated and saved in the `./0_generate_dataset/expand_dataset/processed` and `./0_generate_dataset/expand_dataset/graph` directories, respectively.

After obtaining the graph data from preprocessing, open the `train.ipynb` script located in the `./2_train` directory. Set the appropriate file paths, neural network architecture parameters, and training parameters, then execute the script to begin training. During the training process, a `./2_train/trained_model` directory will be automatically created under `./2_train` to store training logs, loss values, learning rates, and the optimized model learned throughout the training process.

After training is completed, navigate to the `3_predict` directory and place the structure files to be predicted in the `3_predict/predict` folder. Perform a single-point DFT calculation on the target structure to generate the necessary input data for preprocessing. Finally, open the `predict.ipynb` script in the `3_predict` directory, set the path to the trained model and the corresponding computational parameters, and execute the code in the script. This will perform the prediction and compute electronic structure properties such as band structures and density of states.


