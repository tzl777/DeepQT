# DeepQT: Deep Learning Accelerated Quantum Transport Framework

## Overview
We present &zwnj;**DeepQT**&zwnj;, a novel AI-driven framework that revolutionizes quantum transport simulations by replacing the traditional self-consistent procedure with deep learning models. This breakthrough significantly reduces computational complexity while maintaining high accuracy.

## Model Architecture
The DeepQT framework consists of two synergistic sub-models:

1. &zwnj;**DeepQTH**&zwnj; (Quantum Transport Hamiltonian)  
   - Predict the equilibrium Hamiltonian matrix (H<sub>eq</sub>) in the absence of bias  
   - Used for electronic structure prediction and providing intermediate quantities for quantum transport calculations

2. &zwnj;**DeepQTV**&zwnj; (Quantum Transport Total Potential Difference)  
   - Predict the non-equilibrium total potential difference (TPD) under various bias conditions, and compute the Hamiltonian correction (ΔH<sub>neq</sub>) by integrating over the basis functions for each bias condition.
   - After obtaining the Hamiltonians under different bias voltages, they are used to predict quantum transport properties, such as transmission spectra and current-voltage characteristic curves, etc.

## Technical Advantages
The combined &zwnj;**DeepQT**&zwnj; + &zwnj;**DeepQTV**&zwnj; system:
- Accurately predicts device Hamiltonians across diverse bias conditions
- Enables efficient computation of multiple transport properties
- Seamlessly integrates with quantum transport solvers (e.g., TBtrans)
- Reduces computational overhead by orders of magnitude

## Open Source Policy
Due to commercial considerations, we are currently releasing the &zwnj;**DeepQTH**&zwnj; code as open source. This initiative aims to:
- Accelerate research in quantum transport simulations
- Facilitate community-driven improvements
- Demonstrate the framework's capabilities

The DeepQTV model will remain proprietary during the initial commercialization phase.

To balance academic openness with ongoing commercial collaboration, we are currently releasing the core implementation of the DeepQTH model as open source. This release aims to:
- Accelerate research and innovation in electronic-structure modeling;
- Demonstrate the overall functionality and framework of the model;
- Prevent disclosure of sensitive content during the collaboration phase.

After consultation with our commercial partners, the DeepQTV model will be made open source within the next month.


