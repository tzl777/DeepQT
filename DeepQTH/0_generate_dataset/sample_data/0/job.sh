#!/bin/bash
#SBATCH -p cp6
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --error=%J.err

module purge
module add siesta/4.1.5-icc19-ompi-x

siesta <input.fdf > out
