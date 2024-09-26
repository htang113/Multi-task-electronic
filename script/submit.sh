#!/bin/bash
#SBATCH -A m3706_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 5
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1

module load pytorch/2.1.0-cu12

srun --nodes=5 bash torchrun_script.sh
