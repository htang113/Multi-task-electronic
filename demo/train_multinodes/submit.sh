#!/bin/bash
#SBATCH -A m3706_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 4
#SBATCH --gpus=16

module load pytorch/2.1.0-cu12

srun --nodes=4 bash torchrun_script.sh