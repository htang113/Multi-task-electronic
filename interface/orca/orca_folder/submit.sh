#!/bin/bash
#SBATCH -o log-%j
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -p xeon-p8
module load mpi/openmpi-4.1.5

/home/gridsan/htang/orca/orca  run.inp > log
