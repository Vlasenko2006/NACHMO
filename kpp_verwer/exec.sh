#!/bin/bash  
#SBATCH --account=gg0302
#SBATCH -N 4
#SBATCH -n 256
#SBATCH -p compute
#SBATCH -t 04:00:00           
#SBATCH -J  kpp               # Specify job name
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1
#SBATCH -o stdout.txt

#-A gg0302 -N 1 -n 256 -t 04:00:00 -p compute

echo "Start kpp script execution at $(date)"

mpirun --mca opal_common_ucx_opal_mem_hooks 1 ./verwer.exe
