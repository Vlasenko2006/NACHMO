#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=tst
#SBATCH --nodes=1 # unfortunately 3 is the max on strand at the moment. 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=68:59:00
#SBATCH --account=ksm
#SBATCH --partition=pGPU #pCluster
#SBATCH --error=tst.out

#SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                    # Request all memory available on all nodes


srun /gpfs/home/vlasenko/miniconda3/envs/gpuenv/bin/python rms_counter.py data_config.trajectory_length=$1 
