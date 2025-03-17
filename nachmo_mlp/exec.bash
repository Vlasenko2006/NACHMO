#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=tst
#SBATCH --nodes=1 # unfortunately 3 is the max on strand at the moment. 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:59:00
#SBATCH --account=ksm
#SBATCH --partition=pCluster #pGPU
#SBATCH --error=tst.out

##SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
##SBATCH --mem=0                    # Request all memory available on all nodes


#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/home/vlasenko/miniconda3/lib/

#srun /gpfs/home/vlasenko/miniconda3/envs/gpu-hackathon/bin/python main.py data_config.trajectory_length=1


#/gpfs/home/vlasenko/miniconda3/envs/gpu-hackathon/bin/python main.py data_config.trajectory_length=$1 train_config.lr=$2 net_config.activation=$3 net_config.n_hidden=$4 loader_config.batch_size=$5 log_name=$6 exp_name=$7


srun /gpfs/home/vlasenko/miniconda3/envs/gpuenv/bin/python main.py data_config.trajectory_length=$1 train_config.lr=$2 net_config.activation=$3 net_config.n_hidden=$4 loader_config.batch_size=$5 log_name=$6 exp_name=$7 extra_option=$8 description=$9

