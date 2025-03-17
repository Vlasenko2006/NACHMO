#!/bin/bash -f


# SLURM SUBMIT SCRIPT
#SBATCH --job-name=driver-andrey
#SBATCH --nodes=1 # unfortunately 3 is the max on strand at the moment. 
#SBATCH --ntasks-per-node=1
##SBATCH --cpu-per-tasks=1
#SBATCH --time=30:59:00
#SBATCH --account=ksm
#SBATCH --partition=pCluster
#SBATCH --error=e.out



export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/home/vlasenko/miniconda3/lib/


#number of layers 2-5
#actfun: ReLu, pReLu, sigmoid/tanh
#LR: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
#number of time steps during training: 1, 2, 5
#batch_size: 32, 64, 256
#output type: concentrations, updates, or rates
#amount of data used: all, half, 10%



Num_of_layersB=2
Num_of_layersE=5
activation="Nothing"
#yearE=$2
#prec=$3 
#flag=0
c=1
for ((nlayers=Num_of_layersB;nlayers<=Num_of_layersE;nlayers++)); do
 
    if [ $nlayers -eq 2 ]; then
       n_hidden=[19,9]
    elif [ $nlayers -eq 3 ]; then  
       n_hidden=[19,9,9]
    elif [ $nlayers -eq 4 ]; then  
       n_hidden=[19,9,9,9]
    elif [ $nlayers -eq 5 ]; then  
       n_hidden=[19,9,9,9,9]
    fi

    for ((nactivation=0;nactivation<=2;nactivation++)); do
#activation="ReLU"

    if [ $nactivation -eq 0 ]; then
        activation="ReLU"
    fi 

    if [ $nactivation -eq 1 ]; then
        activation="PReLU"
    fi

    if [ $nactivation -eq 2 ]; then
        activation="Sigmoid"
    fi
    lr=0.00001 
            for ((lrate=1;lrate<=5;lrate++)); do

               for ((traj=1;traj<=3;traj++)); do
                   if [ $traj -eq 3 ]; then
                        traj=5
                   fi


                   #srun /gpfs/home/vlasenko/miniconda3/envs/gpu-hackathon/bin/python main.py data_config.trajectory_length=$traj train_config.lr=$lr net_config.activation=$activation net_config.n_hidden=$n_hidden

                    batch_size=32
                    for ((batch_factor=1;batch_factor<=3;batch_factor++)); do
                    #sbatch ./exec.bash $traj $lr $activation $n_hidden
                        if [ $batch_factor -eq 2 ]; then
                            batch_size=64
                        elif [ $batch_factor -eq 3 ]; then
                            batch_size=256
                        fi


                    while [ $(squeue -u vlasenko | wc -l) -ge 99 ]; do
                       sleep 2
                    done
            
                    sbatch ./exec.bash $traj $lr $activation $n_hidden $batch_size $activation
                   # squeue -u vlasenko | wc -l
                    

                    echo  'layers = ' $nlayers, $activation, $lr, $c, $traj, $n_hidden, $batch_size, $activation
                    c=`expr "$c" + "1"`
                    done




               done

               if [ `expr "$lrate" % "2"` -eq 0 ]; then
                   lr=$(bc <<< "$lr * 2")
               else
                   lr=$(bc <<< "$lr * 5")
               fi

            done
    done

done


#echo $a
#echo $a
#echo $a
## | grep 'job'

#sbatch echo 'tmp = ' $SLURM_JOB_ID

