
import os
import sys
import timeit

import numpy as np
import torch
from dataset import RolloutTensorDataset, RolloutDataset
from torch.utils.data import random_split

#from chemical_constants_and_parameters import Smatrix
from dataloader import prepare_dataloader

from Ssurrogate import Ssurrogate

from net_info import net_info


def curiculim_dataloader(train_data,test_data,valid_data, depth,data_config,rollout_length, skip, loader_config):


    train_data = train_data[:,:depth:skip,:] 
    test_data = test_data[:,:depth:skip,:]
    valid_data = valid_data[:,:depth:skip,:]

    if not os.path.isdir('tmpfiles'): os.mkdir('tmpfiles')
   

    train_path = os.getcwd()+"/tmpfiles/train_data"+str(timeit.default_timer())+".npy"
    test_path = os.getcwd()+"/tmpfiles/test_data"+str(timeit.default_timer())+".npy"
    valid_path = os.getcwd()+"/tmpfiles/valid_data"+str(timeit.default_timer())+".npy"
    np.save(train_path,train_data)
    np.save(test_path,test_data)
    np.save(valid_path,valid_data)

    print("data is saved")
    print("rollout_length = ", rollout_length)
    print("train_data.shape = ", train_data.shape)


    if data_config.apply_disk_mapping == True:

        train_dataset = RolloutDataset(train_path,rollout_length=rollout_length, noise = data_config.add_noise)
        print("training data set is prepared")
        test_dataset = RolloutDataset(test_path,rollout_length=rollout_length,noise = data_config.add_noise)
        print("test data set is prepared")
        valid_dataset = RolloutDataset(valid_path,rollout_length=rollout_length,noise = data_config.add_noise)
        print("valid data set is prepared, rollout_length = ", rollout_length,noise = data_config.add_noise )

    else:
 
        train_dataset = RolloutTensorDataset(train_data, trajectory_length=rollout_length,noise = data_config.add_noise)
        test_dataset = RolloutTensorDataset(test_data, trajectory_length=rollout_length,noise = data_config.add_noise)
        valid_dataset = RolloutTensorDataset(valid_data, trajectory_length=rollout_length,noise = data_config.add_noise)


   
   
    print("train_dataset.shape[0] = ", (train_dataset[0][0].shape) )
    print("train_dataset.shape[1] = ", (train_dataset[0][1].shape) )

    print("test_dataset.shape[0] = ", (test_dataset[0][0].shape) )
    print("test_dataset.shape[1] = ", (test_dataset[0][1].shape) )

    print("validation_dataset.shape[0] = ", (valid_dataset[0][0].shape) )
    print("validation_dataset.shape[1] = ", (valid_dataset[0][1].shape) )

#    print("test_dataset.shape = ", test_dataset[0].shape )
#    print("valid_dataset.shape = ", valid_dataset[0].shape )

    train_loader = prepare_dataloader(train_dataset, **loader_config)
    test_loader = prepare_dataloader(test_dataset, **loader_config)
    valid_loader = prepare_dataloader(valid_dataset, **loader_config)


    print("Data loaders have been initiated")

    return train_loader, test_loader, valid_loader

