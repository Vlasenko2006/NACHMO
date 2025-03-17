#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 20:53:28 2025

@author: andrey
"""


from torch.utils.data import random_split

def get_validation_data(data,cfg, split_the_data = True):
    if split_the_data:
        n_test = int(data.shape[0] * cfg.experiment_config["test_frac"])
        n_val = int(data.shape[0] * cfg.experiment_config["val_frac"])
        n_train = data.shape[0] - n_test - n_val
        train_data, valid_data, test_data = random_split(data, (n_train, n_val, n_test))
        train_data, valid_data, test_data = train_data[:], valid_data[:], test_data[:]
        valid_data[: ,:,:]
    else:
        valid_data[: ,:,:] = data
    print("ref.shape = ",valid_data.shape)
    return valid_data

