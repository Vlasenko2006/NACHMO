#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 19:31:12 2025

@author: andrey

double call load_data enables  training neural network on one dataset and test on any
other dtat set retrieveng normalization constants max_c for both of them. Note we 
split these constants on purpose, to see how robust the NN is to varions normalization constants 

"""
from utilities import load_data
from copy import deepcopy
from chemical_mechanism_selector import chemical_mechanism_selector
from Ssurrogate import Ssurrogate
import lightning.pytorch as pl
import torch

def Matrices_for_NN(cfg, new_path_to_data, n_steps, Smatrix,random_seeder =2024):
    
    '''
    n_steps limit the time length of the validation period. 
            In case of cumulative rmse  n_steps < rollout_length + tries
    '''
    pl.seed_everything(random_seeder) 
    
    
    _, max_c_old = load_data(
    cfg.data_config.data_path,
    dtype=eval(cfg.data_config.data_type),
    normalize=cfg.data_config.normalize_data,
    species=cfg.data_config["species"],
    ntimesteps_in_data_set=cfg.data_config.ntimesteps_in_training_set
    )

    data, max_c = load_data(
        new_path_to_data,
        dtype=eval(cfg.data_config.data_type),
        normalize=cfg.data_config.normalize_data,
        species=cfg.data_config["species"],
        ntimesteps_in_data_set=n_steps
    )
    
    for i in range(len(max_c_old)):
        data[:, :, i] = data[:, :, i] * max_c[i] / max_c_old[i]
    
    
    NSmatrix = deepcopy(Smatrix)
    for i in range(len(NSmatrix)):
        NSmatrix[i, :] = NSmatrix[i, :] / max_c_old[i]
        
    data = data[:, :n_steps, :]

    Ssur = Ssurrogate(
        data,
        max_c_old,
        cut=cfg.extra_option,
        dtype=eval(cfg.data_config.data_type),
        subtract_mean=cfg.data_config.subtarct_mean,
        Error_removal_projection=cfg.data_config.Error_removal_projection,
        normalize=cfg.data_config.normalize_data
    )
        

    return data, max_c, max_c_old, NSmatrix, Ssur, data 
