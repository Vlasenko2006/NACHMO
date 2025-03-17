#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Mon Feb 27 13:36:38 2023.

@author: andreyvlasenko
"""
import os
import timeit

import hydra
import lightning.pytorch as pl
import torch
import numpy as np
import timeit
from copy import deepcopy

from chemical_constants_and_parameters import S_oh, S_verwer
from constansts_and_parameters_loader import constants_and_parameters_dataloader
from dataloader import prepare_dataloader
from dataset import RolloutTensorDataset
from models import ChemicalTimeStepper, MLP, RolloutModel, OptLayer
from omegaconf import DictConfig, open_dict
from torch.utils.data import random_split
from utilities import load_data, load_data_reproduce

from MSE_counter_fun import MSE_counter_fun
from Ssurrogate import Ssurrogate
import random
from rms_mechanism_selector import rms_mechanism_selector
from control_estimate import control_estimate
from chemical_mechanism_selector import chemical_mechanism_selector

option_verwer = ["no_gate", "qp_conc", "ssur", "ssur_no_gate", "gate", "err_proj", "err_proj_no_gate", "train_with_err_proj_no_gate"]

#option_oh = ["no_gate", "qp_conc", "ssur", "naive", "smat_proj"]

option_oh = ["no_gate", "qp_conc", "qp_conc_no_gate", "ssur", "gate", "smat_proj_gate", "smat_proj_no_gate"]

mechanism = "Verwer"
option = option_verwer[7]
#current_epoch = 17

#n_samples = 1
#rollout_length = 590
#tries =1

current_epoch = 2843 #17

n_samples = 100
n_steps = 160
rollout_length = 120
tries =40


cuda = "cpu"
random_starts = False


n_steps = rollout_length + tries

print("mechanism = ", mechanism)
print("option = ", option)


path_to_model, path_to_data, path_to_config = rms_mechanism_selector(option, mechanism = mechanism, current_epoch = current_epoch )

path_to_mse = "MSE_oh_long/" + option + "_val" 



@hydra.main(version_base=None, config_path=path_to_config, config_name="hparams")
def main(cfg: DictConfig):

    pl.seed_everything(2024)

#   Set computing device as cpu (no gpu so far)  

    if option == "qp_conc":
         apply_QP = True
    else:
         apply_QP = False

    if cuda == "cpu":
        cfg.net_config.device = "cpu"   #FIXME
        cfg.hardw_settings.accelerator = "cpu"

    if option == "err_proj": 
        #data_config.normalize_data = True
        cfg.stepper_config.try_Ssur_matrix = True
        cfg.stepper_config.learn_updates = False

#   Selector for the type of c hemical scheme we want to compute. It just sets the number of reacting speceis 

    Smatrix, nrates, cfg = chemical_mechanism_selector(cfg)

#   Set the amount of inputs and outputs in the NN
    in_features = len(cfg.data_config["species"])
    out_features = nrates if cfg.stepper_config["learn_rates"] else in_features  



#   Normalize Stoichiometry matrix  
    _, max_c_old = load_data(
                          cfg.data_config.data_path,
                          dtype = eval(cfg.data_config.data_type), 
                          normalize = cfg.data_config.normalize_data, 
                          species = cfg.data_config["species"],
                          ntimesteps_in_data_set = cfg.data_config.ntimesteps_in_training_set
                          )


    data, max_c = load_data(
                            path_to_data, 
                            dtype=eval(cfg.data_config.data_type),
                            normalize=cfg.data_config.normalize_data,
                            species=cfg.data_config["species"], 
                            ntimesteps_in_data_set = n_steps
                            )


    QPL = OptLayer(Smatrix.T, torch.tensor(max_c_old), device = cuda)


    NSmatrix = deepcopy(Smatrix)

    for i in range(0,len(NSmatrix)):
       NSmatrix[i, :]  = NSmatrix[i, :]/max_c_old[i]

    if option == "qp_conc":
        dc_u, dc_est = control_estimate(data, QPL, device = cuda)
        np.save("MSE/dc_u", dc_u)
        np.save("MSE/dc_est", dc_est)   

    for i in range(0,len(max_c_old)):
        data[:,:,i] = data[:,:,i] * max_c[i]/max_c_old[i]

    data = data[:,:n_steps,:]

    Ssur = Ssurrogate(data,
                      max_c_old, 
                      cut=cfg.extra_option,
                      dtype=eval(cfg.data_config.data_type), 
                      subtract_mean=cfg.data_config.subtarct_mean, 
                      Error_removal_projection= True,  #data_config.Error_removal_projection, 
                      normalize=cfg.data_config.normalize_data
                      )

    data = data[:,:n_steps,:]
    n_test = int(data.shape[0] * cfg.experiment_config["test_frac"])
    n_val = int(data.shape[0] * cfg.experiment_config["val_frac"])
    n_train = data.shape[0] - n_test - n_val
    train_data, valid_data, test_data = random_split(data, (n_train, n_val, n_test))
    train_data, valid_data, test_data = train_data[:], valid_data[:], test_data[:]
    ref = valid_data



#   Distributing some constants and Smatrix to devices for estimates on gpu (not needed so far, this is for future, when gpu is enabled)

    if "cfg.penalty_weights" not in locals(): 
        penalty_weights = [False]
    else:
        penalty_weights = cfg.penalty_weights
    if "cfg.cost_weights" not in locals():
        cost_weights = [False]
    else:
        cost_weights = cfg.cost_weights
    if "cfg.stepper_config.apply_QP_correction" not in locals(): 
        with open_dict(cfg):
            cfg.stepper_config.update({"apply_QP_correction": "False"})

    if "stoichiometry_matrix" not in cfg.stepper_config: 
        print("cfg.stepper_config.stoichiometry_matrix not in locals !")
        print(cfg.stepper_config)
    else:
        del cfg.stepper_config.stoichiometry_matrix   #FIXME obsolete settings in config.yaml must be removed

    if apply_QP == "True":
        with open_dict(cfg):
            cfg.stepper_config.update({"apply_QP_correction": "True"})  #cfg.stepper_config.apply_QP_correction = True

    parameters_loader = constants_and_parameters_dataloader(
                                                            cfg.net_config.device, 
                                                            NSmatrix,
                                                            Ssur,
                                                            penalty_weights = penalty_weights, 
                                                            cost_weights = cost_weights,
                                                            dtype = eval(cfg.data_config.data_type)
                                                            )
                                                      
#   Constructing the NN
    net = MLP(in_features, out_features, **cfg.net_config)

    stepper = ChemicalTimeStepper(
                                  net,
                                  QPL,
                                  device=cfg.net_config.device, 
                                  dtype = eval(cfg.net_config.NN_dtype), 
                                  species_list=cfg.data_config["species"],
                                  parameters_loader = parameters_loader,
                                  **cfg.stepper_config
    )

    model = RolloutModel(stepper, cfg.data_config["trajectory_length"], cfg.net_config.device)


#   Instantinate (load weights and biased) the NN
    epoch = current_epoch
    
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu') ) )

    start = timeit.default_timer()
    mse, pred, targets = MSE_counter_fun(
                          model.stepper,
                          ref[:n_samples,:,:],
                          rollout_length = rollout_length,
                          tries = tries,
                          random_starts = random_starts,
                          strategy = cfg.hardw_settings.strategy,
                          top_device = cfg.net_config.device,
                          accelerator = cfg.hardw_settings.accelerator,
                          num_of_nodes = cfg.hardw_settings.num_of_nodes,
                          devices = cfg.hardw_settings.devices
                          )

    end = timeit.default_timer()

    print("Computational time =", end - start)
    print("MSE_shape", mse.shape)
    np.save(path_to_mse, mse)
   


if __name__ == "__main__":
    main()

