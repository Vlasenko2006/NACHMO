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

from chemical_constants_and_parameters import S_oh, S_verwer
from constansts_and_parameters_loader import constants_and_parameters_dataloader
from dataloader import prepare_dataloader
from dataset import RolloutTensorDataset
from models import ChemicalTimeStepper, MLP, RolloutModel, no_grad_stepper
from omegaconf import DictConfig
from torch.utils.data import random_split
from utilities import load_data, load_data_reproduce

from Ssurrogate import Ssurrogate






################################3###### Pankaj, set the pass here ################################3######

#path_to_data = "/gpfs/work/vlasenko/NACHMO_data/concentrations_verwer_long2/"

Naive = 0

if Naive == 0:
    path_to_model = "/gpfs/work/vlasenko/NACHMO_prog/Smatrix5/nachmo_coding/nachmo_mlp/Verwer_2h_nonshifted_proj/Reference_proj_CurCurr__torch.float32_Ssur__mean__tr_length_100_opt_3/version_1/"
#    path_to_model = "/gpfs/work/vlasenko/NACHMO_prog/Smatrix5/nachmo_coding/nachmo_mlp/Verwer_2h_shifted_proj/Reference_proj_CurCurr__torch.float32_Ssur__mean__tr_length_100_opt_3/version_0/"
    name_for_estimates = "Y_proj"
elif Naive == 1:
    path_to_model = "/gpfs/work/vlasenko/NACHMO_prog/Smatrix5/nachmo_coding/nachmo_mlp/Verwer_2h_nonshifted_proj/Reference_naive_CurCurr__torch.float32___mean__tr_length_100_opt_3/version_0/"
#    path_to_model = "/gpfs/work/vlasenko/NACHMO_prog/Smatrix5/nachmo_coding/nachmo_mlp/Verwer_2h_shifted_proj/Reference_naive_CurCurr__torch.float32___mean__tr_length_100_opt_3/version_0/"
    name_for_estimates = "Y_naive"
else:
    path_to_model = "/gpfs/work/vlasenko/NACHMO_prog/Smatrix5/nachmo_coding/nachmo_mlp/Verwer_2h_nonshifted_proj/Reference_naive_CurCurr__torch.float32___mean__tr_length_100_opt_3/version_0/"
#    path_to_model = "/gpfs/work/vlasenko/NACHMO_prog/Smatrix5/nachmo_coding/nachmo_mlp/Verwer_2h_shifted_proj/Reference_naive_CurCurr__torch.float32___mean__tr_length_100_opt_3/version_0/"
    name_for_estimates = "Y_naive2"


current_epoch = 2843  # set epoch
ntimesteps = 220      # set the lenght of rollout
n_samples = 10

################################3###### End of Pankaj, set the pass here ################################3######
PATH = path_to_model + "model" + str(current_epoch) + ".pt"




@hydra.main(version_base=None, config_path=path_to_model, config_name="hparams" )
def main(cfg: DictConfig):

    pl.seed_everything(2024)

#   Loading configs   

    stepper_config = cfg.stepper_config
    net_config = cfg.net_config
    data_config = cfg.data_config
    loader_config = cfg.loader_config
    data_dtype = eval(data_config.data_type)
    NN_dtype = eval(net_config.NN_dtype)


    if Naive ==2: 
        #data_config.normalize_data = True
        stepper_config.try_Ssur_matrix = True
        stepper_config.learn_updates = False

 


#   Set computing device as cpu (no gpu so far)  

    net_config.device = 'cpu'   #FIXME

#   Selector for the type of c hemical scheme we want to compute. It just sets the number of reacting speceis 

    if data_config.scheme == "Verwer":
        data_config.species = data_config.species_verwer
        cfg.data_config.species = data_config.species_verwer
        Smatrix = S_verwer
        nrates=20
         
    if data_config.scheme == "OH":
        data_config.species = data_config.species_oh
        cfg.data_config.species = data_config.species_oh
        Smatrix = S_oh
        nrates = 4

#   Set the amount of inputs and outputs in the NN
    in_features = len(data_config["species"])
    out_features = nrates if stepper_config["learn_rates"] else in_features  


#   Load the data, ref is the reference data, max_c are the normalization coefficients 

    data, max_c = load_data(cfg.data_config.data_path, dtype=data_dtype, normalize=data_config.normalize_data, species=data_config["species"])
#    data, max_c = load_data(path_to_data, dtype=data_dtype, normalize=data_config.normalize_data, species=data_config["species"])
    np.save('max_c', max_c)

    data = data[:,:ntimesteps,:]


    print('data.shape = ', data.shape)
    Ssur = Ssurrogate(data, max_c, cut=cfg.extra_option, dtype=data_dtype, subtract_mean=data_config.subtarct_mean, unit_S=data_config.unit_S)

    data = data[:n_samples,:,:]
    print('data.shape = ', data.shape)
#   Normalize Stoichiometry matrix 
    for i in range(len(Smatrix)):
        Smatrix[i, :] = Smatrix[i, :] / max_c[i] 


#   Distributing some constants and Smatrix to devices for estimates on gpu (not needed so far, this is for future, when gpu is enabled)
    parameters_loader = constants_and_parameters_dataloader(net_config.device, Smatrix, Ssur, weights = np.ones(in_features), dtype = data_dtype)

#   Constructing the NN
    net = MLP(in_features, out_features, **net_config)
    stepper = ChemicalTimeStepper(
        net, device=net_config.device, dtype = NN_dtype, species_list=data_config["species"], parameters_loader = parameters_loader, **stepper_config
    )

    model = RolloutModel(stepper, data_config["trajectory_length"], net_config.device)
#    for var_name in stepper.state_dict():
#        print(var_name, "\t", stepper.state_dict()[var_name])


#   Instantinate (load weights and biased) the NN
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu') ) )


    # Compute your concentrations with a NN
    y = no_grad_stepper(model.stepper, data, net_config.device, n_timepts = ntimesteps) # n_timepts is the ammount of timesteps to compute

    y = np.asarray(y)
    data = np.asarray(data)

    print("data.shape", data.shape, "y.shape", y.shape)  

    for i in range(0,len(data_config.species)):
        print("max. data[:,:,i]", np.max(data[:,:,i]), "y.max", np.max(y[:,:,i]) )

#   Save your results
    np.save(name_for_estimates, y)
    np.save("Y_ref" , data)



if __name__ == "__main__":
    main()
