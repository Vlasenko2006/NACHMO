#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:36:38 2023.

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

##############################################
# Set your path here
##############################################

Approach = "Naive"

if Approach == "Naive":
    path_to_model = "/gpfs/work/NACHMO_prog/Naive/"
    name_for_estimates = "Y_naive"
elif Approach == "Gated":
    path_to_model = "/gpfs/work/NACHMO_prog/Gated/"
    name_for_estimates = "Y_gated"
else:
    raise ValueError("Unknown Approach specified. Use 'Naive' or 'Gated'.")

current_epoch = 2843  # Set epoch
ntimesteps = 220      # Set the length of rollout
n_samples = 10

##############################################
# End of path setup
##############################################

PATH = os.path.join(path_to_model, f"model{current_epoch}.pt")

@hydra.main(version_base=None, config_path=path_to_model, config_name="hparams")
def main(cfg: DictConfig):
    pl.seed_everything(2024)

    # Load configs
    stepper_config = cfg.stepper_config
    net_config = cfg.net_config
    data_config = cfg.data_config
    loader_config = cfg.loader_config
    data_dtype = eval(data_config.data_type)
    NN_dtype = eval(net_config.NN_dtype)

    # Set computing device as CPU (no GPU so far)
    net_config.device = "cpu"  # FIXME: Use GPU if available

    # Select chemical scheme and set up stoichiometry matrix and rates
    if data_config.scheme == "Verwer":
        data_config.species = data_config.species_verwer
        cfg.data_config.species = data_config.species_verwer
        Smatrix = S_verwer
        nrates = 20
    elif data_config.scheme == "OH":
        data_config.species = data_config.species_oh
        cfg.data_config.species = data_config.species_oh
        Smatrix = S_oh
        nrates = 4
    else:
        raise ValueError(f"Unknown chemical scheme: {data_config.scheme}")

    # Set number of inputs/outputs in the NN
    in_features = len(data_config["species"])
    out_features = nrates if stepper_config["learn_rates"] else in_features

    # Load the data. ref is the reference data, max_c are the normalization coefficients
    data, max_c = load_data(
        cfg.data_config.data_path,
        dtype=data_dtype,
        normalize=data_config.normalize_data,
        species=data_config["species"]
    )

    Ssur = Ssurrogate(
        data,
        max_c,
        cut=cfg.extra_option,
        dtype=data_dtype,
        subtract_mean=data_config.subtarct_mean,
        unit_S=data_config.unit_S,
    )

    # Normalize stoichiometry matrix
    for i in range(len(Smatrix)):
        Smatrix[i, :] = Smatrix[i, :] / max_c[i]

    # Prepare constants and Smatrix for estimates (future GPU use)
    parameters_loader = constants_and_parameters_dataloader(
        net_config.device,
        Smatrix,
        Ssur,
        weights=np.ones(in_features),
        dtype=data_dtype,
    )

    # Construct the neural network
    net = MLP(in_features, out_features, **net_config)
    stepper = ChemicalTimeStepper(
        net,
        device=net_config.device,
        dtype=NN_dtype,
        species_list=data_config["species"],
        parameters_loader=parameters_loader,
        **stepper_config,
    )

    model = RolloutModel(stepper, data_config["trajectory_length"], net_config.device)

    # Instantiate (load weights and biases) the NN
    model.load_state_dict(torch.load(PATH, map_location=torch.device("cpu")))

    # Compute your concentrations with NN
    y = no_grad_stepper(
        model.stepper, data, net_config.device, n_timepts=ntimesteps
    )  # n_timepts is the number of timesteps to compute

    y = np.asarray(y)
    data = np.asarray(data)

    print("data.shape", data.shape, "y.shape", y.shape)

    # Save results
    np.save(name_for_estimates, y)
    np.save("Reference", data)

if __name__ == "__main__":
    main()
