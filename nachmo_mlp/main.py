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
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch.utils.data import random_split

from models import ChemicalTimeStepper, MLP, RolloutModel, OptLayer
from train import Lit_train
from utilities import load_data

from setting_hyperparameters_for_tb import setting_hyperparameters_for_tb
from chemical_mechanism_selector import chemical_mechanism_selector
from constansts_and_parameters_loader import constants_and_parameters_dataloader
from Ssurrogate import Ssurrogate

from curiculim_dataloader import curiculim_dataloader
from set_experiments_name import set_experiments_name 

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):  # cfg read from config.yaml, it holds neural network settings, i/o, etc.  
    
    # We initialize the same random initialization in ALL experiments
    # to ensure equal starting conditions 
    pl.seed_everything(2024)
    
    
    
    # initiate tensorboard logger
    tb_logger = TensorBoardLogger(cfg.log_name, name=set_experiments_name(cfg), default_hp_metric=False)
    cfg = setting_hyperparameters_for_tb(cfg)
    


    #Load some network and training settings
    # curriculum - curriculum training schedule
    # nrates - number of reaction rates
    # Smatrix - stoichiometry matrix
    curriculum_scheme = cfg.curriculum_scheme + [cfg.data_config.trajectory_length]
    Smatrix, nrates, cfg = chemical_mechanism_selector(cfg)
    
    # Neural network input-output
    in_features = len(cfg.data_config["species"])
    out_features = nrates if cfg.stepper_config["learn_rates"] else in_features
    
    # Load the dataset and normalize it
    # data - normalized concentrations, 
    # max_c - normalization constants (maximal concentrations)
    data, max_c = load_data(
        cfg.data_config.data_path,
        dtype=eval(cfg.data_config.data_type),
        normalize=cfg.data_config.normalize_data,
        species=cfg.data_config["species"],
        ntimesteps_in_data_set=cfg.data_config.ntimesteps_in_training_set
    ) 
    
    # Initialize quadratic programming layer
    QPL = OptLayer(Smatrix.T,
                   torch.tensor(max_c),
                   device=cfg.net_config.device
                   )


    for i in range(len(Smatrix)):
        Smatrix[i, :] = Smatrix[i, :] / max_c[i]



    # Split into training/ test and validation sets
    n_test = int(data.shape[0] * cfg.experiment_config["test_frac"])
    n_val = int(data.shape[0] * cfg.experiment_config["val_frac"])
    n_train = data.shape[0] - n_test - n_val
    train_data, valid_data, test_data = random_split(data, (n_train, n_val, n_test))
    train_data, valid_data, test_data = train_data[:], valid_data[:], test_data[:]

    # Compute error-projection matricies from the dataset
    Ssur = Ssurrogate(
        train_data,
        max_c, 
        cut=cfg.extra_option,
        dtype=eval(cfg.data_config.data_type), 
        subtract_mean=cfg.data_config.subtarct_mean, 
        Error_removal_projection=cfg.data_config.Error_removal_projection, 
        normalize=cfg.data_config.normalize_data
    )


    parameters_loader = constants_and_parameters_dataloader(
        cfg.net_config.device, 
        Smatrix,
        Ssur,
        penalty_weights=cfg.penalty_weights,
        cost_weights=cfg.cost_weights, 
        dtype=eval(cfg.data_config.data_type)
    )
    
    # Instantinate our Neural network
    net = MLP(in_features, out_features, **cfg.net_config)

    # Add error-removing projection/stoichiometry matrix to
    # the last layer in MLP (if specified) 
    stepper = ChemicalTimeStepper(
        net,
        QPL,
        device=cfg.net_config.device,
        dtype=eval(cfg.net_config.NN_dtype),
        species_list=cfg.data_config["species"],
        parameters_loader=parameters_loader, 
        **cfg.stepper_config
    )
    
    # Start clocking to see how robust is the NN 
    start = timeit.default_timer()
    
    # Start curriculum training
    for rollout_length in curriculum_scheme:
        if rollout_length < curriculum_scheme[-1]:
            n_epochs = 1
        else:
            n_epochs = cfg.train_config.n_epochs

        # Build a multi-step MLP, i.e, the stepper computing concentrations
        # "rollout_length timesteps ahead
        model = RolloutModel(stepper, rollout_length, cfg.net_config.device)
        checkpoint_callback = ModelCheckpoint(save_top_k=-1)
        

        model = Lit_train(
            model,
            valid_data,
            parameters_loader,
            cfg.net_config.device,
            max_c,
            cfg,
            **cfg.train_config,
            **cfg.loss_config,
            **cfg.visualization_config,
        )
        
        
        # Update the train/ valid datasets according to the curriculum
        # Set the time length in each dataset equal to the quiriculum's
        # rollout length
        train_loader, test_loader, valid_loader = curiculim_dataloader(
            train_data, test_data, valid_data, 
            cfg.data_config.ntimesteps_in_training_set,
            cfg.data_config,
            rollout_length,
            cfg.data_config.skip,
            cfg.loader_config,
        )

        trainer = pl.Trainer(
            strategy=cfg.hardw_settings.strategy,
            devices=cfg.hardw_settings.devices,
            accelerator=cfg.hardw_settings.accelerator,
            num_nodes=cfg.hardw_settings.num_of_nodes,
            max_epochs=n_epochs,
            logger=tb_logger,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(model, train_loader, valid_loader)
        stepper = model.stepper

    print("Training is finished")
    stop = timeit.default_timer()
    print("Time:", stop - start)

    os.remove(os.getcwd() + "/tmpfiles/*")

if __name__ == "__main__":
    main()