#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Mon Feb 27 13:36:38 2023.

@author: andreyvlasenko
"""
import os
import sys
import timeit

import numpy as np

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch.utils.data import random_split

#from chemical_constants_and_parameters import Smatrix
from dataloader import prepare_dataloader
from dataset import RolloutTensorDataset, RolloutDataset
from models import ChemicalTimeStepper, MLP, RolloutModel, OptLayer
from train import Lit_train
from utilities import load_data

from setting_hyperparameters_for_tb import setting_hyperparameters_for_tb
from chemical_mechanism_selector import chemical_mechanism_selector
from constansts_and_parameters_loader import constants_and_parameters_dataloader
from Ssurrogate import Ssurrogate

from net_info import net_info
from curiculim_dataloader import curiculim_dataloader
from set_experiments_name import set_experiments_name 






@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(2024)


    curriculum_scheme = cfg.curriculum_scheme + [cfg.data_config.trajectory_length]
    tb_logger = TensorBoardLogger(cfg.log_name, name=set_experiments_name(cfg), default_hp_metric=False)
    cfg = setting_hyperparameters_for_tb(cfg)


    Smatrix, nrates, cfg = chemical_mechanism_selector(cfg)

    in_features = len(cfg.data_config["species"])
    out_features = nrates if cfg.stepper_config["learn_rates"] else in_features  # FIXME out_features=4

    count = 0
    path = cfg.log_name + "/" + set_experiments_name(cfg) + "/version_"
    while os.path.exists(path + str(count)):
        count += 1
    path = path + str(count)
    cfg.path_to_estimates = path

    data, max_c = load_data(
                            cfg.data_config.data_path,
                            dtype=eval(cfg.data_config.data_type),
                            normalize=cfg.data_config.normalize_data,
                            species=cfg.data_config["species"],
                            ntimesteps_in_data_set = cfg.data_config.ntimesteps_in_training_set
                           ) 

    QPL = OptLayer(
                   Smatrix.T, 
                   torch.tensor(max_c), 
                   device = cfg.net_config.device
                   )  

    print("cut=cfg.extra_option = " , cfg.extra_option)

    for i in range(0,len(Smatrix)):
       Smatrix[i, :]  = Smatrix[i, :]/max_c[i]


    Ssur = Ssurrogate(data,
                      max_c, 
                      cut=cfg.extra_option,
                      dtype=eval(cfg.data_config.data_type), 
                      subtract_mean=cfg.data_config.subtarct_mean, 
                      Error_removal_projection=cfg.data_config.Error_removal_projection, 
                      normalize=cfg.data_config.normalize_data
                      )

    n_test = int(data.shape[0] * cfg.experiment_config["test_frac"])
    n_val = int(data.shape[0] * cfg.experiment_config["val_frac"])
    n_train = data.shape[0] - n_test - n_val
    train_data, valid_data, test_data = random_split(data, (n_train, n_val, n_test))
    train_data, valid_data, test_data = train_data[:], valid_data[:], test_data[:] # converts lists to tensors. DO NOT DELETE!!!


    parameters_loader = constants_and_parameters_dataloader(
                                                            cfg.net_config.device, 
                                                            Smatrix,
                                                            Ssur,
                                                            penalty_weights = cfg.penalty_weights,
                                                            cost_weights = cfg.cost_weights, 
                                                            dtype = eval(cfg.data_config.data_type)
                                                            )

    start = timeit.default_timer()
    net = MLP(in_features, out_features, **cfg.net_config)
    Smatrix = parameters_loader[0].dataset.tensors[0]

    stepper = ChemicalTimeStepper(
                                  net,
                                  QPL,
                                  device=cfg.net_config.device,
                                  dtype = eval(cfg.net_config.NN_dtype),
                                  species_list=cfg.data_config["species"],
                                  parameters_loader = parameters_loader, 
                                  **cfg.stepper_config
                                  )


    for rollout_length in curriculum_scheme:

        if rollout_length < curriculum_scheme[-1]:
            n_epochs = 1
        else:
            n_epochs= cfg.train_config.n_epochs

        print("Rollout_length = ", rollout_length , "n_epochs = ", n_epochs)


        model = RolloutModel(stepper, rollout_length, cfg.net_config.device)

        checkpoint_callback = ModelCheckpoint(save_top_k=-1)

        net_info(train_data,valid_data,test_data, cfg, set_experiments_name(cfg), path)

        model = Lit_train(
        model,
        valid_data[:,:,:],
        parameters_loader,
        cfg.net_config.device,
        max_c,
        cfg,
        **cfg.train_config,
        **cfg.loss_config,
        **cfg.visualization_config,
        )


        train_loader, test_loader, valid_loader = curiculim_dataloader(
            train_data,test_data,
            valid_data, 
            cfg.data_config.ntimesteps_in_training_set,
            cfg.data_config,
            rollout_length,
            cfg.data_config.skip,
            cfg.loader_config,
            )



        trainer = pl.Trainer(
            strategy = cfg.hardw_settings.strategy,
            devices = cfg.hardw_settings.devices,
            accelerator = cfg.hardw_settings.accelerator,
            num_nodes = cfg.hardw_settings.num_of_nodes,
            max_epochs=n_epochs,
            logger=tb_logger,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(model, train_loader, valid_loader)
        stepper = model.stepper

    print("Training is finished")

    # tb_logger.log_hyperparams(cfg)
    stop = timeit.default_timer()

    print("Time: ", stop - start)

    os.remove(os.getcwd()+"/tmpfiles/*")


if __name__ == "__main__":
    main()
