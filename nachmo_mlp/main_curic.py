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
from models import ChemicalTimeStepper, MLP, RolloutModel
from train import Lit_train
from utilities import load_data

from setting_hyperparameters_for_tb import setting_hyperparameters_for_tb
from chemical_mechanism_selector import chemical_mechanism_selector
from constansts_and_parameters_loader import constants_and_parameters_dataloader
from Ssurrogate import Ssurrogate

from net_info import net_info
from curiculim_dataloader import curiculim_dataloader






@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(2024)

    DEVICES = cfg.hardw_settings.devices
    ACCELERATOR = cfg.hardw_settings.accelerator
    STRATEGY = cfg.hardw_settings.strategy
    NUM_OF_NODES = cfg.hardw_settings.num_of_nodes

    loss_config = cfg.loss_config
    stepper_config = cfg.stepper_config
    net_config = cfg.net_config
    data_config = cfg.data_config
    loader_config = cfg.loader_config
    train_config = cfg.train_config
    experiment_config = cfg.experiment_config
    visualization_config = cfg.visualization_config
    log_name = cfg.log_name
    Exp_name = cfg.exp_name
    extra_option = cfg.extra_option
    description = cfg.description
    depth = data_config.ntimesteps_in_training_set
    data_dtype = eval(data_config.data_type)
    NN_dtype = eval(net_config.NN_dtype)

    Exp = (
        Exp_name
        + "_batch_"
        + str(loader_config.batch_size)
        + "_trajectory_length_"
        + str(data_config.trajectory_length)
        + "_opt_" + str(extra_option)
    )


    tb_logger = TensorBoardLogger(log_name, name=Exp, default_hp_metric=False)
    cfg = setting_hyperparameters_for_tb(cfg)


    Smatrix, nrates, cfg = chemical_mechanism_selector(cfg)

    in_features = len(data_config["species"])
    out_features = nrates if stepper_config["learn_rates"] else in_features  # FIXME out_features=4

    count = 0
    path = log_name + "/" + Exp + "/version_"
    while os.path.exists(path + str(count)):
        count += 1
    path = path + str(count)
    cfg.path_to_estimates = path


    data, max_c = load_data(cfg.data_config.data_path, dtype=data_dtype, normalize=True, species=data_config["species"])
    data = data[:,:84000,:] #84000
   #Ssur = Ssurrogate(data, max_c, cut=cfg.extra_option, dtype=data_dtype, subtract_mean=False)#NN_dtype)
    Ssur = Ssurrogate(data, max_c, cut=cfg.extra_option, dtype=data_dtype, subtract_mean=data_config.subtarct_mean)#NN_dtype)


    print("Ssur is prepared")



    n_test = int(data.shape[0] * experiment_config["test_frac"])
    n_val = int(data.shape[0] * experiment_config["val_frac"])
    n_train = data.shape[0] - n_test - n_val
    train_data, valid_data, test_data = random_split(data, (n_train, n_val, n_test))
    train_data, valid_data, test_data = train_data[:], valid_data[:], test_data[:] # converts lists to tensors. DO NOT DELETE!!!


    parameters_loader = constants_and_parameters_dataloader(net_config.device, Smatrix, Ssur, dtype = data_dtype)

    start = timeit.default_timer()
    net = MLP(in_features, out_features, **net_config)
    Smatrix = parameters_loader[0].dataset.tensors[0]
    stepper = ChemicalTimeStepper(
        net, device=net_config.device, dtype = NN_dtype, species_list=data_config["species"], parameters_loader = parameters_loader, **stepper_config
    )

    for i in range(0,5):


        print("LOOP =", i)
        if i < 4:
            n_epochs = 3
        else:
            n_epochs= train_config.n_epochs

        if i ==0: rollout_length = 2
        if i ==1: rollout_length = 5
        if i ==2: rollout_length = 10
        if i ==3: rollout_length = 100
        if i ==4: rollout_length = 200

        model = RolloutModel(stepper, rollout_length, net_config.device)

        checkpoint_callback = ModelCheckpoint(save_top_k=-1)

        net_info(train_data,valid_data,test_data, cfg, Exp, path)

        model = Lit_train(
        model,
        valid_data[:20,:,:],
        parameters_loader,
        net_config.device,
        max_c,
        cfg,
        **train_config,
        **loss_config,
        **visualization_config,
        )


        train_loader, test_loader, valid_loader = curiculim_dataloader(train_data,test_data,valid_data, depth,data_config,rollout_length, loader_config)
        trainer = pl.Trainer(
            strategy=STRATEGY,
            devices=DEVICES,
            accelerator=ACCELERATOR,
            num_nodes=NUM_OF_NODES,
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
