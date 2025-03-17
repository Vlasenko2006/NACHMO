#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 21:08:53 2025

@author: andrey
"""

import timeit
import os
import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from constansts_and_parameters_loader import constants_and_parameters_dataloader
from models import ChemicalTimeStepper, MLP, RolloutModel, OptLayer
from MSE_counter_slice_fun import MSE_counter_slice_fun
from chemical_mechanism_selector import chemical_mechanism_selector
from get_validation_data import get_validation_data
from Matrices_for_NN import Matrices_for_NN


def run_validation_with_rmse(cfg: DictConfig, 
                   option: str,
                   path_to_model: str,
                   path_to_data: str,
                   step_slices: list,
                   rollout_length: int,
                   device: str,
                   n_steps: int,
                   reg_fac: float,
                   random_starts: bool
                   ):
    pl.seed_everything(2024)  # This is fixed to assure that train valid and test data sets are always the same
    if "cfg.data_config.Error_removal_projection" not in locals():
        with open_dict(cfg):
            cfg.data_config.update({"Error_removal_projection": "False"})

    apply_QP = option == "qp_conc"

    if device == "cpu":
        cfg.net_config.device = "cpu"
        cfg.hardw_settings.accelerator = "cpu"

    if option in ["err_proj"]:
        cfg.stepper_config.try_Ssur_matrix = True
        cfg.stepper_config.learn_updates = False
        cfg.data_config.Error_removal_projection = True

    Smatrix, nrates, cfg = chemical_mechanism_selector(cfg)

    in_features = len(cfg.data_config["species"])
    out_features = nrates if cfg.stepper_config["learn_rates"] else in_features

    data, max_c, max_c_old, NSmatrix, Ssur, data = Matrices_for_NN(cfg, path_to_data, n_steps, Smatrix,random_seeder =2024)
    QPL = OptLayer(Smatrix.T, batch=1, max_c=torch.tensor(max_c_old), device=device, reg_fac=reg_fac)

    ref = get_validation_data(data, cfg, split_the_data=True)

    if "cfg.stepper_config.apply_QP_correction" not in locals():
        with open_dict(cfg):
            cfg.stepper_config.update({"apply_QP_correction": str(apply_QP)})

    print("apply_QP = ", apply_QP)

    if "stoichiometry_matrix" in cfg.stepper_config:
        del cfg.stepper_config.stoichiometry_matrix

    parameters_loader = constants_and_parameters_dataloader(
        cfg.net_config.device,
        NSmatrix,
        Ssur,
        dtype=eval(cfg.data_config.data_type)
    )

    net = MLP(in_features, out_features, **cfg.net_config)

    stepper = ChemicalTimeStepper(
        net,
        QPL,
        device=cfg.net_config.device,
        dtype=eval(cfg.net_config.NN_dtype),
        species_list=cfg.data_config["species"],
        parameters_loader=parameters_loader,
        **cfg.stepper_config
    )

    model = RolloutModel(stepper, cfg.data_config["trajectory_length"], cfg.net_config.device)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))

    start = timeit.default_timer()
    rmse,rmse_min,rmse_max,rmse_std, len_rmse = MSE_counter_slice_fun(
        model.stepper,
        step_slices,
        ref[: , :, :],
        rollout_length=rollout_length,
        random_starts=random_starts,
        strategy=cfg.hardw_settings.strategy,
        top_device=cfg.net_config.device,
        accelerator=cfg.hardw_settings.accelerator,
        num_of_nodes=cfg.hardw_settings.num_of_nodes,
        devices=cfg.hardw_settings.devices
    )

    end = timeit.default_timer()


    print("Computational time =", end - start)

    return rmse
    

