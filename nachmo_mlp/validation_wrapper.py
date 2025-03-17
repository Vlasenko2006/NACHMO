#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Mon Feb 27 13:36:38 2023.

@author: andreyvlasenko
"""


import os
import numpy as np
from hydra import initialize, compose
from run_validation_with_rmse import run_validation_with_rmse
from parse_experiments import parse_experiments



# This part specifies IO of the saved models and output results
# Experimental setups and corresponding explanation find in 
# conf/list_of_experiments.yaml
 
reg_frac = 1e-6   # must be moved to list_of_experiments.yaml
compute_sliced_rmse = True
device = "cpu"
path_to_models = 'saved_models/'
list_of_experiments = "conf/list_of_experiments.yaml"
output_top_dir = "SMSE3/"

# Get experimental setup from a .yaml file
mechanisms, settings, options = parse_experiments(list_of_experiments)

#iterate over all experimental options and compute rmse and predicted concentrations (in future) 
for step_length in ['singlestep', 'multistep']:
    for mechanism in mechanisms:
        mechanism_settings = [item for item in settings if item['mechanism'] == mechanism]
        for setting in mechanism_settings:
            current_epoch = setting["current_epoch"]
            n_steps = setting["n_steps"]
            slices = setting["slices"]
            path_to_data = setting["path_to_data"]
            rollout_length = setting["rollout_length"]
            tries = setting["tries"]
            random_starts = setting["random_starts"]

            for option in options[mechanism]:
                if not os.path.exists(output_top_dir):
                    os.makedirs(output_top_dir)

                # hparams holds config of each experiment
                config_name = 'hparams'
                model_name  = 'model'+ str(current_epoch) + ".pt" 
                exp_group = f"{mechanism}_{step_length}/"
                path_to_config = os.path.join(path_to_models, exp_group, option)
                path_to_model = os.path.join(path_to_config, model_name)

                output_dir = os.path.join(output_top_dir, exp_group)
                path_to_rmse = output_dir + option + '_rmse'


                # get a relative path to config file for a given experiment
                current_dir = os.getcwd()
                relative_config_dir = os.path.relpath(path_to_config, start=current_dir)

                # The output will be stored here               
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                print("config dir: ", relative_config_dir)
                print("config_name: ", config_name)
                print("output_dir: ", output_dir )
                print("path_to_rmse: ", path_to_rmse )

                # get config file for a given experiment
                with initialize(config_path=relative_config_dir, version_base=None):
                    cfg = compose(config_name=config_name)
                
                # compute rmse and estimated concentrations
                rmse = run_validation_with_rmse(cfg,
                           option, 
                           path_to_model, 
                           path_to_data, 
                           slices,
                           rollout_length,
                           device, 
                           n_steps,
                           reg_frac,
                           random_starts
                           )
               
                np.save(path_to_rmse, rmse)

