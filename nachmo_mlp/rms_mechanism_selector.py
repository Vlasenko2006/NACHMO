#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:56:44 2024

@author: andrey
"""

import os
import numpy as np


option_verwer = ["no_gate", "qp_conc", "ssur", "naive", "err_proj"]

option_oh = ["no_gate", "qp_conc", "ssur", "naive", "smat_proj"]




#current_epoch = 17

#current_epoch = 2843




def rms_mechanism_selector(option,  mechanism = None, current_epoch=None ):


    assert mechanism     != None, "Specify mechanism !"
    assert current_epoch != None, "Specify epoch !"
   

    if mechanism == "Verwer":
        if option == "no_gate":
            sub_path = "No_gate_Curr__torch.float32___mean__tr_length_100_opt_0/version_0/"
    

        if option == "qp_conc":
            sub_path = "Reference_naive_CurCurr__torch.float32___mean__tr_length_100_opt_3/version_0/"
            

        if option == "ssur":
            sub_path = "Reference_proj_CurCurr__torch.float32_Ssur__mean__tr_length_100_opt_3/version_1/"

        if option == "ssur_no_gate":
            sub_path = "Ssur_no_gates_Curr__torch.float32_Ssur__mean__tr_length_100_opt_3/version_0/"


        if option == "gate":
            sub_path = "Reference_naive_CurCurr__torch.float32___mean__tr_length_100_opt_3/version_0/"

        if option == "err_proj":
            sub_path = "Reference_naive_CurCurr__torch.float32___mean__tr_length_100_opt_3/version_0/"

        if option == "err_proj_no_gate":
            sub_path = "No_gate_Curr__torch.float32___mean__tr_length_100_opt_0/version_0/"

        if option == "train_with_err_proj_no_gate":
            sub_path = "Err_proj_no_gates_Curr__torch.float32_Ssur__mean__tr_length_100_opt_3/version_0/"

        path_to_config = "/gpfs/work/vlasenko/NACHMO_prog/prerelease/nachmo_coding/nachmo_mlp/Verwer_long/" + sub_path            
#        path_to_config = "/gpfs/work/vlasenko/NACHMO_prog/Smatrix5/nachmo_coding/nachmo_mlp/Verwer_2h_nonshifted_proj/" + sub_path 
        path_to_data = "/gpfs/work/vlasenko/NACHMO_data/Verwer_10h/"




    if mechanism == "OH":    
        if option == "gate":
            sub_path = "Gates_Curr__torch.float32___mean__tr_length_10_opt_0/version_0/"


        if option == "qp_conc":
            sub_path = "Gates_Curr__torch.float32___mean__tr_length_10_opt_0/version_0/"

        if option == "no_gates":
            sub_path = "No_gatesCurr__torch.float32___mean__tr_length_10_opt_0/version_0/"

        if option == "ssur":
            sub_path = "Ssur_Curr__torch.float32_Ssur__mean__tr_length_10_opt_0/version_0/"

        if option == "smat_proj":
            sub_path = "/Smatrix_gatesCurr__torch.float32___mean__tr_length_10_opt_0/version_0/"

        path_to_config = "/gpfs/work/vlasenko/NACHMO_prog/Smatrix5/nachmo_coding/nachmo_mlp/dyn_oh_control/" + sub_path
        path_to_data = "/gpfs/work/vlasenko/NACHMO_data/dyn_OH/"
          
    path_to_model = path_to_config + "model" + str(current_epoch) + ".pt" 
        
    return path_to_model, path_to_data, path_to_config
        
