#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Thu Mar 30 15:00:28 2023.

@author: andreyvlasenko
"""
import torch


def loss_fun(yhat, y, penalty_factor=0.0, my_weights = False, max_c = None, first_step=1):

    _, nspecs,steps= y.shape

    yhat, y = yhat[:, :, first_step - 1 :], y[:, :, first_step - 1 :]
    residual = yhat - y



 #   print("My Weigths= ", my_weights)

    if not any(my_weights):

      #  print("My Weigths false= ", my_weights)

        penalty = torch.tensor(0., dtype = y.dtype)
        penalty1 = torch.tensor(0., dtype = y.dtype)
        penalty2 = torch.tensor(0., dtype = y.dtype)
        penalty3 = torch.tensor(0., dtype = y.dtype)
        penalty4 = torch.tensor(0., dtype = y.dtype)
    elif all(my_weights):

      #  print("My Weigths true = ", my_weights)
#        penalty1  = (yhat[:,2,:]*max_c[2] + yhat[:,5,:]*max_c[5])/(torch.max(yhat[:,2,:]*max_c[2], yhat[:,5,:]*max_c[5])) 

        penalty1  = my_weights[0] * torch.nn.functional.relu( -1 * torch.diff(yhat[:, 12, first_step :])) # HCHO depleets onlu
        penalty2  = my_weights[1] *torch.nn.functional.relu( -1 * torch.diff(yhat[:, 17, first_step :])) # NO depleets only
        penalty3  = my_weights[2] *torch.nn.functional.relu(  1 * torch.diff(yhat[:, 0,  first_step :]))  # CO builds up only
        penalty4  = my_weights[3] *torch.nn.functional.relu( -1 * torch.diff(yhat[:, 7,  first_step :])) # ALD depleets only
        
 #   exit()
#        for i in range(0,nspecs):
#            residual[:, i, :] = residual[:, i, :]
#            penalty = 0.001 * my_weights[i] * torch.log(yhat / y) 

    L = (residual * residual).mean() +  (penalty1 * penalty1).mean() +  (penalty2 * penalty2).mean() +  (penalty3 * penalty3).mean() +  (penalty4 * penalty4).mean()
    positivity_violation = -torch.clamp(yhat, max=0.0).mean()
    return L + positivity_violation * penalty_factor
