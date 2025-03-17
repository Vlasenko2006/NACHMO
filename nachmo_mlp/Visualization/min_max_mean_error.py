#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 21:33:38 2023

@author: andreyvlasenko
"""



import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def min_max_mean_error(absolute_error):
    

  [ncells,tsteps] = absolute_error.shape

  absolute_err_std = np.zeros([tsteps])
  absolute_err_mean = np.zeros([tsteps])
  absolute_err_min = np.zeros([tsteps])
  absolute_err_max = np.zeros([tsteps])
          
          
          
  for i in range(0,tsteps):  
      aux = absolute_error[:,i]   
      min_error = aux[aux<0] 
      max_error = aux[aux>0] 
      if min_error.size ==0: min_error = 0
      if max_error.size ==0: max_error = 0

      
      absolute_err_std[i] = np.std(absolute_error[:,i])
      absolute_err_min[i] = np.min(aux)
      absolute_err_max[i] = np.max(aux)
      absolute_err_mean[i] = np.mean(abs(absolute_error[:,i]))

  return absolute_err_std, absolute_err_min, absolute_err_max, absolute_err_mean