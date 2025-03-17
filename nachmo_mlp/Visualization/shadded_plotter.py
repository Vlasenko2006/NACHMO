#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:55:14 2023

@author: andreyvlasenko
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib.ticker as ticker

from passmaker import passmaker
from min_max_mean_error import min_max_mean_error
from myticks import myticks

    
    

def shadded_plotter(absolute_error, passf = None, title = " ", xticks = None, xlabel = None, ylabel = None, namef = None):



  matplotlib.rcParams.update(matplotlib.rcParamsDefault)
  plt.rcParams['text.usetex'] = True

  absolute_err_std, absolute_err_min, absolute_err_max, absolute_err_mean = min_max_mean_error(absolute_error)
  
  cells,steps  = absolute_error.shape
  
  
  m = np.min(np.ndarray.flatten(absolute_err_mean - absolute_err_std[:]),axis=0)
  M = np.max(np.ndarray.flatten(absolute_err_mean + absolute_err_std[:]),axis=0)
  
  yticks = np.linspace( np.min([m, np.min(absolute_err_min)]),  np.max([M, np.max(absolute_err_max)]), 5).round(3)

  fig, ax = plt.subplots()

  ax.plot(absolute_err_mean[:], color = 'r', label = "absolute value of MAE")                    
  ax.fill_between(np.array(range(0,len(absolute_err_mean))), absolute_err_min[:], absolute_err_max[:], color = [0.4, 0., 0.4], alpha=0.2, label = 'min-max error')    
  ax.fill_between(np.array(range(0,len(absolute_err_mean))), absolute_err_mean - absolute_err_std[:],absolute_err_mean + absolute_err_std[:], color = [0.4, 0.4, 0.], alpha=0.2, label = 'std-dev error')         
  ax.set_title(title, fontsize="30" )
  ax.set_xlabel(xlabel, fontsize="20")
  ax.set_ylabel(ylabel, fontsize="20")
  ax.set_xticks([0,int(steps/2), steps])
  ax.set_xticklabels(xticks, fontsize="20")
  if np.max(np.abs(yticks))<0.0001:
      ax.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))
  else:
      ax.set_yticklabels(yticks, fontsize="20")
      ax.set_yticks(yticks)
  ax.legend()
  
# ax.axvline(x = 50, color = [0.2, 0.2, 0.2], label = 'avail data',linestyle='--')


  if passf != None:
      passmaker(passf,namef, fig)
  else:
      return fig, ax
