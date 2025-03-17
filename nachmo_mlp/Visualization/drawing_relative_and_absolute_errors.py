#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:36:51 2024

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


def drawing_relative_and_absolute_errors(absolute_error, species, xticks, passf = None, title = " " , xlabel = None, ylabel = None, namef = None, fx = 5, fy = 4 ):

   
  counter = 0
  fig, axs = plt.subplots(fx, fy)

  scale = 20/fx*fy
  
  cells,steps  = absolute_error[:,:,counter].shape
  print("Steps = ", steps)

  for n in range(0,1):

      for i in range(0,fx):
          for j in range(0,fy):
              

             absolute_err_std, absolute_err_min, absolute_err_max, absolute_err_mean = min_max_mean_error(absolute_error[:,:,counter])

  
             m = np.min(np.ndarray.flatten(absolute_err_mean - absolute_err_std[:]),axis=0)
             M = np.max(np.ndarray.flatten(absolute_err_mean + absolute_err_std[:]),axis=0)
              
             
             yticks = np.linspace( np.min( absolute_error ),  np.max(absolute_error ), 3).round(1)

            
             sub_title= species[counter]
    
             yticks = np.linspace( m,  M, 3).round(2)
             
             
             low_bond = absolute_err_mean - absolute_err_std[:]
             #low_bond[low_bond<0] = 0
             upper_bond = absolute_err_mean + absolute_err_std[:]
             
             
             if fx == 1:  ax = axs[j]
             if fy == 1:  ax = axs[i] 
             if fx >1 and fy >1 :  ax = axs[i, j]

             ax.plot(absolute_err_mean[:], color = 'r', label = "absolute value of MAE")   
           #  ax.fill_between(np.array(range(0,len(absolute_err_mean))), absolute_err_min[:], absolute_err_max[:], color = [0.4, 0., 0.4], alpha=0.2, label = 'min-max error')    
             ax.fill_between(np.array(range(0,len(absolute_err_mean))), low_bond, upper_bond, color = [0.4, 0.4, 0.], alpha=0.2, label = 'std-dev error')    
                    
             # ax.fill_between(np.array(range(0,len(y_mean))), low_bound,upper_bound, color = [0.4, 0. , 0.], alpha=0.2, label = 'std-dev error')         
             # ax.fill_between(np.array(range(0,len(y_mean))), ylow_bound,yupper_bound, color = [0., 0.4, 0.4], alpha=0.2, label = 'std-dev error')    
             ymin, ymax = ax.get_ylim()
             #ax.vlines(x=int(steps*0.8), ymin=ymin, ymax=ymax, colors=[0.7,0.7,0.7], ls='--', lw=2)


             ax.set_title(sub_title, fontsize="10" ,y=.95)
             
             if j == 0:
                 ax.set_ylabel(ylabel, fontsize="15")
             
             if i == fx-1:


                 ax.set_xlabel(xlabel, fontsize="15")             
                 ax.set_xticks([0,int(steps/2), steps])
             #    axs[i, j].set_xticks([0, 2042, 5040])               
                 ax.set_xticklabels(xticks, fontsize="12")
             else:
                  ax.set_xticklabels(["","", ""], fontsize =1)

             ax.ticklabel_format(axis='y', useMathText=True)
             counter+=1
             
             
      fig.suptitle(title, fontsize=36) 
      fig.legend(["MAE", "std(MAE)"], loc="center right")  
      
      
      if passf != None:
          passmaker(passf,namef, fig)
      else:
          return fig, ax
      