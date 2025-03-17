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


def drawing_the_best_and_worst_results(rel_err, y,y_ref, species, xticks, passf = None, title = " " , xlabel = None, ylabel = None, namef = None, fx = 5, fy = 4 ):

    
  max_err  = np.argmax(np.sum(np.sum(rel_err, axis = 1),axis = 1))

   
  counter = 0
  fig, axs = plt.subplots(fx, fy)

  scale = 20/fx*fy
  
  cells,steps  = rel_err[:,:,counter].shape
  print("Steps = ", steps)

  for n in range(0,1):

      for i in range(0,fx):
          for j in range(0,fy):
              

             absolute_err_std, absolute_err_min, absolute_err_max, absolute_err_mean = min_max_mean_error(rel_err[:,:,counter])

  
             m = np.min(np.ndarray.flatten(absolute_err_mean - absolute_err_std[:]),axis=0)
             M = np.max(np.ndarray.flatten(absolute_err_mean + absolute_err_std[:]),axis=0)
              
             
             yticks = np.linspace( np.min( rel_err ),  np.max(rel_err ), 3).round(1)

            
             sub_title= species[counter]
    
             yticks = np.linspace( m,  M, 3).round(2)
             
             
             
             
             if fx == 1:  ax = axs[j]
             if fy == 1:  ax = axs[i] 
             if fx >1 and fy >1 :  ax = axs[i, j]

             ax.plot(y_ref[max_err,:,counter], color = 'r', label = "a")      
             ax.plot(y[max_err,:,counter], color = 'b', label = "a")     
             ymin, ymax = ax.get_ylim()
             ymax= 1.1 * ymax

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
      fig.legend(["reference", "estimates"], loc="center right")  
      
      
      if passf != None:
          passmaker(passf,namef, fig)
      else:
          return fig, ax
      