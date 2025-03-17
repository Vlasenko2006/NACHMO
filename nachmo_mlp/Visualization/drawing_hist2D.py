#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:45:36 2024

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


def drawing_hist2D(c_ref, c_est, species, bins=100, xticks =None, passf = None, title = "Estimates vs Reference concentrations" , xlabel = 'Referecne(ppb)', ylabel = 'Estimates(ppb)', namef = None,  fx = 5, fy = 4 ):
    

  counter = 0

  fig, axs = plt.subplots(fx, fy)


  cells,steps  = c_est[:,:,counter].shape


  c    = torch.linspace(0, 12, 11) * 0.1 - 0.2




  for i in range(0,fx):
      for j in range(0,fy):
              
          if fx == 1:  ax = axs[j]
          if fy == 1:  ax = axs[i] 
          if fx >1 and fy >1 :  ax = axs[i, j]


          sub_title = species[i]
          counts, xedges, yedges, h = axs[i,j].hist2d(c_ref[:,:,counter].flatten(), c_est[:,:,counter].flatten(), bins)
        #axs[i,j].set_title(sub_title, fontsize="24")
        # axs[0,j].set_ylabel(ylabel, fontsize="24")
        # axs[i,j].set_xlabel(xlabel, fontsize="24")
    #axs[i].set_xticks(fontsize=20)
    
          ax.set_title(sub_title, fontsize="4" ,y=.95)
              
          if j == 0 and i == int(fx/2):
              ax.set_ylabel(ylabel, fontsize="12")
             
          if i == fx-1 and j == int(fy/2):
              ax.set_xlabel(xlabel, fontsize="12")  
        #if i < fx-1:
           # ax.set_xticklabels(["","", ""], fontsize =1) 
          ax.ticklabel_format(axis='both',style ='sci', scilimits = [0,0] ,useMathText=True)    
          axs[i,j].tick_params(axis='both', which='major', labelsize=12)
    #axs[i].tick_params(axis='both', **kwargs)
          counter = counter + 1

    
#fig.colorbar(h[0], ax=axs[0])
 

#plt.xticks(fontsize=20)
   
  fig.suptitle(title, fontsize=16)

  fig.tight_layout()

  cbar_ax = fig.add_axes([1.05, 0.1, 0.05, 0.8])
  clb = fig.colorbar(h, cax=cbar_ax)
  plt.show()

