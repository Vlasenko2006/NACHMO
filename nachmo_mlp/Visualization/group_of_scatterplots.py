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




def group_of_scatterplots(c_ref,c_est, species, step = 0, alp = 0.125,  xticks = " ", passf = None, title = " " ,  xlabel = "Reference", ylabel = "Estimates", namef = None, fx = 5, fy = 4 ):
    
  if not title:
        titleS = "Scatter plot " 
  else:
        titleS = title 
  if not namef: namefS = "group_scatter_plot_"   
        
    
    

   
  counter = 0
  fig, axs = plt.subplots(fx, fy)

  scale = 20/fx*fy
  
  cells,steps  = c_est[:,:,counter].shape
  print("Steps = ", steps)

  c    = torch.linspace(0, 12, 11) * 0.1 - 0.2

  for n in range(0,1):

      for i in range(0,fx):
          for j in range(0,fy):
              
              reference = c_ref[:,step,counter]
              y = c_est[:,step,counter]
        
              y=y/max(abs(reference))  
              reference=reference/max(abs(reference))        
        

              yticks = np.linspace( -0.2,  np.max(y), 5).round(3)        
              xticks = np.linspace( -0.2,  np.max(reference), 5).round(3)  
              sub_title= species[counter]
             
             
             
              if fx == 1:  ax = axs[j]
              if fy == 1:  ax = axs[i] 
              if fx >1 and fy >1 :  ax = axs[i, j]

              ax.scatter(reference, y, alpha=alp)
              ax.plot(c, c, "r")
              ax.plot(c, c*0, "k", linestyle='--')
              ax.plot(c*0, c, "k", linestyle='--')
        
        
             # ax.set_xlabel(xlabel,fontsize="30")
             # ax.set_ylabel(ylabel,fontsize="30")
             # ax.set_xticks(xticks)
             # ax.set_xticklabels(xticks, fontsize="20")
             # ax.set_yticks(yticks)
             # ax.set_yticklabels(yticks, fontsize="20")
             # ax.set_title(titleS, fontsize="30")                               


              ax.set_title(sub_title, fontsize="10" ,y=.95)
              
              if j == 0 and i == int(fx/2):
                 ax.set_ylabel(ylabel, fontsize="15")
             
              if i == fx-1 and j == int(fy/2):
                  ax.set_xlabel(xlabel, fontsize="15")  
              if i < fx-1:
                  ax.set_xticklabels(["","", ""], fontsize =1) 
                  
             #      ax.set_xticks([0,int(steps/2), steps])
             # #    axs[i, j].set_xticks([0, 2042, 5040])               
             #      ax.set_xticklabels(xticks, fontsize="12")
             #  else:
             #       ax.set_xticklabels(["","", ""], fontsize =1)

             #  ax.ticklabel_format(axis='y', useMathText=True)
              counter+=1
             
             
      fig.suptitle(title, fontsize=26)  
      
      
      if passf != None:
          passmaker(passf,namef, fig)
      else:
          return fig, ax
      
        
      
        
