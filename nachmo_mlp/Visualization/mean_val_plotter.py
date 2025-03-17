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


def mean_val_plotter(ref,y, species, xticks, passf = None, title = " " , xlabel = None, ylabel = None, namef = None, scheme = "", fx = 5, fy = 4 ):

  print("ref.shape = ", ref.shape)    
  print("scheme = ", scheme)    
  
  
  counter = 0
  fig, axs = plt.subplots(fx, fy)

  scale = 20/fx*fy
  
  cells,steps  = y[:,:,counter].shape
  print("Steps = ", steps)


  for n in range(0,1):

      for i in range(0,fx):
          for j in range(0,fy):
              
              
             y_std, y_min, y_max, y_mean= min_max_mean_error(y[:,:,counter])
             ref_std, ref_min, ref_max, ref_mean= min_max_mean_error(ref[:,:,counter])
             
             print("ref.shape = ", ref.shape) 
             print("ref_mean.shape = ", ref_mean.shape)
             print("i = ", i, " j = ", j )
             
             low_bound = ref_mean- ref_std[:]
             low_bound[low_bound<0] = 0.
             upper_bound = ref_mean+ ref_std[:]
             
             
             ylow_bound = y_mean- y_std[:]
             ylow_bound[ylow_bound<0] = 0.
             yupper_bound = y_mean+ y_std[:]           
             
             
             
             m = np.min(np.ndarray.flatten(low_bound),axis=0)
             M = np.max(np.ndarray.flatten(upper_bound),axis=0)
             
             yticks = np.linspace( np.min(y),  np.max(y), 3).round(1)

            
             sub_title= species[counter]
    
             yticks = np.linspace( m,  M, 3).round(2)
             
             
             
             if fx == 1:  ax = axs[j]
             if fy == 1:  ax = axs[i] 
             if fx >1 and fy >1 :  ax = axs[i, j]

             ax.plot(ref_mean[:], color = 'r', label = "absolute value of MAE")              
             ax.plot(y_mean[:], color = 'b', label = "absolute value of MAE")                        
             ax.fill_between(np.array(range(0,len(y_mean))), low_bound,upper_bound, color = [0.4, 0. , 0.], alpha=0.2, label = 'std-dev error')         
             ax.fill_between(np.array(range(0,len(y_mean))), ylow_bound,yupper_bound, color = [0., 0.4, 0.4], alpha=0.2, label = 'std-dev error')    
             ymin, ymax = ax.get_ylim()
             ax.set_title(sub_title, fontsize="10" ,y=.95)
             
             if j == 0:
                 ax.set_ylabel(ylabel, fontsize="15")
             
             if i == fx-1:


                 ax.set_xlabel(xlabel, fontsize="15")             
                 ax.set_xticks([0,int(steps/2), steps])

                 if scheme == "dyn_OH":
                     ax.vlines(x=int(steps*0.8), ymin=0, ymax=ymax, colors=[0.7,0.7,0.7], ls='--', lw=2)
                     ax.set_xticks([0,int(0.8*steps/2),int(steps*0.8) ,steps])
                 else:
                     ax.set_xticks([0,int(steps/2), steps])
                     
                 ax.set_xticklabels(xticks, fontsize="12")
             else:
                  ax.set_xticklabels(["","", ""], fontsize =1)


             #ax.set_yticks(yticks)
             ax.set_ylim(bottom=0)
             ax.ticklabel_format(axis='y', useMathText=True)
             counter+=1
             
             
      fig.suptitle(title, fontsize=36) 
      fig.legend(["true", "estimates", "std(true)", "std(est)"], loc="center right")  
      
      
      if passf != None:
          passmaker(passf,namef, fig)
      else:
          return fig, ax
      