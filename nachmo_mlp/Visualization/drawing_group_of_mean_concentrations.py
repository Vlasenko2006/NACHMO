#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:35:34 2023

@author: andrey
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:20:05 2023

@author: andrey
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



from shadded_plotter import shadded_plotter



def load_data(path, species=None):
    """Load data from data files for each chemical species.

    returned data has size (n_trajectories, n_timepts, n_species)
    """
    c = [np.load(path + s + ".npy") for s in species]  # load chemical concentrations
    for d in c:
        assert d.ndim == 2
    max_c = [d.max() for d in c]
    c_norm = np.stack([c / m for c, m in zip(c, max_c)], axis=2)
  
    return c_norm, max_c






def explicit_max(l, mode):
    l = list(abs(torch.squeeze(torch.tensor(l))))
    
    if mode == "max":
        val = max(l)
    if mode == "min":
        val = min(l)
    idx = l.index(val)
    return int(idx), val




def plotter(ref,y,xticks, xlabel, ylabel, title, species, fx = 5, fy = 4 ):
    
   
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
             axs[i, j].plot(ref_mean[:], color = 'r', label = "absolute value of MAE")              
             axs[i, j].plot(y_mean[:], color = 'b', label = "absolute value of MAE")                        
             axs[i, j].fill_between(np.array(range(0,len(y_mean))), low_bound,upper_bound, color = [0.4, 0. , 0.], alpha=0.2, label = 'std-dev error')         
             axs[i, j].fill_between(np.array(range(0,len(y_mean))), ylow_bound,yupper_bound, color = [0., 0.4, 0.4], alpha=0.2, label = 'std-dev error')         


             axs[i, j].set_title(sub_title, fontsize="10" ,y=.95)
             
             if j == 0:
                 axs[i, j].set_ylabel(ylabel, fontsize="15")
             
             if i == fx-1:


                 axs[i, j].set_xlabel(xlabel, fontsize="15")             
                 axs[i, j].set_xticks([0,int(steps/2), steps])
             #    axs[i, j].set_xticks([0, 2042, 5040])               
                 axs[i, j].set_xticklabels(xticks, fontsize="12")
             else:
                  axs[i, j].set_xticklabels(["","", ""], fontsize =1)

             axs[i, j].ticklabel_format(axis='y', useMathText=True)
             counter+=1
             
             
      fig.suptitle(title, fontsize=26) 
      fig.legend(["true", "estimates", "std(true)"], loc="upper right")  


species = [ "CO","HNO3","SO4","XO2","O1D","SO2","O3P","ALD2","PAN","CH3O","N2O5","NO3","HCHO","O3","C2O3","HO2","NO2","NO","CH3O2","OH"]





m  = 240
s = 2
count = 0

sp = 1
epoch = 7

path = "/Users/andreyvlasenko/tst/data/ref/npy/" #'../../../data/'

impath = path + 'figs/'



#y  = np.load(path+ 'data.npy')

l = 11000

#y = y[l:l+210,:,:]

#ref = y
y = np.load(path+"Y" + str(epoch) + ".npy") 

# ref = np.load(path+"Y_ref_"+ str(epoch) +".npy") 
 
# y1 = np.load(path+"Y" + str(epoch+0) + ".npy") 

# max_c = np.load(path+"max_c.npy")

# max_c = np.squeeze(max_c)


ref, max_c = load_data(path, species)
#y = ref

cell,steps,sp = y.shape

y = y[:,:-1:10,:]
cell,steps,sp = y.shape

steps = steps #int(steps/2)
ref = ref[:,:steps,:]
y = y[:,:steps,:]
for i in range(0,len(species)):
    y[:,:,i] = y[:,:,i]*max_c[i]
    ref[:,:,i] = ref[:,:,i]*max_c[i]




_,l,_ =ref.shape 

l = int(5*l/6) +70 #2500*8

l0 = 2200


xticks = ['0','3','6']
xlabel = "month"
ylabel = " ppb"
title = " Mean of estimated and true concentrations"

plotter(ref[:,l0:l,:], y[:,l0:l,:],xticks, xlabel, ylabel, title, species, 5, 4 )
        
    



