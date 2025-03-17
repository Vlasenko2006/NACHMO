#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:40:58 2024

@author: andreyvlasenko
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal

import matplotlib.colors as mcolors


import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys




from passmaker import passmaker
from drawing_spagetti import drawing_spagetti
from drawing_spagetti_absolute_error import drawing_spagetti_absolute_error
from drawing_spagetti_relative_error import drawing_spagetti_relative_error
from drawing_scatter_plot_dyn_oh import drawing_scatter_plot_dyn_oh
from drawing_histogram import drawing_histogram
from drawing_scatter_plot_dyn_oh import drawing_scatter_plot_dyn_oh
from mean_val_plotter import mean_val_plotter
from group_of_scatterplots import group_of_scatterplots

sys.path.append(os.path.abspath(os.path.join('..', '')))

from utilities import load_data






epoch = 2

species = ["OH", "HO2", "H2O2"]

species = ["CO","HNO3","SO4","XO2","O1D","SO2","O3P","ALD2","PAN","CH3O","N2O5","NO3","HCHO","O3","C2O3","HO2","NO2","NO","CH3O2","OH"]


if len(species) == 3:

   # pass_to_figures = "/Users/andreyvlasenko/tst/dyn_OH_paper/Same_distribution/"  
   # # path = "/Users/andreyvlasenko/tst/data/data_for_paper/dyn_OH_paper/Different_distribution/" #'../../../dyn_OH/'
   # path = "/Users/andreyvlasenko/tst/data/data_for_paper/dyn_OH_paper/Same_distribution/"
   
   pass_to_figures = "/Users/andreyvlasenko/tst/Figs_OH_Different/"  
   path = "/Users/andreyvlasenko/tst/data/data_for_paper/dyn_OH_paper/Different_distribution/" 
   
   
#   reffile = path + "Y_ref_"+ str(epoch) +".npy"
   reffile = path + "reference.npy"
   estfile = path + "estimates.npy" #path+"YD" + str(epoch) + ".npy"
   
   
   aux = np.load(estfile) 

   if os.path.isfile(reffile):
       aux1 = np.load(reffile) 
       max_c1 = np.load(path+"max_c.npy")        
   else:
       print("file does not exist")
       aux1, max_c1 = load_data(path, dtype=torch.float32 ,species =species)
       

   aux  = np.asarray(aux[:,:6000,:])
   aux1 = np.asarray(aux1[:,:6000,:])
   aux  = np.asarray(aux[:50,:,:])
   aux1 = np.asarray(aux1[:50,:,:])
   Xticks = ['0(0)','30(3K)','60(6K)',"86K"]
   Xticks = ['0(0)','12(770)',"24(1440)"]   
#   xlabel = "Seconds(steps)"
   xlabel = "minutes(steps)"
   fx = 1
   fy = 3
   scheme ="dyn_OH1"

else:
  # pass_to_figures = "Figs_Verwer_tst/"  
   # pass_to_figures = "/Users/andreyvlasenko/tst/Figs_Verwer_tst/"  
   # path = "/Users/andreyvlasenko/tst/data/data_for_paper/Verwer_paper/Same_distribution/"
   
   # pass_to_figures = "/Users/andreyvlasenko/tst/Figs_Verwer_same/"  
   # path = "/Users/andreyvlasenko/tst/data/data_for_paper/Verwer_paper/same_distribution/"
   

   pass_to_figures = "/Users/andreyvlasenko/tst/Figs_Verwer_different/"  
   path = "/Users/andreyvlasenko/tst/data/data_for_paper/Verwer_paper/Different_distribution/"

   
   #path = "/Users/andreyvlasenko/tst/data/ref/npy/" #Same_distribution/" #'../../../Verwer/'
   reffile = path+"Y_ref_"+ str(epoch) +".npy"
   estfile = path+"estimates.npy"   
   aux = np.load(estfile) 

   aux1, max_c1 = load_data(path, dtype=torch.float32 ,species =species) 
   aux  = np.asarray(aux[:,:-1:10,:])
   aux1 = np.asarray(aux1)
   scheme ="Verwer"



   Xticks = ['0(0)','6(26K)','12(52K)']
   Xticks2M = ['0(0)','1(4.5K)','2(9K)']

   xlabel = "months(steps)"
   fx = 5
   fy = 4


ylabel = "ppb"

max_c = np.load(path+"max_c.npy") 
    
[ncells,timesteps0,conc] = aux.shape
[ncells,timesteps1,conc] = aux1.shape
timesteps = np.min([timesteps0,timesteps1])


if len(species) > 3:
    timesteps = timesteps - 160
    timesteps2m = int(timesteps/12)*2



for i in range(0,len(species)):
    aux[:,:,i]  = aux[:,:,i]  * max_c[i]
    aux1[:,:,i] = aux1 [:,:,i] * max_c1[i]
    print()




#print("aux.shape = ", aux.shape, "aux1.shape = ", aux1.shape)


skip = int(timesteps/400)


if skip == 0: skip = 1


skip = 1

c_est = aux[:,0:timesteps:skip,:]
c_ref = aux1[:,0:timesteps:skip,:]

if len(species)>3:
    skip2m = int(timesteps2m/400)
    c_est2m = aux[:,0:timesteps2m:skip,:]
    c_ref2m = aux1[:,0:timesteps2m:skip,:]


abs_err = c_est - c_ref

rel_err = np.zeros(abs_err.shape)



c_est[np.isnan(c_est)] = 0


ylabel = 'Estimates(ppb)'
xlabel = 'Referecne(ppb)'
title = "Estimates vs Reference concentrations"

passf =pass_to_figures + 'Group_of_scatterplots'
namef = "2D_Hist"


counter = 0

fig, axs = plt.subplots(fx, fy)


cells,steps  = c_est[:,:,counter].shape


c    = torch.linspace(0, 12, 11) * 0.1 - 0.2


dc_ref= c_ref[:,1:,:] - c_ref[:,:-1,:] 
dc_est= c_est[:,1:,:] - c_est[:,:-1,:] 



for i in range(0,fx):
    for j in range(0,fy):
              
        if fx == 1:  ax = axs[j]
        if fy == 1:  ax = axs[i] 
        if fx >1 and fy >1 :  ax = axs[i, j]


        sub_title = species[i]
        counts, xedges, yedges, h = axs[i,j].hist2d(dc_ref[:,:,counter].flatten(), dc_est[:,:,counter].flatten(), bins=100)
        #axs[i,j].set_title(sub_title, fontsize="24")
        # axs[0,j].set_ylabel(ylabel, fontsize="24")
        # axs[i,j].set_xlabel(xlabel, fontsize="24")
    #axs[i].set_xticks(fontsize=20)
    
        ax.set_title(sub_title, fontsize="15" ,y=.95)
              
        if j == 0 and i == int(fx/2):
            ax.set_ylabel(ylabel, fontsize="20")
             
        if i == fx-1 and j == int(fy/2):
            ax.set_xlabel(xlabel, fontsize="20")  
        #if i < fx-1:
           # ax.set_xticklabels(["","", ""], fontsize =1) 
        ax.ticklabel_format(axis='both',style ='sci', scilimits = [0,0] ,useMathText=True)    
        axs[i,j].tick_params(axis='both', which='major', labelsize=15)
    #axs[i].tick_params(axis='both', **kwargs)
        counter = counter + 1

    
#fig.colorbar(h[0], ax=axs[0])
 

#plt.xticks(fontsize=20)
   
fig.suptitle(title, fontsize=36)

fig.tight_layout()

cbar_ax = fig.add_axes([1.05, 0.1, 0.05, 0.8])
clb = fig.colorbar(h, cax=cbar_ax)

clb.ax.set_title("Occurance", fontsize = "18")


plt.show()


if passf != None:
    passmaker(passf,namef, fig)



