#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:20:05 2023

@author: andrey
"""


import matplotlib.pyplot as plt
import numpy as np
import torch




def shadded_plotter(relative_err_std, relative_err_min, relative_err_max, relative_err_mean, passf = None):
  counter = 0



  plt.plot(relative_err_mean[:], color = 'r', label = "mean_MAE")                    
  plt.fill_between(np.array(range(0,len(relative_err_mean))), relative_err_min[:,counter], relative_err_max[:,counter], color = [0.2, 0.2, 0.2], alpha=0.2, label = 'min-max area')           
  plt.set_title(species[counter+1])
# axs[i, j].axvline(x = 50, color = [0.2, 0.2, 0.2], label = 'avail data',linestyle='--')

  plt.set(xlabel='timesteps', ylabel='conc')

  if passf != None:
      plt.savefig(passf + '.png')




def plotter(y, passf = None, title = " ", xticks = None, xlabel = None, ylabel = None ):
     
    m= max(y[:,0])
    cells,steps  = y.shape
    c = 0
    fig, ax = plt.subplots()
    for l in range(0,50):
    
        col = y[l,0]
        plt.plot(y[l,:], color = (0,0,col))
        
        
        
        
    plt.title(title, fontsize="20" )
    ax.set_xlabel(xlabel, fontsize="20")
    ax.set_ylabel(ylabel, fontsize="20")
    ax.set_xticks([0,int(steps/2), steps])
    ax.set_xticklabels(xticks, fontsize="20")
        
    if passf != None:
        fig.savefig(passf + ".png")



def min_max_mean_error(c,c_ref):
    

  [ncells,tsteps,nspecies] = c_ref.shape
  relative_err_cell = np.zeros([ncells, tsteps, nspecies])
  relative_err = np.zeros([ncells,tsteps,nspecies])
  relative_err_std = np.zeros([tsteps,nspecies])
  relative_err_mean = np.zeros([tsteps,nspecies])
  relative_err_min = np.zeros([tsteps,nspecies])
  relative_err_max = np.zeros([tsteps,nspecies])


  for i in range(0,ncells):
      for j in range(0,nspecies):   
          relative_err_cell[i,:,j] = (c[i,:,j]-c_ref[i,:,j])
          
          
          
  for i in range(0,tsteps):
      for j in range(0,nspecies):             
          relative_err_std[i,j] = np.std(relative_err_cell[:,i,j])
          relative_err_min[i,j] = np.min(relative_err_cell[:,i,j])
          relative_err_max[i,j] = np.max(relative_err_cell[:,i,j])
          relative_err_mean[i,j] = np.mean(abs(relative_err_cell[:,i,j]))

  return relative_err_std, relative_err_min, relative_err_max, relative_err_mean




def scatter_plot_dyn_oh(c_est, c_ref, step = 0, alp = 0.125):
    c = torch.linspace(0, 12, 11) * 0.1 - 0.2
            

    for gas in range(0,len(species)):
        reference = c_ref[:,step,gas]
        y = c_est[:,step,gas]
        
        fig, ax = plt.subplots()
        
        plt.scatter(reference, y, alpha=alp)
        plt.plot(c, c, "r")
        
        ax.set_xlabel("Reference")
        ax.set_ylabel("Estimates")
        plt.title("Scatter plot of " + species[gas] + " after "  + str(step) + " steps")  
        
        plt.show()



def spagetti(c_est, c_ref = None, passf = None, xticks = None, xlabel = None, ylabel = None ):
    
    
    
    
    for i in range(0,len(species)):
    
        title = "Estimated concentrations of " + species[i]     
    
        plotter(c_est[:,:,i], passf = passf, title = title, xticks = xticks, xlabel = xlabel, ylabel = ylabel)

        title = "Reference concentrations of " + species[i]   
        
        plotter(c_ref[:,:,i], passf = passf, title = title, xticks = xticks, xlabel = xlabel, ylabel = ylabel )




def spagetti_relative_error(c_est, passf = None, xticks = None, xlabel = None, ylabel = "Error in % "):
    
    
    
    
    for i in range(0,len(species)):
    
        title = "Relative error in " + species[i] + "estimates"     
    
        plotter(c_est[:,:,i], passf = passf, title = title, xticks = xticks, xlabel = xlabel, ylabel = ylabel)



epoch = 10
stride = 1

species = ["oh", "ho2", "h2o2"]


s = 0
count = 0

sp = 1
path = 'Data/'
max_c = np.asarray([1,1,1])#np.load(path+"max_c.npy")

aux = np.load(path+"Y" + str(epoch) + ".npy") 
aux1 = np.load(path+"Y_ref_"+ str(epoch) +".npy") 

for i in range(0,len(max_c)):
    aux[:,:,i]  =  aux[:,:,i]  *max_c[i]
    aux1[:,:,i] =  aux1[:,:,i] * max_c[i]

[ncells,timesteps,conc] = aux.shape


skip = int(timesteps/400)

if skip == 0: skip = 1

c_ref = aux[:,0:timesteps:skip,:]
c_est = aux1[:,0:timesteps:skip,:]

abs_err = c_est - c_ref

rel_err = np.zeros(abs_err.shape)


for i in range(0,len(species)):
    for j in range(0,len(c_ref)): 
        rel_err[j,:,i] = 100 * abs_err[j,:,i]/np.max(c_ref[j,:,i])


spagetti(c_est, c_ref, passf = None, xticks = ['0(0)','30(125K)','60(250K)'],  xlabel = "Seconds(steps)" , ylabel = "Concentrations(ppb)")


spagetti_relative_error(rel_err, passf = None, xticks = ['0(0)','30(125K)','60(250K)'],  xlabel = "Seconds(steps)" , ylabel = "Error in % ")

scatter_plot_dyn_oh(c_est, c_ref, step = 0, alp = 0.25)




