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


def mean_val_plotter_oh(lref,lest, species, Xticks, passf = None, title = " " , Xlabel = None, ylabel = None, namef = None, scheme = "", fx = 1, fy = 3 ):

  
 
  fig, axs = plt.subplots(fx, fy)

  sup = ["m.seconds","seconds","hours"]
  for n in range(0,1):

      for i in range(0,fx):
          ref = lref[i] 
          y   = lest[i] 

          counter = i
         # print("ref.shape = ", ref.shape) [ref[:,:20,:],c_ref, ref_day]
             
          for j in range(0,fy):
              
             ref = lref[j] 
             y   = lest[j]              
             xticks   = Xticks[j] 
             xlabel   = Xlabel[j]
              
             ncells,steps,_ = y.shape
             
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
             
             
             
             if fx == 1:  ax = axs[j]
             if fy == 1:  ax = axs[i] 
             if fx >1 and fy >1 :  ax = axs[i, j]

             ax.plot(ref_mean[:], color = 'r', label = "absolute value of MAE")              
             ax.plot(y_mean[:], color = 'b', label = "absolute value of MAE")                        
             ax.fill_between(np.array(range(0,len(y_mean))), low_bound,upper_bound, color = [0.4, 0. , 0.], alpha=0.2, label = 'std-dev error')         
             ax.fill_between(np.array(range(0,len(y_mean))), ylow_bound,yupper_bound, color = [0., 0.4, 0.4], alpha=0.2, label = 'std-dev error')    
             #ymin, ymax = ax.get_ylim()
             ax.set_title(sub_title, fontsize="10" ,y=1)
             ax.set_xticks([0,int(steps/2), steps])
            # ax.set_ylim([min(ylow_bound),max(yupper_bound)])

             if j == 0:
                 ax.set_ylabel(ylabel, fontsize="15")
             
             if i == fx-1:


                 ax.set_xlabel(xlabel, fontsize="15")                

                 ax.set_xticklabels(xticks, fontsize="12")
             else:

                 ax.set_xticklabels(["","", ""], fontsize =1)
            # ax.ticklabel_format(axis='y', useMathText=True)
             #counter+=1


             
             # if j == 0:
             #     ax.set_ylabel(ylabel, fontsize="15")
             
             # if i == fx-1:


             #     ax.set_xlabel(xlabel, fontsize="15")             
             #     ax.set_xticks([0,int(steps/2), steps])
                     
             #     ax.set_xticklabels(xticks, fontsize="12")
             # else:
             #      ax.set_xticklabels(["","", ""], fontsize =1)


             #ax.set_yticks(yticks)
             ax.set_ylim(bottom=0)
             ax.ticklabel_format(axis='y',style ='sci', scilimits = [0,0] ,useMathText=True)
            # counter+=1
             
             
      fig.suptitle(title, fontsize=36) 
      fig.legend(["true", "estimates", "std(true)", "std(est)"], loc="center right")  
      
      
      if passf != None:
          passmaker(passf,namef, fig)
      else:
          return fig, ax
      
        


def load_data(path):
   reffile = path + "reference.npy"
   estfile = path + "estimates.npy" #path+"YD" + str(epoch) + ".npy"
   
   
   est = np.load(estfile) 

   if os.path.isfile(reffile):
       ref = np.load(reffile) 
       max_c = np.load(path+"max_c.npy")        
   else:
       print("file does not exist")
       ref, max_c = load_data(path, dtype=torch.float32 ,species =species)
       
   return est, ref, max_c 
       

epoch = 2

species = ["OH", "HO2", "H2O2"]




if len(species) == 3:

   pass_to_figures = "/Users/andreyvlasenko/tst/Figs_OH_Same/"  
   # path = "/Users/andreyvlasenko/tst/data/data_for_paper/dyn_OH_paper/Different_distribution/" #'../../../dyn_OH/'
   path = "/Users/andreyvlasenko/tst/data/data_for_paper/dyn_OH_paper/no_Smatrix/"
   path_day = "/Users/andreyvlasenko/tst/data/data_for_paper/dyn_OH_paper/one_day_estimates/"
#   reffile = path + "Y_ref_"+ str(epoch) +".npy"

#   Xticks = ['0(0)','30(3K)','60(6K)',"86K"]
   Xticks = ['0(0)','12(770)',"24(1440)"]   
#   xlabel = "Seconds(steps)"
   xlabel = "minutes(steps)"
   fx = 1
   fy = 3
   scheme ="dyn_OH1"
   
   
   est, ref, max_c = load_data(path)
   est_day, ref_day, max_c = load_data(path_day)


   est  = np.asarray(est[:50,:6000,:])
   ref  = np.asarray(ref[:50,:6000,:])
   
   
   
   ylabel = "ppb"

   max_c = np.load(path+"max_c.npy") 
       
   [ncells,timesteps0,conc] = est.shape
   [ncells,timesteps1,conc] = ref.shape
   timesteps = np.min([timesteps0,timesteps1])


   if len(species) > 3:
       timesteps = timesteps - 160
       timesteps2m = int(timesteps/12)*2



   for i in range(0,len(species)):
       est[:,:,i]  = est[:,:,i]  * max_c[i]
       ref[:,:,i]  = ref[:,:,i] *  max_c[i]
       est_day[:,:,i]  = est_day[:,:,i]  * max_c[i]
       ref_day[:,:,i]  = ref_day[:,:,i] *  max_c[i]
       print()




   #print("aux.shape = ", aux.shape, "aux1.shape = ", aux1.shape)


   skip = int(timesteps/400)


   if skip == 0: skip = 1

   c_est = est[:,0:timesteps:skip,:]
   c_ref = ref[:,0:timesteps:skip,:]


lref = [ref[:,:20,:],c_ref, ref_day]
lest = [est[:,:20,:],c_est, est_day]


Xticks0 = ['0(0)','1K(10)',"2K(20)"] 
Xticks1 = ['0(0)','30(3K)','60(6K)']
Xticks2 = ['0(0)','12(770)',"24(1440)"]   

Xlabel = ["m.seconds(steps)","seconds(steps)","hours(steps)"]
Xticks = [Xticks0,Xticks1,Xticks2]

#   xlabel = "Seconds(steps)"
xlabel = "minutes(steps)"
fx = 3
fy = 3
scheme ="dyn_OH1"



passF = pass_to_figures + "Group/"

mean_val_plotter_oh(lref,lest, species, Xticks = Xticks, passf = passF, title = "Reference and estimated mean concentrations " , Xlabel = Xlabel, ylabel = "ppb" , namef = "Mean_conc", scheme = "", fx = fx, fy = fy )






















