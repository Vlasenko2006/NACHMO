#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 21:35:40 2023

@author: andreyvlasenko
"""




import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from passmaker import passmaker


def drawing_scatter_plot_dyn_oh(species, c_est, c_ref, step = 0, alp = 0.125,  passf = None, xlabel = "Reference", ylabel = "Estimates", title = None, namef = None ):
    
    c    = torch.linspace(0, 12, 11) * 0.1 - 0.2
            
    for i in range(0,len(species)):
        
        
        if not title:
            titleS = "Concentrations of " + species[i]  + " at step " + str(step) 
        else:
            titleS = title + species[i] 
        if not namef: namefS = "scatter_plot_" + species[i]  
        
        reference = c_ref[:,step,i]
        y = c_est[:,step,i]
        
        y=y/max(abs(reference))  
        reference=reference/max(abs(reference))        
        

        yticks = np.linspace( -0.2,  np.max(y), 5).round(3)        
        xticks = np.linspace( -0.2,  np.max(reference), 5).round(3)  
        
        fig, ax = plt.subplots()
        
        ax.scatter(reference, y, alpha=alp)
        ax.plot(c, c, "r")
        ax.plot(c, c*0, "k", linestyle='--')
        ax.plot(c*0, c, "k", linestyle='--')
        
        
        ax.set_xlabel(xlabel,fontsize="30")
        ax.set_ylabel(ylabel,fontsize="30")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize="20")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize="20")
        ax.set_title(titleS, fontsize="30")  
        
        
        if passf != None:
            passmaker(passf,namefS, fig)
