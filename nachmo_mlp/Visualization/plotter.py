#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:57:25 2023

@author: andreyvlasenko
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from passmaker import passmaker
from min_max_mean_error import min_max_mean_error



def plotter(y, passf = None,title = " ", xticks = None, xlabel = None, ylabel = None, namef = None ):
     

    cells,steps  = y.shape


    if np.max(y)< 10: 
       yticks = np.linspace( np.min(y),  np.max(y), 5).round(3)


    if np.max(y)< 100 and np.max(y) >= 10: 
       yticks = np.linspace( np.min(y),  np.max(y), 5).round(2)

    if np.max(y)< 1000 and np.max(y) >= 100: 
       yticks = np.linspace( np.min(y),  np.max(y), 5).round(1)

    if np.max(y) >= 1000: 
       yticks = np.linspace( np.min(y),  np.max(y), 5).round(0)
    print("np.max(y) = ", np.max(y), title)
   
    fig, ax = plt.subplots()
    for l in range(0,len(y)):
    

        col = np.abs(y[l,0]/max(y[:,0]))
        
        if max(y[:,0]) == 0 : col = np.max(np.abs(y[l,:]))/np.max(np.abs(y[:,:])) # this is for colouring error graphs
        if col > 1 : col = 1 # this is for colouring error graphs

        plt.plot(y[l,:], color = (0,0,col))
        
        
        
        
    ax.set_title(title, fontsize="30" )
    ax.set_xlabel(xlabel, fontsize="20")
    ax.set_ylabel(ylabel, fontsize="20")
    ax.set_xticks([0,int(steps/2), steps])
    ax.set_xticklabels(xticks, fontsize="20")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize="20")
        
    if passf != None:
        passmaker(passf,namef, fig)
